import json
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors import safe_open

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.diffusion_calibration import diffusion_loss

logger = logging.getLogger(__name__)


def _resolve_model_class(model_path: str, trust_remote_code: bool = True):
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        auto_map = getattr(cfg, "auto_map", {})
        if isinstance(auto_map, dict):
            if "AutoModelForCausalLM" not in auto_map and "AutoModel" in auto_map:
                logger.info(f"Model '{model_path}' defines AutoModel in auto_map (not AutoModelForCausalLM). Using AutoModel.")
                return AutoModel
    except Exception as e:
        logger.debug(f"Could not inspect auto_map for {model_path}: {e}")
    return AutoModelForCausalLM

def _llada_compat_patches(model_path: str) -> None:
    """Apply compatibility shims for dynamic modeling code if needed."""
    # Patch 0: Sanitize auto_map in local directory if AutoConfig and AutoModel point to different namespaces
    if os.path.isdir(model_path):
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg_dict = json.load(f)
                auto_map = cfg_dict.get("auto_map", {})
                if isinstance(auto_map, dict):
                    auto_cfg = auto_map.get("AutoConfig", "")
                    auto_model = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel", "")
                    if "--" in auto_model and "--" not in auto_cfg:
                        repo_prefix = auto_model.split("--")[0]
                        cfg_dict["auto_map"]["AutoConfig"] = f"{repo_prefix}--{auto_cfg}"
                        with open(cfg_path, "w") as f:
                            json.dump(cfg_dict, f, indent=2)
                        logger.info(f"Fixed mismatched auto_map AutoConfig in {cfg_path} with repo prefix '{repo_prefix}'.")
                    elif "--" in auto_cfg and "--" not in auto_model:
                        repo_prefix = auto_cfg.split("--")[0]
                        if "AutoModel" in auto_map:
                            cfg_dict["auto_map"]["AutoModel"] = f"{repo_prefix}--{auto_map['AutoModel']}"
                        if "AutoModelForCausalLM" in auto_map:
                            cfg_dict["auto_map"]["AutoModelForCausalLM"] = f"{repo_prefix}--{auto_map['AutoModelForCausalLM']}"
                        with open(cfg_path, "w") as f:
                            json.dump(cfg_dict, f, indent=2)
                        logger.info(f"Fixed mismatched auto_map AutoModel in {cfg_path} with repo prefix '{repo_prefix}'.")
            except Exception as e:
                logger.debug(f"Could not sanitize auto_map in {cfg_path}: {e}")

    # Patch 1: all_tied_weights_keys property (with setter for post_init compatibility)
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def _get_tied_keys(self):
            val = getattr(self, "_all_tied_weights_keys", None)
            if val is None:
                val = getattr(self, "_tied_weights_keys", {})
            return val if isinstance(val, dict) else {}

        def _set_tied_keys(self, value):
            self._all_tied_weights_keys = value

        PreTrainedModel.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)

    # Patch 2: tie_weights keyword argument compatibility & heterogeneous layer sizes
    try:
        cfg = None
        class_ref = "modeling_llada.LLaDAModelLM"
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(cfg, "auto_map") and "AutoModelForCausalLM" in cfg.auto_map:
                class_ref = cfg.auto_map["AutoModelForCausalLM"]
        except Exception:
            pass

        model_cls = get_class_from_dynamic_module(class_ref, model_path)
        if model_cls is not None:
            if hasattr(model_cls, "tie_weights"):
                orig = model_cls.tie_weights
                def _safe_tie(self, *a, **kw):
                    try:
                        return orig(self, *a, **kw)
                    except TypeError:
                        return orig(self)
                model_cls.tie_weights = _safe_tie

            # Support heterogeneous per-layer pruned intermediate sizes
            layer_hidden_sizes = {}
            if cfg is not None and hasattr(cfg, "mlp_hidden_sizes") and cfg.mlp_hidden_sizes:
                if isinstance(cfg.mlp_hidden_sizes, dict):
                    layer_hidden_sizes = {int(k): int(v) for k, v in cfg.mlp_hidden_sizes.items()}
                elif isinstance(cfg.mlp_hidden_sizes, list):
                    layer_hidden_sizes = {i: int(v) for i, v in enumerate(cfg.mlp_hidden_sizes)}

            if not layer_hidden_sizes and os.path.isdir(model_path):
                st_path = os.path.join(model_path, "model.safetensors")
                if os.path.isfile(st_path) and safe_open is not None:
                    try:
                        with safe_open(st_path, framework="pt") as sf:
                            for k in sf.keys():
                                if ".ff_out.weight" in k:
                                    parts = k.split(".")
                                    if "blocks" in parts:
                                        b_idx = int(parts[parts.index("blocks") + 1])
                                        layer_hidden_sizes[b_idx] = sf.get_tensor(k).shape[1]
                    except Exception as exc:
                        logger.debug(f"Could not inspect safetensors for layer hidden sizes: {exc}")

            if layer_hidden_sizes:
                mod = sys.modules[model_cls.__module__]
                LLaDABlock = getattr(mod, "LLaDABlock", None)
                LLaDALlamaBlock = getattr(mod, "LLaDALlamaBlock", None)

                if LLaDABlock is not None and not getattr(LLaDABlock, "_heterogeneous_patched", False):
                    orig_block_init = LLaDABlock.__init__
                    def _patched_block_init(self, layer_id, c, cache):
                        orig_block_init(self, layer_id, c, cache)
                        if layer_id in layer_hidden_sizes:
                            self.hidden_size = layer_hidden_sizes[layer_id]
                            if hasattr(self, "ff_out") and isinstance(self.ff_out, torch.nn.Linear):
                                self.ff_out = torch.nn.Linear(
                                    int(self.act.output_multiplier * self.hidden_size),
                                    c.d_model,
                                    bias=c.include_bias,
                                    device=c.init_device
                                )
                    LLaDABlock.__init__ = _patched_block_init
                    LLaDABlock._heterogeneous_patched = True

                if LLaDALlamaBlock is not None and not getattr(LLaDALlamaBlock, "_heterogeneous_patched", False):
                    orig_llama_init = LLaDALlamaBlock.__init__
                    def _patched_llama_init(self, layer_id, c, cache):
                        orig_llama_init(self, layer_id, c, cache)
                        if layer_id in layer_hidden_sizes:
                            h = layer_hidden_sizes[layer_id]
                            self.hidden_size = h
                            self.ff_proj = torch.nn.Linear(c.d_model, h, bias=c.include_bias, device=c.init_device)
                            self.up_proj = torch.nn.Linear(c.d_model, h, bias=c.include_bias, device=c.init_device)
                            self.ff_out = torch.nn.Linear(
                                int(self.act.output_multiplier * h),
                                c.d_model,
                                bias=c.include_bias,
                                device=c.init_device
                            )
                    LLaDALlamaBlock.__init__ = _patched_llama_init
                    LLaDALlamaBlock._heterogeneous_patched = True
                logger.info(f"Patched LLaDA blocks to support heterogeneous pruned intermediate sizes across {len(layer_hidden_sizes)} layers.")
    except Exception as exc:
        logger.warning(f"Error in _llada_compat_patches: {exc}")

def load_llada_model_and_tokenizer(
    model_path_or_config,
    dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = True,
):
    """
    Dedicated loader for LLaDA discrete diffusion LM with all dynamic module compatibility patches,
    use_cache config initialization, and mask_token_id resolution.
    Accepts either a string (model path/HF ID) or a PruningConfig object.
    """
    if hasattr(model_path_or_config, "model_name_or_path"):
        model_path = model_path_or_config.model_name_or_path
        device = getattr(model_path_or_config, "device", device)
        dtype = getattr(model_path_or_config, "dtype", dtype)
    else:
        model_path = str(model_path_or_config)

    _llada_compat_patches(model_path)

    torch_dtype = getattr(torch, dtype, torch.bfloat16)

    logger.info(f"Loading LLaDA tokenizer from '{model_path}' ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading LLaDA model from '{model_path}' ({dtype}) on {device} ...")
    model_cls = _resolve_model_class(model_path, trust_remote_code=trust_remote_code)
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )

    # LLaDA config compatibility: use_cache may be omitted in config
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Resolve mask_token_id
    mask_token_id = (
        getattr(model.config, "mask_token_id", None)
        or getattr(tokenizer, "mask_token_id", None)
        or 126336
    )
    tokenizer.mask_token_id = mask_token_id

    model.eval()
    logger.info(
        f"LLaDA Model loaded. Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B | mask_token_id={mask_token_id}"
    )
    return model, tokenizer


def load_model_and_tokenizer(config):
    logger.info(f"Loading model: {config.model_name_or_path}")
    _llada_compat_patches(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, config.dtype, torch.float16)
    model_cls = _resolve_model_class(config.model_name_or_path, trust_remote_code=True)
    model = model_cls.from_pretrained(
        config.model_name_or_path,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(
        f"Model loaded. Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
    )
    return model, tokenizer


def save_model(model: nn.Module, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


def sanitize_auto_map(model_dir: str) -> None:
    """Fix auto_map in a saved config.json to prevent a dynamic-module namespace mismatch on reload."""
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg_dict = json.load(f)
            auto_map = cfg_dict.get("auto_map", {})
            if isinstance(auto_map, dict):
                auto_cfg = auto_map.get("AutoConfig", "")
                auto_model = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel", "")
                if "--" in auto_model and "--" not in auto_cfg:
                    repo_prefix = auto_model.split("--")[0]
                    cfg_dict["auto_map"]["AutoConfig"] = f"{repo_prefix}--{auto_cfg}"
                    with open(cfg_path, "w") as f:
                        json.dump(cfg_dict, f, indent=2)
        except Exception as exc:
            logging.getLogger(__name__).warning(f"Could not sanitize auto_map in {cfg_path}: {exc}")


# Output heads: LLaDA `model.transformer.ff_out` (same short name as the block MLP output), Dream / HF `lm_head`.
_OUTPUT_HEAD_REGEX = r"(?:.*\.)?transformer\.ff_out|(?:.*\.)?lm_head"


# ======================================================================== #
#  Diffusion Masking & Loss                                                #
# ======================================================================== #

def masking_ratio(t: int, total_timesteps: int = 64) -> float:
    """Ratio r(t) = t / T of tokens to mask at timestep t in [1, total_timesteps] (linear schedule)."""
    return t / total_timesteps


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    t: int,
    total_timesteps: int = 64,
    generator: Optional[torch.Generator] = None,
    protect_ends: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrupts clean token sequences by masking a fraction r(t) with mask_token_id.
    protect_ends keeps the first and last position unmasked (Dream's convention).

    Returns:
        noisy_ids: corrupted tensor (B, L)
        mask: boolean tensor (B, L) where True indicates masked positions
    """
    r = masking_ratio(t, total_timesteps)
    rand = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask = rand < r
    if protect_ends:
        mask[:, 0] = False
        mask[:, -1] = False

    # Ensure at least 1 token is masked if r > 0 to avoid zero-loss degenerate batches
    if r > 0 and not mask.any():
        lo, hi = (1, input_ids.shape[1] - 1) if protect_ends else (0, input_ids.shape[1])
        rand_idx = torch.randint(lo, hi, (input_ids.shape[0], 1), device=input_ids.device)
        mask.scatter_(1, rand_idx, True)

    noisy_ids = input_ids.clone()
    noisy_ids[mask] = mask_token_id
    return noisy_ids, mask


def compute_diffusion_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    mask_token_id: int,
    total_timesteps: int = 64,
    generator: Optional[torch.Generator] = None,
    shift_logits: bool = False,
    protect_ends: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Samples diffusion timesteps uniformly t ~ U(1, T), applies masking, 
    and computes Cross-Entropy loss strictly on the masked token positions.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Sample random timestep t for the batch (or per sequence)
    # Sampling per-sequence gives richer gradient signals across timesteps
    t_samples = torch.randint(1, total_timesteps + 1, (batch_size,), device=device, generator=generator)

    noisy_batch = []
    mask_batch = []
    for b in range(batch_size):
        t_val = int(t_samples[b].item())
        seq = input_ids[b:b+1]
        noisy_seq, mask_seq = mask_tokens(
            seq,
            mask_token_id=mask_token_id,
            t=t_val,
            total_timesteps=total_timesteps,
            generator=generator,
            protect_ends=protect_ends,
        )
        noisy_batch.append(noisy_seq)
        mask_batch.append(mask_seq)

    noisy_ids = torch.cat(noisy_batch, dim=0)
    mask = torch.cat(mask_batch, dim=0)

    loss = diffusion_loss(model, noisy_ids, input_ids, mask, shift_logits=shift_logits)

    metrics = {
        "loss": loss.item(),
        "avg_t": t_samples.float().mean().item(),
        "mask_ratio": mask.float().mean().item(),
    }
    return loss, metrics

