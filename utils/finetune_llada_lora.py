"""
LoRA Fine-tuning script for Pruned / Base LLaDA Diffusion Language Models.

Supports:
- Fine-tuning on a subset of C4, WikiText-2 or Alpaca (instruction data), or a balanced mixture
  (mixed = C4 + WikiText-2; mixed_alpaca = C4 + WikiText-2 + Alpaca).
- Dynamic timestep sampling (t ~ U(1, T)) with the linear masking ratio r(t) = t / T.
- PEFT LoRA parameter-efficient fine-tuning with configurable target modules and rank.
- Mixed precision (bfloat16 / fp16), cosine LR scheduling with warmup.
- Validation evaluation and saving adapter checkpoints or merged model weights.
"""

import os
import re
import sys
import math
import time
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, for `data`
from data.diffusion_calibration import diffusion_loss  # noqa: E402
from utils.dataloader_utils import (  # noqa: E402
    DATASET_COMPONENTS,
    TextSequenceDataset,
    format_alpaca,
    load_alpaca_samples,
    load_c4_samples,
    load_wikitext_samples,
    prepare_data_loaders,
)
from utils.model_utils import (  # noqa: E402
    compute_diffusion_loss,
    mask_tokens,
    masking_ratio,
    sanitize_auto_map,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _llada_compat_patches(model_path: str) -> None:
    """Apply compatibility shims for LLaDA dynamic modeling code."""
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def _get_tied_keys(self):
            val = getattr(self, "_all_tied_weights_keys", None)
            if val is None:
                val = getattr(self, "_tied_weights_keys", {})
            return val if isinstance(val, dict) else {}

        def _set_tied_keys(self, value):
            self._all_tied_weights_keys = value

        PreTrainedModel.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)
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

            # Fallback: inspect actual tensor shapes in model.safetensors if available
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
                                    device=c.init_device,
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
                                device=c.init_device,
                            )
                    LLaDALlamaBlock.__init__ = _patched_llama_init
                    LLaDALlamaBlock._heterogeneous_patched = True
                logger.info(f"Patched LLaDA blocks to support heterogeneous pruned intermediate sizes across {len(layer_hidden_sizes)} layers.")
    except Exception as e:
        logger.warning(f"Error applying LLaDA compat patches: {e}")
#  Evaluation Routine                                                                                                                      #
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    mask_token_id: int,
    total_timesteps: int,
    device: torch.device,
    dtype: torch.dtype,
    shift_logits: bool = False,
    protect_ends: bool = False,
) -> Dict[str, float]:
    """Evaluates diffusion loss and approximate perplexity on validation dataset."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            loss, _ = compute_diffusion_loss(
                model=model,
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                total_timesteps=total_timesteps,
                shift_logits=shift_logits,
                protect_ends=protect_ends,
            )
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(min(avg_loss, 100)) 

    return {"val_loss": avg_loss, "val_perplexity": ppl}
#  Main Training Routine                                                                                                                #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Pruned LLaDA Diffusion Models")
    
    # Model & Tokenizer
    parser.add_argument("--model_name_or_path", type=str, default="./llada_2048_64timesteps_shrunk",
                        help="Path or HuggingFace repo ID of the model to fine-tune")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Tokenizer path (defaults to model_name_or_path)")
    
    # Data configuration
    parser.add_argument("--dataset", type=str, default="mixed",
                        choices=["c4", "wikitext2", "alpaca", "mixed", "mixed_alpaca"],
                        help="Dataset to train on: c4, wikitext2, alpaca, mixed (c4 + wikitext2) or "
                             "mixed_alpaca (c4 + wikitext2 + alpaca)")
    parser.add_argument("--num_c4_samples", type=int, default=1000,
                        help="Number of C4 samples to load for training")
    parser.add_argument("--num_wiki_samples", type=int, default=1000,
                        help="Number of WikiText-2 samples to load for training")
    parser.add_argument("--num_alpaca_samples", type=int, default=1000,
                        help="Number of packed Alpaca sequences (seq_len tokens each) to load for training")
    parser.add_argument("--alpaca_name", type=str, default="tatsu-lab/alpaca",
                        help="HF dataset with instruction/input/output fields (e.g. yahma/alpaca-cleaned)")
    parser.add_argument("--num_eval_samples", type=int, default=128,
                        help="Number of validation samples to evaluate on")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for tokenized chunks")
    
    # Diffusion parameters
    parser.add_argument("--total_timesteps", type=int, default=64,
                        help="Total diffusion timesteps T (e.g. 64 or 1000)")
    
    # LoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="ff_proj,up_proj,ff_out",
                        help="Comma-separated names of linear modules to apply LoRA to")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1
    , help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every N optimizer steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N optimizer steps")
    
    # Environment & Output
    parser.add_argument("--output_dir", type=str, default="./finetuned_llada_lora", help="Output directory")
    parser.add_argument("--merge_and_save", action="store_true", help="Merge LoRA weights and save standalone model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision (recommended)")


    parser.add_argument("--lora_output_head", action=argparse.BooleanOptionalAction, default=True,
                        help="Also put LoRA on the (unpruned) output head (LLaDA transformer.ff_out / lm_head); "
                             "--no-lora_output_head restricts LoRA to the transformer blocks")

    parser.add_argument("--mask_pruned_attention", action="store_true", default=False,
                        help="Also adapt the attention projections (q/k/v and the output projection), but only on "
                             "the channels that survived pruning: the pruned all-zero rows/columns stay exactly zero "
                             "through training and merging")

    return parser


def parse_args():
    return build_parser().parse_args()


# Output head outside the blocks: LLaDA `transformer.ff_out`, Dream `lm_head`
_OUTPUT_HEAD_REGEX = r".*(?:transformer\.ff_out|lm_head)"
_ATTN_ROW_PRUNED = ("q_proj", "k_proj", "v_proj")
_ATTN_COL_PRUNED = ("attn_out", "o_proj")

def lora_target_regex(names: List[str], output_head: bool = False) -> str:
    blocks = r".*\.(?:blocks|layers)\.\d+\.(?:\w+\.)?(?:" + "|".join(re.escape(n) for n in names) + ")"
    return f"{blocks}|{_OUTPUT_HEAD_REGEX}" if output_head else blocks

def mask_pruned_attention_lora(model) -> Dict[str, Tuple[str, torch.Tensor]]:
    """
    Confine the LoRA update of the attention projections to the channels that survived pruning, so the pruned
    (all-zero) rows of q/k/v and columns of the output projection stay exactly zero through training and merging.

    dW = B @ A, with B (d_out x r) and A (r x d_in):
      q/k/v      : zero the rows of B of the pruned output channels (and mask their gradient);
      attn_out   : zero the columns of A of the pruned input channels (and mask their gradient).
    """
    masks: Dict[str, Tuple[str, torch.Tensor]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer):
            continue
        short = name.split(".")[-1]
        if short not in _ATTN_ROW_PRUNED + _ATTN_COL_PRUNED:
            continue
        w = mod.get_base_layer().weight.detach()
        adapter = next(iter(mod.lora_A))
        if short in _ATTN_ROW_PRUNED:
            pruned = (w.abs().sum(dim=1) == 0)
            target, kind = mod.lora_B[adapter].weight, "row"
            keep = (~pruned).to(target.dtype).unsqueeze(1)
            with torch.no_grad():
                target[pruned] = 0
        else:
            pruned = (w.abs().sum(dim=0) == 0)
            target, kind = mod.lora_A[adapter].weight, "col"
            keep = (~pruned).to(target.dtype).unsqueeze(0)
            with torch.no_grad():
                target[:, pruned] = 0
        target.register_hook(lambda grad, keep=keep.to(target.device): grad * keep)
        masks[name.replace("base_model.model.", "", 1)] = (kind, pruned.cpu())
    n_pruned = sum(int(m.sum()) for _, m in masks.values())
    logger.info(f"--mask_pruned_attention: masked {len(masks)} attention LoRA modules, {n_pruned} pruned channels in total.")
    if n_pruned == 0:
        logger.warning("No all-zero attention rows/columns found: the model has no pruned attention, so nothing was masked.")
    return masks

def verify_pruned_attention_zero(merged_model, masks: Dict[str, Tuple[str, torch.Tensor]]) -> int:
    """Count pruned attention channels that are no longer exactly zero in the merged model (expected: 0)."""
    modules = dict(merged_model.named_modules())
    bad = 0
    for name, (kind, pruned) in masks.items():
        mod = modules.get(name)
        if mod is None or not hasattr(mod, "weight"):
            logger.warning(f"verify_pruned_attention_zero: module {name} not found in the merged model.")
            continue
        w = mod.weight.detach().cpu()
        part = w[pruned] if kind == "row" else w[:, pruned]
        bad += int((part != 0).sum())
    if bad:
        logger.warning(f"{bad} entries of pruned attention channels are non-zero after merging.")
    else:
        logger.info("Verified: every pruned attention row/column is still exactly zero in the merged model.")
    return bad

def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    dtype = torch.float32
    if args.bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    logger.info(f"Target Device: {device} | Precision: {dtype}")

    # 1. Load Tokenizer
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Resolve mask_token_id
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = 126336  # standard LLaDA mask token ID
    logger.info(f"Using mask_token_id: {mask_token_id}")

    # 2. Load Base Model
    logger.info(f"Loading base model from {args.model_name_or_path}...")
    _llada_compat_patches(args.model_name_or_path)

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    try:
        if int(transformers.__version__.split(".")[0]) >= 5:
            load_kwargs["dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype
    except Exception:
        load_kwargs["torch_dtype"] = dtype

    load_kwargs["device_map"] = {"": device}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **load_kwargs,
    )
    base_model.config.use_cache = False

    # 3. Setup LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if args.mask_pruned_attention:
        target_modules += [m for m in ("q_proj", "k_proj", "v_proj", "attn_out") if m not in target_modules]
    logger.info(f"Applying LoRA to target modules: {target_modules}"
                + (" + output head" if args.lora_output_head else " (output head excluded)"))


    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_regex(target_modules, output_head=args.lora_output_head),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    attn_masks = mask_pruned_attention_lora(model) if args.mask_pruned_attention else None

    # 4. Prepare Datasets & Loaders
    train_loader, val_loader = prepare_data_loaders(
        dataset_name=args.dataset,
        num_c4_samples=args.num_c4_samples,
        num_wiki_samples=args.num_wiki_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        seed=args.seed,
        num_eval_samples=args.num_eval_samples,
        num_alpaca_samples=args.num_alpaca_samples,
        alpaca_name=args.alpaca_name,
    )

    # 5. Optimizer and Scheduler
    total_train_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_train_steps, 1),
    )

    logger.info(f"Total training steps: {total_train_steps} (Warmup steps: {warmup_steps})")

    # Initial Evaluation
    logger.info("Running initial zero-shot validation...")
    val_metrics = evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, dtype)
    logger.info(f"Initial Validation -> Loss: {val_metrics['val_loss']:.4f} | Perplexity: {val_metrics['val_perplexity']:.2f}")

    # 6. Training Loop
    global_step = 0
    best_val_loss = float("inf")
    model.train()
    optimizer.zero_grad()

    logger.info("Starting LoRA fine-tuning...")
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.num_epochs} ---")
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                loss, metrics = compute_diffusion_loss(
                    model=model,
                    input_ids=input_ids,
                    mask_token_id=mask_token_id,
                    total_timesteps=args.total_timesteps,
                )
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = lr_scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "avg_t": f"{metrics['avg_t']:.1f}",
                    "lr": f"{current_lr:.2e}",
                })

                # Periodic Evaluation
                if global_step % args.eval_steps == 0:
                    val_metrics = evaluate(
                        model=model,
                        val_loader=val_loader,
                        mask_token_id=mask_token_id,
                        total_timesteps=args.total_timesteps,
                        device=device,
                        dtype=dtype,
                    )
                    logger.info(
                        f"\nStep {global_step} | Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"Val PPL: {val_metrics['val_perplexity']:.2f}"
                    )

                    # Save best checkpoint
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        best_dir = os.path.join(args.output_dir, "best_checkpoint")
                        logger.info(f"Saving best adapter checkpoint to {best_dir}...")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                    model.train()

                # Periodic Saving
                if global_step % args.save_steps == 0:
                    step_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                    logger.info(f"Saving step checkpoint to {step_dir}...")
                    model.save_pretrained(step_dir)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} finished | Avg Train Loss: {avg_epoch_loss:.4f}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.2f} minutes.")

    # 7. Final Evaluation & Save
    final_val = evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, dtype)
    logger.info(f"Final Validation -> Loss: {final_val['val_loss']:.4f} | Perplexity: {final_val['val_perplexity']:.2f}")

    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    logger.info(f"Saving final LoRA adapter to {final_adapter_dir}...")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Save training summary
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset,
        "num_c4_samples": args.num_c4_samples,
        "num_wiki_samples": args.num_wiki_samples,
        "num_alpaca_samples": args.num_alpaca_samples,
        "alpaca_name": args.alpaca_name,
        "seq_len": args.seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "lora_output_head": args.lora_output_head,
        "mask_pruned_attention": args.mask_pruned_attention,
        "total_timesteps": args.total_timesteps,
        "final_val_loss": final_val["val_loss"],
        "final_val_perplexity": final_val["val_perplexity"],
        "best_val_loss": best_val_loss,
        "training_time_minutes": total_time / 60.0,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Merge weights if requested
    if args.merge_and_save:
        merged_dir = os.path.join(args.output_dir, "merged_model")
        logger.info(f"Merging LoRA weights into base model and saving to {merged_dir}...")
        merged_model = model.merge_and_unload()
        if attn_masks is not None:
            verify_pruned_attention_zero(merged_model, attn_masks)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        sanitize_auto_map(merged_dir)

        logger.info("Merged model saved successfully!")

    logger.info(f"All done! Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
