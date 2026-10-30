import json
import logging
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from utils.model_utils import sanitize_auto_map

logger = logging.getLogger(__name__)


def dream_compat_patches(model_path: str) -> None:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(cfg, "intermediate_sizes", None):
        return
    model_cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], model_path)
    module = sys.modules[model_cls.__module__]
    layer_cls = getattr(module, "DreamDecoderLayer")
    if getattr(layer_cls, "_heterogeneous_patched", False):
        return
    orig_init = layer_cls.__init__

    def patched_init(self, config, layer_idx, *args, **kwargs):
        orig_init(self, config, layer_idx, *args, **kwargs)
        sizes = getattr(config, "intermediate_sizes", None)
        if sizes and int(sizes[layer_idx]) != self.mlp.intermediate_size:
            h, d = int(sizes[layer_idx]), self.mlp.hidden_size
            self.mlp.intermediate_size = h
            self.mlp.gate_proj = nn.Linear(d, h, bias=False)
            self.mlp.up_proj = nn.Linear(d, h, bias=False)
            self.mlp.down_proj = nn.Linear(h, d, bias=False)

    layer_cls.__init__ = patched_init
    layer_cls._heterogeneous_patched = True
    logger.info(f"Patched Dream decoder layers to support per-layer MLP widths ({len(cfg.intermediate_sizes)} layers).")


def _point_auto_map_at_source_repo(output_dir: str, source: str) -> None:
    cfg_path = os.path.join(output_dir, "config.json")
    if not source or os.path.exists(source) or not os.path.isfile(cfg_path):
        return 
    with open(cfg_path) as f:
        cfg = json.load(f)
    auto_map = cfg.get("auto_map")
    if isinstance(auto_map, dict):
        cfg["auto_map"] = {k: (v if "--" in v else f"{source}--{v}") for k, v in auto_map.items()}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


def save_shrunk_dream(model: nn.Module, tokenizer, output_dir: str) -> List[int]:
    """
    Save `model` to `output_dir` with every MLP channel that is zero in gate_proj, up_proj and down_proj removed.
    A channel is kept if any of its three weights is non-zero, so the shrunk model computes exactly what the
    zero-masked model does.  Returns the new MLP width of every layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    state = model.state_dict()
    n_layers = model.config.num_hidden_layers
    keep: Dict[int, torch.Tensor] = {}
    for i in range(n_layers):
        p = f"model.layers.{i}.mlp."
        alive = (state[p + "gate_proj.weight"].abs().sum(dim=1) > 0) \
              | (state[p + "up_proj.weight"].abs().sum(dim=1) > 0) \
              | (state[p + "down_proj.weight"].abs().sum(dim=0) > 0)
        keep[i] = torch.where(alive)[0]
    sizes = [int(keep[i].numel()) for i in range(n_layers)]

    new_state: Dict[str, torch.Tensor] = {}
    for key, w in state.items():
        parts = key.split(".")
        if len(parts) > 4 and parts[1] == "layers" and parts[3] == "mlp":
            idx = keep[int(parts[2])].to(w.device)
            if parts[4] in ("gate_proj", "up_proj"):
                w = w[idx, :]
            elif parts[4] == "down_proj":
                w = w[:, idx]
        new_state[key] = w.detach().cpu().contiguous()

    old = (getattr(model.config, "intermediate_size", None), getattr(model.config, "intermediate_sizes", None))
    model.config.intermediate_size, model.config.intermediate_sizes = max(sizes), sizes
    try:
        model.config.save_pretrained(output_dir)
    finally:
        model.config.intermediate_size, model.config.intermediate_sizes = old
        if old[1] is None:
            del model.config.intermediate_sizes
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _point_auto_map_at_source_repo(output_dir, getattr(model.config, "_name_or_path", None))
    sanitize_auto_map(output_dir)

    for fname in os.listdir(output_dir):
        if (fname.startswith("model") and fname.endswith(".safetensors")) or fname in (
                "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"):
            os.remove(os.path.join(output_dir, fname))
    save_file(new_state, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})

    full = model.config.intermediate_size if old[0] is None else old[0]
    kept = sum(sizes)
    logger.info(f"Shrunk Dream MLPs: {kept}/{full * n_layers} channels kept "
                f"({100 * (1 - kept / (full * n_layers)):.1f}% removed), widths {min(sizes)}..{max(sizes)} per layer "
                f"-> {output_dir}")
    return sizes
