# core/base_pruner.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from config.base_config import PruningConfig

logger = logging.getLogger(__name__)

class BasePruner(ABC):
    """
    Abstract base class for all pruning methods.
    """

    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model  = model
        self.config = config
        self.device = config.device
        self.state : Dict[str, Any] = {}

    def _get_arg(self, key: str, default: Any = None) -> Any:
        """
        Look `key` up in config.method_kwargs, then as a top-level PruningConfig field, then in the
        calibration / sparsity sub-configs; the first non-None value wins, else `default`.
        """
        sources = [self.config.method_kwargs, self.config, self.config.calibration, self.config.sparsity]
        for src in sources:
            val = src.get(key) if isinstance(src, dict) else getattr(src, key, None)
            if val is not None:
                return val
        return default
    #  Step 1 – Prepare                                                 
    def prepare(self) -> None:
        """
        Method-specific model preparation before calibration.
        Default: eval mode, freeze all parameters.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
    #  Step 2 – Collect                                                 
    @abstractmethod
    def collect(self, calibration_loader) -> None:
        """
        Run forward (and optionally backward) passes over calibration data.
        Store anything needed for scoring in self.state.
        """
        ...
    #  Step 3 – Compute scores                                  
    @abstractmethod
    def compute_scores(self) -> Dict[str, torch.Tensor]:
        """
        Use self.state to produce per-weight importance scores.
        """
        ...
    #  Step 4 – Apply pruning                                     
    def apply_pruning(
        self,
        scores  : Dict[str, torch.Tensor],
        strategy: Any = None,
    ) -> None:
        """
        Default implementation: compute binary mask via strategy (or thresholding), zero weights.
        """
        for name, module in self._get_prunable_layers():
            if name not in scores:
                logger.warning(f"No score found for layer '{name}'. Skipping.")
                continue
            if strategy is not None and hasattr(strategy, "compute_mask"):
                mask = strategy.compute_mask(scores[name], name)
            else:
                s = scores[name]
                sp = self._layer_sparsity(name)
                k = max(1, int(s.numel() * sp))
                thresh = torch.kthvalue(s.flatten(), k).values
                mask = s > thresh
            self._apply_mask(module, mask)
            achieved = 1.0 - mask.float().mean().item()
            logger.debug(f"  {name}: sparsity = {achieved:.2%}")
    #  Step 5 – Cleanup                                                 
    def cleanup(self) -> None:
        """Remove hooks, clear state, free GPU memory."""
        self.state.clear()
        torch.cuda.empty_cache()
        logger.debug("Pruner state cleared.")
    #  Shared helpers                                                   
    def _get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Return (name, module) pairs that are candidates for pruning,
        respecting config.layers_to_prune and config.layers_to_skip.
        """
        prunable = []
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if "lm_head" in name or "embed" in name:
                continue
            # Skip vocabulary projection layers (e.g. LLaDA output head has out_features=126464)
            if hasattr(module, "out_features") and module.out_features > 16384:
                continue
            if hasattr(module, "in_features") and module.in_features > 16384:
                continue
            if self.config.layers_to_skip and any(
                skip in name for skip in self.config.layers_to_skip
            ):
                continue
            if self.config.layers_to_prune and not any(
                target in name for target in self.config.layers_to_prune
            ):
                continue
            prunable.append((name, module))
        return prunable

    def _apply_mask(self, module: nn.Module, mask: torch.Tensor) -> None:
        """Apply boolean keep-mask to module weight in-place."""
        with torch.no_grad():
            module.weight.data *= mask.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

    def _layer_sparsity(self, layer_name: str) -> float:
        """Resolve the target sparsity for a given layer."""
        cfg = self.config.sparsity
        if cfg.scope == cfg.scope.LAYERWISE_CUSTOM:
            return cfg.layer_sparsity_map.get(layer_name, cfg.target_sparsity)
        return cfg.target_sparsity
