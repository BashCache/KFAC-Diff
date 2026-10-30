"""Small pruning helpers shared by the compare / baseline pruning scripts."""

from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn

from metrics.diffusion_pruning_score_store import PruningScoreStore


def make_strategies(total_timesteps: int = 1000) -> List[Dict]:
    """
    Temporal score-aggregation strategies, each a dict with
        name   : label
        method : 'sum' | 'mean' | 'log_decay'
        kwargs : extra kwargs for PruningScoreStore.aggregate()
        _custom: optional tag for a custom weight / pooling function (plus _mv_lambda / _mv_mode for mean_var)
    """
    return [
        {"name": "sum", "method": "sum", "kwargs": {}},
        {"name": "mean", "method": "mean", "kwargs": {}},
        {"name": "log_decay_a0_1", "method": "log_decay", "kwargs": {"alpha_min": 0.0, "alpha_max": 1.0}},
        {"name": "log_decay_a1_0", "method": "log_decay", "kwargs": {"alpha_min": 1.0, "alpha_max": 0.0}},
        {"name": "log_decay_reversed", "method": "sum", "kwargs": {}, "_custom": "log_reversed"},
        {"name": "sqrt_decay", "method": "sum", "kwargs": {}, "_custom": "sqrt_decay"},
        {"name": "linear_ramp", "method": "sum", "kwargs": {}, "_custom": "linear_ramp"},
        {"name": "inv_linear", "method": "sum", "kwargs": {}, "_custom": "inv_linear"},
        {"name": "inv_quadratic", "method": "sum", "kwargs": {}, "_custom": "inv_quadratic"},
        {"name": "exp_decay", "method": "sum", "kwargs": {}, "_custom": "exp_decay"},
        {"name": "cosine_decay", "method": "sum", "kwargs": {}, "_custom": "cosine_decay"},
        {"name": "gaussian_bandpass", "method": "sum", "kwargs": {}, "_custom": "gaussian_bandpass"},
        {"name": "triangular_window", "method": "sum", "kwargs": {}, "_custom": "triangular_window"},
        {"name": "mask_entropy", "method": "sum", "kwargs": {}, "_custom": "mask_entropy"},
        {"name": "snr_weighted", "method": "sum", "kwargs": {}, "_custom": "snr_weighted"},
        {"name": "max_pooling", "method": "sum", "kwargs": {}, "_custom": "max_pooling"},
        {"name": "softmax_weighted", "method": "sum", "kwargs": {}, "_custom": "softmax_weighted"},
        {"name": "topk_mean_25", "method": "sum", "kwargs": {}, "_custom": "topk_mean_25"},
        {"name": "mean_plus_1p0std", "method": "sum", "kwargs": {}, "_custom": "mean_var", "_mv_lambda": 1.0, "_mv_mode": "add"},
        {"name": "mean_minus_0p5std", "method": "sum", "kwargs": {}, "_custom": "mean_var", "_mv_lambda": 0.5, "_mv_mode": "subtract"},
    ]


def zero_rows(module: nn.Module, keep_mask: torch.Tensor) -> None:
    """Zero out rows (output neurons) where keep_mask is False."""
    with torch.no_grad():
        prune = ~keep_mask.to(device=module.weight.device)
        module.weight.data[prune] = 0.0
        if module.bias is not None:
            module.bias.data[prune] = 0.0


def zero_cols(module: nn.Module, keep_mask: torch.Tensor) -> None:
    """Zero out columns (input neurons) where keep_mask is False."""
    with torch.no_grad():
        prune = ~keep_mask.to(device=module.weight.device)
        module.weight.data[:, prune] = 0.0


def clone_raw_store(store: PruningScoreStore) -> PruningScoreStore:
    """
    New PruningScoreStore with the same raw scores but empty _aggregated, so each strategy can call
    aggregate() independently.
    """
    new = PruningScoreStore()
    new._raw         = defaultdict(dict, {ln: dict(ts_dict) for ln, ts_dict in store._raw.items()})
    new._timesteps   = list(store._timesteps)
    new._layer_order = list(store._layer_order)
    return new
