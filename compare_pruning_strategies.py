"""
compare_pruning_strategies.py

Compare multiple score-aggregation strategies for Kronecker-OBD structural
pruning of LLaDA, using *pre-computed* KFAC statistics.

Inputs
------
  --kfac_store  : .pt file with raw {layer -> {timestep -> {diag_G, diag_WAW, ...}}}
  --scores_pkl  : optional .pkl PruningScoreStore with pre-computed raw per-timestep scores
  --model       : HuggingFace id or local path to LLaDA checkpoint

For each aggregation strategy, the script:
  1. Re-aggregates scores from raw per-timestep data
  2. Applies coupled structured pruning (same logic as KFACOBDDiffusionPruner)
  3. Evaluates perplexity on wikitext2 + c4
  4. Restores weights for the next strategy

Usage
-----
python compare_pruning_strategies.py \\
    --model           ./llada_pruned_2048_64timesteps \\
    --kfac_store      kfac_store_activations_gradients_backup_2048_64timesteps.pt \\
    --scores_pkl      llada_pruned_2048_64timesteps/pruning_scores.pkl \\
    --sparsity        0.3 \\
    --total_timesteps 1000 \\
    --output_dir      ./strategy_comparison_results \\
    --device          cuda
"""

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics.diffusion_pruning_score_store import PruningScoreStore
from utils.pruning_utils import clone_raw_store, zero_cols, zero_rows
from utils.sparsity import measure_layer_sparsity, measure_model_sparsity, measure_sparsity
from utils.model_utils import load_llada_model_and_tokenizer
from evaluation.eval_diffusion import evaluate_perplexity
from methods.kfac_obd_diffusion import (
    _kfac_obd_row_scores,
    _kfac_obd_col_scores,
    UP_PROJ_KEYWORDS,
    DOWN_PROJ_KEYWORDS,
    ATTN_Q_KEYWORDS,
    ATTN_K_KEYWORDS,
    ATTN_V_KEYWORDS,
    ATTN_OUT_KEYWORDS,
    ALL_BLOCK_KEYWORDS,
    _is_up_proj_layer,
    _is_down_proj_layer,
    _is_q_proj_layer,
    _is_k_proj_layer,
    _is_v_proj_layer,
    _is_attn_out_layer,
    _get_block_prefix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ======================================================================== #
#  Strategy registry                                                          #
# ======================================================================== #

def make_strategies(total_timesteps: int = 1000) -> List[Dict]:
    """
    Temporal score-aggregation strategies compared in this script.
    """
    return [
        {"name": "log_decay_reversed", "method": "sum", "kwargs": {}, "_custom": "log_reversed"},
        {"name": "inv_linear", "method": "sum", "kwargs": {}, "_custom": "inv_linear"},
        {"name": "snr_weighted", "method": "sum", "kwargs": {}, "_custom": "snr_weighted"},
        {"name": "linear_ramp", "method": "sum", "kwargs": {}, "_custom": "linear_ramp"},
        {"name": "mean_plus_1p0std", "method": "sum", "kwargs": {}, "_custom": "mean_var", "_mv_lambda": 1.0, "_mv_mode": "add"},
        {"name": "max_pooling", "method": "sum", "kwargs": {}, "_custom": "max_pooling"},
    ]


def _build_custom_weights(
    tag           : str,
    timesteps     : List[int],
    total_timesteps: int,
) -> Dict[int, float]:
    """Build custom per-timestep weight dict for non-standard aggregations."""
    T = float(total_timesteps)
    weights: Dict[int, float] = {}

    if tag == "log_reversed":
        for t in timesteps:
            weights[t] = math.log(float(t) + 1.0)
    elif tag == "linear_ramp":
        T_max = float(max(timesteps)) if timesteps else T
        for t in timesteps:
            weights[t] = float(t) / T_max
    elif tag == "inv_linear":
        T_max = float(max(timesteps)) if timesteps else T
        for t in timesteps:
            weights[t] = max(0.0, 1.0 - float(t) / T_max)
    elif tag == "snr_weighted":
        T_max = float(max(timesteps)) if timesteps else T
        for t in timesteps:
            weights[t] = max(0.0, float(T_max - t) / (float(t) + 10.0))
    else:
        raise ValueError(f"Unknown custom weight tag: {tag!r}")

    return weights


# ======================================================================== #
#  Score computation from raw KFAC store                                     #
# ======================================================================== #

def _compute_raw_scores_from_kfac(
    kfac_store     : Dict,
    model          : nn.Module,
    prunable_layers: List[Tuple[str, nn.Module]],
    timestep_list  : List[int],
) -> PruningScoreStore:
    """
    Compute per-layer, per-timestep Kronecker-OBD row/column scores from the
    raw KFAC store and populate a fresh PruningScoreStore.
    """
    store = PruningScoreStore()

    for name, module in prunable_layers:
        W = module.weight.data.float()
        ts_store = kfac_store.get(name, {})

        if not ts_store:
            logger.warning(f"  No KFAC data for '{name}' – skipping.")
            continue

        is_col_score = _is_down_proj_layer(name) or _is_attn_out_layer(name)

        n_scored = 0
        for t in timestep_list:
            stats = ts_store.get(t)
            if not stats or stats.get("diag_G") is None:
                continue

            score = _kfac_obd_col_scores(W, stats) if is_col_score else _kfac_obd_row_scores(W, stats)
            store.add(name, t, score)
            n_scored += 1

        logger.debug(f"  {name}: scored at {n_scored}/{len(timestep_list)} timesteps")

    logger.info(f"  Raw scores computed for {len(store.layer_names)} layers.")
    return store


# ======================================================================== #
#  Mean-Variance aggregation                                                 #
# ======================================================================== #

def _aggregate_mean_variance(
    store          : PruningScoreStore,
    lam            : float = 1.0,
    mode           : str   = "subtract",
) -> Dict[str, torch.Tensor]:
    """
    Aggregate per-timestep scores using mean ± λ·std combination.

    For each layer and each neuron/element i:
        s_i = mu(score_i)  ±  λ · std(score_i)
    """
    aggregated: Dict[str, torch.Tensor] = {}

    for layer_name in store.layer_names:
        ts_scores = store._raw.get(layer_name, {})
        if not ts_scores:
            continue

        sorted_ts = sorted(ts_scores.keys())
        stacked = torch.stack(
            [ts_scores[t] for t in sorted_ts], dim=0
        ).float()  # (T, n_neurons)  or  (T, d_out, d_in)

        mu  = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        if mode == "subtract":
            agg = mu - lam * std
        elif mode == "add":
            agg = mu + lam * std
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'subtract' or 'add'.")

        agg = agg - agg.min()
        aggregated[layer_name] = agg

    logger.info(
        f"[MeanVar] Aggregated {len(aggregated)} layers  "
        f"(λ={lam}, mode={mode})"
    )
    return aggregated


def _aggregate_custom_non_linear(
    store: PruningScoreStore,
    tag  : str,
) -> Dict[str, torch.Tensor]:
    """Aggregate across timesteps using non-linear pooling (max-pooling)."""
    aggregated: Dict[str, torch.Tensor] = {}

    for layer_name in store.layer_names:
        ts_scores = store._raw.get(layer_name, {})
        if not ts_scores:
            continue

        stacked = torch.stack(
            [ts_scores[t] for t in sorted(ts_scores.keys())], dim=0
        ).float()  # (T, n_neurons) or (T, d_out, d_in)

        if tag == "max_pooling":
            agg = stacked.max(dim=0).values
        else:
            agg = stacked.mean(dim=0)

        aggregated[layer_name] = agg

    logger.info(f"[NonLinearAgg] Aggregated {len(aggregated)} layers using '{tag}'")
    return aggregated


# ======================================================================== #
#  Aggregation (router)                                                      #
# ======================================================================== #

def _aggregate_store(
    store          : PruningScoreStore,
    strategy_cfg   : Dict,
    total_timesteps: int,
) -> Dict[str, torch.Tensor]:
    """Apply one aggregation strategy to a fresh score store copy."""
    method     = strategy_cfg["method"]
    extra      = strategy_cfg.get("kwargs", {})
    custom_tag = strategy_cfg.get("_custom")

    # ── Mean-Variance strategies get their own aggregator ─────────────── #
    if custom_tag == "mean_var":
        return _aggregate_mean_variance(
            store,
            lam            = strategy_cfg.get("_mv_lambda", 1.0),
            mode           = strategy_cfg.get("_mv_mode",   "subtract"),
        )

    # ── Non-linear aggregations get their own aggregator ──────────────── #
    if custom_tag == "max_pooling":
        return _aggregate_custom_non_linear(store, custom_tag)

    custom_weights: Optional[Dict[int, float]] = None
    if custom_tag is not None:
        custom_weights = _build_custom_weights(custom_tag, store.timesteps, total_timesteps)

    store.aggregate(
        method          = method,
        total_timesteps = total_timesteps,
        custom_weights  = custom_weights,
        **extra,
    )
    return store.all_aggregated()


# ======================================================================== #
#  Pruning application                                                                                                                    #
# ======================================================================== #

_LayerList = List[Tuple[str, nn.Module]]


def _row_scores_1d(s: torch.Tensor) -> torch.Tensor:
    """Collapse a 2-D score matrix to per-row (output unit) scores."""
    return s.mean(dim=1) if s.dim() == 2 else s


def _col_scores_1d(s: torch.Tensor) -> torch.Tensor:
    """Collapse a 2-D score matrix to per-column (input unit) scores."""
    return s.mean(dim=0) if s.dim() == 2 else s


def _topk_keep_mask(scores_1d: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask: prune ceil(n * sparsity) lowest-scoring units, keeping at least one."""
    n_units = scores_1d.numel()
    n_prune = max(1, int(math.ceil(n_units * sparsity)))
    keep_count = max(1, n_units - n_prune)
    top_idx = torch.topk(scores_1d, k=keep_count, largest=True).indices
    mask = torch.zeros(n_units, dtype=torch.bool, device=scores_1d.device)
    mask[top_idx] = True
    return mask


def _zero_and_record(
    layers: _LayerList,
    mask: torch.Tensor,
    axis: str,
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
    truncate: bool = False,
) -> None:
    """Zero rows (axis='row') or columns (axis='col') of each layer and record it as pruned."""
    for name, module in layers:
        if axis == "row":
            zero_rows(module, mask[: module.weight.shape[0]] if truncate else mask)
        else:
            zero_cols(module, mask)
        processed.add(name)
        resolved_sparsities[name] = target_sparsity


def _group_layers_by_block(prunable_layers: _LayerList) -> Dict[str, _LayerList]:
    """Group prunable layers by transformer block prefix."""
    blocks: Dict[str, _LayerList] = defaultdict(list)
    for name, module in prunable_layers:
        blocks[_get_block_prefix(name)].append((name, module))
    return blocks


def _couple_mlp_scores(
    u_1d: List[torch.Tensor],
    d_1d: List[torch.Tensor],
    coupling_mode: str,
) -> Optional[torch.Tensor]:
    """Combine up/gate row scores and down column scores into one score per intermediate channel."""
    if coupling_mode == "sum":
        return torch.stack(u_1d).sum(dim=0) + torch.stack(d_1d).sum(dim=0)

    avg_up = torch.stack(u_1d).mean(dim=0)
    avg_down = torch.stack(d_1d).mean(dim=0)
    if avg_up.shape == avg_down.shape:
        if coupling_mode == "geometric":
            return torch.sqrt(torch.clamp(avg_up * avg_down, min=1e-12))
        return 0.5 * (avg_up + avg_down)
    return None


def _prune_mlp_block(
    block_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    target_sparsity: float,
    coupling_mode: str,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Coupled channel pruning: rows of up/gate projections and columns of the down projection."""
    up_layers   = [(n, m) for n, m in block_layers if _is_up_proj_layer(n)]
    down_layers = [(n, m) for n, m in block_layers if _is_down_proj_layer(n)]
    if not (up_layers and down_layers):
        return

    all_up   = [scores.get(n) for n, _ in up_layers   if scores.get(n) is not None]
    all_down = [scores.get(n) for n, _ in down_layers if scores.get(n) is not None]
    if not (all_up and all_down):
        return

    coupled = _couple_mlp_scores(
        [_row_scores_1d(s) for s in all_up],
        [_col_scores_1d(s) for s in all_down],
        coupling_mode,
    )
    if coupled is None:
        return
    mask_1d = _topk_keep_mask(coupled, target_sparsity)

    _zero_and_record(up_layers, mask_1d, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(down_layers, mask_1d, "col", target_sparsity, processed, resolved_sparsities)


def _compute_head_damages(
    q_score_1d: torch.Tensor,
    o_score_1d: torch.Tensor,
    k_score_1d: Optional[torch.Tensor],
    v_score_1d: Optional[torch.Tensor],
    n_heads: int,
    head_dim: int,
    g: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-query-head damage: Q + O slice sums plus shared K/V slice sums split across the g heads in each group."""
    head_damages = torch.zeros(n_heads, device=device, dtype=dtype)
    for h_idx in range(n_heads):
        c_start = h_idx * head_dim
        c_end = (h_idx + 1) * head_dim
        dq = q_score_1d[c_start:c_end].sum()
        do = o_score_1d[c_start:c_end].sum()

        h_kv = h_idx // g
        kv_start = h_kv * head_dim
        kv_end = (h_kv + 1) * head_dim

        dk = k_score_1d[kv_start:kv_end].sum() / float(g) if k_score_1d is not None else 0.0
        dv = v_score_1d[kv_start:kv_end].sum() / float(g) if v_score_1d is not None else 0.0

        head_damages[h_idx] = dq + dk + dv + do
    return head_damages


def _prune_attn_heads(
    model: nn.Module,
    q_layers: _LayerList,
    k_layers: _LayerList,
    v_layers: _LayerList,
    o_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    q_score: torch.Tensor,
    q_score_1d: torch.Tensor,
    o_score_1d: torch.Tensor,
    d_model: int,
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Head-wise (GQA-aware) attention pruning."""
    cfg = getattr(model, "config", None)
    n_heads = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 32)))
    n_kv_heads = getattr(cfg, "n_kv_heads", getattr(cfg, "num_key_value_heads", n_heads))
    if n_kv_heads is None:
        n_kv_heads = n_heads

    head_dim = d_model // n_heads
    g = max(1, n_heads // n_kv_heads)

    k_score = scores.get(k_layers[0][0]) if k_layers else None
    v_score = scores.get(v_layers[0][0]) if v_layers else None
    k_score_1d = _row_scores_1d(k_score) if k_score is not None else None
    v_score_1d = _row_scores_1d(v_score) if v_score is not None else None

    head_damages = _compute_head_damages(
        q_score_1d, o_score_1d, k_score_1d, v_score_1d,
        n_heads, head_dim, g, q_score.device, q_score.dtype,
    )
    head_keep_mask = _topk_keep_mask(head_damages, target_sparsity)

    q_o_mask_1d = head_keep_mask.repeat_interleave(head_dim)

    kv_head_keep_mask = torch.zeros(n_kv_heads, dtype=torch.bool, device=q_score.device)
    for h_kv in range(n_kv_heads):
        kv_head_keep_mask[h_kv] = head_keep_mask[h_kv * g : (h_kv + 1) * g].any()
    kv_mask_1d = kv_head_keep_mask.repeat_interleave(head_dim)

    _zero_and_record(q_layers, q_o_mask_1d, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(k_layers, kv_mask_1d, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(v_layers, kv_mask_1d, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(o_layers, q_o_mask_1d, "col", target_sparsity, processed, resolved_sparsities)


def _prune_attn_channels(
    q_layers: _LayerList,
    k_layers: _LayerList,
    v_layers: _LayerList,
    o_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    q_score: torch.Tensor,
    q_score_1d: torch.Tensor,
    o_score_1d: torch.Tensor,
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Channel-wise coupled pruning across Q, K, V output rows and O input columns."""
    k_score = scores.get(k_layers[0][0]) if k_layers else None
    v_score = scores.get(v_layers[0][0]) if v_layers else None
    k_score_1d = _row_scores_1d(k_score) if k_score is not None else None
    v_score_1d = _row_scores_1d(v_score) if v_score is not None else None

    channel_damages = q_score_1d + o_score_1d
    if k_score_1d is not None and k_score_1d.shape == q_score_1d.shape:
        channel_damages += k_score_1d
    if v_score_1d is not None and v_score_1d.shape == q_score_1d.shape:
        channel_damages += v_score_1d

    mask_1d = _topk_keep_mask(channel_damages, target_sparsity)

    _zero_and_record(q_layers, mask_1d, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(k_layers, mask_1d, "row", target_sparsity, processed, resolved_sparsities, truncate=True)
    _zero_and_record(v_layers, mask_1d, "row", target_sparsity, processed, resolved_sparsities, truncate=True)
    _zero_and_record(o_layers, mask_1d, "col", target_sparsity, processed, resolved_sparsities)


def _prune_attn_block(
    model: nn.Module,
    block_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    target_sparsity: float,
    attn_coupling_mode: str,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Coupled attention pruning for one block, dispatching to head- or channel-wise mode."""
    if attn_coupling_mode not in ("head", "channel"):
        return

    q_layers = [(n, m) for n, m in block_layers if _is_q_proj_layer(n)]
    k_layers = [(n, m) for n, m in block_layers if _is_k_proj_layer(n)]
    v_layers = [(n, m) for n, m in block_layers if _is_v_proj_layer(n)]
    o_layers = [(n, m) for n, m in block_layers if _is_attn_out_layer(n)]
    if not (q_layers and o_layers):
        return

    q_score = scores.get(q_layers[0][0])
    o_score = scores.get(o_layers[0][0])
    if q_score is None or o_score is None:
        return

    q_score_1d = _row_scores_1d(q_score)
    o_score_1d = _col_scores_1d(o_score)
    d_model = q_score_1d.numel()

    if attn_coupling_mode == "head":
        _prune_attn_heads(
            model, q_layers, k_layers, v_layers, o_layers, scores,
            q_score, q_score_1d, o_score_1d, d_model,
            target_sparsity, processed, resolved_sparsities,
        )
    else:
        _prune_attn_channels(
            q_layers, k_layers, v_layers, o_layers, scores,
            q_score, q_score_1d, o_score_1d,
            target_sparsity, processed, resolved_sparsities,
        )


def _prune_unpaired_layers(
    prunable_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Fallback: attention and unpaired layers row pruning."""
    for name, module in prunable_layers:
        if name in processed:
            continue
        agg = scores.get(name)
        if agg is None:
            continue
        mask_1d = _topk_keep_mask(_row_scores_1d(agg), target_sparsity)
        zero_rows(module, mask_1d)
        resolved_sparsities[name] = target_sparsity


def _apply_coupled_structured_pruning(
    model          : nn.Module,
    scores         : Dict[str, torch.Tensor],
    prunable_layers: List[Tuple[str, nn.Module]],
    target_sparsity: float,
    coupling_mode  : str = "geometric",
    attn_coupling_mode: str = "head",
) -> Dict[str, float]:
    """
    Apply coupled row/column structured pruning.
    Supports arithmetic, geometric (default), and sum coupling for MLP,
    and 'head', 'channel', or 'none' for Attention.
    Weights are zeroed in-place.

    Returns a dict mapping layer_name -> resolved_target_sparsity.
    """
    resolved_sparsities: Dict[str, float] = {}
    processed: set = set()

    for block_layers in _group_layers_by_block(prunable_layers).values():
        _prune_mlp_block(block_layers, scores, target_sparsity, coupling_mode, processed, resolved_sparsities)
        _prune_attn_block(model, block_layers, scores, target_sparsity, attn_coupling_mode, processed, resolved_sparsities)

    _prune_unpaired_layers(prunable_layers, scores, target_sparsity, processed, resolved_sparsities)
    return resolved_sparsities


# ======================================================================== #
#  Global sparsity pruning                                                   #
# ======================================================================== #

def _apply_global_sparsity_pruning(
    model          : nn.Module,
    scores         : Dict[str, torch.Tensor],
    prunable_layers: List[Tuple[str, nn.Module]],
    global_sparsity: float,
    coupling_mode  : str = "geometric",
    attn_coupling_mode: str = "head",
) -> Dict[str, float]:
    """
    Apply coupled structured pruning using a **single global score threshold**.

    Key design: the threshold is computed from the EFFECTIVE score of each
    pruning decision — the coupled MLP score, per-head aggregated attention score, 
    or individual layer score for unpaired layers. 
    """
    resolved_sparsities: Dict[str, float] = {}

    # ── Build 1-D score vectors for every layer ───────────────────────── #
    layer_scores_1d: Dict[str, torch.Tensor] = {}
    for name, _ in prunable_layers:
        agg = scores.get(name)
        if agg is None:
            continue
        is_col = _is_down_proj_layer(name) or _is_attn_out_layer(name)
        score_1d = agg.mean(dim=0).float() if (agg.dim() == 2 and is_col) \
               else agg.mean(dim=1).float() if agg.dim() == 2 \
               else agg.float()
        layer_scores_1d[name] = score_1d

    if not layer_scores_1d:
        logger.warning("[GlobalSparsity] No scores available — skipping pruning.")
        return resolved_sparsities

    # ── Model config (needed for attention head coupling) ─────────────── #
    cfg        = getattr(model, "config", None)
    n_heads    = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 32)))
    n_kv_heads = getattr(cfg, "n_kv_heads", getattr(cfg, "num_key_value_heads", n_heads)) or n_heads
    g          = max(1, n_heads // n_kv_heads)

    # ── Build block groups (same logic as coupled pruning) ────────────── #
    blocks: Dict[str, List] = defaultdict(list)
    for name, module in prunable_layers:
        blocks[_get_block_prefix(name)].append((name, module))

    # ─────────────────────────────────────────────────────────────────── #
    #  PASS 1: Collect effective scores per pruning group                  #
    # ─────────────────────────────────────────────────────────────────── #
    group_records: List[Dict] = []
    accounted: set = set()

    for prefix, block_layers in blocks.items():
        up_layers   = [(n, m) for n, m in block_layers if _is_up_proj_layer(n)]
        down_layers = [(n, m) for n, m in block_layers if _is_down_proj_layer(n)]

        # ── MLP group ────────────────────────────────────────────────── #
        if up_layers and down_layers:
            all_up   = [layer_scores_1d[n] for n, _ in up_layers   if n in layer_scores_1d]
            all_down = [layer_scores_1d[n] for n, _ in down_layers if n in layer_scores_1d]

            if all_up and all_down:
                if coupling_mode == "sum":
                    eff = torch.stack(all_up).sum(dim=0) + torch.stack(all_down).sum(dim=0)
                else:
                    avg_up   = torch.stack(all_up).mean(dim=0)
                    avg_down = torch.stack(all_down).mean(dim=0)
                    if avg_up.shape == avg_down.shape:
                        if coupling_mode == "geometric":
                            eff = torch.sqrt(torch.clamp(avg_up * avg_down, min=1e-12))
                        else:
                            eff = 0.5 * (avg_up + avg_down)
                    else:
                        eff = torch.stack(all_up).mean(dim=0)

                group_records.append({
                    "type"       : "mlp",
                    "prefix"     : prefix,
                    "eff_score"  : eff,
                    "up_layers"  : up_layers,
                    "down_layers": down_layers,
                })
                for n, _ in up_layers + down_layers:
                    accounted.add(n)

        # ── Attention group ───────────────────────────────────────────── #
        if attn_coupling_mode in ("head", "channel"):
            q_layers = [(n, m) for n, m in block_layers if _is_q_proj_layer(n)]
            k_layers = [(n, m) for n, m in block_layers if _is_k_proj_layer(n)]
            v_layers = [(n, m) for n, m in block_layers if _is_v_proj_layer(n)]
            o_layers = [(n, m) for n, m in block_layers if _is_attn_out_layer(n)]

            if q_layers and o_layers:
                q_n = q_layers[0][0]
                o_n = o_layers[0][0]
                q_s = layer_scores_1d.get(q_n)
                o_s = layer_scores_1d.get(o_n)

                if q_s is not None and o_s is not None:
                    d_model  = q_s.numel()
                    head_dim = d_model // n_heads

                    if attn_coupling_mode == "head":
                        k_s = layer_scores_1d.get(k_layers[0][0]) if k_layers else None
                        v_s = layer_scores_1d.get(v_layers[0][0]) if v_layers else None

                        head_eff = torch.zeros(n_heads, device=q_s.device, dtype=q_s.dtype)
                        for h in range(n_heads):
                            c_s, c_e   = h * head_dim, (h + 1) * head_dim
                            h_kv       = h // g
                            kv_s, kv_e = h_kv * head_dim, (h_kv + 1) * head_dim
                            head_eff[h] = (
                                q_s[c_s:c_e].mean() + o_s[c_s:c_e].mean()
                                + (k_s[kv_s:kv_e].mean() / g if k_s is not None else 0.0)
                                + (v_s[kv_s:kv_e].mean() / g if v_s is not None else 0.0)
                            )

                        group_records.append({
                            "type"    : "head",
                            "prefix"  : prefix,
                            "eff_score": head_eff,
                            "head_dim": head_dim,
                            "q_layers": q_layers,
                            "k_layers": k_layers,
                            "v_layers": v_layers,
                            "o_layers": o_layers,
                        })

                    else:  # channel
                        k_s_c = layer_scores_1d.get(k_layers[0][0]) if k_layers else None
                        v_s_c = layer_scores_1d.get(v_layers[0][0]) if v_layers else None
                        ch_eff = q_s + o_s
                        if k_s_c is not None and k_s_c.shape == q_s.shape:
                            ch_eff = ch_eff + k_s_c
                        if v_s_c is not None and v_s_c.shape == q_s.shape:
                            ch_eff = ch_eff + v_s_c

                        group_records.append({
                            "type"    : "channel",
                            "prefix"  : prefix,
                            "eff_score": ch_eff,
                            "q_layers": q_layers,
                            "k_layers": k_layers,
                            "v_layers": v_layers,
                            "o_layers": o_layers,
                        })

                    for n, _ in q_layers + k_layers + v_layers + o_layers:
                        accounted.add(n)

    # ── Remaining / unpaired layers ──────────────────────────────────── #
    for name, module in prunable_layers:
        if name in accounted or name not in layer_scores_1d:
            continue
        group_records.append({
            "type"     : "individual",
            "name"     : name,
            "module"   : module,
            "eff_score": layer_scores_1d[name],
        })

    # ─────────────────────────────────────────────────────────────────── #
    #  Global threshold from effective-score pool                          #
    # ─────────────────────────────────────────────────────────────────── #
    all_eff_flat = torch.cat([rec["eff_score"].reshape(-1) for rec in group_records])
    total_units  = all_eff_flat.numel()
    n_prune      = max(0, int(math.floor(total_units * global_sparsity)))

    if n_prune == 0:
        logger.info("[GlobalSparsity] 0 units selected for pruning — nothing to do.")
        return resolved_sparsities

    threshold = torch.kthvalue(all_eff_flat, n_prune).values.item()
    logger.info(
        f"[GlobalSparsity] target={global_sparsity:.1%}  "
        f"total_effective_units={total_units}  n_prune={n_prune}  "
        f"global_threshold={threshold:.6g}"
    )

    # ─────────────────────────────────────────────────────────────────── #
    #  PASS 2: Apply threshold to effective scores → zero weights          #
    # ─────────────────────────────────────────────────────────────────── #
    total_kept = 0

    for rec in group_records:
        eff  = rec["eff_score"]
        mask = (eff > threshold)          # keep-mask on effective score
        total_kept += mask.float().sum().item()
        gtype = rec["type"]

        if gtype == "mlp":
            actual_sp = 1.0 - mask.float().mean().item()
            for u_name, u_module in rec["up_layers"]:
                zero_rows(u_module, mask)
                resolved_sparsities[u_name] = actual_sp
            for d_name, d_module in rec["down_layers"]:
                zero_cols(d_module, mask)
                resolved_sparsities[d_name] = actual_sp

        elif gtype == "head":
            hd       = rec["head_dim"]
            h_keep   = mask               # (n_heads,) bool
            actual_sp = 1.0 - h_keep.float().mean().item()

            q_o_mask = h_keep.repeat_interleave(hd)
            kv_keep  = torch.zeros(n_kv_heads, dtype=torch.bool, device=eff.device)
            for h_kv in range(n_kv_heads):
                kv_keep[h_kv] = h_keep[h_kv * g : (h_kv + 1) * g].any()
            kv_mask = kv_keep.repeat_interleave(hd)

            for q_n, q_m in rec["q_layers"]:
                zero_rows(q_m, q_o_mask)
                resolved_sparsities[q_n] = actual_sp
            for k_n, k_m in rec["k_layers"]:
                zero_rows(k_m, kv_mask)
                resolved_sparsities[k_n] = actual_sp
            for v_n, v_m in rec["v_layers"]:
                zero_rows(v_m, kv_mask)
                resolved_sparsities[v_n] = actual_sp
            for o_n, o_m in rec["o_layers"]:
                zero_cols(o_m, q_o_mask)
                resolved_sparsities[o_n] = actual_sp

        elif gtype == "channel":
            actual_sp = 1.0 - mask.float().mean().item()
            for q_n, q_m in rec["q_layers"]:
                zero_rows(q_m, mask)
                resolved_sparsities[q_n] = actual_sp
            for k_n, k_m in rec["k_layers"]:
                zero_rows(k_m, mask[:k_m.weight.shape[0]])
                resolved_sparsities[k_n] = actual_sp
            for v_n, v_m in rec["v_layers"]:
                zero_rows(v_m, mask[:v_m.weight.shape[0]])
                resolved_sparsities[v_n] = actual_sp
            for o_n, o_m in rec["o_layers"]:
                zero_cols(o_m, mask)
                resolved_sparsities[o_n] = actual_sp

        else:  # individual
            actual_sp = 1.0 - mask.float().mean().item()
            zero_rows(rec["module"], mask)
            resolved_sparsities[rec["name"]] = actual_sp

    logger.info(
        f"[GlobalSparsity] done — kept {total_kept:.0f}/{total_units} effective units "
        f"({total_kept / total_units:.1%} kept, {1 - total_kept / total_units:.1%} pruned)"
    )
    return resolved_sparsities



# ======================================================================== #
#  Weight snapshot & restore                                                                                                        #
# ======================================================================== #

def _snapshot_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Deep-copy every parameter to CPU for later restore."""
    return {
        name: param.data.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def _restore_weights(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore weights from snapshot (CPU → device in-place)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name].to(param.device))


# ======================================================================== #
#  Shared helpers                                                                                                                               #
# ======================================================================== #

def _get_prunable_layers(
    model: nn.Module,
    layers_to_prune: Optional[List[str]] = None,
    layers_to_skip: Optional[List[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """Same filter logic as BasePruner._get_prunable_layers."""
    prunable = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        if hasattr(module, "out_features") and module.out_features > 16384:
            continue
        if hasattr(module, "in_features") and module.in_features > 16384:
            continue
        if layers_to_skip and any(skip in name for skip in layers_to_skip):
            continue
        if layers_to_prune and not any(target in name for target in layers_to_prune):
            continue
        prunable.append((name, module))
    return prunable


def _json_default(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


# ======================================================================== #
#  Summary & plotting                                                                                                                       #
# ======================================================================== #

def _print_summary(results: Dict, eval_datasets: List[str]) -> None:
    col_w   = 22
    ds_cols = [f"perplexity_{ds}" for ds in eval_datasets]
    header  = f"{'Strategy':<{col_w}}" + "".join(f"{d:>16}" for d in ds_cols) + f"{'Sparsity':>12}"
    logger.info("")
    logger.info("=" * 70)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(header)
    logger.info("-" * len(header))

    for name, res in results.items():
        row = f"{name:<{col_w}}"
        for ds in eval_datasets:
            val = res.get(f"perplexity_{ds}", float("nan"))
            row += f"{val:>16.4f}"
        sp = res.get("actual_sparsity_global", res.get("sparsity_global", float("nan")))
        row += f"{sp:>11.2%}"
        logger.info(row)
    logger.info("=" * 70)


def _plot_comparison(
    results       : Dict,
    eval_datasets : List[str],
    output_dir    : str,
) -> None:
    if not _HAS_MATPLOTLIB:
        logger.warning("  matplotlib not available – skipping plot.")
        return

    strategy_names = [n for n in results if n != "baseline"]
    n_strats = len(strategy_names)
    n_ds     = len(eval_datasets)
    if n_strats == 0:
        return

    x       = np.arange(n_strats)
    bar_w   = 0.8 / max(n_ds, 1)
    offsets = np.linspace(-0.4 + bar_w / 2, 0.4 - bar_w / 2, n_ds) if n_ds > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(max(10, n_strats * 1.5), 6))
    colors  = plt.cm.tab10.colors

    for i, (ds, offset) in enumerate(zip(eval_datasets, offsets)):
        ppls = [results[n].get(f"perplexity_{ds}", float("nan")) for n in strategy_names]
        bars = ax.bar(x + offset, ppls, bar_w * 0.9,
                      label=f"PPL {ds}", color=colors[i % len(colors)], alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    # Baseline dashed reference lines
    for i, ds in enumerate(eval_datasets):
        if "baseline" in results:
            bl = results["baseline"].get(f"perplexity_{ds}")
            if bl is not None and not math.isnan(bl):
                ax.axhline(
                    bl, color=colors[i % len(colors)],
                    linestyle="--", linewidth=1.2, alpha=0.7,
                    label=f"Baseline {ds} ({bl:.1f})",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Perplexity ↓", fontsize=11)
    ax.set_title("Kronecker-OBD Aggregation Strategy Comparison", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "strategy_comparison.png")
    fig.savefig(plot_path, dpi=150)
    logger.info(f"  Plot saved to: {plot_path}")
    plt.close(fig)


# ======================================================================== #
#  Model Checkpoint & Parameter Saving                                                                                   #
# ======================================================================== #

def save_model(
    model: nn.Module,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    save: bool = True,
) -> None:
    """
    Physically shrink and save the structured-pruned model checkpoint, config,
    and tokenizer to output_dir.

    Args:
        model: PyTorch / HuggingFace model with zeroed structured pruned weights.
        output_dir: Output directory to save the physically shrunk model.
        tokenizer: Optional tokenizer to save.
        save: If False, skips saving.
    """
    if not save:
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"  Saving and physically shrinking model to '{output_dir}' ...")

    # 1. Save standard model files (config, code, tokenizer, etc.) via HuggingFace
    if hasattr(model, "save_pretrained") and callable(model.save_pretrained):
        try:
            model.save_pretrained(output_dir, safe_serialization=True)
        except TypeError:
            model.save_pretrained(output_dir)

    tok = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    if tok is not None and hasattr(tok, "save_pretrained") and callable(tok.save_pretrained):
        tok.save_pretrained(output_dir)

    # 2. Physically shrink weights by removing pruned (zeroed) intermediate channels
    state_dict = model.state_dict()
    surviving_indices_per_layer: Dict[int, torch.Tensor] = {}

    # Identify surviving channels per layer
    for key, tensor in state_dict.items():
        parts = key.split(".")
        layer_idx = None
        proj_name = None

        if "blocks" in parts:
            b_idx = parts.index("blocks")
            if len(parts) > b_idx + 2 and parts[b_idx + 1].isdigit():
                layer_idx = int(parts[b_idx + 1])
                proj_name = parts[b_idx + 2]
        elif "layers" in parts:
            l_idx = parts.index("layers")
            if len(parts) > l_idx + 3 and parts[l_idx + 1].isdigit() and parts[l_idx + 2] == "mlp":
                layer_idx = int(parts[l_idx + 1])
                proj_name = parts[l_idx + 3]

        if layer_idx is not None and proj_name is not None:
            if proj_name in ("ff_proj", "up_proj", "gate_proj"):
                mag = tensor.abs().sum(dim=1)
                mask = (mag > 0)
                if layer_idx not in surviving_indices_per_layer:
                    surviving_indices_per_layer[layer_idx] = mask
                else:
                    surviving_indices_per_layer[layer_idx] = surviving_indices_per_layer[layer_idx] | mask
            elif proj_name in ("ff_out", "down_proj"):
                mag = tensor.abs().sum(dim=0)
                mask = (mag > 0)
                if layer_idx not in surviving_indices_per_layer:
                    surviving_indices_per_layer[layer_idx] = mask
                else:
                    surviving_indices_per_layer[layer_idx] = surviving_indices_per_layer[layer_idx] | mask

    # 3. Slice state dict and determine new intermediate size
    new_state_dict: Dict[str, torch.Tensor] = {}
    new_intermediate_size = None

    for layer_idx, mask in surviving_indices_per_layer.items():
        surv = torch.where(mask)[0]
        if new_intermediate_size is None:
            new_intermediate_size = len(surv)
        logger.info(f"    Layer {layer_idx} surviving intermediate size: {len(surv)}")

    for key, tensor in state_dict.items():
        parts = key.split(".")
        layer_idx = None
        proj_name = None

        if "blocks" in parts:
            b_idx = parts.index("blocks")
            if len(parts) > b_idx + 2 and parts[b_idx + 1].isdigit():
                layer_idx = int(parts[b_idx + 1])
                proj_name = parts[b_idx + 2]
        elif "layers" in parts:
            l_idx = parts.index("layers")
            if len(parts) > l_idx + 3 and parts[l_idx + 1].isdigit() and parts[l_idx + 2] == "mlp":
                layer_idx = int(parts[l_idx + 1])
                proj_name = parts[l_idx + 3]

        if layer_idx is not None and layer_idx in surviving_indices_per_layer:
            surv_idx = torch.where(surviving_indices_per_layer[layer_idx])[0].to(tensor.device)
            if proj_name in ("ff_proj", "up_proj", "gate_proj"):
                if key.endswith(".weight") or tensor.dim() == 2:
                    tensor = tensor[surv_idx, :]
                elif key.endswith(".bias") or tensor.dim() == 1:
                    tensor = tensor[surv_idx]
            elif proj_name in ("ff_out", "down_proj"):
                if key.endswith(".weight") or tensor.dim() == 2:
                    tensor = tensor[:, surv_idx]

        new_state_dict[key] = tensor.contiguous()

    # 4. Update config.json with shrunk intermediate size
    config_file = os.path.join(output_dir, "config.json")
    if new_intermediate_size is not None and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            if "mlp_hidden_size" in cfg or "blocks" in str(state_dict.keys()):
                cfg["mlp_hidden_size"] = new_intermediate_size
            if "intermediate_size" in cfg:
                cfg["intermediate_size"] = new_intermediate_size
            # Also store per-layer hidden sizes for heterogeneous global pruning
            layer_sizes = {str(l_idx): len(torch.where(m)[0]) for l_idx, m in surviving_indices_per_layer.items()}
            if layer_sizes:
                cfg["mlp_hidden_sizes"] = layer_sizes
            with open(config_file, "w") as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"    Updated config mlp_hidden_size to {new_intermediate_size} (and saved mlp_hidden_sizes for {len(layer_sizes)} layers)")
        except Exception as exc:
            logger.warning(f"    Could not update config.json: {exc}")

    # 5. Clean up old safetensor shards and any bin files, then save physically shrunk safetensors
    for fname in os.listdir(output_dir):
        if (fname.startswith("model-") and fname.endswith(".safetensors")) or fname in (
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ):
            try:
                os.remove(os.path.join(output_dir, fname))
            except OSError:
                pass

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(new_state_dict, safetensors_path, metadata={"format": "pt"})
    logger.info(f"    Physically shrunk model.safetensors saved to {output_dir}")
    logger.info(f"  Physically shrunk model successfully saved to '{output_dir}'.")


# Alias for backward compatibility
save_model_with_params = save_model



# ======================================================================== #
#  CLI                                                                       #
# ======================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare pruning aggregation strategies on LLaDA "
                    "using pre-computed KFAC stores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      required=True,
                   help="HF model id or local path (e.g. ./llada_pruned_2048_64timesteps).")
    p.add_argument("--kfac_store", required=True,
                   help="Path to .pt file with raw KFAC activations/gradients.")
    p.add_argument("--scores_pkl", default=None,
                   help="Optional path to pre-computed PruningScoreStore .pkl  "
                        "(if provided, raw score computation is skipped).")
    p.add_argument("--sparsity",        type=float, default=0.3,
                   help="Default per-layer target pruning ratio (fraction of neurons to prune per layer).")
    p.add_argument("--global_sparsity",  type=float, default=None,
                   help="When set, compute scores across ALL layers jointly and remove the bottom X%% "
                        "globally (e.g. --global_sparsity 0.3 removes the least-important 30%% of "
                        "structured units across the whole model). Overrides --sparsity for the "
                        "threshold calculation while still respecting coupled MLP/Attention pruning.")
    p.add_argument("--total_timesteps", type=int,   default=1000,
                   help="Total diffusion timesteps T (for weight schedules).")
    p.add_argument("--eval_datasets",   nargs="+",  default=["wikitext2", "c4"],
                   help="Perplexity evaluation datasets.")
    p.add_argument("--eval_seq_len",    type=int,   default=1024,
                   help="Sequence length for perplexity evaluation.")
    p.add_argument("--output_dir",      default="./strategy_comparison_results",
                   help="Directory for JSON results and plots.")
    p.add_argument("--device",  default="cuda",   choices=["cuda", "cpu"])
    p.add_argument("--dtype",   default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--strategies", nargs="*", default=None,
                   help="Subset of strategy names to run (default: all registered).")
    p.add_argument("--coupling_mode", default="geometric", choices=["arithmetic", "geometric", "sum"],
                   help="Coupling function for structured MLP pruning (up_proj and down_proj channels). 'sum' is canonical OBD sum damage.")
    p.add_argument("--attn_coupling_mode", default="head", choices=["head", "channel", "none"],
                   help="Coupling mode for Attention layers: 'head' (head-wise pruning), 'channel' (coupled channel-wise pruning across Q,K,V,O), or 'none' (independent per-layer row pruning).")

    # ── Layer selection ───────────────────────────────────────────────── #
    p.add_argument("--layers_to_prune", nargs="*", default=[],
                   metavar="NAME",
                   help="Substring patterns of layer names to prune. "
                        "Empty = prune all prunable Linear layers.")
    p.add_argument("--layers_to_skip",  nargs="*", default=[],
                   metavar="NAME",
                   help="Substring patterns of layer names to SKIP (never prune, "
                        "e.g. --layers_to_skip attn or q_proj k_proj v_proj attn_out).")

    # ── Evaluation options ────────────────────────────────────────────── #
    p.add_argument("--mc_num", type=int, default=128,
                   help="Monte Carlo masking iterations for perplexity. Must be divisible by --mc_batch_size.")
    p.add_argument("--mc_batch_size", type=int, default=32,
                   help="MC mini-batch size (mc_num must be divisible by this).")

    # ── Checkpoint saving ────────────────────────────────────────────── #
    p.add_argument("--save_model", action="store_true", default=False,
                   help="Physically shrink and save the pruned model checkpoint and tokenizer for each strategy.")
    return p.parse_args()


# ======================================================================== #
#  Main                                                                      #
# ======================================================================== #

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # ── 1. Load model + tokenizer ─────────────────────────────────────── #
    logger.info("=" * 65)
    logger.info("  [1/8] Loading model …")
    model, tokenizer = load_llada_model_and_tokenizer(
        model_path_or_config=args.model,
        dtype=args.dtype,
        device=args.device,
    )
    model.tokenizer = tokenizer

    # ── 2. Build evaluation helper ────────────────────────────────────── #
    def _run_eval(model) -> Dict[str, Any]:
        """Perplexity on each --eval_datasets entry."""
        res: Dict[str, Any] = {}
        for ds in args.eval_datasets:
            try:
                ppl = evaluate_perplexity(
                    model         = model,
                    tokenizer     = tokenizer,
                    dataset       = ds,
                    seq_len       = args.eval_seq_len,
                    device        = args.device,
                    mc_num        = args.mc_num,
                    mc_batch_size = args.mc_batch_size,
                )
            except Exception as exc:
                logger.warning(f"  Perplexity({ds}) failed: {exc}")
                ppl = float("nan")
            res[f"perplexity_{ds}"] = ppl
            logger.info(f"  Perplexity on {ds}: {ppl:.3f}")
        return res

    # ── 3. Snapshot clean weights for restore between strategies ──────── #
    logger.info("  [2/8] Snapshotting clean model weights …")
    clean_snapshot = _snapshot_weights(model)

    # ── 4. Strategy results container ───────────────────────────────────── #
    results: Dict[str, Dict] = {}

    # ── 5. Load KFAC store ─────────────────────────────────────────────── #
    logger.info(f"  [4/8] Loading KFAC store from '{args.kfac_store}' …")
    kfac_store: Dict = torch.load(args.kfac_store, map_location="cpu", weights_only=False)
    logger.info(f"    Layers in KFAC store: {len(kfac_store)}")

    # ── 5b. Load optional pre-computed score store ────────────────────── #
    precomputed_store: Optional[PruningScoreStore] = None
    if args.scores_pkl and os.path.isfile(args.scores_pkl):
        logger.info(f"  [4b] Loading pre-computed scores from '{args.scores_pkl}' …")
        precomputed_store = PruningScoreStore.load(args.scores_pkl)
        logger.info(
            f"    Layers: {len(precomputed_store.layer_names)}  |  "
            f"Timesteps: {precomputed_store.timesteps}"
        )

    # ── 6. Prunable layers & timestep list ────────────────────────────── #
    prunable_layers = _get_prunable_layers(
        model,
        layers_to_prune=args.layers_to_prune,
        layers_to_skip=args.layers_to_skip,
    )
    logger.info(f"  [5/8] Prunable layers selected ({len(prunable_layers)}):")
    mlp_layers = [n for n, _ in prunable_layers if _is_up_proj_layer(n) or _is_down_proj_layer(n)]
    other_layers = [n for n, _ in prunable_layers if n not in mlp_layers]
    logger.info(f"    MLP layers ({len(mlp_layers)}): {mlp_layers[:4]}{'…' if len(mlp_layers) > 4 else ''}")
    logger.info(f"    Other / Attention layers ({len(other_layers)}): {other_layers[:4]}{'…' if len(other_layers) > 4 else ''}")
    for n, _ in prunable_layers:
        logger.debug(f"      - {n}")

    # Infer timestep list from KFAC store
    all_ts: set = set()
    for ts_dict in kfac_store.values():
        all_ts.update(ts_dict.keys())
    timestep_list = sorted(all_ts)
    logger.info(
        f"    Timesteps detected ({len(timestep_list)}): "
        f"{timestep_list[:10]}{'…' if len(timestep_list) > 10 else ''}"
    )

    # ── 7. Compute (or reuse) raw per-timestep scores ─────────────────── #
    if precomputed_store is not None:
        raw_store = precomputed_store
        logger.info("  [6/8] Reusing pre-computed raw score store.")
    else:
        logger.info("  [6/8] Computing per-timestep OBD scores …")
        t0 = time.time()
        raw_store = _compute_raw_scores_from_kfac(
            kfac_store, model, prunable_layers, timestep_list
        )
        logger.info(f"    Done in {time.time() - t0:.1f}s")

    # ── 8. Strategy loop ──────────────────────────────────────────────── #
    all_strategies  = make_strategies(args.total_timesteps)
    requested_names = set(args.strategies) if args.strategies else None
    if requested_names:
        all_strategies = [s for s in all_strategies if s["name"] in requested_names]

    logger.info(f"  [7/8] Running {len(all_strategies)} strategies …")

    for strat_cfg in all_strategies:
        name = strat_cfg["name"]
        logger.info("")
        logger.info("-" * 65)
        logger.info(f"  Strategy: {name}")
        logger.info("-" * 65)

        # Re-aggregate from a fresh copy so strategies don't pollute each other
        fresh_store = clone_raw_store(raw_store)

        t0 = time.time()
        aggregated = _aggregate_store(fresh_store, strat_cfg, args.total_timesteps)
        logger.info(
            f"    Aggregation: {time.time() - t0:.2f}s  |  "
            f"layers: {len(aggregated)}"
        )

        # Apply pruning in-place
        t0 = time.time()
        if args.global_sparsity is not None:
            # ── Global-threshold pruning: pool scores across ALL layers ── #
            logger.info(
                f"    [global_sparsity={args.global_sparsity:.1%}] "
                "Computing single threshold across all layers …"
            )
            resolved_target_sp = _apply_global_sparsity_pruning(
                model,
                aggregated,
                prunable_layers,
                global_sparsity=args.global_sparsity,
                coupling_mode=args.coupling_mode,
                attn_coupling_mode=args.attn_coupling_mode,
            )
        else:
            # ── Per-layer pruning (original behaviour) ────────────────── #
            resolved_target_sp = _apply_coupled_structured_pruning(
                model,
                aggregated,
                prunable_layers,
                args.sparsity,
                coupling_mode=args.coupling_mode,
                attn_coupling_mode=args.attn_coupling_mode,
            )
        actual_sp = measure_sparsity(model)
        layer_sp = measure_layer_sparsity(model)
        model_sp = measure_model_sparsity(model)
        pruned_layer_names = [ln for ln, sp in layer_sp.items() if sp > 0.0]
        logger.info(
            f"    Pruning: {time.time() - t0:.2f}s  |  "
            f"actual global sparsity: {actual_sp:.2%}  |  "
            f"pruned layers: {len(pruned_layer_names)}"
        )
        logger.info(
            f"    Model sparsity: {model_sp['overall_sparsity']:.2%} of all {model_sp['total_params'] / 1e9:.2f}B params "
            f"(incl. embeddings) | {model_sp['sparsity_excl_embeddings']:.2%} excl. embeddings | "
            f"{model_sp['sparsity_excl_embeddings_and_head']:.2%} excl. embeddings + output head  |  "
            + "  ".join(f"{k}: {v['sparsity']:.1%} of {v['params'] / 1e9:.2f}B"
                        for k, v in model_sp["categories"].items())
        )

        # ── Save physically shrunk pruned model if requested ─────────── #
        if args.save_model:
            strat_out_dir = os.path.join(args.output_dir, f"model_{name}")
            save_model(model, strat_out_dir, tokenizer=tokenizer, save=args.save_model)

        # Evaluate using eval_diffusion
        logger.info("    Evaluating …")
        t0      = time.time()
        ppl_res = _run_eval(model)
        eval_t  = time.time() - t0

        for ds in args.eval_datasets:
            logger.info(
                f"      PPL {ds}: {ppl_res.get('perplexity_' + ds, float('nan')):.4f}"
            )

        results[name] = {
            **ppl_res,
            "actual_sparsity_global" : actual_sp,
            "overall_model_sparsity" : model_sp["overall_sparsity"],
            "model_sparsity_excl_embeddings"          : model_sp["sparsity_excl_embeddings"],
            "model_sparsity_excl_embeddings_and_head" : model_sp["sparsity_excl_embeddings_and_head"],
            "sparsity_breakdown"     : model_sp["categories"],
            "total_params"           : model_sp["total_params"],
            "pruned_layers"          : pruned_layer_names,
            "target_layer_sparsity"  : {ln: round(sp, 6) for ln, sp in resolved_target_sp.items()},
            "layer_sparsity"         : {ln: round(sp, 6) for ln, sp in layer_sp.items() if sp > 0.0},
            "eval_time_s"            : round(eval_t, 2),
            "strategy_method"        : strat_cfg["method"],
            "strategy_kwargs"        : strat_cfg.get("kwargs", {}),
        }

        # Restore weights before the next strategy
        logger.info("    Restoring clean weights …")
        _restore_weights(model, clean_snapshot)

    # ── 9. Summary table ──────────────────────────────────────────────── #
    _print_summary(results, args.eval_datasets)

    # ── 10. Save JSON ─────────────────────────────────────────────────── #
    logger.info("  [8/8] Saving results …")
    out_path = os.path.join(args.output_dir, "strategy_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info(f"    JSON saved: {out_path}")

    # ── 11. Plot ─────────────────────────────────────────────────────── #
    try:
        _plot_comparison(results, args.eval_datasets, args.output_dir)
    except Exception as exc:
        logger.warning(f"  Could not generate plot: {exc}")


if __name__ == "__main__":
    main()
