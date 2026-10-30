"""
Compare multiple score-aggregation strategies for Kronecker-OBD structural
pruning of Dream-7B-Base (diffusion language model), using pre-computed KFAC statistics.

Dream Diffusion Evaluation Differences:
---------------------------------------
1. Output Logit Alignment:
   In Dream, the diffusion model generates outputs such that position t predicts token t+1:
       logits = model(batch).logits
       logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
   Without this shift, predictions are misaligned by 1 token, causing random guessing (~25% / 50%)
   and invalid perplexities.

2. Special Tokens:
   - mask_token_id: 151666 (<|mask|>)
   - bos_token_id:  151665 (<|beginoftext|>)
   - eos_token_id:  151643 (<|endoftext|>)

3. Continuous-Time Noise Process:
   Dream samples continuous noise timesteps t in [0, 1] with p_mask = (1 - eps) * t + eps.
   BOS (pos 0) and EOS (pos -1) are explicitly unmasked:
       mask_indices[:, 0] = False
       mask_indices[:, -1] = False

Inputs:
-------
  --model       : HuggingFace ID or local path (e.g. Dream-org/Dream-v0-Base-7B)
  --kfac_store  : .pt file with raw {layer -> {timestep -> {diag_A, diag_G, ...}}}
  --scores_pkl  : optional .pkl with pre-computed raw per-timestep scores
  --output_dir  : directory where results JSON and plot will be saved

Usage:
------
python compare_pruning_strategies_dream.py \\
    --model Dream-org/Dream-v0-Base-7B \\
    --kfac_store activations_gradients_pt_files/kfac_store_activations_gradients_backup_2048_ctxlength_64timesteps_dream_base.pt \\
    --scores_pkl dream_base_scores_log_decay/pruning_scores_INVERSE.pkl \\
    --global_sparsity 0.30 \\
    --output_dir ./strategy_comparison_results_dream_30 \\
    --device cuda
"""

import argparse
import copy
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics.diffusion_pruning_score_store import PruningScoreStore
from utils.pruning_utils import clone_raw_store, zero_cols, zero_rows
from utils.dream_utils import save_shrunk_dream
from utils.sparsity import measure_layer_sparsity, measure_model_sparsity, measure_sparsity

# Re-use layer keywords & score computation helpers
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
from evaluation.eval_dream import (  # noqa: F401  (Dream evaluation lives in its own module)
    DREAM_MASK_TOKEN_ID_DEFAULT,
    DREAM_BOS_TOKEN_ID_DEFAULT,
    DREAM_EOS_TOKEN_ID_DEFAULT,
    evaluate_perplexity_dream,
    load_dream_model_and_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================== #
#  Pruning Strategy Registry                                                                                                       #
# ======================================================================== #

def make_strategies(total_timesteps: int = 1000) -> List[Dict]:
    """
    Temporal score-aggregation strategies compared in this script. Each is a dict with
        name   : label
        method : base method for PruningScoreStore.aggregate() (unused when _custom is set)
        kwargs : extra kwargs for PruningScoreStore.aggregate()
        _custom: tag for a custom weight / pooling function (plus _mv_lambda / _mv_mode for mean_var)
    """
    return [
        {"name": "log_decay_reversed", "method": "sum", "kwargs": {}, "_custom": "log_reversed"},
        {"name": "inv_linear", "method": "sum", "kwargs": {}, "_custom": "inv_linear"},
        {"name": "snr_weighted", "method": "sum", "kwargs": {}, "_custom": "snr_weighted"},
        {"name": "linear_ramp", "method": "sum", "kwargs": {}, "_custom": "linear_ramp"},
        {"name": "mean_plus_1p0std", "method": "sum", "kwargs": {}, "_custom": "mean_var", "_mv_lambda": 1.0, "_mv_mode": "add"},
        {"name": "max_pooling", "method": "sum", "kwargs": {}, "_custom": "max_pooling"},
    ]


def _build_custom_weights(tag: str, timesteps: List[int], total_timesteps: int = 1000) -> Dict[int, float]:
    """Compute normalized scalar weights w(t) for each calibration timestep."""
    T_max = max(total_timesteps, max(timesteps))
    raw: Dict[int, float] = {}

    for t in timesteps:
        t_f = float(t)
        if tag == "log_reversed":
            w = math.log(1.0 + t_f)
        elif tag == "linear_ramp":
            w = max(0.01, t_f / T_max)
        elif tag == "inv_linear":
            w = max(0.01, 1.0 - t_f / T_max)
        elif tag == "snr_weighted":
            w = (T_max - t_f) / (t_f + 10.0)
        else:
            raise ValueError(f"Unknown custom weight tag: {tag!r}")
        raw[t] = max(1e-8, float(w))

    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}


def _compute_raw_scores_from_kfac(
    kfac_store: Dict[str, Dict[int, Dict[str, Optional[torch.Tensor]]]],
    model: nn.Module,
    total_timesteps: int = 1000,
) -> PruningScoreStore:
    """Compute raw Kronecker-OBD scores per (layer, timestep) from KFAC statistics."""
    weights_by_name: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights_by_name[name] = module.weight.detach()

    store = PruningScoreStore()

    for layer_name, per_t in kfac_store.items():
        W = weights_by_name.get(layer_name)
        if W is None:
            continue
        is_col = _is_down_proj_layer(layer_name) or _is_attn_out_layer(layer_name)

        for t, stats in per_t.items():
            if is_col:
                score = _kfac_obd_col_scores(W, stats)
            else:
                score = _kfac_obd_row_scores(W, stats)
            store.add(layer_name, int(t), score.cpu())

    return store


def _aggregate_mean_variance(
    raw_scores: Dict[str, Dict[int, torch.Tensor]],
    lam: float = 1.0,
    mode: str = "add",
) -> Dict[str, torch.Tensor]:
    """Aggregate per-timestep scores with mean +/- lambda * std."""
    agg: Dict[str, torch.Tensor] = {}
    for layer_name, t_dict in raw_scores.items():
        if not t_dict:
            continue
        sorted_ts = sorted(t_dict.keys())
        stacked = torch.stack([t_dict[t].float() for t in sorted_ts], dim=0)

        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=True)

        if mode == "add":
            agg[layer_name] = torch.clamp(mean + lam * std, min=0.0)
        else:
            agg[layer_name] = torch.clamp(mean - lam * std, min=0.0)
    return agg


def _aggregate_max_pooling(
    raw_scores: Dict[str, Dict[int, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Aggregate per-timestep scores with element-wise max over timesteps."""
    agg: Dict[str, torch.Tensor] = {}
    for layer_name, t_dict in raw_scores.items():
        if not t_dict:
            continue
        stacked = torch.stack(list(t_dict.values()), dim=0).float()
        agg[layer_name] = stacked.max(dim=0).values
    return agg


def _aggregate_store(
    store: PruningScoreStore,
    strategy: Dict,
    total_timesteps: int = 1000,
) -> Dict[str, torch.Tensor]:
    """Dispatch aggregation strategy to produce final 1-D scores per layer."""
    custom_tag = strategy.get("_custom")

    if custom_tag == "mean_var":
        return _aggregate_mean_variance(
            store._raw,
            lam=strategy.get("_mv_lambda", 1.0),
            mode=strategy.get("_mv_mode", "add"),
        )

    if custom_tag == "max_pooling":
        return _aggregate_max_pooling(store._raw)

    if custom_tag is not None:
        timesteps = sorted(next(iter(store._raw.values())).keys())
        weights = _build_custom_weights(custom_tag, timesteps, total_timesteps)
        store.aggregate(method="weighted", custom_weights=weights)
        return store.all_aggregated()

    store.aggregate(method=strategy["method"], total_timesteps=total_timesteps, **strategy.get("kwargs", {}))
    return store.all_aggregated()


# ======================================================================== #
#  Coupled & Global Pruning Logic                                                                                              #
# ======================================================================== #

_LayerList = List[Tuple[str, nn.Module]]


def _row_scores_1d(s: torch.Tensor) -> torch.Tensor:
    """Collapse a 2-D score matrix to per-row (output unit) scores."""
    return s.mean(dim=1) if s.dim() == 2 else s


def _col_scores_1d(s: torch.Tensor) -> torch.Tensor:
    """Collapse a 2-D score matrix to per-column (input unit) scores."""
    return s.mean(dim=0) if s.dim() == 2 else s


def _topk_keep_mask(scores_1d: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask: prune ceil(n * sparsity) lowest-scoring units, always keeping at least one."""
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
    """Zero rows (axis='row') or columns (axis='col') of each layer and record it as pruned.

    If truncate is True, the row mask is sliced to each layer's output dim (used for GQA K/V).
    """
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
) -> torch.Tensor:
    """Combine up/gate row scores and down column scores into one score per intermediate channel."""
    if coupling_mode == "sum":
        return torch.stack(u_1d).sum(dim=0) + torch.stack(d_1d).sum(dim=0)

    avg_up   = torch.stack(u_1d).mean(dim=0)
    avg_down = torch.stack(d_1d).mean(dim=0)
    if avg_up.shape != avg_down.shape:
        return avg_up
    if coupling_mode == "geometric":
        return torch.sqrt(torch.clamp(avg_up * avg_down, min=1e-12))
    return 0.5 * (avg_up + avg_down)


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
) -> torch.Tensor:
    """Per-query-head damage: Q + O slice sums plus shared K/V slice sums split across the g heads in each group."""
    head_damages = torch.zeros(n_heads, device=q_score_1d.device)
    for h in range(n_heads):
        s_idx = h * head_dim
        e_idx = (h + 1) * head_dim
        dq = q_score_1d[s_idx:e_idx].sum()
        do = o_score_1d[s_idx:e_idx].sum()
        h_kv = h // g
        kv_s = h_kv * head_dim
        kv_e = (h_kv + 1) * head_dim
        dk = k_score_1d[kv_s:kv_e].sum() / float(g) if k_score_1d is not None else 0.0
        dv = v_score_1d[kv_s:kv_e].sum() / float(g) if v_score_1d is not None else 0.0
        head_damages[h] = dq + do + dk + dv
    return head_damages


def _prune_attn_heads(
    q_layers: _LayerList,
    k_layers: _LayerList,
    v_layers: _LayerList,
    o_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    q_score_1d: torch.Tensor,
    o_score_1d: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Head-wise (GQA-aware) attention pruning; a KV head is kept if any of its query heads is kept."""
    g = max(1, n_heads // n_kv_heads)

    k_score = scores.get(k_layers[0][0]) if k_layers else None
    v_score = scores.get(v_layers[0][0]) if v_layers else None
    k_score_1d = _row_scores_1d(k_score) if k_score is not None else None
    v_score_1d = _row_scores_1d(v_score) if v_score is not None else None

    head_damages = _compute_head_damages(q_score_1d, o_score_1d, k_score_1d, v_score_1d, n_heads, head_dim, g)
    head_keep = _topk_keep_mask(head_damages, target_sparsity)

    q_o_mask = head_keep.repeat_interleave(head_dim)
    kv_head_keep = head_keep.view(n_kv_heads, g).any(dim=1)
    kv_mask = kv_head_keep.repeat_interleave(head_dim)

    _zero_and_record(q_layers, q_o_mask, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(k_layers, kv_mask, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(v_layers, kv_mask, "row", target_sparsity, processed, resolved_sparsities)
    _zero_and_record(o_layers, q_o_mask, "col", target_sparsity, processed, resolved_sparsities)


def _prune_attn_channels(
    q_layers: _LayerList,
    k_layers: _LayerList,
    v_layers: _LayerList,
    o_layers: _LayerList,
    q_score_1d: torch.Tensor,
    o_score_1d: torch.Tensor,
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Channel-wise attention pruning coupled across Q/K/V rows and O columns (MHA only)."""
    channel_damages = q_score_1d + o_score_1d
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

    cfg = getattr(model, "config", None)
    n_heads = getattr(cfg, "num_attention_heads", 28)
    n_kv_heads = getattr(cfg, "num_key_value_heads", 4)
    head_dim = d_model // n_heads

    effective_attn_mode = attn_coupling_mode
    if attn_coupling_mode == "channel" and n_heads != n_kv_heads:
        logger.warning(
            f"  [AttnPruning] attn_coupling_mode='channel' is invalid for GQA "
            f"(n_heads={n_heads} != n_kv_heads={n_kv_heads}). "
            f"Falling back to 'head' coupling mode to prevent attention coordinate corruption."
        )
        effective_attn_mode = "head"

    if effective_attn_mode == "head":
        _prune_attn_heads(
            q_layers, k_layers, v_layers, o_layers, scores,
            q_score_1d, o_score_1d, n_heads, n_kv_heads, head_dim,
            target_sparsity, processed, resolved_sparsities,
        )
    else:
        _prune_attn_channels(
            q_layers, k_layers, v_layers, o_layers,
            q_score_1d, o_score_1d,
            target_sparsity, processed, resolved_sparsities,
        )


def _prune_unpaired_layers(
    prunable_layers: _LayerList,
    scores: Dict[str, torch.Tensor],
    target_sparsity: float,
    processed: set,
    resolved_sparsities: Dict[str, float],
) -> None:
    """Fallback: independent row pruning for any layer not handled by MLP/attention coupling."""
    for name, module in prunable_layers:
        if name in processed or name not in scores:
            continue
        mask = _topk_keep_mask(_row_scores_1d(scores[name]), target_sparsity)
        zero_rows(module, mask)
        resolved_sparsities[name] = target_sparsity


def _apply_coupled_structured_pruning(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    prunable_layers: List[Tuple[str, nn.Module]],
    target_sparsity: float,
    coupling_mode: str = "geometric",
    attn_coupling_mode: str = "head",
) -> Dict[str, float]:
    """Apply coupled structured pruning across MLP and Attention layers."""
    resolved_sparsities: Dict[str, float] = {}
    processed: set = set()

    for block_layers in _group_layers_by_block(prunable_layers).values():
        _prune_mlp_block(block_layers, scores, target_sparsity, coupling_mode, processed, resolved_sparsities)
        _prune_attn_block(model, block_layers, scores, target_sparsity, attn_coupling_mode, processed, resolved_sparsities)

    _prune_unpaired_layers(prunable_layers, scores, target_sparsity, processed, resolved_sparsities)
    return resolved_sparsities


def _apply_global_sparsity_pruning(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    prunable_layers: List[Tuple[str, nn.Module]],
    global_sparsity: float,
    coupling_mode: str = "geometric",
    attn_coupling_mode: str = "head",
) -> Dict[str, float]:
    """Apply global threshold structured pruning across the entire model."""
    resolved_sparsities: Dict[str, float] = {}
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
        logger.warning("[GlobalSparsity] No scores available.")
        return resolved_sparsities

    cfg = getattr(model, "config", None)
    n_heads = getattr(cfg, "num_attention_heads", 28)
    n_kv_heads = getattr(cfg, "num_key_value_heads", 4)
    g = max(1, n_heads // n_kv_heads)

    blocks: Dict[str, List] = defaultdict(list)
    for name, module in prunable_layers:
        blocks[_get_block_prefix(name)].append((name, module))

    group_records: List[Dict] = []
    accounted: set = set()

    for prefix, block_layers in blocks.items():
        up_layers   = [(n, m) for n, m in block_layers if _is_up_proj_layer(n)]
        down_layers = [(n, m) for n, m in block_layers if _is_down_proj_layer(n)]

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
                        eff = avg_up

                group_records.append({
                    "group_key": f"{prefix}_mlp",
                    "effective_score": eff,
                    "group_type": "mlp",
                    "layers": up_layers + down_layers,
                    "up_layers": up_layers,
                    "down_layers": down_layers,
                })
                for n, _ in up_layers + down_layers:
                    accounted.add(n)

        # Attention groups
        if attn_coupling_mode in ("head", "channel"):
            q_layers = [(n, m) for n, m in block_layers if _is_q_proj_layer(n)]
            k_layers = [(n, m) for n, m in block_layers if _is_k_proj_layer(n)]
            v_layers = [(n, m) for n, m in block_layers if _is_v_proj_layer(n)]
            o_layers = [(n, m) for n, m in block_layers if _is_attn_out_layer(n)]

            if q_layers and o_layers:
                q_name, _ = q_layers[0]
                o_name, _ = o_layers[0]
                q_s = layer_scores_1d.get(q_name)
                o_s = layer_scores_1d.get(o_name)

                if q_s is not None and o_s is not None:
                    d_model = q_s.numel()
                    if attn_coupling_mode == "head":
                        head_dim = d_model // n_heads
                        head_damages = torch.zeros(n_heads, device=q_s.device)
                        k_s = layer_scores_1d.get(k_layers[0][0]) if k_layers else None
                        v_s = layer_scores_1d.get(v_layers[0][0]) if v_layers else None

                        for h in range(n_heads):
                            s_idx = h * head_dim
                            e_idx = (h + 1) * head_dim
                            dq = q_s[s_idx:e_idx].mean()
                            do = o_s[s_idx:e_idx].mean()
                            h_kv = h // g
                            kv_s = h_kv * head_dim
                            kv_e = (h_kv + 1) * head_dim
                            dk = k_s[kv_s:kv_e].mean() if k_s is not None else 0.0
                            dv = v_s[kv_s:kv_e].mean() if v_s is not None else 0.0
                            head_damages[h] = 0.5 * (dq + do + dk + dv)

                        group_records.append({
                            "group_key": f"{prefix}_attn_head",
                            "effective_score": head_damages,
                            "group_type": "head",
                            "q_layers": q_layers,
                            "k_layers": k_layers,
                            "v_layers": v_layers,
                            "o_layers": o_layers,
                            "head_dim": head_dim,
                            "n_heads": n_heads,
                            "n_kv_heads": n_kv_heads,
                            "g": g,
                            "param_cost": (2 * d_model * head_dim) + (2 * d_model * head_dim / g),
                        })
                    else:
                        eff = q_s + o_s
                        group_records.append({
                            "group_key": f"{prefix}_attn_channel",
                            "effective_score": eff,
                            "group_type": "channel",
                            "q_layers": q_layers,
                            "k_layers": k_layers,
                            "v_layers": v_layers,
                            "o_layers": o_layers,
                            "param_cost": 2 * d_model,
                        })
                    for n, _ in q_layers + k_layers + v_layers + o_layers:
                        accounted.add(n)

        # MLP groups param_cost tag
        for rec in group_records:
            if rec["group_type"] == "mlp" and "param_cost" not in rec:
                # MLP channel removes 3 * hidden_size parameters (gate, up, down)
                rec["param_cost"] = 3 * block_layers[0][1].weight.shape[1] if block_layers else 3 * 3584

    # Remaining unpaired layers
    for name, module in prunable_layers:
        if name not in accounted and name in layer_scores_1d:
            group_records.append({
                "group_key": name,
                "effective_score": layer_scores_1d[name],
                "group_type": "individual",
                "layer_name": name,
                "module": module,
                "param_cost": module.weight.shape[1] if hasattr(module, "weight") else 1,
            })

    # Parameter-weighted global threshold pooling
    flat_items = []
    for rec in group_records:
        cost = rec.get("param_cost", 1)
        for val in rec["effective_score"]:
            flat_items.append((val.item(), cost))

    flat_items.sort(key=lambda x: x[0])
    total_prunable_params = sum(cost for _, cost in flat_items)
    target_params_to_prune = total_prunable_params * global_sparsity

    accum = 0.0
    cutoff_idx = 0
    for idx, (score_val, cost) in enumerate(flat_items):
        accum += cost
        if accum >= target_params_to_prune:
            cutoff_idx = idx
            break

    threshold = flat_items[cutoff_idx][0]

    # Apply masks
    for rec in group_records:
        gtype = rec["group_type"]
        eff = rec["effective_score"]
        keep_mask = eff > threshold
        if not keep_mask.any():
            keep_mask[eff.argmax()] = True

        frac_pruned = float((~keep_mask).sum().item()) / float(eff.numel())

        if gtype == "mlp":
            for u_n, u_m in rec["up_layers"]:
                zero_rows(u_m, keep_mask)
                resolved_sparsities[u_n] = frac_pruned
            for d_n, d_m in rec["down_layers"]:
                zero_cols(d_m, keep_mask)
                resolved_sparsities[d_n] = frac_pruned

        elif gtype == "head":
            head_dim = rec["head_dim"]
            n_kv_heads = rec["n_kv_heads"]
            g_factor = rec["g"]
            q_o_mask = keep_mask.repeat_interleave(head_dim)
            kv_head_keep = keep_mask.view(n_kv_heads, g_factor).any(dim=1)
            kv_mask = kv_head_keep.repeat_interleave(head_dim)

            for q_n, q_m in rec["q_layers"]:
                zero_rows(q_m, q_o_mask)
                resolved_sparsities[q_n] = frac_pruned
            for k_n, k_m in rec["k_layers"]:
                zero_rows(k_m, kv_mask)
                resolved_sparsities[k_n] = frac_pruned
            for v_n, v_m in rec["v_layers"]:
                zero_rows(v_m, kv_mask)
                resolved_sparsities[v_n] = frac_pruned
            for o_n, o_m in rec["o_layers"]:
                zero_cols(o_m, q_o_mask)
                resolved_sparsities[o_n] = frac_pruned

        elif gtype == "channel":
            for q_n, q_m in rec["q_layers"]:
                zero_rows(q_m, keep_mask)
                resolved_sparsities[q_n] = frac_pruned
            for k_n, k_m in rec["k_layers"]:
                zero_rows(k_m, keep_mask[:k_m.weight.shape[0]])
                resolved_sparsities[k_n] = frac_pruned
            for v_n, v_m in rec["v_layers"]:
                zero_rows(v_m, keep_mask[:v_m.weight.shape[0]])
                resolved_sparsities[v_n] = frac_pruned
            for o_n, o_m in rec["o_layers"]:
                zero_cols(o_m, keep_mask)
                resolved_sparsities[o_n] = frac_pruned

        elif gtype == "individual":
            zero_rows(rec["module"], keep_mask)
            resolved_sparsities[rec["layer_name"]] = frac_pruned

    return resolved_sparsities


# ======================================================================== #
#  Weight Snapshot & Helper Functions                                                                                     #
# ======================================================================== #

def _snapshot_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Take a memory-efficient CPU clone of all linear layer weights and biases."""
    snapshot: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            snapshot[f"{name}.weight"] = module.weight.detach().cpu().clone()
            if getattr(module, "bias", None) is not None:
                snapshot[f"{name}.bias"] = module.bias.detach().cpu().clone()
    return snapshot


def _restore_weights(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore clean weights onto the model in-place."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w_key = f"{name}.weight"
            if w_key in snapshot:
                module.weight.data.copy_(snapshot[w_key].to(module.weight.device))
            b_key = f"{name}.bias"
            if b_key in snapshot and getattr(module, "bias", None) is not None:
                module.bias.data.copy_(snapshot[b_key].to(module.bias.device))


def _get_prunable_layers(
    model: nn.Module,
    layers_to_prune: List[str],
    layers_to_skip: List[str],
) -> List[Tuple[str, nn.Module]]:
    """Return all Linear modules matching inclusion/exclusion patterns."""
    prunable = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if layers_to_skip and any(skip in name for skip in layers_to_skip):
            continue
        if layers_to_prune:
            if any(p in name for p in layers_to_prune):
                prunable.append((name, module))
        else:
            prunable.append((name, module))
    return prunable


def _print_summary(results: Dict[str, Any], eval_datasets: List[str]) -> None:
    """Print an ASCII summary table of strategy comparisons."""
    logger.info("=" * 90)
    logger.info("  DREAM PRUNING RESULTS SUMMARY")
    logger.info("=" * 90)
    header = f"{'Strategy':<28} " + " ".join([f"{f'ppl_{d}':>10}" for d in eval_datasets])
    header += "  Sparsity"
    logger.info(header)
    logger.info("-" * len(header))

    for strat_name, metrics in results.items():
        row = f"{strat_name:<28} "
        for d in eval_datasets:
            ppl = metrics.get(f"perplexity_{d}", float("nan"))
            row += f"{ppl:>10.2f} " if not math.isnan(ppl) else f"{'nan':>10} "
        sp = metrics.get("actual_sparsity", float("nan")) * 100
        row += f" {sp:>8.2f}%"
        logger.info(row)
    logger.info("=" * 90)




# ======================================================================== #
#  CLI Arguments                                                           #
# ======================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare pruning aggregation strategies on Dream-7B-Base "
                    "using pre-computed KFAC stores and official Dream evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Dream-org/Dream-v0-Base-7B",
                   help="HuggingFace model ID or local checkpoint path.")
    p.add_argument("--kfac_store", required=True,
                   help="Path to .pt file containing pre-computed KFAC activations and gradients.")
    p.add_argument("--scores_pkl", default=None,
                   help="Optional path to pre-computed PruningScoreStore (.pkl) to bypass score computation.")
    p.add_argument("--sparsity", type=float, default=0.3,
                   help="Default per-layer target pruning ratio (fraction of neurons to prune per layer).")
    p.add_argument("--global_sparsity", type=float, default=None,
                   help="When set, compute scores across ALL layers jointly and remove the bottom X%% "
                        "globally (e.g. --global_sparsity 0.3 removes the least-important 30%% of "
                        "structured units across the whole model). Overrides --sparsity for the "
                        "threshold calculation while still respecting coupled MLP/Attention pruning.")
    p.add_argument("--total_timesteps", type=int, default=1000,
                   help="Total diffusion timesteps T for weight schedules.")
    p.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"],
                   help="Perplexity evaluation datasets (wikitext2, c4).")
    p.add_argument("--eval_seq_len", type=int, default=512,
                   help="Sequence length for continuous diffusion perplexity evaluation.")
    p.add_argument("--output_dir", default="./strategy_comparison_results_dream",
                   help="Directory for results JSON and plots.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="*", default=None,
                   help="Subset of strategy names to run (default: all registered).")
    p.add_argument("--coupling_mode", default="geometric", choices=["arithmetic", "geometric", "sum"],
                   help="Coupling mode for MLP.")
    p.add_argument("--attn_coupling_mode", default="head", choices=["head", "channel", "none"],
                   help="Coupling mode for Attention layers (supports Dream GQA head-wise pruning).")
    p.add_argument("--layers_to_prune", nargs="*", default=[],
                   help="Substring patterns of layer names to prune (empty = prune all prunable linear layers).")
    p.add_argument("--layers_to_skip", nargs="*", default=[],
                   help="Substring patterns of layer names to skip.")
    p.add_argument("--mc_num", type=int, default=128,
                   help="Monte Carlo masking iterations for Dream perplexity evaluation.")
    p.add_argument("--mc_batch_size", type=int, default=32,
                   help="Batch size for MC evaluation (must divide mc_num).")
    p.add_argument("--add_bos_token", action="store_true", default=True,
                   help="Prepend BOS token (<|beginoftext|>) during evaluation (required by Dream).")
    p.add_argument("--save_model", action="store_true", default=False,
                   help="Physically shrink (MLP channels only; attention stays full size with zeros) and save the "
                        "pruned model checkpoint for each strategy.")
    p.add_argument("--skip_baseline_eval", action="store_true", default=False,
                   help="Skip baseline evaluation (0%% sparsity).")
    return p.parse_args()


# ======================================================================== #
#  Main Execution Loop                                                     #
# ======================================================================== #

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 1. Load Model & Tokenizer
    logger.info("=" * 65)
    logger.info("  [1/7] Loading Dream model & tokenizer …")
    logger.info("=" * 65)
    model, tokenizer = load_dream_model_and_tokenizer(
        model_path=args.model,
        dtype=args.dtype,
        device=args.device,
    )

    # 2. Evaluation Helper
    def _run_eval(curr_model: nn.Module) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        # Perplexity evaluation using Dream continuous diffusion ELBO
        for ds in args.eval_datasets:
            try:
                ppl = evaluate_perplexity_dream(
                    model=curr_model,
                    tokenizer=tokenizer,
                    dataset=ds,
                    seq_len=args.eval_seq_len,
                    device=args.device,
                    mc_num=args.mc_num,
                    mc_batch_size=args.mc_batch_size,
                )
                metrics[f"perplexity_{ds}"] = ppl
            except Exception as exc:
                logger.warning(f"  Perplexity evaluation on {ds} failed: {exc}")
                metrics[f"perplexity_{ds}"] = float("nan")

        return metrics

    # 3. Snapshot Clean Weights
    logger.info("  [2/7] Taking snapshot of clean weights …")
    clean_snapshot = _snapshot_weights(model)

    # 4. Load or Compute Raw Scores
    logger.info("  [3/7] Loading KFAC / score store …")
    master_store: Optional[PruningScoreStore] = None

    if args.scores_pkl and os.path.isfile(args.scores_pkl):
        logger.info(f"    Loading pre-computed score store from {args.scores_pkl} …")
        with open(args.scores_pkl, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, PruningScoreStore):
            master_store = data
        elif isinstance(data, dict) and "raw" in data:
            master_store = PruningScoreStore()
            for l_n, per_t in data["raw"].items():
                for t_val, s_t in per_t.items():
                    master_store.add(l_n, int(t_val), s_t.cpu())
            logger.info(f"    Loaded raw scores for {len(master_store._raw)} layers from dict.")

    prunable_layers = _get_prunable_layers(
        model,
        args.layers_to_prune or [],
        args.layers_to_skip or [],
    )
    expected_layer_names = {n for n, _ in prunable_layers}

    if master_store is not None and args.kfac_store and os.path.isfile(args.kfac_store):
        missing_layers = expected_layer_names - set(master_store._raw.keys())
        if missing_layers:
            logger.warning(
                f"    [ScoreStore] scores_pkl is missing {len(missing_layers)} layers "
                f"(loaded {len(master_store._raw)} / {len(expected_layer_names)} prunable layers). "
                f"Computing missing scores from KFAC store '{args.kfac_store}' ..."
            )
            kfac_data = torch.load(args.kfac_store, map_location="cpu")
            weights_by_name = {n: m.weight.detach() for n, m in model.named_modules() if isinstance(m, nn.Linear)}
            added = 0
            for layer_name in missing_layers:
                per_t = kfac_data.get(layer_name)
                W = weights_by_name.get(layer_name)
                if per_t is None or W is None:
                    continue
                is_col = _is_down_proj_layer(layer_name) or _is_attn_out_layer(layer_name)
                for t, stats in per_t.items():
                    if is_col:
                        score = _kfac_obd_col_scores(W, stats)
                    else:
                        score = _kfac_obd_row_scores(W, stats)
                    master_store.add(layer_name, int(t), score.cpu())
                added += 1
            logger.info(f"    [ScoreStore] Augmented {added} missing layers from KFAC store. Total layers in store: {len(master_store._raw)}.")

    if master_store is None:
        logger.info(f"    Computing raw scores from KFAC store {args.kfac_store} …")
        kfac_data = torch.load(args.kfac_store, map_location="cpu")
        master_store = _compute_raw_scores_from_kfac(kfac_data, model, total_timesteps=args.total_timesteps)
        logger.info(f"    Computed raw scores for {len(master_store._raw)} layers.")

    # 5. Evaluate Baseline (0% Sparsity)
    comparison_results: Dict[str, Any] = {}
    json_path = os.path.join(args.output_dir, "strategy_comparison.json")

    if not args.skip_baseline_eval:
        logger.info("  [4/7] Evaluating unpruned baseline …")
        baseline_eval = _run_eval(model)
        comparison_results["baseline_unpruned"] = {
            **baseline_eval,
            "actual_sparsity": 0.0,
            "aggregation_time": 0.0,
            "pruning_time": 0.0,
        }
    else:
        logger.info("  [4/7] Skipping unpruned baseline evaluation (--skip_baseline_eval).")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as f:
                    prev_data = json.load(f)
                if "baseline_unpruned" in prev_data:
                    comparison_results["baseline_unpruned"] = prev_data["baseline_unpruned"]
                    logger.info("    Carrying over baseline_unpruned from existing strategy_comparison.json.")
            except Exception:
                pass

    # 6. Build Strategies & Run Iterations
    all_strategies = make_strategies(total_timesteps=args.total_timesteps)
    if args.strategies:
        selected_names = set(args.strategies)
        all_strategies = [s for s in all_strategies if s["name"] in selected_names]

    logger.info(f"  [5/7] Comparing {len(all_strategies)} pruning strategies …")
    prunable_layers = _get_prunable_layers(model, args.layers_to_prune, args.layers_to_skip)
    logger.info(f"    Prunable linear layers: {len(prunable_layers)}")

    for idx, strategy in enumerate(all_strategies, 1):
        strat_name = strategy["name"]
        logger.info("-" * 65)
        logger.info(f"  [{idx}/{len(all_strategies)}] Strategy: {strat_name}")
        logger.info("-" * 65)

        # A. Aggregate scores
        t0_agg = time.time()
        temp_store = clone_raw_store(master_store)
        aggregated_scores = _aggregate_store(temp_store, strategy, total_timesteps=args.total_timesteps)
        agg_time = time.time() - t0_agg

        # B. Apply Pruning
        t0_prune = time.time()
        if args.global_sparsity is not None:
            logger.info(
                f"    [global_sparsity={args.global_sparsity:.1%}] "
                "Computing single threshold across all layers …"
            )
            resolved_sp = _apply_global_sparsity_pruning(
                model=model,
                scores=aggregated_scores,
                prunable_layers=prunable_layers,
                global_sparsity=args.global_sparsity,
                coupling_mode=args.coupling_mode,
                attn_coupling_mode=args.attn_coupling_mode,
            )
        else:
            resolved_sp = _apply_coupled_structured_pruning(
                model=model,
                scores=aggregated_scores,
                prunable_layers=prunable_layers,
                target_sparsity=args.sparsity,
                coupling_mode=args.coupling_mode,
                attn_coupling_mode=args.attn_coupling_mode,
            )
        prune_time = time.time() - t0_prune
        actual_sparsity = measure_sparsity(model)
        layer_sp = measure_layer_sparsity(model)
        model_sp = measure_model_sparsity(model)
        logger.info(f"    Pruned. Actual global sparsity: {actual_sparsity * 100:.2f}% ({len(resolved_sp)} layers)")
        logger.info(
            f"    Model sparsity: {model_sp['overall_sparsity']:.2%} of all {model_sp['total_params'] / 1e9:.2f}B params "
            f"(incl. embeddings) | {model_sp['sparsity_excl_embeddings']:.2%} excl. embeddings | "
            f"{model_sp['sparsity_excl_embeddings_and_head']:.2%} excl. embeddings + lm_head  |  "
            + "  ".join(f"{k}: {v['sparsity']:.1%} of {v['params'] / 1e9:.2f}B"
                        for k, v in model_sp["categories"].items())
        )


        # C. Evaluate
        eval_metrics = _run_eval(model)
        comparison_results[strat_name] = {
            **eval_metrics,
            "actual_sparsity": actual_sparsity,
            "overall_model_sparsity": model_sp["overall_sparsity"],
            "model_sparsity_excl_embeddings": model_sp["sparsity_excl_embeddings"],
            "model_sparsity_excl_embeddings_and_head": model_sp["sparsity_excl_embeddings_and_head"],
            "sparsity_breakdown": model_sp["categories"],
            "total_params": model_sp["total_params"],
            "target_layer_sparsity": {ln: round(sp, 6) for ln, sp in resolved_sp.items()},
            "layer_sparsity": {ln: round(sp, 6) for ln, sp in layer_sp.items() if sp > 0.0},
            "aggregation_time": agg_time,
            "pruning_time": prune_time,
        }

        # D. Save Checkpoint (Optional)
        if args.save_model:
            save_path = os.path.join(args.output_dir, f"model_{strat_name}")
            save_shrunk_dream(model, tokenizer, save_path)
            logger.info(f"    Saved shrunk pruned model to {save_path}")

        # E. Restore Clean Weights
        logger.info("    Restoring clean weights for next strategy …")
        _restore_weights(model, clean_snapshot)

    # 7. Print Summary and Save Artifacts
    logger.info("  [6/7] Generating summary …")
    _print_summary(comparison_results, args.eval_datasets)

    json_path = os.path.join(args.output_dir, "strategy_comparison.json")
    with open(json_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"  [7/7] Results saved to: {json_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
