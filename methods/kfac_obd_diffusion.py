"""
Kronecker-OBD Structural Pruner for Discrete Diffusion Language Models.

Discrete diffusion forward process
At timestep t, a fraction r(t) of the input tokens are replaced with
a special [MASK] token.  The model predicts the original token at each
masked position.  The training objective is:

    L(t) = -E [ Σ_{i: mask_i=1}  log p_θ(x0_i | x_t) ]

This is the discrete analogue of the score-matching / denoising loss
used in continuous diffusion.

K-FAC factors collected per (layer, timestep)
For a Linear layer with input X ∈ R^{d_in} and output H ∈ R^{d_out}:

    A(t) = (1/N) Σ_n  x_n x_n^T      ∈ R^{d_in  × d_in }
    G(t) = (1/N) Σ_n  g_n g_n^T      ∈ R^{d_out × d_out}

where g_n = ∂L/∂H_n (gradient of loss w.r.t. pre-activation output).

Kronecker-OBD saliency scores
Row scores:
    s_row(t) = 0.5 · diag(G(t)) ⊙ diag(W A(t) W^T)

Column scores:
    s_col(t) = 0.5 · diag(A(t)) ⊙ diag(W^T G(t) W)

Aggregated score (used for pruning decision)
    s = Σ_t  s(t)          
"""

import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from collectors.diffusion_collector import DiffusionKFACCollector
from config.base_config import PruningConfig, SparsityScope, SparsityType
from core.base_pruner import BasePruner
from core.registry import registry
from data.diffusion_calibration import (
    diffusion_loss,
    get_diffusion_calibration_loader,
    get_mask_token_id,
    get_timestep_schedule,
    mask_tokens,
    masking_ratio,
)
from metrics.diffusion_pruning_score_store import PruningScoreStore
from tqdm import tqdm

logger = logging.getLogger(__name__)
#  Kronecker-OBD saliency score computation                                                                       #
def _kfac_obd_row_scores(
    W: torch.Tensor,
    stats: Dict[str, Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    Kronecker-OBD row saliency score:
        s_row = 0.5 * diag(G) * diag(W A W^T)
    diag(W A W^T) is directly accumulated as mean squared output activations.
    """
    dev = W.device
    diag_G = stats["diag_G"].float().to(dev)
    diag_WAW = stats["diag_WAW"].float().to(dev)
    return 0.5 * diag_G * diag_WAW


def _kfac_obd_col_scores(
    W: torch.Tensor,
    stats: Dict[str, Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    Kronecker-OBD column saliency score:
        s_col = 0.5 * diag(A) * diag(W^T G W)
    """
    dev = W.device
    diag_A = stats["diag_A"].float().to(dev)
    diag_WtGW = stats.get("diag_WtGW")
    if diag_WtGW is not None:
        diag_WtGW = diag_WtGW.float().to(dev)
    else:
        diag_G = stats["diag_G"].float().to(dev)
        diag_WtGW = (diag_G.unsqueeze(1) * (W.float() ** 2)).sum(dim=0)
    return 0.5 * diag_A * diag_WtGW

UP_PROJ_KEYWORDS = ("ff_proj", "up_proj", "gate_proj", "w1", "w3", "fc1", "c_fc")
DOWN_PROJ_KEYWORDS = ("ff_out", "down_proj", "w2", "fc2")
ATTN_Q_KEYWORDS = ("q_proj", "w_q", "query")
ATTN_K_KEYWORDS = ("k_proj", "w_k", "key")
ATTN_V_KEYWORDS = ("v_proj", "w_v", "value")
ATTN_OUT_KEYWORDS = ("attn_out", "o_proj", "w_o", "c_proj_attn", "out_proj")
ALL_BLOCK_KEYWORDS = UP_PROJ_KEYWORDS + DOWN_PROJ_KEYWORDS + ATTN_Q_KEYWORDS + ATTN_K_KEYWORDS + ATTN_V_KEYWORDS + ATTN_OUT_KEYWORDS

def _is_up_proj_layer(name: str, keywords: Tuple[str, ...] = UP_PROJ_KEYWORDS) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

def _is_down_proj_layer(name: str, keywords: Tuple[str, ...] = DOWN_PROJ_KEYWORDS) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

def _is_q_proj_layer(name: str, keywords: Tuple[str, ...] = ATTN_Q_KEYWORDS) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

def _is_k_proj_layer(name: str, keywords: Tuple[str, ...] = ATTN_K_KEYWORDS) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

def _is_v_proj_layer(name: str, keywords: Tuple[str, ...] = ATTN_V_KEYWORDS) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

def _is_attn_out_layer(name: str, keywords: Tuple[str, ...] = ATTN_OUT_KEYWORDS) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)

def _get_block_prefix(name: str, keywords: Tuple[str, ...] = ALL_BLOCK_KEYWORDS) -> str:
    name_lower = name.lower()
    for kw in keywords:
        if kw in name_lower:
            idx = name_lower.find(kw)
            return name[:idx].rstrip(".")
    return name

@registry.register("kfac_obd_diffusion")
class KFACOBDDiffusionPruner(BasePruner):
    def __init__(self, model: nn.Module, config: PruningConfig):
        super().__init__(model, config)
        self._collector        : Optional[DiffusionKFACCollector] = None
        self._kfac_store       : Dict[str, Dict[int, Dict[str, Optional[torch.Tensor]]]] = {}
        self.score_store       : PruningScoreStore                        = PruningScoreStore()
        self._timestep_list    : List[int]                                = []
        self.score_dir         : Optional[str]                            = None
        self.mask_token_id     : Optional[int]                            = None
        self.total_timesteps   : int                                      = 1000
        self.num_timesteps     : int                                      = 32
        self.timestep_strategy : str                                      = "uniform"
        self.damp              : float                                    = 1e-3
        self.seed              : int                                      = 42
        self.agg_method        : str                                      = "sum"
        self.alpha_min         : float                                    = 0.0
        self.alpha_max         : float                                    = 1.0

    def _is_up_proj(self, name: str) -> bool:
        keywords = tuple(self._get_arg("up_proj_keywords", UP_PROJ_KEYWORDS))
        return _is_up_proj_layer(name, keywords)

    def _is_down_proj(self, name: str) -> bool:
        keywords = tuple(self._get_arg("down_proj_keywords", DOWN_PROJ_KEYWORDS))
        return _is_down_proj_layer(name, keywords)

    def _is_attn_out(self, name: str) -> bool:
        keywords = tuple(self._get_arg("attn_out_keywords", ATTN_OUT_KEYWORDS))
        return _is_attn_out_layer(name, keywords)

    def _is_q_proj(self, name: str) -> bool:
        keywords = tuple(self._get_arg("attn_q_keywords", ATTN_Q_KEYWORDS))
        return _is_q_proj_layer(name, keywords)

    def _is_k_proj(self, name: str) -> bool:
        keywords = tuple(self._get_arg("attn_k_keywords", ATTN_K_KEYWORDS))
        return _is_k_proj_layer(name, keywords)

    def _is_v_proj(self, name: str) -> bool:
        keywords = tuple(self._get_arg("attn_v_keywords", ATTN_V_KEYWORDS))
        return _is_v_proj_layer(name, keywords)

    def _get_block_prefix_for_layer(self, name: str) -> str:
        up_kw = tuple(self._get_arg("up_proj_keywords", UP_PROJ_KEYWORDS))
        down_kw = tuple(self._get_arg("down_proj_keywords", DOWN_PROJ_KEYWORDS))
        q_kw = tuple(self._get_arg("attn_q_keywords", ATTN_Q_KEYWORDS))
        k_kw = tuple(self._get_arg("attn_k_keywords", ATTN_K_KEYWORDS))
        v_kw = tuple(self._get_arg("attn_v_keywords", ATTN_V_KEYWORDS))
        o_kw = tuple(self._get_arg("attn_out_keywords", ATTN_OUT_KEYWORDS))
        return _get_block_prefix(name, up_kw + down_kw + q_kw + k_kw + v_kw + o_kw)

    def _layer_sparsity(self, layer_name: str) -> float:
        """Resolve the target sparsity for a given layer."""
        sparsity_cfg = getattr(self.config, "sparsity", None)
        if sparsity_cfg is not None:
            scope = getattr(sparsity_cfg, "scope", None)
            layerwise_enum = getattr(SparsityScope, "LAYERWISE_CUSTOM", None)
            if scope == layerwise_enum or scope == "layerwise_custom":
                layer_map = getattr(sparsity_cfg, "layer_sparsity_map", {})
                if layer_name in layer_map:
                    return float(layer_map[layer_name])
            target_sp = getattr(sparsity_cfg, "target_sparsity", None)
            if target_sp is not None:
                return float(target_sp)

        target = self._get_arg("sparsity", self._get_arg("target_sparsity", 0.5))
        return float(target)

    #  Step 1 – Prepare
    def prepare(self) -> None:
        """
        Prepare model parameters for gradient-based K-FAC collection and instantiate the collector.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        prunable = self._get_prunable_layers()
        layer_names = [n for n, _ in prunable]

        logger.info(
            f"  [KFACOBDDiff] Target layers for K-FAC collection ({len(layer_names)}):"
        )
        for n in layer_names:
            logger.info(f"    - {n}")

        self._kfac_save_path = self._get_arg("kfac_save_path", None)

        self._collector = DiffusionKFACCollector(self.model, layer_names, self.device)

        # Extract config values using config args
        self.num_timesteps = self._get_arg("num_timesteps", self._get_arg("n_timesteps", 32))
        self.timestep_strategy = self._get_arg("timestep_strategy", self._get_arg("timestep_schedule", "uniform"))
        self.total_timesteps = self._get_arg("total_timesteps", self._get_arg("total_steps", 1000))
        self.damp = self._get_arg("damp", 1e-3)
        self.seed = self._get_arg("seed", getattr(self.config, "seed", 42))
        self.agg_method = self._get_arg("agg_method", self._get_arg("score_agg_method", self._get_arg("agg_strategy", "sum")))
        self.alpha_min = float(self._get_arg("alpha_min", 0.0))
        self.alpha_max = float(self._get_arg("alpha_max", 1.0))

        mask_token_id = self._get_arg("mask_token_id")
        if mask_token_id is None:
            mask_token_id = get_mask_token_id(getattr(self.model, "tokenizer", self.model))
        self.mask_token_id = mask_token_id

        self._timestep_list = get_timestep_schedule(
            total_timesteps = self.total_timesteps,
            num_timesteps   = self.num_timesteps,
            strategy        = self.timestep_strategy,
        )
        logger.info(
            f"  [KFACOBDDiff] Timestep schedule ({self.timestep_strategy}): "
            f"{self._timestep_list}"
        )
        logger.info(
            f"  [KFACOBDDiff] Score aggregation method: {self.agg_method} "
            + (f"(alpha_min={self.alpha_min}, alpha_max={self.alpha_max})" if self.agg_method in ("log_decay", "log_weighted", "weighted", "log") else "")
        )

        score_dir = self._get_arg("score_dir")
        if score_dir is None:
            out_dir = self._get_arg("output_dir", getattr(self.config, "output_dir", None))
            if out_dir:
                score_dir = os.path.join(out_dir, "scores")

        self.score_dir = os.path.abspath(score_dir) if score_dir else None


    #  Step 2 – Collect
    def collect(self, calibration_loader) -> None:
        """
        For each (input_ids, timesteps_batch) from calibration_loader:
          1. Apply noise mask X_t ~ q(X_t | X_0, t).
          2. collector.set_timestep(t_val)
          3. Forward pass -> compute ELBO loss.
          4. backward() -> hooks accumulate G.
          5. collector.snapshot(t_val)
        """
        logger.info("  [KFACOBDDiff] Collecting K-FAC statistics across timesteps ...")
        assert self._collector is not None, "Call prepare() before collect()."

        if self.mask_token_id is None:
            mask_token_id = self._get_arg("mask_token_id")
            if mask_token_id is None:
                mask_token_id = get_mask_token_id(getattr(self.model, "tokenizer", self.model))
            self.mask_token_id = mask_token_id

        total_batches = 0
        pbar = tqdm(calibration_loader, desc="[KFACOBDDiff] Collecting K-FAC", unit="batch", dynamic_ncols=True)
        with self._collector.active():
            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                    t_batch   = batch[1] if len(batch) > 1 else None
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    t_batch   = batch.get("timesteps", None)
                else:
                    input_ids = batch
                    t_batch   = None

                input_ids = input_ids.to(self.device)
                batch_size = input_ids.shape[0]

                if t_batch is not None:
                    t_iter = [t_batch[i].item() for i in range(batch_size)]
                else:
                    t_iter = self._timestep_list

                for t_val in t_iter:
                    t_val = int(t_val)

                    noisy_ids, mask = mask_tokens(
                        input_ids       = input_ids,
                        mask_token_id   = self.mask_token_id,
                        t               = t_val,
                        total_timesteps = self.total_timesteps,
                        seed            = self.seed,
                    )

                    self._collector.set_timestep(t_val)

                    try:
                        loss = diffusion_loss(self.model, noisy_ids, input_ids, mask)

                        loss.backward()

                        self._collector.snapshot(t_val)
                        self.model.zero_grad(set_to_none=True)

                    except Exception as exc:
                        logger.warning(
                            f"  [KFACOBDDiff] Error at t={t_val}: {exc}. Skipping."
                        )
                        self.model.zero_grad(set_to_none=True)
                        continue

                total_batches += 1
                pbar.set_postfix({"batches": total_batches, "timesteps/batch": len(t_iter)})

        self._kfac_store = self._collector.get_all()
        n_layer_ts = sum(len(ts) for ts in self._kfac_store.values())
        logger.info(
            f"  [KFACOBDDiff] Collection complete. "
            f"(layer × timestep) entries: {n_layer_ts}"
        )

        # Calculate and log the size of _kfac_store
        total_bytes = 0
        for ts_dict in self._kfac_store.values():
            for mode_dict in ts_dict.values():
                for t in mode_dict.values():
                    if t is not None:
                        total_bytes += t.element_size() * t.nelement()
        logger.info(f"  [KFACOBDDiff] _kfac_store memory footprint: {total_bytes / (1024**2):.2f} MB")

        # Save diagonal KFAC store
        diag_save_path = getattr(self, "_kfac_save_path", None)
        if not diag_save_path:
            diag_save_path = "kfac_store_activations_gradients_backup_2048_ctxlength_64timesteps.pt"
        os.makedirs(os.path.dirname(os.path.abspath(diag_save_path)), exist_ok=True)
        torch.save(self._kfac_store, diag_save_path)
        logger.info(f"  [KFACOBDDiff] Diagonal KFAC store saved → {diag_save_path}")


    #  Step 3 – Compute scores
    def compute_scores(self) -> Dict[str, torch.Tensor]:
        """
        For every (layer, timestep):
          - Computes 1D row scores for output-pruned projections (up_proj, gate_proj, q_proj, k_proj, v_proj)
          - Computes 1D column scores for input-pruned projections (down_proj, attn_out)

        These are stored in PruningScoreStore, then aggregated across timesteps.
        Optionally saves the score store.
        """
        logger.info("  [KFACOBDDiff] Computing OBD scores ...")
        prunable_layers = self._get_prunable_layers()
        for name, module in tqdm(prunable_layers, desc="[KFACOBDDiff] Computing OBD scores", unit="layer", dynamic_ncols=True):
            W        = module.weight.data.float()
            ts_store = self._kfac_store.get(name, {})

            if not ts_store:
                logger.warning(
                    f"  [KFACOBDDiff] No K-FAC data for '{name}'. Skipping."
                )
                continue

            n_scored = 0
            for t in self._timestep_list:
                stats = ts_store.get(t)

                if not stats or stats.get("diag_G") is None:
                    logger.debug(
                        f"    {name} @ t={t}: missing factor – skipped."
                    )
                    continue

                if self._is_down_proj(name) or self._is_attn_out(name):
                    score = _kfac_obd_col_scores(W, stats)
                else:
                    score = _kfac_obd_row_scores(W, stats)

                self.score_store.add(name, t, score)
                n_scored += 1

            logger.debug(
                f"    {name}: scored at {n_scored}/{len(self._timestep_list)} "
                f"timesteps."
            )

        # Aggregate across timesteps
        self.score_store.aggregate(
            method          = self.agg_method,
            total_timesteps = self.total_timesteps,
            alpha_min       = self.alpha_min,
            alpha_max       = self.alpha_max,
        )

        logger.info(self.score_store.summary())

        # Persist
        if self.score_dir is not None:
            os.makedirs(self.score_dir, exist_ok=True)

            store_path = os.path.join(self.score_dir, "pruning_scores_INVERSE.pkl")
            self.score_store.save(store_path)
            logger.info(f"  [KFACOBDDiff] Score store saved: {store_path}")

        return self.score_store.all_aggregated()

    #  Step 4 – Apply pruning
    def apply_pruning(
        self,
        scores  : Dict[str, torch.Tensor],
        strategy: Any = None,
    ) -> None:
        """
        Apply structured pruning masks for paired MLP layers (up_proj / ff_proj and
        down_proj / ff_out) and Attention layers (q, k, v, attn_out).
        """
        prunable_layers = self._get_prunable_layers()

        # Group prunable layers by transformer block prefix
        blocks = defaultdict(list)
        for name, module in prunable_layers:
            prefix = self._get_block_prefix_for_layer(name)
            blocks[prefix].append((name, module))

        processed_layers = set()
        coupling_mode = self._get_arg("coupling_mode", "geometric")
        attn_coupling_mode = self._get_arg("attn_coupling_mode", "head")

        for prefix, block_layers in blocks.items():
            self._prune_mlp_coupled(block_layers, scores, coupling_mode, processed_layers)
            self._prune_attn_coupled(block_layers, scores, attn_coupling_mode, processed_layers)

        self._prune_fallback(prunable_layers, scores, strategy, processed_layers)

    def _prune_mlp_coupled(
        self,
        block_layers: List[Tuple[str, nn.Module]],
        scores: Dict[str, torch.Tensor],
        coupling_mode: str,
        processed_layers: Set[str],
    ) -> None:
        """Coupled channel pruning for MLP up-projections and down-projections."""
        up_layers = [(n, m) for n, m in block_layers if self._is_up_proj(n)]
        down_layers = [(n, m) for n, m in block_layers if self._is_down_proj(n)]

        if not (up_layers and down_layers):
            return

        all_up_scores = [scores.get(n) for n, _ in up_layers if scores.get(n) is not None]
        all_down_scores = [scores.get(n) for n, _ in down_layers if scores.get(n) is not None]

        if not (all_up_scores and all_down_scores):
            return

        u_scores_1d = [s.mean(dim=1) if s.dim() == 2 else s for s in all_up_scores]
        d_scores_1d = [s.mean(dim=0) if s.dim() == 2 else s for s in all_down_scores]

        if coupling_mode == "sum":
            coupled_score = torch.stack(u_scores_1d).sum(dim=0) + torch.stack(d_scores_1d).sum(dim=0)
        elif coupling_mode == "geometric":
            avg_up = torch.stack(u_scores_1d).mean(dim=0)
            avg_down = torch.stack(d_scores_1d).mean(dim=0)
            coupled_score = torch.sqrt(torch.clamp(avg_up * avg_down, min=1e-12))
        else:
            avg_up = torch.stack(u_scores_1d).mean(dim=0)
            avg_down = torch.stack(d_scores_1d).mean(dim=0)
            coupled_score = 0.5 * (avg_up + avg_down)

        d_ff           = coupled_score.numel()
        layer_sparsity = self._layer_sparsity(up_layers[0][0])
        k              = max(1, int(math.ceil(d_ff * layer_sparsity)))
        keep_count     = max(1, d_ff - k)
        top_idx        = torch.topk(coupled_score, k=keep_count, largest=True).indices
        mask_1d        = torch.zeros(d_ff, dtype=torch.bool, device=coupled_score.device)
        mask_1d[top_idx] = True

        # Apply to up-projections (rows)
        for u_name, u_module in up_layers:
            W_u = u_module.weight
            mask_up = mask_1d.unsqueeze(1).expand_as(W_u)
            self._apply_mask(u_module, mask_up)
            if u_module.bias is not None:
                with torch.no_grad():
                    u_module.bias.data *= mask_1d.to(
                        device=u_module.bias.device,
                        dtype=u_module.bias.dtype
                    )
            actual_sp = (~mask_up).float().mean().item()
            logger.info(
                f"  [KFACOBDDiff] {u_name} (Coupled Row): "
                f"pruned to {actual_sp:.1%} sparsity"
            )
            processed_layers.add(u_name)

        # Apply to down-projections (cols)
        for d_name, d_module in down_layers:
            W_d = d_module.weight
            mask_down = mask_1d.unsqueeze(0).expand_as(W_d)
            self._apply_mask(d_module, mask_down)
            actual_sp = (~mask_down).float().mean().item()
            logger.info(
                f"  [KFACOBDDiff] {d_name} (Coupled Col): "
                f"pruned to {actual_sp:.1%} sparsity"
            )
            processed_layers.add(d_name)

    def _prune_attn_coupled(
        self,
        block_layers: List[Tuple[str, nn.Module]],
        scores: Dict[str, torch.Tensor],
        attn_coupling_mode: str,
        processed_layers: Set[str],
    ) -> None:
        """Coupled pruning for Attention layers (head-wise or channel-wise)."""
        if attn_coupling_mode not in ("head", "channel"):
            return

        q_layers = [(n, m) for n, m in block_layers if self._is_q_proj(n)]
        k_layers = [(n, m) for n, m in block_layers if self._is_k_proj(n)]
        v_layers = [(n, m) for n, m in block_layers if self._is_v_proj(n)]
        o_layers = [(n, m) for n, m in block_layers if self._is_attn_out(n)]

        if not (q_layers and o_layers):
            return

        q_name, q_module = q_layers[0]
        o_name, o_module = o_layers[0]
        q_score = scores.get(q_name)
        o_score = scores.get(o_name)

        if q_score is None or o_score is None:
            return

        q_score_1d = q_score.mean(dim=1) if q_score.dim() == 2 else q_score
        o_score_1d = o_score.mean(dim=0) if o_score.dim() == 2 else o_score
        d_model = q_score_1d.numel()

        if attn_coupling_mode == "head":
            # Head-wise coupled pruning
            cfg = getattr(self.model, "config", None)
            n_heads = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 32)))
            n_kv_heads = getattr(cfg, "n_kv_heads", getattr(cfg, "num_key_value_heads", n_heads))
            if n_kv_heads is None:
                n_kv_heads = n_heads

            head_dim = d_model // n_heads
            g = max(1, n_heads // n_kv_heads)

            head_damages = torch.zeros(n_heads, device=q_score.device, dtype=q_score.dtype)

            k_score_1d = None
            if k_layers and scores.get(k_layers[0][0]) is not None:
                ks = scores.get(k_layers[0][0])
                k_score_1d = ks.mean(dim=1) if ks.dim() == 2 else ks

            v_score_1d = None
            if v_layers and scores.get(v_layers[0][0]) is not None:
                vs = scores.get(v_layers[0][0])
                v_score_1d = vs.mean(dim=1) if vs.dim() == 2 else vs

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

            layer_sp = self._layer_sparsity(q_name)
            k_prune = max(1, int(math.ceil(n_heads * layer_sp)))
            keep_count = max(1, n_heads - k_prune)
            top_head_idx = torch.topk(head_damages, k=keep_count, largest=True).indices

            head_keep_mask = torch.zeros(n_heads, dtype=torch.bool, device=q_score.device)
            head_keep_mask[top_head_idx] = True

            q_o_mask_1d = head_keep_mask.repeat_interleave(head_dim)

            kv_head_keep_mask = torch.zeros(n_kv_heads, dtype=torch.bool, device=q_score.device)
            for h_kv in range(n_kv_heads):
                kv_head_keep_mask[h_kv] = head_keep_mask[h_kv * g : (h_kv + 1) * g].any()
            kv_mask_1d = kv_head_keep_mask.repeat_interleave(head_dim)

            for q_n, q_m in q_layers:
                mask_q = q_o_mask_1d.unsqueeze(1).expand_as(q_m.weight)
                self._apply_mask(q_m, mask_q)
                processed_layers.add(q_n)
                actual_sp = (~mask_q).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {q_n} (Coupled Head Row): pruned to {actual_sp:.1%} sparsity")

            for k_n, k_m in k_layers:
                mask_k = kv_mask_1d.unsqueeze(1).expand_as(k_m.weight)
                self._apply_mask(k_m, mask_k)
                processed_layers.add(k_n)
                actual_sp = (~mask_k).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {k_n} (Coupled Head Row): pruned to {actual_sp:.1%} sparsity")

            for v_n, v_m in v_layers:
                mask_v = kv_mask_1d.unsqueeze(1).expand_as(v_m.weight)
                self._apply_mask(v_m, mask_v)
                processed_layers.add(v_n)
                actual_sp = (~mask_v).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {v_n} (Coupled Head Row): pruned to {actual_sp:.1%} sparsity")

            for o_n, o_m in o_layers:
                mask_o = q_o_mask_1d.unsqueeze(0).expand_as(o_m.weight)
                self._apply_mask(o_m, mask_o)
                processed_layers.add(o_n)
                actual_sp = (~mask_o).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {o_n} (Coupled Head Col): pruned to {actual_sp:.1%} sparsity")

        elif attn_coupling_mode == "channel":
            # Channel-wise coupled pruning across Q, K, V output rows and O input columns
            k_score_1d = scores.get(k_layers[0][0]) if k_layers and scores.get(k_layers[0][0]) is not None else None
            if k_score_1d is not None:
                k_score_1d = k_score_1d.mean(dim=1) if k_score_1d.dim() == 2 else k_score_1d

            v_score_1d = scores.get(v_layers[0][0]) if v_layers and scores.get(v_layers[0][0]) is not None else None
            if v_score_1d is not None:
                v_score_1d = v_score_1d.mean(dim=1) if v_score_1d.dim() == 2 else v_score_1d

            channel_damages = q_score_1d + o_score_1d
            if k_score_1d is not None and k_score_1d.shape == q_score_1d.shape:
                channel_damages += k_score_1d
            if v_score_1d is not None and v_score_1d.shape == q_score_1d.shape:
                channel_damages += v_score_1d

            layer_sp = self._layer_sparsity(q_name)
            k_prune = max(1, int(math.ceil(d_model * layer_sp)))
            keep_count = max(1, d_model - k_prune)
            top_idx = torch.topk(channel_damages, k=keep_count, largest=True).indices

            mask_1d = torch.zeros(d_model, dtype=torch.bool, device=q_score.device)
            mask_1d[top_idx] = True

            for q_n, q_m in q_layers:
                mask_q = mask_1d.unsqueeze(1).expand_as(q_m.weight)
                self._apply_mask(q_m, mask_q)
                processed_layers.add(q_n)
                actual_sp = (~mask_q).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {q_n} (Coupled Channel Row): pruned to {actual_sp:.1%} sparsity")

            for k_n, k_m in k_layers:
                sub_mask = mask_1d[:k_m.weight.shape[0]]
                mask_k = sub_mask.unsqueeze(1).expand_as(k_m.weight)
                self._apply_mask(k_m, mask_k)
                processed_layers.add(k_n)
                actual_sp = (~mask_k).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {k_n} (Coupled Channel Row): pruned to {actual_sp:.1%} sparsity")

            for v_n, v_m in v_layers:
                sub_mask = mask_1d[:v_m.weight.shape[0]]
                mask_v = sub_mask.unsqueeze(1).expand_as(v_m.weight)
                self._apply_mask(v_m, mask_v)
                processed_layers.add(v_n)
                actual_sp = (~mask_v).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {v_n} (Coupled Channel Row): pruned to {actual_sp:.1%} sparsity")

            for o_n, o_m in o_layers:
                mask_o = mask_1d.unsqueeze(0).expand_as(o_m.weight)
                self._apply_mask(o_m, mask_o)
                processed_layers.add(o_n)
                actual_sp = (~mask_o).float().mean().item()
                logger.info(f"  [KFACOBDDiff] {o_n} (Coupled Channel Col): pruned to {actual_sp:.1%} sparsity")

    def _prune_fallback(
        self,
        prunable_layers: List[Tuple[str, nn.Module]],
        scores: Dict[str, torch.Tensor],
        strategy: Any,
        processed_layers: Set[str],
    ) -> None:
        """Fallback pruning for any uncoupled or unpaired prunable layers."""
        for name, module in prunable_layers:
            if name in processed_layers:
                continue
            agg = scores.get(name)
            if agg is None:
                continue
            if strategy is not None and hasattr(strategy, "compute_mask"):
                W_shape = module.weight.shape
                score_2d = agg if (agg.dim() == 2 and agg.shape == W_shape) else agg.unsqueeze(1).expand(W_shape)
                mask = strategy.compute_mask(score_2d, name)
            else:
                is_col = self._is_down_proj(name) or self._is_attn_out(name)
                score_1d = (agg.mean(0) if agg.dim() == 2 else agg) if is_col else (agg.mean(1) if agg.dim() == 2 else agg)
                sp = self._layer_sparsity(name)
                d = score_1d.numel()
                k = max(1, int(math.ceil(d * sp)))
                keep_count = max(1, d - k)
                top_idx = torch.topk(score_1d, k=keep_count, largest=True).indices
                mask_1d = torch.zeros(d, dtype=torch.bool, device=score_1d.device)
                mask_1d[top_idx] = True
                mask = mask_1d.unsqueeze(0).expand_as(module.weight) if is_col else mask_1d.unsqueeze(1).expand_as(module.weight)
            self._apply_mask(module, mask)
            actual_sp = (~mask).float().mean().item()
            logger.info(
                f"  [KFACOBDDiff] {name}: pruned to {actual_sp:.1%} sparsity"
            )


    #  Step 5 – Cleanup
    def cleanup(self) -> None:
        self._kfac_store.clear()
        self.score_store = PruningScoreStore()
        self.state.clear()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        logger.info("  [KFACOBDDiff] Cleanup complete.")
