"""
kfac_obs_diffusion.py
=====================
Kronecker-Factored Optimal Brain Surgeon (KFAC-OBS) Structural Pruner
for Discrete Diffusion Language Models.

Theory
------
KFAC-OBS computes the exact OBS saliency and weight reconstruction using the full Kronecker-factored
inverse Hessian.

The Fisher / Gauss-Newton Hessian for a Linear layer W (d_out x d_in) is:
    H ≈ A ⊗ G      (d_in·d_out x d_in·d_out)

where:
    A = (1/N) Σ_n  x_n x_n^T      in R^{d_in  x d_in }   (input covariance)
    G = (1/N) Σ_n  g_n g_n^T      in R^{d_out x d_out}   (gradient covariance)

Kronecker-OBS Saliency & Weight Reconstruction
-----------------------------------------------
Removing ROW r  (output neuron r, structured output pruning):
    Saliency:  L_r = 0.5 * W[r,:] @ A @ W[r,:].T / G_inv[r,r]
    Update:    ΔW[i,:] -= (G_inv[i,r] / G_inv[r,r]) * W[r,:]   for all i

Removing COLUMN c  (input neuron c, structured input pruning):
    Saliency:  L_c = 0.5 * W[:,c].T @ G @ W[:,c] / A_inv[c,c]
    Update:    ΔW[:,j] -= W[:,c] * (A_inv[c,j] / A_inv[c,c])   for all j

Weight updates are applied greedily in ascending saliency order (lowest-damage
units pruned first, with reconstruction reducing damage for remaining units).

Implementation note
-------------------
Full A (d_in x d_in) and G (d_out x d_out) matrices are accumulated across ALL
calibration timesteps (aggregated, not stored per timestep) via the
DiffusionKFACCollector's full_matrix=True mode.
"""

import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Timestep weighting helpers                                                                                                     #
# ======================================================================== #

def _build_timestep_weights(
    tag       : str,
    timesteps : List[int],
    T_total   : int   = 1000,
    alpha_min : float = 0.0,
    alpha_max : float = 1.0,
) -> Dict[int, float]:
    """Compute normalised per-timestep weights for score aggregation."""
    T_max = float(max(timesteps)) if timesteps else float(T_total)
    w: Dict[int, float] = {}
    for t in timesteps:
        r = max(0.0, 1.0 - float(t) / T_max)
        if tag == "sum":
            w[t] = 1.0
        elif tag == "mean":
            w[t] = 1.0 / max(len(timesteps), 1)
        elif tag == "inv_linear":
            w[t] = r
        elif tag == "inv_sqrt":
            w[t] = math.sqrt(r)
        elif tag == "inv_quadratic":
            w[t] = r * r
        elif tag == "exp_decay":
            w[t] = math.exp(-2.0 * float(t) / T_max)
        elif tag == "cosine_decay":
            w[t] = 0.5 * (1.0 + math.cos(math.pi * float(t) / T_max))
        elif tag in ("log_decay", "log_weighted"):
            log_T  = math.log(T_total) if T_total > 1 else 1.0
            t_term = max(1.0, float(T_total - t + 1))
            w[t]   = alpha_min + ((alpha_max - alpha_min) / log_T) * math.log(t_term)
        elif tag == "snr":
            w[t] = max(0.0, float(T_max - t) / (float(t) + 10.0))
        else:
            logger.warning(f"[KFAC-OBS] Unknown weight tag '{tag}', falling back to 'sum'.")
            w[t] = 1.0
    total = sum(w.values()) or 1.0
    return {t: v / total for t, v in w.items()}


# ======================================================================== #
#  Exact K-FAC OBS score + weight reconstruction                                                               #
# ======================================================================== #

def _safe_inv(M: torch.Tensor, damp: float) -> torch.Tensor:
    """
    Damped inversion of a symmetric PSD matrix M.
    Adds damp * ||M||_F / d * I for scale-relative damping,
    then inverts via Cholesky. Falls back to torch.linalg.inv.
    """
    d = M.shape[0]
    # Scale-relative damping: avoids over/under-damping across layers
    scale = M.norm(p="fro").item() / d + 1e-8
    M_d = M + (damp * scale) * torch.eye(d, dtype=M.dtype, device=M.device)
    try:
        L = torch.linalg.cholesky(M_d)
        return torch.cholesky_inverse(L)
    except Exception:
        return torch.linalg.inv(M_d)


def _obs_row_saliencies(
    W     : torch.Tensor,   # (d_out, d_in)
    A     : torch.Tensor,   # (d_in,  d_in)
    G_inv : torch.Tensor,   # (d_out, d_out)  pre-computed G^{-1}
) -> torch.Tensor:
    """
    Exact K-FAC OBS row saliency for each output neuron r:
        L_r = 0.5 * W[r,:] @ A @ W[r,:].T / G_inv[r,r]
    Returns shape (d_out,).
    """
    WA       = W.float() @ A.float()             # (d_out, d_in)
    WAW_diag = (WA * W.float()).sum(dim=1)        # (d_out,)
    denom    = G_inv.diag().clamp(min=1e-12)      # (d_out,)
    return 0.5 * WAW_diag / denom


def _obs_col_saliencies(
    W     : torch.Tensor,   # (d_out, d_in)
    G     : torch.Tensor,   # (d_out, d_out)
    A_inv : torch.Tensor,   # (d_in,  d_in)   pre-computed A^{-1}
) -> torch.Tensor:
    """
    Exact K-FAC OBS column saliency for each input neuron c:
        L_c = 0.5 * W[:,c].T @ G @ W[:,c] / A_inv[c,c]
    Returns shape (d_in,).
    """
    GW        = G.float() @ W.float()             # (d_out, d_in)
    WtGW_diag = (W.float() * GW).sum(dim=0)       # (d_in,)
    denom     = A_inv.diag().clamp(min=1e-12)      # (d_in,)
    return 0.5 * WtGW_diag / denom


def _obs_reconstruct_rows(
    W      : torch.Tensor,   # (d_out, d_in)  modified IN PLACE
    G_inv  : torch.Tensor,   # (d_out, d_out)
    rows   : torch.Tensor,   # LongTensor of row indices to prune (ascending saliency order)
) -> None:
    """
    Greedy K-FAC OBS weight reconstruction for row pruning.
    For each pruned row r:
        W[i, :] -= (G_inv[i, r] / G_inv[r, r]) * W[r, :]   for all i
        W[r, :] = 0
    Applied sequentially (greedy): each update lowers damage for subsequent prunings.

    NOTE: W[r,:] is cloned BEFORE the in-place subtraction so that the
    update for row i=r does not use the already-modified value.
    """
    for r in rows:
        r     = r.item()
        g_rr  = G_inv[r, r].clamp(min=1e-12)
        w_r   = W[r, :].clone()              # snapshot before in-place update
        coeff = G_inv[:, r] / g_rr           # (d_out,)
        W    -= coeff.unsqueeze(1) * w_r.unsqueeze(0)
        W[r, :] = 0.0


def _obs_reconstruct_cols(
    W      : torch.Tensor,   # (d_out, d_in)  modified IN PLACE
    A_inv  : torch.Tensor,   # (d_in,  d_in)
    cols   : torch.Tensor,   # LongTensor of col indices to prune (ascending saliency order)
) -> None:
    """
    Greedy K-FAC OBS weight reconstruction for column pruning.
    For each pruned col c:
        W[:, j] -= W[:, c] * (A_inv[c, j] / A_inv[c, c])   for all j
        W[:, c] = 0

    NOTE: W[:,c] is cloned BEFORE the in-place subtraction so that the
    update for column j=c does not use the already-modified value.
    """
    for c in cols:
        c     = c.item()
        a_cc  = A_inv[c, c].clamp(min=1e-12)
        w_c   = W[:, c].clone()              # snapshot before in-place update
        coeff = A_inv[c, :] / a_cc           # (d_in,)
        W    -= w_c.unsqueeze(1) * coeff.unsqueeze(0)
        W[:, c] = 0.0


# ======================================================================== #
#  KFAC-OBS Pruner                                                                                                                              #
# ======================================================================== #

@registry.register("kfac_obs_diffusion")
class KFACOBSDiffusionPruner(BasePruner):
    """
    KFAC-OBS Structural Pruner for Discrete Diffusion LMs.

    Uses exact Kronecker-factored OBS saliency scores and performs weight
    reconstruction after each pruned row/column to compensate remaining weights.
    """

    def __init__(self, model: nn.Module, config: PruningConfig):
        super().__init__(model, config)
        self._collector         : Optional[DiffusionKFACCollector]                        = None
        self._kfac_store        : Dict[str, Dict[int, Dict[str, Optional[torch.Tensor]]]] = {}
        self._full_matrix_store : Dict[str, Dict[str, Optional[torch.Tensor]]]            = {}
        self._inv_cache         : Dict[str, Dict[str, torch.Tensor]]                      = {}
        self.score_store        : PruningScoreStore                                        = PruningScoreStore()
        self._timestep_list     : List[int]                                                = []
        self.score_dir          : Optional[str]                                            = None
        self.mask_token_id      : Optional[int]                                            = None
        self.total_timesteps    : int                                                      = 1000
        self.num_timesteps      : int                                                      = 32
        self.timestep_strategy  : str                                                      = "uniform"
        self.damp               : float                                                    = 1e-3
        self.seed               : int                                                      = 42
        self.agg_method         : str                                                      = "log_decay"
        self.alpha_min          : float                                                    = 0.0
        self.alpha_max          : float                                                    = 1.0

    # ── Layer-type helpers ─────────────────────────────────────────────── #
    def _is_up_proj(self, name: str) -> bool:
        return _is_up_proj_layer(name, tuple(self._get_arg("up_proj_keywords", UP_PROJ_KEYWORDS)))

    def _is_down_proj(self, name: str) -> bool:
        return _is_down_proj_layer(name, tuple(self._get_arg("down_proj_keywords", DOWN_PROJ_KEYWORDS)))

    def _is_q_proj(self, name: str) -> bool:
        return _is_q_proj_layer(name, tuple(self._get_arg("attn_q_keywords", ATTN_Q_KEYWORDS)))

    def _is_k_proj(self, name: str) -> bool:
        return _is_k_proj_layer(name, tuple(self._get_arg("attn_k_keywords", ATTN_K_KEYWORDS)))

    def _is_v_proj(self, name: str) -> bool:
        return _is_v_proj_layer(name, tuple(self._get_arg("attn_v_keywords", ATTN_V_KEYWORDS)))

    def _is_attn_out(self, name: str) -> bool:
        return _is_attn_out_layer(name, tuple(self._get_arg("attn_out_keywords", ATTN_OUT_KEYWORDS)))

    def _get_block_prefix_for_layer(self, name: str) -> str:
        all_kw = (
            tuple(self._get_arg("up_proj_keywords",   UP_PROJ_KEYWORDS))
            + tuple(self._get_arg("down_proj_keywords", DOWN_PROJ_KEYWORDS))
            + tuple(self._get_arg("attn_q_keywords",   ATTN_Q_KEYWORDS))
            + tuple(self._get_arg("attn_k_keywords",   ATTN_K_KEYWORDS))
            + tuple(self._get_arg("attn_v_keywords",   ATTN_V_KEYWORDS))
            + tuple(self._get_arg("attn_out_keywords", ATTN_OUT_KEYWORDS))
        )
        return _get_block_prefix(name, all_kw)

    def _layer_sparsity(self, layer_name: str) -> float:
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
        return float(self._get_arg("sparsity", self._get_arg("target_sparsity", 0.3)))

    # ------------------------------------------------------------------ #
    #  Step 1 – Prepare                                                 #
    # ------------------------------------------------------------------ #
    def prepare(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        prunable    = self._get_prunable_layers()
        layer_names = [n for n, _ in prunable]

        logger.info(f"  [KFAC-OBS] Target layers for K-FAC collection ({len(layer_names)}):")
        for n in layer_names:
            logger.info(f"    - {n}")

        # full_matrix=True: accumulate full A and G covariance matrices
        self._collector = DiffusionKFACCollector(
            self.model, layer_names, self.device,
            full_matrix=True,
        )

        self.total_timesteps   = int(self._get_arg("total_timesteps",   self._get_arg("total_steps", 1000)))
        self.num_timesteps     = int(self._get_arg("num_timesteps",     self._get_arg("n_timesteps", 32)))
        self.timestep_strategy = str(self._get_arg("timestep_strategy", self._get_arg("timestep_schedule", "uniform")))
        self.damp              = float(self._get_arg("damp",            1e-3))
        self.seed              = int(self._get_arg("seed",              getattr(self.config, "seed", 42)))
        self.agg_method        = str(self._get_arg("agg_method",        self._get_arg("score_agg_method", self._get_arg("agg_strategy", "log_decay"))))
        self.alpha_min         = float(self._get_arg("alpha_min",       0.0))
        self.alpha_max         = float(self._get_arg("alpha_max",       1.0))

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
            f"  [KFAC-OBS] Prepared: {len(layer_names)} layers, "
            f"{len(self._timestep_list)} timesteps ({self.timestep_strategy}), "
            f"damp={self.damp}  [full_matrix=True → exact OBS weight reconstruction]"
        )

        score_dir = self._get_arg("score_dir")
        if score_dir is None:
            out_dir = self._get_arg("output_dir", getattr(self.config, "output_dir", None))
            if out_dir:
                score_dir = os.path.join(out_dir, "scores")

        self.score_dir = os.path.abspath(score_dir) if score_dir else None

    # ------------------------------------------------------------------ #
    #  Step 2 – Collect                                                 #
    # ------------------------------------------------------------------ #
    def collect(self, calibration_loader) -> None:
        logger.info("  [KFAC-OBS] Collecting K-FAC statistics across timesteps ...")
        assert self._collector is not None, "Call prepare() before collect()."

        if self.mask_token_id is None:
            mask_token_id = self._get_arg("mask_token_id")
            if mask_token_id is None:
                mask_token_id = get_mask_token_id(getattr(self.model, "tokenizer", self.model))
            self.mask_token_id = mask_token_id

        total_batches = 0
        pbar = tqdm(calibration_loader, desc="[KFAC-OBS] Collecting K-FAC", unit="batch", dynamic_ncols=True)
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

                input_ids  = input_ids.to(self.device)
                batch_size = input_ids.shape[0]

                t_iter = [t_batch[i].item() for i in range(batch_size)] if t_batch is not None else self._timestep_list

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
                        del loss, noisy_ids, mask
                    except Exception as exc:
                        logger.warning(f"  [KFAC-OBS] Error at t={t_val}: {exc}. Skipping.")
                        self.model.zero_grad(set_to_none=True)
                        continue

                total_batches += 1
                pbar.set_postfix({"batches": total_batches, "timesteps/batch": len(t_iter)})
                if torch.cuda.is_available() and total_batches % 4 == 0:
                    torch.cuda.empty_cache()

        self._kfac_store        = self._collector.get_all()
        self._full_matrix_store = self._collector.get_full_matrices()

        n_layer_ts = sum(len(ts) for ts in self._kfac_store.values())
        logger.info(f"  [KFAC-OBS] Collection complete. (layer × timestep) diagonal entries: {n_layer_ts}")

        total_bytes = sum(
            t.element_size() * t.nelement()
            for mats in self._full_matrix_store.values()
            for t in mats.values()
            if t is not None
        )
        logger.info(
            f"  [KFAC-OBS] Full-matrix store in-memory size: {total_bytes / (1024**2):.1f} MB  "
            f"(NOT saved to disk — inverses will be computed and saved instead)"
        )

    # ------------------------------------------------------------------ #
    #  Step 3 – Compute scores                                            #
    # ------------------------------------------------------------------ #
    def compute_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute exact K-FAC OBS saliency scores and cache G_inv / A_inv
        per layer for reuse in apply_pruning weight reconstruction.

        Row-pruned layers (up_proj, q/k/v):
            score_r = 0.5 * W[r,:] @ A @ W[r,:].T / G_inv[r,r]

        Column-pruned layers (down_proj, attn_out):
            score_c = 0.5 * W[:,c].T @ G @ W[:,c] / A_inv[c,c]

        Memory note
        -----------
        A and G are loaded one layer at a time, inverted, then immediately freed
        from _full_matrix_store so that peak memory is bounded to ~2 layers
        rather than the entire store.
        """
        logger.info("  [KFAC-OBS] Computing exact OBS scores and inverses ...")
        self._inv_cache = {}

        prunable_layers = self._get_prunable_layers()
        scores: Dict[str, torch.Tensor] = {}

        for name, module in tqdm(prunable_layers, desc="[KFAC-OBS] OBS scores", unit="layer", dynamic_ncols=True):
            W    = module.weight.data.float()   # (d_out, d_in)
            mats = self._full_matrix_store.get(name, {})
            A    = mats.get("A")                # (d_in,  d_in) or None
            G    = mats.get("G")                # (d_out, d_out) or None

            if A is None or G is None:
                logger.warning(f"  [KFAC-OBS] No full matrices for '{name}'. Falling back to diagonal OBD score.")
                ts_store = self._kfac_store.get(name, {})
                for t in self._timestep_list:
                    stats = ts_store.get(t)
                    if stats and stats.get("diag_G") is not None:
                        fn = _kfac_obd_col_scores if (self._is_down_proj(name) or self._is_attn_out(name)) else _kfac_obd_row_scores
                        scores[name] = fn(W, stats)
                        break
                continue

            dev = W.device
            A   = A.float().to(dev)
            G   = G.float().to(dev)

            if self._is_down_proj(name) or self._is_attn_out(name):
                # Column pruning path: need A_inv for saliency and reconstruction
                A_inv                 = _safe_inv(A, self.damp)
                scores[name]          = _obs_col_saliencies(W, G, A_inv).cpu()
                self._inv_cache[name] = {"A_inv": A_inv.cpu()}
                del A_inv
            else:
                # Row pruning path: need G_inv for saliency and reconstruction
                G_inv                 = _safe_inv(G, self.damp)
                scores[name]          = _obs_row_saliencies(W, A, G_inv).cpu()
                self._inv_cache[name] = {"G_inv": G_inv.cpu()}
                del G_inv

            # Free the raw A/G tensors for this layer immediately — they are no
            # longer needed now that the inverse is cached.
            self._full_matrix_store[name] = {"A": None, "G": None}
            del A, G
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"    {name}: score range [{scores[name].min():.3e}, {scores[name].max():.3e}]")

        logger.info(f"  [KFAC-OBS] Scores computed for {len(scores)}/{len(prunable_layers)} layers.")

        if self.score_dir is not None:
            os.makedirs(self.score_dir, exist_ok=True)
            # Save saliency scores
            torch.save(scores, os.path.join(self.score_dir, "obs_scores.pt"))
            logger.info(f"  [KFAC-OBS] Scores saved to {self.score_dir}/obs_scores.pt")
            # Save inverses — required to re-run apply_pruning without re-collecting
            inv_save_path = os.path.join(self.score_dir, "obs_inv_cache.pt")
            torch.save(self._inv_cache, inv_save_path)
            logger.info(f"  [KFAC-OBS] Inverse cache saved to {inv_save_path}")

        return scores

    # ---------------------------------------------------------------------------------------------------- #
    #  Step 4 – Apply pruning with OBS weight reconstruction            #
    # ---------------------------------------------------------------------------------------------------- #
    def apply_pruning(
        self,
        scores  : Dict[str, torch.Tensor],
        strategy = None,
    ) -> None:
        """
        Apply structured pruning with exact K-FAC OBS weight reconstruction.

        Units are pruned in ascending saliency order (greedy OBS).  After each
        pruned row/column, surviving weights are updated to absorb the damage.

        Coupling logic (MLP geometric/arithmetic/sum, head/channel attention)
        determines which units share a mask; OBS reconstruction is then applied
        independently within each projection matrix.
        """
        prunable_layers = self._get_prunable_layers()

        blocks = defaultdict(list)
        for name, module in prunable_layers:
            prefix = self._get_block_prefix_for_layer(name)
            blocks[prefix].append((name, module))

        processed_layers: set        = set()
        coupling_mode      = self._get_arg("coupling_mode", "geometric")
        attn_coupling_mode = self._get_arg("attn_coupling_mode", "head")

        # ─── helper: build mask_1d from coupled scores ─────────────────── #
        def _couple_and_mask(u_scores_1d, d_scores_1d, d_ff, layer_sp):
            if coupling_mode == "sum":
                coupled = torch.stack(u_scores_1d).sum(0) + torch.stack(d_scores_1d).sum(0)
            elif coupling_mode == "geometric":
                avg_u = torch.stack(u_scores_1d).mean(0)
                avg_d = torch.stack(d_scores_1d).mean(0)
                coupled = torch.sqrt(torch.clamp(avg_u * avg_d, min=1e-12))
            else:
                avg_u = torch.stack(u_scores_1d).mean(0)
                avg_d = torch.stack(d_scores_1d).mean(0)
                coupled = 0.5 * (avg_u + avg_d)

            n_prune   = max(1, int(math.ceil(d_ff * layer_sp)))
            sorted_   = torch.argsort(coupled)   # ascending = lowest damage first
            prune_idx = sorted_[:n_prune]
            mask_1d   = torch.zeros(d_ff, dtype=torch.bool, device=coupled.device)
            mask_1d[sorted_[n_prune:]] = True
            return mask_1d, prune_idx

        # ─── helper: OBS update + apply_mask for a weight matrix ─────── #
        do_reconstruct = bool(self._get_arg("reconstruct", True))

        def _obs_and_mask(name, module, mask_1d, prune_idx, is_col):
            W  = module.weight.data.float()
            cache = self._inv_cache.get(name, {})
            if is_col:
                A_inv = cache.get("A_inv")
                if do_reconstruct and A_inv is not None:
                    _obs_reconstruct_cols(W, A_inv.to(W.device), prune_idx.to(W.device))
                    module.weight.data.copy_(W.to(module.weight.dtype))
                mask_2d = mask_1d.unsqueeze(0).expand_as(module.weight)
            else:
                G_inv = cache.get("G_inv")
                if do_reconstruct and G_inv is not None:
                    _obs_reconstruct_rows(W, G_inv.to(W.device), prune_idx.to(W.device))
                    module.weight.data.copy_(W.to(module.weight.dtype))
                mask_2d = mask_1d.unsqueeze(1).expand_as(module.weight)
            self._apply_mask(module, mask_2d)
            return mask_2d

        for prefix, block_layers in blocks.items():

            # ── 1. MLP Channels Coupling ───────────────────────────────── #
            up_layers   = [(n, m) for n, m in block_layers if self._is_up_proj(n)]
            down_layers = [(n, m) for n, m in block_layers if self._is_down_proj(n)]

            if up_layers and down_layers:
                all_up_s   = [scores.get(n) for n, _ in up_layers   if scores.get(n) is not None]
                all_down_s = [scores.get(n) for n, _ in down_layers if scores.get(n) is not None]

                if all_up_s and all_down_s:
                    u_1d   = [s.mean(1) if s.dim() == 2 else s for s in all_up_s]
                    d_1d   = [s.mean(0) if s.dim() == 2 else s for s in all_down_s]
                    d_ff   = u_1d[0].numel()
                    sp     = self._layer_sparsity(up_layers[0][0])
                    mask_1d, prune_idx = _couple_and_mask(u_1d, d_1d, d_ff, sp)

                    for u_name, u_module in up_layers:
                        mask_up = _obs_and_mask(u_name, u_module, mask_1d, prune_idx, is_col=False)
                        if u_module.bias is not None:
                            with torch.no_grad():
                                u_module.bias.data *= mask_1d.to(device=u_module.bias.device, dtype=u_module.bias.dtype)
                        logger.info(f"  [KFAC-OBS] {u_name} (OBS Row): pruned to {(~mask_up).float().mean():.1%} sparsity")
                        processed_layers.add(u_name)

                    for d_name, d_module in down_layers:
                        mask_dn = _obs_and_mask(d_name, d_module, mask_1d, prune_idx, is_col=True)
                        logger.info(f"  [KFAC-OBS] {d_name} (OBS Col): pruned to {(~mask_dn).float().mean():.1%} sparsity")
                        processed_layers.add(d_name)

            # ── 2. Attention Coupling ──────────────────────────────────── #
            if attn_coupling_mode in ("head", "channel"):
                q_layers = [(n, m) for n, m in block_layers if self._is_q_proj(n)]
                k_layers = [(n, m) for n, m in block_layers if self._is_k_proj(n)]
                v_layers = [(n, m) for n, m in block_layers if self._is_v_proj(n)]
                o_layers = [(n, m) for n, m in block_layers if self._is_attn_out(n)]

                if q_layers and o_layers:
                    q_name, q_module = q_layers[0]
                    o_name, o_module = o_layers[0]
                    q_score = scores.get(q_name)
                    o_score = scores.get(o_name)

                    if q_score is not None and o_score is not None:
                        q1d = q_score.mean(1) if q_score.dim() == 2 else q_score
                        o1d = o_score.mean(0) if o_score.dim() == 2 else o_score
                        d_model = q1d.numel()

                        k1d = None
                        if k_layers and scores.get(k_layers[0][0]) is not None:
                            ks = scores.get(k_layers[0][0])
                            k1d = ks.mean(1) if ks.dim() == 2 else ks

                        v1d = None
                        if v_layers and scores.get(v_layers[0][0]) is not None:
                            vs = scores.get(v_layers[0][0])
                            v1d = vs.mean(1) if vs.dim() == 2 else vs

                        if attn_coupling_mode == "head":
                            cfg        = getattr(self.model, "config", None)
                            n_heads    = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 32)))
                            n_kv_heads = getattr(cfg, "n_kv_heads", getattr(cfg, "num_key_value_heads", n_heads)) or n_heads
                            head_dim   = d_model // n_heads
                            g          = max(1, n_heads // n_kv_heads)

                            head_dmg = torch.zeros(n_heads, device=q_score.device, dtype=q_score.dtype)
                            for h in range(n_heads):
                                cs, ce   = h * head_dim, (h + 1) * head_dim
                                hkv      = h // g
                                ks_      = hkv * head_dim
                                ke_      = (hkv + 1) * head_dim
                                head_dmg[h] = (q1d[cs:ce].sum() + o1d[cs:ce].sum()
                                               + (k1d[ks_:ke_].sum() / g if k1d is not None else 0.0)
                                               + (v1d[ks_:ke_].sum() / g if v1d is not None else 0.0))

                            sp            = self._layer_sparsity(q_name)
                            n_prune_h     = max(1, int(math.ceil(n_heads * sp)))
                            sorted_h      = torch.argsort(head_dmg)   # ascending
                            prune_h       = sorted_h[:n_prune_h]

                            hkm = torch.zeros(n_heads, dtype=torch.bool, device=q_score.device)
                            hkm[sorted_h[n_prune_h:]] = True
                            q_o_mask = hkm.repeat_interleave(head_dim)

                            kv_hkm = torch.zeros(n_kv_heads, dtype=torch.bool, device=q_score.device)
                            for hkv in range(n_kv_heads):
                                kv_hkm[hkv] = hkm[hkv * g : (hkv + 1) * g].any()
                            kv_mask = kv_hkm.repeat_interleave(head_dim)

                            prune_ch_q  = (~q_o_mask).nonzero(as_tuple=False).squeeze(1)
                            prune_ch_kv = (~kv_mask).nonzero(as_tuple=False).squeeze(1)

                            for q_n, q_m in q_layers:
                                m = _obs_and_mask(q_n, q_m, q_o_mask, prune_ch_q, is_col=False)
                                processed_layers.add(q_n)
                                logger.info(f"  [KFAC-OBS] {q_n} (OBS Head Row): {(~m).float().mean():.1%}")
                            for k_n, k_m in k_layers:
                                m = _obs_and_mask(k_n, k_m, kv_mask, prune_ch_kv, is_col=False)
                                processed_layers.add(k_n)
                                logger.info(f"  [KFAC-OBS] {k_n} (OBS Head Row): {(~m).float().mean():.1%}")
                            for v_n, v_m in v_layers:
                                m = _obs_and_mask(v_n, v_m, kv_mask, prune_ch_kv, is_col=False)
                                processed_layers.add(v_n)
                                logger.info(f"  [KFAC-OBS] {v_n} (OBS Head Row): {(~m).float().mean():.1%}")
                            for o_n, o_m in o_layers:
                                m = _obs_and_mask(o_n, o_m, q_o_mask, prune_ch_q, is_col=True)
                                processed_layers.add(o_n)
                                logger.info(f"  [KFAC-OBS] {o_n} (OBS Head Col): {(~m).float().mean():.1%}")

                        elif attn_coupling_mode == "channel":
                            ch_dmg = q1d + o1d
                            if k1d is not None and k1d.shape == q1d.shape:
                                ch_dmg += k1d
                            if v1d is not None and v1d.shape == q1d.shape:
                                ch_dmg += v1d

                            sp        = self._layer_sparsity(q_name)
                            n_prune_c = max(1, int(math.ceil(d_model * sp)))
                            sorted_c  = torch.argsort(ch_dmg)
                            prune_c   = sorted_c[:n_prune_c]
                            mask_1d_c = torch.zeros(d_model, dtype=torch.bool, device=q_score.device)
                            mask_1d_c[sorted_c[n_prune_c:]] = True

                            for q_n, q_m in q_layers:
                                m = _obs_and_mask(q_n, q_m, mask_1d_c, prune_c, is_col=False)
                                processed_layers.add(q_n)
                                logger.info(f"  [KFAC-OBS] {q_n} (OBS Ch Row): {(~m).float().mean():.1%}")
                            for k_n, k_m in k_layers:
                                sub_p = prune_c[prune_c < k_m.weight.shape[0]]
                                sm    = mask_1d_c[:k_m.weight.shape[0]]
                                m = _obs_and_mask(k_n, k_m, sm, sub_p, is_col=False)
                                processed_layers.add(k_n)
                                logger.info(f"  [KFAC-OBS] {k_n} (OBS Ch Row): {(~m).float().mean():.1%}")
                            for v_n, v_m in v_layers:
                                sub_p = prune_c[prune_c < v_m.weight.shape[0]]
                                sm    = mask_1d_c[:v_m.weight.shape[0]]
                                m = _obs_and_mask(v_n, v_m, sm, sub_p, is_col=False)
                                processed_layers.add(v_n)
                                logger.info(f"  [KFAC-OBS] {v_n} (OBS Ch Row): {(~m).float().mean():.1%}")
                            for o_n, o_m in o_layers:
                                m = _obs_and_mask(o_n, o_m, mask_1d_c, prune_c, is_col=True)
                                processed_layers.add(o_n)
                                logger.info(f"  [KFAC-OBS] {o_n} (OBS Ch Col): {(~m).float().mean():.1%}")

        # ── Fallback for any unpaired / uncoupled layers ──────────────── #
        for name, module in prunable_layers:
            if name in processed_layers:
                continue
            agg = scores.get(name)
            if agg is None:
                continue

            is_col   = self._is_down_proj(name) or self._is_attn_out(name)
            score_1d = (agg.mean(0) if agg.dim() == 2 else agg) if is_col else (agg.mean(1) if agg.dim() == 2 else agg)
            sp       = self._layer_sparsity(name)
            d        = score_1d.numel()
            n_prune  = max(1, int(math.ceil(d * sp)))
            sorted_  = torch.argsort(score_1d)
            prune_idx = sorted_[:n_prune]
            mask_1d  = torch.zeros(d, dtype=torch.bool, device=score_1d.device)
            mask_1d[sorted_[n_prune:]] = True

            mask_2d = _obs_and_mask(name, module, mask_1d, prune_idx, is_col=is_col)
            logger.info(f"  [KFAC-OBS] {name} (OBS fallback): pruned to {(~mask_2d).float().mean():.1%} sparsity")

    # ------------------------------------------------------------------ #
    #  Step 5 – Cleanup                                                   #
    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        self._kfac_store.clear()
        self._full_matrix_store.clear()
        self._inv_cache.clear()
        self.score_store = PruningScoreStore()
        self.state.clear()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("  [KFAC-OBS] Cleanup complete.")
