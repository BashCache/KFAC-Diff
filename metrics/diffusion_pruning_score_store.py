"""
Stores pruning scores across ALL layers and ALL timesteps.

Score tensor layout per layer:
    scores[layer_name][timestep] = tensor of shape (n_rows,) or (d_out, d_in)
                                   one scalar per row (structural) or full 2-D

After calling aggregate(), we also store:
    aggregated[layer_name] = score tensor aggregated across all timesteps.

Supported aggregation methods
------------------------------
  'sum'          : unweighted sum:                 s = Σ_t s(t)
  'mean'/'average': uniform average:               s = (1/|T|) Σ_t s(t)
  'max_pooling'  : element-wise max across t:      s = max_t s(t)
  'log_decay'    : log-decay timestep weighting:   s = Σ_t α_t · s(t)
  'snr'          : SNR(t)-weighted sum (normalised):
                     SNR(t) = (1 - r(t)) / r(t),  r(t) = t / T
                     s = Σ_t [SNR(t) / Z] · s(t)
                   Up-weights low-noise (signal-rich) timesteps.
"""

import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import math

logger = logging.getLogger(__name__)

class PruningScoreStore:
    """
    Central store for Kronecker-OBD pruning scores.

    Structure
    ---------
    _raw[layer_name][timestep_idx] = score_tensor  (n_rows,)
    _aggregated[layer_name]        = score_tensor  (n_rows,)   # aggregated over T
    _timesteps                     = sorted list of unique timestep values
    _layer_names                   = insertion-ordered list of layer names
    """

    def __init__(self):
        self._raw        : Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        self._aggregated : Dict[str, torch.Tensor]            = {}
        self._timesteps  : List[int]  = []
        self._layer_order: List[str]  = []   # insertion order

    def add(
        self,
        layer_name: str,
        timestep  : int,
        score     : torch.Tensor,   # (n_rows,)
    ) -> None:
        """
        Store the OBD score for one (layer, timestep) pair.
        score must be a 1-D tensor of length = number of rows in that layer.
        """
        if layer_name not in self._layer_order:
            self._layer_order.append(layer_name)
        if timestep not in self._raw[layer_name]:
            if timestep not in self._timesteps:
                self._timesteps.append(timestep)

        self._raw[layer_name][timestep] = score.detach().float().cpu()

    def aggregate(
        self,
        method: str = "sum",
        total_timesteps: int = 1000,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        custom_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        For every layer, aggregate per-timestep scores into a single tensor
        and store in self._aggregated[layer_name].

        Parameters
        ----------
        method : 'sum' | 'mean' | 'average' | 'max_pooling' | 'log_decay' | 'snr'

            - 'sum'            : s = Σ_t s(t)
            - 'mean'/'average' : s = (1/|T|) Σ_t s(t)
            - 'max_pooling'    : s = max_t s(t)  (element-wise)
            - 'log_decay'      : s = Σ_t α_t · s(t),
                                   α_t = alpha_min + (alpha_max-alpha_min)/ln(T)
                                         * ln(T - t + 1)
            - 'snr'            : s = Σ_t w_t · s(t),
                                   w_t = SNR(t) / Z,
                                   SNR(t) = (1 - r(t)) / r(t),  r(t) = t/T
                                   (up-weights low-noise, signal-rich timesteps)

        total_timesteps : T in the diffusion process (e.g. 1000).
        alpha_min       : minimum weight at t=T (for log_decay).
        alpha_max       : maximum weight at t=1 (for log_decay).
        custom_weights  : optional dict {timestep -> weight}; overrides method.
        """
        self._timesteps.sort()
        self._aggregated.clear()

        # ── max_pooling: bypass the weight-vector path entirely ──────────── #
        if custom_weights is None and method == "max_pooling":
            for layer_name in self._layer_order:
                ts_scores = self._raw[layer_name]
                if not ts_scores:
                    continue
                t_sorted = sorted(ts_scores.keys())
                stacked = torch.stack(
                    [ts_scores[t].float() for t in t_sorted], dim=0
                )  # (T, n_rows)  or  (T, d_out, d_in)
                self._aggregated[layer_name] = stacked.max(dim=0).values

            logger.info(
                f"[ScoreStore] Aggregated scores (max_pooling) for "
                f"{len(self._aggregated)} layers over "
                f"{len(self._timesteps)} timesteps."
            )
            return

        # ── weight-based path ────────────────────────────────────────────── #
        weights: Dict[int, float] = {}
        if custom_weights is not None:
            weights = custom_weights

        elif method in ("log_decay", "log_weighted", "weighted", "log"):
            log_T = math.log(total_timesteps) if total_timesteps > 1 else 1.0
            for t in self._timesteps:
                t_term = max(1.0, float(total_timesteps - t + 1))
                alpha_t = alpha_min + ((alpha_max - alpha_min) / log_T) * math.log(t_term)
                weights[t] = float(alpha_t)
            logger.info(
                f"[ScoreStore] Using log-decay timestep weights (T={total_timesteps}, "
                f"alpha_min={alpha_min}, alpha_max={alpha_max}): "
                f"{ {t: round(w, 4) for t, w in sorted(weights.items())} }"
            )

        elif method == "snr":
            # SNR(t) = (1 - r(t)) / r(t),  r(t) = t / T  (linear masking schedule)
            # High SNR at low t (few masks) → up-weights signal-rich timesteps.
            # Weights are normalised so Σ_t w_t = 1.
            raw_w: Dict[int, float] = {}
            for t in self._timesteps:
                r = max(t / max(total_timesteps, 1), 1e-6)
                raw_w[t] = (1.0 - r) / r
            total_w = sum(raw_w.values()) + 1e-12
            for t in self._timesteps:
                weights[t] = raw_w[t] / total_w
            logger.info(
                f"[ScoreStore] Using SNR timestep weights (T={total_timesteps}): "
                f"{ {t: round(w, 4) for t, w in sorted(weights.items())} }"
            )

        elif method in ("mean", "average"):
            n_t = max(len(self._timesteps), 1)
            for t in self._timesteps:
                weights[t] = 1.0 / n_t

        elif method == "sum":
            for t in self._timesteps:
                weights[t] = 1.0

        else:
            raise ValueError(
                f"Unknown aggregation method '{method}'. "
                "Choose: 'sum' | 'mean' | 'average' | 'max_pooling' | "
                "'log_decay' | 'snr'"
            )

        for layer_name in self._layer_order:
            ts_scores = self._raw[layer_name]
            if not ts_scores:
                continue

            t_sorted = sorted(ts_scores.keys())
            weighted_scores = [
                ts_scores[t] * weights.get(t, 1.0)
                for t in t_sorted
            ]
            stacked = torch.stack(weighted_scores, dim=0)   # (T, n_rows) or (T, d_out, d_in)
            self._aggregated[layer_name] = stacked.sum(dim=0)

        logger.info(
            f"[ScoreStore] Aggregated scores ({method}) for "
            f"{len(self._aggregated)} layers over "
            f"{len(self._timesteps)} timesteps."
        )

    # ------------------------------------------------------------------ #
    #  Reading                                                             #
    # ------------------------------------------------------------------ #
    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_order)

    @property
    def timesteps(self) -> List[int]:
        return list(self._timesteps)

    def raw(
        self,
        layer_name: str,
        timestep  : int,
    ) -> Optional[torch.Tensor]:
        return self._raw.get(layer_name, {}).get(timestep)

    def raw_matrix(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Returns (T, n_rows) matrix for one layer.
        Rows = timesteps (sorted), Cols = neurons/rows in that layer.
        """
        ts_scores = self._raw.get(layer_name)
        if not ts_scores:
            return None
        t_sorted = sorted(ts_scores.keys())
        return torch.stack([ts_scores[t] for t in t_sorted], dim=0)

    def aggregated(self, layer_name: str) -> Optional[torch.Tensor]:
        return self._aggregated.get(layer_name)

    def all_aggregated(self) -> Dict[str, torch.Tensor]:
        return dict(self._aggregated)
    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "raw"        : dict(self._raw),
            "aggregated" : self._aggregated,
            "timesteps"  : self._timesteps,
            "layer_order": self._layer_order,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"[ScoreStore] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PruningScoreStore":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        store               = cls()
        store._raw          = defaultdict(dict, payload["raw"])
        store._aggregated   = payload["aggregated"]
        store._timesteps    = payload["timesteps"]
        store._layer_order  = payload["layer_order"]
        logger.info(f"[ScoreStore] Loaded from {path}")
        return store

    # ------------------------------------------------------------------ #
    #  Debug summary                                                     #
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = [
            f"PruningScoreStore",
            f"  Layers    : {len(self._layer_order)}",
            f"  Timesteps : {self._timesteps}",
        ]
        for ln in self._layer_order:
            agg = self._aggregated.get(ln)
            shape = tuple(agg.shape) if agg is not None else "?"
            lines.append(f"  {ln:<60s}: {shape}")
        return "\n".join(lines)