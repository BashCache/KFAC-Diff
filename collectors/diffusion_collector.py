"""
Timestep-aware K-FAC statistics collector for discrete diffusion language models.

At each denoising timestep t, a fraction r(t) of input tokens are replaced with [MASK] tokens. 
We run forward + backward and accumulate:
    A(t) = (1/N) X(t)^T X(t)       -- input activation covariance (d_in, d_in)
    G(t) = (1/N) dL/dH^T · dL/dH   -- gradient covariance         (d_out, d_out)

where X(t) is the activation input to each Linear layer at timestep t.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TimestepKFACStats:
    """
    Accumulates exact 1D diagonal statistics for Kronecker-OBD:
      - diag_A   : (1/N) sum(inp^2)           shape (d_in,)
      - diag_WAW : (1/N) sum(out^2)           shape (d_out,)
      - diag_G   : (1/N) sum(grad_out^2)      shape (d_out,)
      - diag_WtGW: (1/N) sum(grad_in^2)       shape (d_in,)
    """
    def __init__(self, device: torch.device):
        self.device    = device
        self.diag_A    : Optional[torch.Tensor] = None  
        self.diag_WAW  : Optional[torch.Tensor] = None
        self.diag_G    : Optional[torch.Tensor] = None
        self.diag_WtGW : Optional[torch.Tensor] = None
        self.n_tokens  : int                    = 0

    def accumulate_forward(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        inp = inp.detach()
        out = out.detach()
        if inp.dim() >= 3:
            inp = inp.reshape(-1, inp.shape[-1])
        if out.dim() >= 3:
            out = out.reshape(-1, out.shape[-1])

        d_a   = (inp.float() ** 2).sum(dim=0)
        d_waw = (out.float() ** 2).sum(dim=0)

        if self.diag_A is None:
            self.diag_A   = d_a
            self.diag_WAW = d_waw
        else:
            self.diag_A   += d_a
            self.diag_WAW += d_waw
        self.n_tokens += inp.shape[0]

    def accumulate_backward(self, grad_in: Optional[torch.Tensor], grad_out: torch.Tensor) -> None:
        grad_out = grad_out.detach()
        if grad_out.dim() >= 3:
            grad_out = grad_out.reshape(-1, grad_out.shape[-1])
        d_g = (grad_out.float() ** 2).sum(dim=0)
        if self.diag_G is None:
            self.diag_G = d_g
        else:
            self.diag_G += d_g

        if grad_in is not None and isinstance(grad_in, torch.Tensor):
            g_in = grad_in.detach()
            if g_in.dim() >= 3:
                g_in = g_in.reshape(-1, g_in.shape[-1])
            d_wtgw = (g_in.float() ** 2).sum(dim=0)
            if self.diag_WtGW is None:
                self.diag_WtGW = d_wtgw
            else:
                self.diag_WtGW += d_wtgw

    def finalise(self) -> Dict[str, Optional[torch.Tensor]]:
        n = max(self.n_tokens, 1)
        return {
            "diag_A"   : (self.diag_A / n).cpu() if self.diag_A is not None else None,
            "diag_WAW" : (self.diag_WAW / n).cpu() if self.diag_WAW is not None else None,
            "diag_G"   : (self.diag_G / n).cpu() if self.diag_G is not None else None,
            "diag_WtGW": (self.diag_WtGW / n).cpu() if self.diag_WtGW is not None else None,
        }

    def reset(self) -> None:
        self.diag_A    = None
        self.diag_WAW  = None
        self.diag_G    = None
        self.diag_WtGW = None
        self.n_tokens  = 0


class FullMatrixKFACStats:
    """
    Accumulates full Kronecker-factor matrices for exact K-FAC OBS:
      - A   : (1/N) X^T X    shape (d_in,  d_in)   -- input activation covariance
      - G   : (1/N) g^T g    shape (d_out, d_out)   -- output gradient covariance

    Memory footprint: d_in^2 + d_out^2 floats per layer per timestep.
    """
    def __init__(self, device: torch.device):
        self.device   = device
        self.A        : Optional[torch.Tensor] = None 
        self.G        : Optional[torch.Tensor] = None
        self.n_tokens : int                    = 0

    def accumulate_forward(self, inp: torch.Tensor) -> None:
        inp = inp.detach()
        if inp.dim() >= 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        # A += X^T X  (d_in, d_in)
        contrib = inp.t().mm(inp)
        if self.A is None:
            self.A = contrib
        else:
            self.A += contrib
        self.n_tokens += inp.shape[0]

    def accumulate_backward(self, grad_out: torch.Tensor) -> None:
        grad_out = grad_out.detach()
        if grad_out.dim() >= 3:
            grad_out = grad_out.reshape(-1, grad_out.shape[-1])
        grad_out = grad_out.float()
        # G += g^T g  (d_out, d_out)
        contrib = grad_out.t().mm(grad_out)
        if self.G is None:
            self.G = contrib
        else:
            self.G += contrib

    def finalise(self) -> Dict[str, Optional[torch.Tensor]]:
        n = max(self.n_tokens, 1)
        return {
            "A": (self.A / n).cpu() if self.A is not None else None,
            "G": (self.G / n).cpu() if self.G is not None else None,
        }

    def reset(self) -> None:
        self.A        = None
        self.G        = None
        self.n_tokens = 0


class DiffusionKFACCollector:
    """
    Registers forward and backward hooks on all target Linear layers
    of a discrete diffusion language model.

    Set full_matrix=True to accumulate full A and G covariance matrices.
    Statistics are aggregated across ALL timesteps into a single matrix pair
    to keep memory manageable.
    """

    def __init__(
        self,
        model       : nn.Module,
        layer_names : List[str],
        device      : torch.device,
        full_matrix : bool = False,
    ):
        self.model        = model
        self.device       = device
        self.target_names = set(layer_names)
        self.full_matrix  = full_matrix
        self._hooks       : list = []
        self._current_t   : int  = -1

        # Per-timestep diagonal stats (always accumulated)
        self._stats: Dict[str, Dict[int, TimestepKFACStats]] = {
            n: {} for n in layer_names
        }

        # Single aggregated full-matrix stats (only when full_matrix=True)
        # Aggregated across all timesteps: A_agg, G_agg
        self._full_stats: Dict[str, FullMatrixKFACStats] = {
            n: FullMatrixKFACStats(device) for n in layer_names
        } if full_matrix else {}

    def set_timestep(self, t: int) -> None:
        self._current_t = t
        for n in self.target_names:
            if t not in self._stats[n]:
                self._stats[n][t] = TimestepKFACStats(self.device)

    def snapshot(self, t: int) -> None:
        pass

    @contextmanager
    def active(self):
        self._register_hooks()
        try:
            yield self
        finally:
            self._remove_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name not in self.target_names:
                continue
            if isinstance(module, nn.Linear):
                self._hooks.append(
                    module.register_forward_hook(self._fwd_hook(name))
                )
                self._hooks.append(
                    module.register_full_backward_hook(self._bwd_hook(name))
                )

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _fwd_hook(self, name: str):
        def hook(module, inp, out):
            t = self._current_t
            if t in self._stats[name]:
                self._stats[name][t].accumulate_forward(inp[0], out)
            if self.full_matrix and name in self._full_stats:
                self._full_stats[name].accumulate_forward(inp[0])
        return hook

    def _bwd_hook(self, name: str):
        def hook(module, grad_in, grad_out):
            t = self._current_t
            if t in self._stats[name]:
                g_out = grad_out[0] if (grad_out and len(grad_out) > 0 and grad_out[0] is not None) else None
                g_in = grad_in[0] if (grad_in and len(grad_in) > 0 and grad_in[0] is not None) else None
                if g_out is not None:
                    self._stats[name][t].accumulate_backward(g_in, g_out)
            if self.full_matrix and name in self._full_stats:
                g_out = grad_out[0] if (grad_out and len(grad_out) > 0 and grad_out[0] is not None) else None
                if g_out is not None:
                    self._full_stats[name].accumulate_backward(g_out)
        return hook

    def get_all(self) -> Dict[str, Dict[int, Dict[str, Optional[torch.Tensor]]]]:
        result = {}
        for name, ts_dict in self._stats.items():
            result[name] = {}
            for t, stats in ts_dict.items():
                result[name][t] = stats.finalise()
        return result

    def get_full_matrices(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        return {name: stats.finalise() for name, stats in self._full_stats.items()}