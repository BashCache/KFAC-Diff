"""
Semi-structured (N:M) Kronecker-OBD pruning of LLaDA with per-timestep score
aggregation (snr_weighted), using *pre-computed* K-FAC stats.

Per-weight diagonal Kronecker-OBD saliency at timestep t:

    s_ij(t) = 0.5 * G_ii(t) * A_jj(t) * W_ij^2

The score is not separable across timesteps (sum_t w_t G_i(t) A_j(t) != (sum w G)(sum w A)),
so the outer product is formed per timestep and folded on the fly:

    S = sum_t w_t s(t) / sum_t w_t,   w_t = max(0, (T_max - t) / (t + 10))   (snr_weighted)

Only one (d_out, d_in) accumulator per layer is alive at a time.  For each layer the
N:M mask keeps the N highest-scoring weights in every group of M consecutive input
weights of a row (2:4 runs along dim=1, the layout sparse tensor cores need).

Usage
-----
python compare_semi_structured.py \\
    --model       ./llada_pruned_2048_64timesteps \\
    --kfac_store  kfac_store_activations_gradients_backup_2048_64timesteps.pt \\
    --pattern     2:4 \\
    --output_dir  ./semi_structured_results
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from compare_pruning_strategies import (
    _get_prunable_layers,
    _json_default,
    _restore_weights,
    _snapshot_weights,
)
from evaluation.eval_diffusion import evaluate_perplexity
from safetensors.torch import save_file

from utils.model_utils import load_llada_model_and_tokenizer
from utils.sparsity import measure_layer_sparsity, measure_model_sparsity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SNR_WEIGHTED = "snr_weighted"


# ======================================================================== #
#  Scores                                                                                                                                                 #
# ======================================================================== #

def _weight_scores(W2: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """s_ij = 0.5 * G_ii * A_jj * W_ij^2, with W2 = W^2 already on device (float32)."""
    dev = W2.device
    g = stats["diag_G"].float().to(dev)
    a = stats["diag_A"].float().to(dev)
    return 0.5 * g[:, None] * a[None, :] * W2


def snr_weighted_weights(timesteps: List[int]) -> Dict[int, float]:
    """w_t = max(0, (T_max - t) / (t + 10)); T_max = largest timestep in the store."""
    t_max = float(max(timesteps))
    return {t: max(0.0, (t_max - t) / (t + 10.0)) for t in timesteps}


def aggregate_layer_scores(
    W: torch.Tensor,
    ts_store: Dict[int, Dict[str, torch.Tensor]],
    timesteps: List[int],
) -> Optional[torch.Tensor]:
    """S = sum_t w_t s(t) / sum_t w_t, accumulated one timestep at a time."""
    usable = [t for t in timesteps
              if t in ts_store and ts_store[t].get("diag_G") is not None
              and ts_store[t].get("diag_A") is not None]
    if not usable:
        return None

    W2 = W.float().pow(2)
    weights = snr_weighted_weights(timesteps)
    z = sum(weights[t] for t in usable) + 1e-30
    acc = torch.zeros_like(W2)
    for t in usable:
        acc.add_(_weight_scores(W2, ts_store[t]), alpha=weights[t] / z)
    return acc


# ======================================================================== #
#  N:M mask                                                                                                                                               #
# ======================================================================== #

def nm_mask(score: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Boolean keep-mask: top-n of every m consecutive entries along dim=1.
    A trailing remainder of d_in % m columns (if any) is kept dense.
    """
    d_out, d_in = score.shape
    full = (d_in // m) * m
    mask = torch.ones_like(score, dtype=torch.bool)
    if full == 0:
        return mask
    g = score[:, :full].reshape(d_out, full // m, m)
    idx = g.topk(n, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    mask[:, :full] = keep.reshape(d_out, full)
    return mask


def prune_model_nm(
    model: nn.Module,
    kfac_store: Dict,
    prunable: List[Tuple[str, nn.Module]],
    timesteps: List[int],
    n: int,
    m: int,
    device: str,
) -> Dict[str, float]:
    """Prune every prunable layer in place; return {layer: achieved sparsity}."""
    done: Dict[str, float] = {}
    for name, module in prunable:
        W = module.weight.data
        if W.shape[1] < m:
            logger.warning(f"    {name}: d_in={W.shape[1]} < M={m}; skipped.")
            continue
        Wd = W.to(device).float()
        ts_store = kfac_store.get(name)
        if not ts_store:
            logger.warning(f"    {name}: no K-FAC data; skipped.")
            continue
        score = aggregate_layer_scores(Wd, ts_store, timesteps)
        if score is None:
            logger.warning(f"    {name}: no usable timesteps; skipped.")
            continue
        mask = nm_mask(score, n, m)
        with torch.no_grad():
            module.weight.data.mul_(mask.to(device=W.device, dtype=W.dtype))
        done[name] = 1.0 - mask.float().mean().item()
        del Wd, score, mask
    return done


# ======================================================================== #
#  Saving                                                                                                                                                #
# ======================================================================== #

def check_nm_pattern(
    model: nn.Module,
    prunable_names: set,
    n: int,
    m: int,
) -> Dict[str, int]:
    """Count groups with more than N nonzeros per layer (0 everywhere = valid N:M)."""
    bad: Dict[str, int] = {}
    for name, module in model.named_modules():
        if name not in prunable_names or not hasattr(module, "weight"):
            continue
        W = module.weight.data
        d_out, d_in = W.shape
        full = (d_in // m) * m
        if full == 0:
            continue
        nnz = (W[:, :full].reshape(d_out, full // m, m) != 0).sum(dim=-1)
        n_bad = int((nnz > n).sum().item())
        if n_bad:
            bad[name] = n_bad
    return bad


def save_pruned_model(
    model: nn.Module,
    tokenizer: Any,
    out_dir: str,
    meta: Dict[str, Any],
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"    Saving pruned model to '{out_dir}' ...")

    saved = False
    if hasattr(model, "save_pretrained") and callable(model.save_pretrained):
        try:
            model.save_pretrained(out_dir, safe_serialization=True)
            saved = True
        except Exception as exc:
            logger.warning(f"    save_pretrained failed ({exc}); falling back to raw safetensors.")
    if not saved:
        state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
        save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "save_pretrained"):
            cfg.save_pretrained(out_dir)

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "semi_structured_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    logger.info(f"    Saved (dense shapes, zeros at pruned positions) -> {out_dir}")


# ======================================================================== #
#  Evaluation                                                                                                                                       #
# ======================================================================== #

def run_eval(model, tokenizer, args) -> Dict[str, Any]:
    """Perplexity on each --eval_datasets entry (unless --skip_perplexity)."""
    res: Dict[str, Any] = {}
    if not args.skip_perplexity:
        for ds in args.eval_datasets:
            try:
                ppl = evaluate_perplexity(
                    model=model, tokenizer=tokenizer, dataset=ds, seq_len=args.eval_seq_len,
                    device=args.device, mc_num=args.mc_num, mc_batch_size=args.mc_batch_size,
                )
            except Exception as exc:
                logger.warning(f"  Perplexity({ds}) failed: {exc}")
                ppl = float("nan")
            res[f"perplexity_{ds}"] = ppl
            logger.info(f"  Perplexity on {ds}: {ppl:.3f}")
    return res


def add_ppl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip_perplexity", action="store_true", help="Skip perplexity evaluation.")
    p.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"],
                   help="Perplexity evaluation datasets.")
    p.add_argument("--eval_seq_len", type=int, default=1024,
                   help="Sequence length for perplexity evaluation.")
    p.add_argument("--mc_num", type=int, default=128,
                   help="Monte Carlo masking iterations for perplexity. Must be divisible by --mc_batch_size.")
    p.add_argument("--mc_batch_size", type=int, default=16,
                   help="MC mini-batch size (mc_num must be divisible by this).")


def print_summary(results: Dict[str, Dict]) -> None:
    keys = sorted({k for r in results.values() for k in r if k.startswith("perplexity_")})
    logger.info("=" * 80)
    logger.info("  " + f"{'strategy':<28s}" + "".join(f"{k:>16s}" for k in keys) + f"{'sparsity':>10s}")
    for name, r in results.items():
        row = "".join(f"{r.get(k, float('nan')):>16.4f}" for k in keys)
        logger.info("  " + f"{name:<28s}" + row + f"{r['prunable_layer_sparsity']:>10.2%}")


# ======================================================================== #
#  Main                                                                                                                                                     #
# ======================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semi-structured (N:M) K-FAC OBD pruning of LLaDA with timestep aggregation strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HF model id or local path.")
    p.add_argument("--kfac_store", required=True, help=".pt file with {layer: {timestep: {diag_G, diag_A, ...}}}.")
    p.add_argument("--pattern", default="2:4", help="N:M pattern, e.g. 2:4 or 4:8 (keep N of every M along d_in).")
    p.add_argument("--save_model", action="store_true",
                   help="Save each pruned model to <output_dir>/model_<strategy>_<N>of<M>/ before evaluation.")
    p.add_argument("--layers_to_prune", nargs="*", default=[])
    p.add_argument("--layers_to_skip", nargs="*", default=[])
    p.add_argument("--output_dir", default="./semi_structured_results")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)

    add_ppl_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    n, m = (int(x) for x in args.pattern.split(":"))
    assert 0 < n < m, f"invalid pattern {args.pattern}"
    logger.info(f"N:M pattern {n}:{m}  (target sparsity {1 - n / m:.1%} of every pruned layer)")

    model, tokenizer = load_llada_model_and_tokenizer(
        model_path_or_config=args.model, dtype=args.dtype, device=args.device,
    )
    model.tokenizer = tokenizer
    clean_snapshot = _snapshot_weights(model)

    kfac_store: Dict = torch.load(args.kfac_store, map_location="cpu", weights_only=False)
    timesteps = sorted({t for ts in kfac_store.values() for t in ts})
    logger.info(f"K-FAC store: {len(kfac_store)} layers, {len(timesteps)} timesteps: "
                f"{timesteps[:10]}{'...' if len(timesteps) > 10 else ''}")

    prunable = _get_prunable_layers(model, args.layers_to_prune, args.layers_to_skip)
    prunable_names = {name for name, _ in prunable}
    logger.info(f"Prunable layers: {len(prunable)}")

    results: Dict[str, Dict] = {}
    for strat in (SNR_WEIGHTED,):
        logger.info("-" * 65)
        logger.info(f"  Strategy: {strat}  ({n}:{m})")
        logger.info("-" * 65)

        t0 = time.time()
        target = prune_model_nm(model, kfac_store, prunable, timesteps, n, m, args.device)
        prune_t = time.time() - t0

        layer_sp = measure_layer_sparsity(model)
        model_sp = measure_model_sparsity(model)
        pruned_sp = [layer_sp[ln] for ln in prunable_names if ln in layer_sp]
        mean_sp = sum(pruned_sp) / max(len(pruned_sp), 1)
        logger.info(f"    Pruned {len(target)} layers in {prune_t:.1f}s | mean prunable-layer sparsity "
                    f"{mean_sp:.2%} | model sparsity {model_sp['overall_sparsity']:.2%} "
                    f"({model_sp['sparsity_excl_embeddings_and_head']:.2%} excl. embeddings+head)")

        if args.save_model:
            bad = check_nm_pattern(model, prunable_names, n, m)
            if bad:
                logger.warning(f"    N:M check: {sum(bad.values())} groups exceed {n} nonzeros "
                               f"in {len(bad)} layers (e.g. {next(iter(bad))}).")
            else:
                logger.info(f"    N:M check passed: no group has more than {n} nonzeros of {m}.")
            save_pruned_model(
                model, tokenizer,
                os.path.join(args.output_dir, f"model_{strat}_{n}of{m}"),
                meta={"strategy": strat, "keep_n": n, "group_m": m, "source_model": args.model,
                      "kfac_store": args.kfac_store, "prunable_layer_sparsity": mean_sp,
                      "overall_model_sparsity": model_sp["overall_sparsity"],
                      "nm_violations": bad},
            )

        t0 = time.time()
        res = run_eval(model, tokenizer, args)
        results[strat] = {
            **res,
            "pattern": f"{n}:{m}",
            "prunable_layer_sparsity": mean_sp,
            "overall_model_sparsity": model_sp["overall_sparsity"],
            "model_sparsity_excl_embeddings": model_sp["sparsity_excl_embeddings"],
            "model_sparsity_excl_embeddings_and_head": model_sp["sparsity_excl_embeddings_and_head"],
            "layers_pruned": len(target),
            "prune_time_s": round(prune_t, 2),
            "eval_time_s": round(time.time() - t0, 2),
        }

        _restore_weights(model, clean_snapshot)

        with open(os.path.join(args.output_dir, "semi_structured_comparison.json"), "w") as f:
            json.dump(results, f, indent=2, default=_json_default)

    print_summary(results)


if __name__ == "__main__":
    main()
