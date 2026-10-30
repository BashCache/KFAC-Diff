"""
Semi-structured (N:M) Kronecker-OBD pruning of Dream-v0-Base-7B with snr_weighted timestep aggregation,
using pre-computed K-FAC statistics.  
Per-weight diagonal Kronecker-OBD saliency at timestep t:

    s_ij(t) = 0.5 * G_ii(t) * A_jj(t) * W_ij^2
    S       = sum_t w_t s(t) / sum_t w_t,    w_t = max(0, (T_max - t) / (t + 10))    (snr_weighted)

For every pruned layer the N:M mask keeps the N highest-scoring weights in each group of M consecutive input
weights of a row (--pattern N:M = KEEP N of every M; 2:4 = 50% sparse, 8:10 = 20% sparse).

With --save_model each pruned model is written to <output_dir>/model_<strategy>_<N>of<M>/ with its original
dense shapes (zeros at pruned positions), the Dream config/tokenizer, and semi_structured_meta.json.

Usage
-----
python compare_semi_structured_dream.py \\
    --model     Dream-org/Dream-v0-Base-7B \\
    --kfac_store activations_gradients_pt_files/kfac_store_activations_gradients_backup_2048_ctxlength_64timesteps_dream_base.pt \\
    --pattern   2:4 \\
    --output_dir ./semi_structured_results_dream/2of4
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safetensors.torch import save_file

from compare_pruning_strategies_dream import _restore_weights, _snapshot_weights
from compare_semi_structured import (
    SNR_WEIGHTED,
    _json_default,
    check_nm_pattern,
    prune_model_nm,
)
from evaluation.eval_dream import (
    evaluate_perplexity_dream,
    load_dream_model_and_tokenizer,
)
from utils.dream_utils import _point_auto_map_at_source_repo
from utils.model_utils import sanitize_auto_map
from utils.sparsity import measure_layer_sparsity, measure_model_sparsity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================== #
#  Layer selection                                                                                                                             #
# ======================================================================== #

def dream_prunable_layers(
    model: nn.Module,
    layers_to_prune: List[str],
    layers_to_skip: List[str],
) -> List[Tuple[str, nn.Module]]:
    """Linear layers inside the decoder blocks (never lm_head / embeddings), by name."""
    out = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if ".layers." not in f".{name}." or "lm_head" in name or "embed" in name:
            continue
        if layers_to_skip and any(s in name for s in layers_to_skip):
            continue
        if layers_to_prune and not any(p in name for p in layers_to_prune):
            continue
        out.append((name, module))
    return out


# ======================================================================== #
#  Saving                                                                                                                                                 #
# ======================================================================== #

def save_dream_semistructured(
    model: nn.Module,
    tokenizer: Any,
    out_dir: str,
    meta: Dict[str, Any],
) -> None:
    """
    Dense-shape Dream checkpoint with zeros at the pruned positions.  Follows utils.dream_utils.save_shrunk_dream
    (config, generation config, tokenizer, auto_map fix-up, one model.safetensors) but removes nothing.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"    Saving pruned Dream model to '{out_dir}' ...")
    state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}

    model.config.save_pretrained(out_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    _point_auto_map_at_source_repo(out_dir, getattr(model.config, "_name_or_path", None))
    sanitize_auto_map(out_dir)

    for fname in os.listdir(out_dir): 
        if (fname.startswith("model") and fname.endswith(".safetensors")) or fname in (
                "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"):
            os.remove(os.path.join(out_dir, fname))
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    with open(os.path.join(out_dir, "semi_structured_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    logger.info(f"    Saved (dense shapes, zeros at pruned positions) -> {out_dir}")


# ======================================================================== #
#  Evaluation                                                                                                                                       #
# ======================================================================== #

def run_eval(model: nn.Module, tokenizer, args) -> Dict[str, Any]:
    """Perplexity on each --eval_datasets entry (unless --skip_perplexity)."""
    res: Dict[str, Any] = {}
    if not args.skip_perplexity:
        for ds in args.eval_datasets:
            try:
                ppl = evaluate_perplexity_dream(
                    model=model, tokenizer=tokenizer, dataset=ds, seq_len=args.eval_seq_len,
                    device=args.device, mc_num=args.mc_num, mc_batch_size=args.mc_batch_size,
                )
            except Exception as exc:
                logger.warning(f"  Perplexity({ds}) failed: {exc}")
                ppl = float("nan")
            res[f"perplexity_{ds}"] = ppl
            logger.info(f"  Perplexity on {ds}: {ppl:.3f}")
    return res


def print_summary(results: Dict[str, Dict]) -> None:
    keys = sorted({k for r in results.values() for k in r if k.startswith("perplexity_")})
    logger.info("=" * 80)
    logger.info("  " + f"{'strategy':<28s}" + "".join(f"{k:>16s}" for k in keys) + f"{'sparsity':>10s}")
    for name, r in results.items():
        row = "".join(f"{r.get(k, float('nan')):>16.4f}" for k in keys)
        logger.info("  " + f"{name:<28s}" + row + f"{r['prunable_layer_sparsity']:>10.2%}")


# ======================================================================== #
#  CLI / main                                                                                                                                           #
# ======================================================================== #

def add_ppl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip_perplexity", action="store_true", help="Skip perplexity evaluation.")
    p.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"],
                   help="Perplexity evaluation datasets (wikitext2, c4).")
    p.add_argument("--eval_seq_len", type=int, default=512,
                   help="Sequence length for continuous diffusion perplexity evaluation.")
    p.add_argument("--mc_num", type=int, default=128,
                   help="Monte Carlo masking iterations for Dream MC evaluation (standard setting: mc_num 128, batch 32).")
    p.add_argument("--mc_batch_size", type=int, default=32,
                   help="Batch size for MC evaluation (must divide mc_num).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semi-structured (N:M) K-FAC OBD pruning of Dream-v0-Base-7B (snr_weighted aggregation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Dream-org/Dream-v0-Base-7B", help="HF model id or local path.")
    p.add_argument("--kfac_store", required=True,
                   help=".pt file with {layer: {timestep: {diag_G, diag_A, ...}}} collected on Dream.")
    p.add_argument("--pattern", default="2:4",
                   help="N:M pattern: KEEP N of every M consecutive input weights (2:4 = 50%% sparse, 8:10 = 20%%).")
    p.add_argument("--save_model", action="store_true",
                   help="Save each pruned model to <output_dir>/model_<strategy>_<N>of<M>/ before evaluation.")
    p.add_argument("--layers_to_prune", nargs="*", default=[],
                   help="Substring patterns of layer names to prune (empty = all block linears).")
    p.add_argument("--layers_to_skip", nargs="*", default=[],
                   help="Substring patterns of layer names to leave dense (e.g. o_proj q_proj).")
    p.add_argument("--output_dir", default="./semi_structured_results_dream")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)

    add_ppl_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    n, m = (int(x) for x in args.pattern.split(":"))
    assert 0 < n < m, f"invalid pattern {args.pattern}"
    logger.info(f"N:M pattern {n}:{m} (keep {n} of every {m}; target sparsity {1 - n / m:.1%} of every pruned layer)")

    model, tokenizer = load_dream_model_and_tokenizer(
        model_path=args.model, dtype=args.dtype, device=args.device,
    )
    clean_snapshot = _snapshot_weights(model)

    kfac_store: Dict = torch.load(args.kfac_store, map_location="cpu", weights_only=False)
    timesteps = sorted({t for ts in kfac_store.values() for t in ts})
    logger.info(f"K-FAC store: {len(kfac_store)} layers, {len(timesteps)} timesteps: "
                f"{timesteps[:6]}...{timesteps[-2:]}")

    prunable = dream_prunable_layers(model, args.layers_to_prune, args.layers_to_skip)
    prunable_names = {name for name, _ in prunable}
    missing = sorted(prunable_names - set(kfac_store))
    logger.info(f"Prunable layers: {len(prunable)} (missing from K-FAC store: {len(missing)})")
    if missing:
        logger.warning(f"  Not in the store, these stay dense: {missing[:5]}{'...' if len(missing) > 5 else ''}")

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
            save_dream_semistructured(
                model, tokenizer,
                os.path.join(args.output_dir, f"model_{strat}_{n}of{m}"),
                meta={"model_family": "dream", "strategy": strat, "keep_n": n, "group_m": m,
                      "source_model": args.model, "kfac_store": args.kfac_store,
                      "prunable_layer_sparsity": mean_sp,
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

        with open(os.path.join(args.output_dir, "semi_structured_comparison_dream.json"), "w") as f:
            json.dump(results, f, indent=2, default=_json_default)

    print_summary(results)


if __name__ == "__main__":
    main()
