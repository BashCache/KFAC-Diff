# collect_calibration_diffusion_llada.py

"""
K-FAC statistics collection and scoring for discrete diffusion language
models (LLaDA and compatible).

Runs the calibration pass over the diffusion timestep schedule, collects the
per-timestep K-FAC statistics (activations A and gradients G) for every
prunable Linear layer, then computes and stores the pruning scores for the
selected method.  Pruning and evaluation are done afterwards from the stored
statistics / scores (see compare_pruning_strategies.py).

Example
-------
# Minimal run
python collect_calibration_diffusion_llada.py \
    --model  GSAI-ML/LLaDA-8B-Instruct \
    --output_dir ./llada_pruned

# Full control
python collect_calibration_diffusion_llada.py \
    --model              GSAI-ML/LLaDA-8B-Instruct \
    --output_dir         ./llada_kfac \
    --method             kfac_obd_diffusion \
    --cal_dataset        c4 \
    --cal_samples        128 \
    --cal_batch_size     4 \
    --seq_len            128 \
    --num_timesteps      32 \
    --timestep_strategy  uniform \
    --total_timesteps    1000 \
    --agg_method         log_decay \
    --alpha_min          0.0 \
    --alpha_max          1.0 \
    --damp               1e-3 \
    --kfac_save_path     ./llada_kfac/kfac_store_diag.pt \
    --score_dir          ./llada_kfac/scores \
    --seed               42 \
    --log_level          INFO
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

# ── make sure project root is on sys.path ─────────────────────────────── #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── project imports ────────────────────────────────────────────────────── #
from config.base_config import CalibrationConfig, PruningConfig, SparsityConfig
from core.registry import registry
from data.diffusion_calibration import get_diffusion_calibration_loader
from utils.logging_utils import setup_logging
from utils.model_utils import load_llada_model_and_tokenizer

# Register all methods (including kfac_obd_diffusion)
import methods  # noqa: F401

logger = logging.getLogger(__name__)


# ======================================================================== #
#  CLI                                                                       #
# ======================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect per-timestep K-FAC statistics (activations and gradients) "
            "and compute pruning scores for discrete diffusion language models (LLaDA)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────── #
    model_g = p.add_argument_group("Model")
    model_g.add_argument(
        "--model",
        required=True,
        metavar="PATH_OR_HF_ID",
        help="HuggingFace model id or local path to the LLaDA checkpoint.",
    )
    model_g.add_argument(
        "--method",
        default="kfac_obd_diffusion",
        choices=["kfac_obd_diffusion", "kfac_obd_trajectory", "kfac_obs_diffusion", "kfac_obd_iterative"],
        help=(
            "Scoring/pruning method to use. "
            "'kfac_obd_diffusion' = independent noise sampling (teacher forcing); "
            "'kfac_obd_trajectory' = inference-trajectory-following collection; "
            "'kfac_obs_diffusion' = full Kronecker-factored OBS with weight reconstruction; "
            "'kfac_obd_iterative' = multi-round iterative K-FAC OBD pruning."
        ),
    )
    model_g.add_argument(
        "--output_dir",
        default="./diffusion_kfac",
        help="Directory where the K-FAC store and scores are saved.",
    )
    model_g.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run collection on.",
    )
    model_g.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.  bfloat16 is recommended for LLaDA.",
    )
    model_g.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True to from_pretrained().",
    )

    # ── Calibration data ──────────────────────────────────────────────── #
    cal_g = p.add_argument_group("Calibration data")
    cal_g.add_argument(
        "--cal_dataset",
        default="c4",
        choices=["c4", "wikitext2"],
        help="Text corpus used to collect K-FAC statistics.",
    )
    cal_g.add_argument(
        "--cal_samples",
        type=int,
        default=128,
        metavar="N",
        help="Number of token sequences used for calibration.",
    )
    cal_g.add_argument(
        "--cal_batch_size",
        type=int,
        default=4,
        metavar="B",
    )
    cal_g.add_argument(
        "--seq_len",
        type=int,
        default=128,
        metavar="L",
        help="Token sequence length for calibration sequences.",
    )

    # ── Diffusion / noise schedule ────────────────────────────────────── #
    diff_g = p.add_argument_group("Diffusion schedule")
    diff_g.add_argument(
        "--num_timesteps",
        type=int,
        default=32,
        metavar="T_CAL",
        help=(
            "Number of denoising timesteps at which K-FAC statistics "
            "are collected (e.g. 32 | 64 | 128)."
        ),
    )
    diff_g.add_argument(
        "--timestep_strategy",
        default="uniform",
        choices=["uniform", "log", "quadratic"],
        help="How to space the calibration timesteps across [1, T].",
    )
    diff_g.add_argument(
        "--total_timesteps",
        type=int,
        default=1000,
        metavar="T",
        help="Total number of timesteps T in the diffusion process.",
    )
    diff_g.add_argument(
        "--mask_token_id",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Explicit [MASK] token id.  "
            "If not set, resolved automatically from the tokenizer."
        ),
    )
    diff_g.add_argument(
        "--agg_method",
        default="sum",
        type=str,
        help=(
            "Score aggregation method across timesteps, e.g.: "
            "'sum', 'mean', 'inv_linear', 'inv_quadratic', 'exp_decay', "
            "'cosine_decay', 'log_decay', 'snr', or hybrid '*_mean_plus_std'."
        ),
    )
    diff_g.add_argument(
        "--alpha_min",
        type=float,
        default=0.0,
        metavar="ALPHA_MIN",
        help="Minimum weight for log_decay score aggregation (at t=T).",
    )
    diff_g.add_argument(
        "--alpha_max",
        type=float,
        default=1.0,
        metavar="ALPHA_MAX",
        help="Maximum weight for log_decay score aggregation (at t=1).",
    )
    diff_g.add_argument(
        "--mv_lambda",
        type=float,
        default=0.5,
        help="Lambda scaling factor for mean_plus_std / mean_minus_std hybrid aggregation.",
    )

    # ── K-FAC ─────────────────────────────────────────────────────────── #
    kfac_g = p.add_argument_group("K-FAC")
    kfac_g.add_argument(
        "--damp",
        type=float,
        default=1e-3,
        metavar="λ",
        help="Damping coefficient for Hessian inversion.",
    )
    kfac_g.add_argument(
        "--kfac_save_path",
        default=None,
        metavar="PATH",
        help=(
            "Path to save the per-timestep diagonal KFAC store (.pt). "
            "Defaults to <output_dir>/kfac_store_diag.pt."
        ),
    )

    # ── Score storage ────────────────────────────────────────────────── #
    score_g = p.add_argument_group("Score storage")
    score_g.add_argument(
        "--score_dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory where scores are saved.  "
            "Defaults to <output_dir>/scores."
        ),
    )

    # ── Layer selection ───────────────────────────────────────────────── #
    layer_g = p.add_argument_group("Layer selection")
    layer_g.add_argument(
        "--layers_to_prune",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "Substring patterns of layer names to collect statistics for.  "
            "Empty = all Linear layers."
        ),
    )
    layer_g.add_argument(
        "--layers_to_skip",
        nargs="*",
        default=[],
        metavar="NAME",
        help="Substring patterns of layer names to SKIP.",
    )

    # ── Iterative pruning (kfac_obd_iterative only) ────────────────────── #
    iter_g = p.add_argument_group("Iterative pruning (kfac_obd_iterative)")
    iter_g.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help=(
            "Number of iterative pruning rounds.  >1 selects "
            "kfac_obd_iterative automatically."
        ),
    )
    iter_g.add_argument(
        "--sparsity",
        type=float,
        default=0.3,
        metavar="RATIO",
        help="Final target pruning ratio reached after all rounds.",
    )
    iter_g.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save the iteratively pruned model to disk.",
    )

    # ── Misc ──────────────────────────────────────────────────────────── #
    misc_g = p.add_argument_group("Misc")
    misc_g.add_argument("--seed",      type=int,  default=42)
    misc_g.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    misc_g.add_argument("--log_file",  default=None,
                        metavar="PATH",
                        help="Optional path to write logs to a file.")

    return p.parse_args()


# ======================================================================== #
#  Config builder                                                                                                                               #
# ======================================================================== #

def build_config(args: argparse.Namespace) -> PruningConfig:
    """
    Translate CLI arguments into the PruningConfig the collector reads.
    """
    method_kwargs = {
        # K-FAC
        "damp"             : args.damp,
        "kfac_save_path"   : args.kfac_save_path,
        # Timestep schedule
        "num_timesteps"    : args.num_timesteps,
        "timestep_strategy": args.timestep_strategy,
        "total_timesteps"  : args.total_timesteps,
        # Timestep score aggregation
        "agg_method"       : args.agg_method,
        "strategy"         : args.agg_method,
        "alpha_min"        : args.alpha_min,
        "alpha_max"        : args.alpha_max,
        "mv_lambda"        : args.mv_lambda,
        # Calibration text data
        "cal_dataset"      : args.cal_dataset,
        "cal_samples"      : args.cal_samples,
        "cal_batch_size"   : args.cal_batch_size,
        "seq_len"          : args.seq_len,
        # Score storage
        "score_dir"        : args.score_dir,
        # Iterative pruning
        "num_rounds"       : args.num_rounds,
    }

    # Optional explicit mask token id
    if args.mask_token_id is not None:
        method_kwargs["mask_token_id"] = args.mask_token_id

    method_name = "kfac_obd_iterative" if args.num_rounds > 1 else args.method
    return PruningConfig(
        # ── core ──────────────────────────────────────────────────────── #
        method             = method_name,
        model_name_or_path = args.model,
        device             = args.device,
        dtype              = args.dtype,
        seed               = args.seed,
        output_dir         = args.output_dir,
        save_model         = not args.no_save,

        # ── calibration ────────────────────────── #
        calibration = CalibrationConfig(
            dataset     = args.cal_dataset,
            num_samples = args.cal_samples,
            seq_len     = args.seq_len,
            batch_size  = args.cal_batch_size,
            seed        = args.seed,
        ),
        sparsity = SparsityConfig(
            target_sparsity = args.sparsity,
        ),
        layers_to_prune = args.layers_to_prune,
        layers_to_skip  = args.layers_to_skip,
        method_kwargs   = method_kwargs,
    )


# ======================================================================== #
#  Model + tokenizer loading (LLaDA-specific)                               #
# ======================================================================== #

def load_llada(args: argparse.Namespace):
    """
    Load the LLaDA model and tokenizer using utils.model_utils.load_llada_model_and_tokenizer.
    Attaches model.tokenizer so that KFACOBDDiffusionPruner.prepare() can resolve it automatically.
    """
    model, tokenizer = load_llada_model_and_tokenizer(
        model_path_or_config=args.model,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    model.tokenizer = tokenizer
    return model, tokenizer


# ======================================================================== #
#  Diffusion-aware calibration loader                                        #
# ======================================================================== #

def build_calibration_loader(args: argparse.Namespace, tokenizer):
    """
    Build a DataLoader of clean token sequences.
    Masking is applied per-timestep inside the collector, not here.
    """
    return get_diffusion_calibration_loader(
        dataset_name = args.cal_dataset,
        num_samples  = args.cal_samples,
        seq_len      = args.seq_len,
        batch_size   = args.cal_batch_size,
        seed         = args.seed,
        tokenizer    = tokenizer,
    )


# ======================================================================== #
#  Seed                                                                      #
# ======================================================================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Global seed set to {seed}.")


def run(args: argparse.Namespace) -> None:
    if args.kfac_save_path is None:
        args.kfac_save_path = os.path.join(args.output_dir, "kfac_store_diag.pt")
    if args.score_dir is None:
        args.score_dir = os.path.join(args.output_dir, "scores")
    if args.num_rounds > 1:
        args.method = "kfac_obd_iterative"

    set_seed(args.seed)
    _print_banner(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load model + tokenizer ─────────────────────────────────────── #
    logger.info("[1/5] Loading model and tokenizer ...")
    model, tokenizer = load_llada(args)
    config = build_config(args)

    # ── 2. Build calibration loader ───────────────────────────────────── #
    logger.info("[2/5] Building calibration data loader ...")
    cal_loader = build_calibration_loader(args, tokenizer)

    # ── 3. Instantiate pruner ─────────────────────────────────────────── #
    logger.info(f"[3/5] Instantiating K-FAC pruner with method {args.method} ...")
    pruner_cls = registry.get(args.method)
    pruner     = pruner_cls(model, config)

    # ── Iterative: collect → score → prune, repeated num_rounds times ── #
    if args.method == "kfac_obd_iterative" or hasattr(pruner, "run_iterative_pruning"):
        logger.info(f"[4/5] Running multi-round iterative K-FAC OBD pruning ({args.num_rounds} rounds) ...")
        pruner.run_iterative_pruning(cal_loader)
        pruner.cleanup()

        if not args.no_save:
            logger.info(f"[5/5] Saving pruned model to '{args.output_dir}' ...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        logger.info("Done.")
        return

    pruner.prepare()

    # ── 4. Collect K-FAC statistics ───────────────────────────────────── #
    logger.info("[4/5] Collecting K-FAC statistics across timesteps ...")
    pruner.collect(cal_loader)

    # ── 5. Compute + store scores ─────────────────────────────────────── #
    logger.info(f"[5/5] Computing scores (saved to '{args.score_dir}') ...")
    pruner.compute_scores()
    pruner.cleanup()

    logger.info("Done.")


# ======================================================================== #
#  Helpers                                                                   #
# ======================================================================== #

def _print_banner(args: argparse.Namespace) -> None:
    lines = [
        "",
        "=" * 65,
        "  LLaDA Discrete Diffusion  –  K-FAC Collection + Scoring",
        "=" * 65,
        f"  Model              : {args.model}",
        f"  Method             : {args.method}"
        + (f"  ({args.num_rounds} rounds, target sparsity {args.sparsity:.0%})" if args.method == "kfac_obd_iterative" else ""),
        f"  Output             : {args.output_dir}",
        f"  K-FAC store        : {args.kfac_save_path}",
        f"  Score dir          : {args.score_dir}",
        f"  Calibration data   : {args.cal_dataset}  "
        f"(n={args.cal_samples}, seq={args.seq_len})",
        f"  Timesteps          : {args.num_timesteps}  "
        f"({args.timestep_strategy})  T={args.total_timesteps}",
        f"  Score aggregation  : {args.agg_method}"
        + (f" (alpha_min={args.alpha_min}, alpha_max={args.alpha_max})" if args.agg_method == "log_decay" else ""),
        f"  Damp               : {args.damp}",
        f"  Device / dtype     : {args.device} / {args.dtype}",
        f"  Seed               : {args.seed}",
        "=" * 65,
        "",
    ]
    for line in lines:
        logger.info(line)


# ======================================================================== #
#  Entry point                                                               #
# ======================================================================== #

def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    run(args)


if __name__ == "__main__":
    main()