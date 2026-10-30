"""
collect_calibration_diffusion_dream.py

K-FAC statistics collection and scoring for Dream Diffusion Language Models
(e.g., Dream-org/Dream-v0-Base-7B and Dream-org/Dream-v0-Instruct-7B).

Runs the calibration pass over the diffusion timestep schedule, collects the
per-timestep K-FAC statistics (activations A and gradients G) for every
prunable Linear layer, then computes and stores the Kronecker-OBD scores.
Pruning and evaluation are done afterwards from the stored statistics /
scores (see compare_pruning_strategies_dream.py).

Core Differences from LLaDA:
----------------------------
1. Denoising Loss & Gradient Alignment:
   Dream's raw output logits are offset by 1 position:
       logits = model(masked_ids).logits
       logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
   Without this shift, cross-entropy compares token i+1 predictions against token i,
   corrupting the backward pass and producing noisy gradient covariance matrices G(t).

2. Protected Tokens:
   BOS (<|beginoftext|> = 151665) and EOS (<|endoftext|> = 151643) are explicitly unmasked
   during calibration:
       mask[:, 0] = False
       mask[:, -1] = False

3. Special Tokens:
   - mask_token_id: 151666 (<|mask|>)
   - bos_token_id:  151665 (<|beginoftext|>)
   - eos_token_id:  151643 (<|endoftext|>)

Usage:
------
python collect_calibration_diffusion_dream.py \\
    --model Dream-org/Dream-v0-Base-7B \\
    --output_dir ./dream_kfac \\
    --cal_dataset c4 \\
    --cal_samples 128 \\
    --seq_len 2048 \\
    --num_timesteps 64 \\
    --total_timesteps 1000 \\
    --agg_method log_decay
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collectors.diffusion_collector import DiffusionKFACCollector
from data.diffusion_calibration import (
    get_diffusion_calibration_loader,
    get_timestep_schedule,
    mask_tokens,
)
from utils.logging_utils import setup_logging
from metrics.diffusion_pruning_score_store import PruningScoreStore
from methods.kfac_obd_diffusion import (
    _kfac_obd_row_scores,
    _kfac_obd_col_scores,
    _is_down_proj_layer,
    _is_attn_out_layer,
)
from compare_pruning_strategies_dream import (
    load_dream_model_and_tokenizer,
    DREAM_MASK_TOKEN_ID_DEFAULT,
    _get_prunable_layers,
)

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Dream Denoising Loss                                                    #
# ======================================================================== #

def _dream_loss(
    model: nn.Module,
    masked_ids: torch.Tensor,
    original_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute discrete diffusion denoising loss for Dream.
    Crucial fix: Applies Dream logit shift so position i accurately predicts token i.
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(masked_ids)
        logits = out.logits if hasattr(out, "logits") else out
        # Shift logits: logits[:, i] aligns with token i
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    B, L, V = logits.shape
    logits_flat = logits.reshape(B * L, V)
    targets_flat = original_ids.reshape(B * L)

    if mask is not None and mask.sum() > 0:
        mask_flat = mask.reshape(B * L)
        loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], reduction="mean")
    else:
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

    return loss


# ======================================================================== #
#  Dream K-FAC OBD Pruner                                                  #
# ======================================================================== #

class DreamKFACOBDPruner:
    """
    Timestep-aware K-FAC statistics collection and Kronecker-OBD scoring for
    Dream Diffusion Language Models, using the shifted denoising loss and
    protected BOS/EOS tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        mask_token_id: int,
        total_timesteps: int,
        num_timesteps: int,
        timestep_strategy: str,
        seed: int,
        layers_to_prune: List[str],
        layers_to_skip: List[str],
        agg_method: str = "log_decay",
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        score_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.mask_token_id = mask_token_id
        self.total_timesteps = total_timesteps
        self.num_timesteps = num_timesteps
        self.timestep_strategy = timestep_strategy
        self.seed = seed
        self.layers_to_prune = layers_to_prune
        self.layers_to_skip = layers_to_skip
        self.agg_method = agg_method
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.score_dir = score_dir

        self._collector: Optional[DiffusionKFACCollector] = None
        self._kfac_store: Dict[str, Dict[int, Dict[str, Optional[torch.Tensor]]]] = {}
        self.score_store: PruningScoreStore = PruningScoreStore()
        self._timestep_list: List[int] = []

    def prepare(self) -> None:
        """Prepare model parameters for gradient-based K-FAC collection and instantiate the collector."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._timestep_list = get_timestep_schedule(
            total_timesteps=self.total_timesteps,
            num_timesteps=self.num_timesteps,
            strategy=self.timestep_strategy,
        )
        prunable = _get_prunable_layers(self.model, self.layers_to_prune, self.layers_to_skip)
        layer_names = [name for name, _ in prunable]

        self._collector = DiffusionKFACCollector(
            model=self.model,
            layer_names=layer_names,
            device=self.device,
            full_matrix=False,
        )
        logger.info(f"  [DreamKFAC] Prepared collector for {len(layer_names)} layers across {len(self._timestep_list)} timesteps.")

    def collect(self, calibration_loader) -> Dict[str, Dict[int, Dict[str, Optional[torch.Tensor]]]]:
        """Collect K-FAC statistics across timesteps with Dream-aligned loss and protected BOS/EOS."""
        assert self._collector is not None, "Call prepare() before collect()."

        pbar = tqdm(calibration_loader, desc="[DreamKFAC] Collecting K-FAC", unit="batch", dynamic_ncols=True)
        with self._collector.active():
            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch

                input_ids = input_ids.to(self.device)

                for t_val in self._timestep_list:
                    t_val = int(t_val)
                    noisy_ids, mask = mask_tokens(
                        input_ids=input_ids,
                        mask_token_id=self.mask_token_id,
                        t=t_val,
                        total_timesteps=self.total_timesteps,
                        seed=self.seed,
                    )

                    # Dream invariant: BOS (pos 0) and EOS (pos -1) must never be masked
                    mask[:, 0] = False
                    mask[:, -1] = False
                    noisy_ids[:, 0] = input_ids[:, 0]
                    noisy_ids[:, -1] = input_ids[:, -1]

                    self._collector.set_timestep(t_val)

                    try:
                        loss = _dream_loss(
                            model=self.model,
                            masked_ids=noisy_ids,
                            original_ids=input_ids,
                            mask=mask,
                        )
                        loss.backward()
                        self._collector.snapshot(t_val)
                        self.model.zero_grad(set_to_none=True)
                    except Exception as exc:
                        logger.warning(f"  [DreamKFAC] Error at t={t_val}: {exc}. Skipping.")
                        self.model.zero_grad(set_to_none=True)
                        continue

        self._kfac_store = self._collector.get_all()
        n_layer_ts = sum(len(ts) for ts in self._kfac_store.values())
        logger.info(f"  [DreamKFAC] K-FAC collection complete. (layer × timestep) entries: {n_layer_ts}")
        return self._kfac_store

    def compute_scores(self) -> Dict[str, torch.Tensor]:
        """Compute Kronecker-OBD scores per timestep, aggregate across timesteps and persist them."""
        logger.info("  [DreamKFAC] Computing Kronecker-OBD scores from KFAC store ...")
        weights_by_name = {n: m.weight.detach() for n, m in self.model.named_modules() if isinstance(m, nn.Linear)}

        for layer_name, per_t in self._kfac_store.items():
            W = weights_by_name.get(layer_name)
            if W is None:
                continue
            is_col = _is_down_proj_layer(layer_name) or _is_attn_out_layer(layer_name)
            for t, stats in per_t.items():
                if is_col:
                    score = _kfac_obd_col_scores(W, stats)
                else:
                    score = _kfac_obd_row_scores(W, stats)
                self.score_store.add(layer_name, int(t), score.cpu())

        # Aggregate across timesteps
        logger.info(f"  [DreamKFAC] Aggregating scores using strategy '{self.agg_method}' ...")
        if self.agg_method == "log_decay":
            self.score_store.aggregate(
                method="log_decay",
                total_timesteps=self.total_timesteps,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
            )
        else:
            self.score_store.aggregate(method=self.agg_method, total_timesteps=self.total_timesteps)

        if self.score_dir:
            try:
                os.makedirs(self.score_dir, exist_ok=True)
                store_save_path = os.path.join(self.score_dir, "pruning_scores.pkl")
                self.score_store.save(store_save_path)
                logger.info(f"  [DreamKFAC] Saved score store ({len(self.score_store._raw)} layers) to '{store_save_path}'.")
            except Exception as e:
                logger.warning(f"Could not persist score store: {e}")

        return self.score_store.all_aggregated()

    def cleanup(self) -> None:
        if self._collector is not None:
            self._collector._remove_hooks()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ======================================================================== #
#  CLI Arguments                                                           #
# ======================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect per-timestep K-FAC statistics (activations and gradients) and compute Kronecker-OBD scores for Dream Diffusion LMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_g = p.add_argument_group("Model")
    model_g.add_argument("--model", default="Dream-org/Dream-v0-Base-7B",
                         help="HuggingFace model ID or local checkpoint path.")
    model_g.add_argument("--output_dir", default="./dream_kfac",
                         help="Directory where the K-FAC store and scores are saved.")
    model_g.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    model_g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    model_g.add_argument("--trust_remote_code", action="store_true", default=True)

    # Calibration & Diffusion Schedule
    cal_g = p.add_argument_group("Calibration data")
    cal_g.add_argument("--cal_dataset", default="c4", choices=["c4", "wikitext2"],
                       help="Dataset used to collect K-FAC statistics.")
    cal_g.add_argument("--cal_samples", type=int, default=128,
                       help="Number of calibration token sequences.")
    cal_g.add_argument("--cal_batch_size", type=int, default=4)
    cal_g.add_argument("--seq_len", type=int, default=512,
                       help="Sequence length for calibration sequences.")

    diff_g = p.add_argument_group("Diffusion schedule")
    diff_g.add_argument("--num_timesteps", type=int, default=32,
                        help="Number of denoising timesteps for K-FAC collection.")
    diff_g.add_argument("--timestep_strategy", default="uniform", choices=["uniform", "log", "quadratic"])
    diff_g.add_argument("--total_timesteps", type=int, default=1000)
    diff_g.add_argument("--mask_token_id", type=int, default=DREAM_MASK_TOKEN_ID_DEFAULT)
    diff_g.add_argument("--agg_method", default="log_decay",
                        help="Temporal score aggregation method (sum, mean, log_decay, inv_linear, etc.).")
    diff_g.add_argument("--alpha_min", type=float, default=0.0)
    diff_g.add_argument("--alpha_max", type=float, default=1.0)

    # K-FAC storage
    kfac_g = p.add_argument_group("K-FAC")
    kfac_g.add_argument("--kfac_save_path", default=None,
                        help="Path to save collected K-FAC store. Defaults to <output_dir>/kfac_store.pt.")

    # Score storage
    score_g = p.add_argument_group("Score storage")
    score_g.add_argument("--score_dir", default=None, help="Where the per-timestep score store is saved (default <output_dir>/scores).")

    # Layer selection
    layer_g = p.add_argument_group("Layer selection")
    layer_g.add_argument("--layers_to_prune", nargs="*", default=[], help="Substring patterns of layers to collect for.")
    layer_g.add_argument("--layers_to_skip", nargs="*", default=[], help="Substring patterns of layers to skip.")

    # Misc
    misc_g = p.add_argument_group("Misc")
    misc_g.add_argument("--seed", type=int, default=42)
    misc_g.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    misc_g.add_argument("--log_file", default=None)

    return p.parse_args()


# ======================================================================== #
#  Main Execution                                                          #
# ======================================================================== #

def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_kfac_path = args.kfac_save_path or os.path.join(args.output_dir, "kfac_store.pt")
    score_dir = args.score_dir or os.path.join(args.output_dir, "scores")

    logger.info("=" * 65)
    logger.info("  DREAM DISCRETE DIFFUSION  –  K-FAC COLLECTION + SCORING")
    logger.info("=" * 65)
    logger.info(f"  Model              : {args.model}")
    logger.info(f"  K-FAC store        : {save_kfac_path}")
    logger.info(f"  Score dir          : {score_dir}")
    logger.info(f"  Calibration data   : {args.cal_dataset} (n={args.cal_samples}, seq={args.seq_len})")
    logger.info(f"  Timesteps          : {args.num_timesteps} ({args.timestep_strategy})  T={args.total_timesteps}")
    logger.info(f"  Score aggregation  : {args.agg_method}")
    logger.info(f"  Device / Dtype     : {args.device} / {args.dtype}")
    logger.info("=" * 65)

    # 1. Load Model & Tokenizer
    logger.info("[1/5] Loading Dream model and tokenizer ...")
    model, tokenizer = load_dream_model_and_tokenizer(
        model_path=args.model,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    model.tokenizer = tokenizer

    # 2. Build calibration loader
    logger.info("[2/5] Building calibration data loader ...")
    cal_loader = get_diffusion_calibration_loader(
        dataset_name=args.cal_dataset,
        num_samples=args.cal_samples,
        seq_len=args.seq_len,
        batch_size=args.cal_batch_size,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    # 3. Collect K-FAC statistics
    logger.info("[3/5] Collecting K-FAC statistics across timesteps ...")
    pruner = DreamKFACOBDPruner(
        model=model,
        device=args.device,
        mask_token_id=args.mask_token_id,
        total_timesteps=args.total_timesteps,
        num_timesteps=args.num_timesteps,
        timestep_strategy=args.timestep_strategy,
        seed=args.seed,
        layers_to_prune=args.layers_to_prune,
        layers_to_skip=args.layers_to_skip,
        agg_method=args.agg_method,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        score_dir=score_dir,
    )
    pruner.prepare()
    kfac_store = pruner.collect(cal_loader)

    # 4. Save K-FAC store
    logger.info(f"[4/5] Saving K-FAC store to '{save_kfac_path}' ...")
    os.makedirs(os.path.dirname(os.path.abspath(save_kfac_path)), exist_ok=True)
    torch.save(kfac_store, save_kfac_path)

    # 5. Compute + store scores
    logger.info(f"[5/5] Computing Kronecker-OBD scores (saved to '{score_dir}') ...")
    pruner.compute_scores()
    pruner.cleanup()
    logger.info("Done.")


if __name__ == "__main__":
    main()
