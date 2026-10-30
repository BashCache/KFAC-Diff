"""Perplexity + lm-eval evaluation helpers"""

import argparse
import logging
import traceback
from typing import Any, Dict

import torch.nn as nn

import lm_eval as _lm_eval_lib

from evaluation.eval_dream import _wrap_dream_for_lm_eval, evaluate_perplexity_dream
from evaluation.lm_eval_utils import evaluate_tasks_each

logger = logging.getLogger(__name__)


def run_eval(model: nn.Module, tokenizer, args, on_update=None) -> Dict[str, Any]:
    """`on_update(res)` (optional) is called after every perplexity dataset and every lm-eval task."""
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
            if on_update:
                on_update(res)

    if args.lm_eval_tasks:
        try:
            lm = _wrap_dream_for_lm_eval(
                model=model, tokenizer=tokenizer, batch_size=args.mc_batch_size,
                mc_num=args.mc_num, device=args.device,
            )
            def _on_task(task, flat):
                res[f"acc_{task}"] = flat[f"acc_{task}"]
                logger.info(f"  {task}: acc/em={res[f'acc_{task}']:.4f}")
                if on_update:
                    on_update(res)

            evaluate_tasks_each(lm, args.lm_eval_tasks, limit=args.lm_eval_limit,
                                simple_evaluate=_lm_eval_lib.simple_evaluate, on_task=_on_task)
        except Exception as exc:
            logger.error(f"  lm_eval error: {exc}\n{traceback.format_exc()}")
    return res


def add_eval_args(p: argparse.ArgumentParser) -> None:
    """Evaluation options, shared with eval_semi_structured_dream.py (same names/defaults as compare_pruning_strategies_dream.py)."""
    # ── Evaluation options (names / defaults as in compare_pruning_strategies_dream.py) ── #
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip perplexity evaluation (only the lm-eval tasks are run).")
    p.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"],
                   help="Perplexity evaluation datasets (wikitext2, c4).")
    p.add_argument("--eval_seq_len", type=int, default=512,
                   help="Sequence length for continuous diffusion perplexity evaluation.")
    p.add_argument("--mc_num", type=int, default=128,
                   help="Monte Carlo masking iterations for Dream MC evaluation (standard setting: mc_num 128, batch 32).")
    p.add_argument("--mc_batch_size", type=int, default=32,
                   help="Batch size for MC evaluation (must divide mc_num).")
    p.add_argument("--lm_eval_tasks", nargs="+",
                   default=["arc_easy", "winogrande", "piqa", "arc_challenge", "hellaswag", "boolq", "gsm8k"],
                   help="lm-eval tasks to evaluate.")
    p.add_argument("--lm_eval_limit", type=lambda x: None if x.lower() == "none" else int(x), default=100,
                   help="Max samples per lm-eval task (None = full evaluation).")
