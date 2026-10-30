import argparse
import logging
import traceback
from typing import Any, Dict

import torch

import lm_eval as _lm_eval_lib
from lm_eval.api.model import LM as _LM_base

from evaluation.eval_diffusion import (
    MASK_TOKEN_ID_DEFAULT,
    LLaDiffusionLM as _LLaDiffusionLM,
    evaluate_perplexity,
)
from evaluation.lm_eval_utils import evaluate_tasks_each

logger = logging.getLogger(__name__)


def wrap_model_for_lm_eval(model, tokenizer, batch_size: int, mc_num: int, device: str, steps: int = 64,
                           cfg: float = 0.5):
    lm = object.__new__(_LLaDiffusionLM)

    if _LM_base is not None:
        _LM_base.__init__(lm)

    # Replicate the attribute layout set by LLaDiffusionLM.__init__
    lm.model              = model
    lm.tokenizer          = tokenizer
    lm.mask_id            = (
        getattr(model.config, "mask_token_id", None)
        or getattr(tokenizer, "mask_token_id", None)
        or MASK_TOKEN_ID_DEFAULT
    )
    lm._device            = torch.device(device)
    lm.batch_size_per_gpu = int(batch_size)
    lm.mc_num             = int(mc_num)
    assert lm.mc_num % lm.batch_size_per_gpu == 0, (
        f"mc_num ({mc_num}) must be divisible by batch_size ({batch_size})"
    )
    lm.is_check_greedy    = False
    lm.cfg                = float(cfg)
    lm.steps              = int(steps)
    lm.gen_length         = 1024
    lm.block_length       = 1024
    lm.remasking          = "low_confidence"
    return lm


def run_eval(model, tokenizer, args, on_update=None) -> Dict[str, Any]:
    """`on_update(res)` (optional) is called after every perplexity dataset and every lm-eval task."""
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
            if on_update:
                on_update(res)

    if args.lm_eval_tasks:
        try:
            lm = wrap_model_for_lm_eval(
                model=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size,
                mc_num=args.mc_num, device=args.device, steps=args.lm_eval_steps, cfg=args.cfg,
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
    """Evaluation options, shared with eval_semi_structured.py (same names/defaults as compare_pruning_strategies.py)."""
    # Evaluation options (same names, defaults and help as compare_pruning_strategies.py)
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip perplexity evaluation (only the lm-eval tasks are run).")
    p.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"],
                   help="Perplexity evaluation datasets.")
    p.add_argument("--eval_seq_len", type=int, default=1024,
                   help="Sequence length for perplexity evaluation.")
    p.add_argument("--mc_num", type=int, default=128,
                   help="Monte Carlo masking iterations (lm-eval log-likelihoods and perplexity). Must be divisible by "
                        "--lm_eval_batch_size.  Standard setting: mc_num 128, lm_eval_batch_size 16.")
    p.add_argument("--mc_batch_size", type=int, default=16,
                   help="MC mini-batch size (mc_num must be divisible by this).")
    p.add_argument("--lm_eval_tasks", nargs="+",
                   default=["arc_easy", "winogrande", "piqa", "arc_challenge", "hellaswag", "boolq", "openbookqa", "gsm8k"],
                   help="lm-eval tasks to run after each pruning run (empty list to skip).")
    p.add_argument("--lm_eval_limit", type=lambda x: None if x.lower() == "none" else int(x), default=100,
                   help="Max samples per lm-eval task (None = all).")
    p.add_argument("--lm_eval_batch_size", type=int, default=16,
                   help="Batch size for lm-eval MC log-likelihood scoring. Must divide --mc_num.")
    p.add_argument("--lm_eval_steps", type=int, default=256,
                   help="Number of denoising steps for LLaDiffusionLM evaluation.")
    p.add_argument("--cfg", type=float, default=0.5,
                   help="Classifier-free guidance for lm-eval scoring/generation: uncond + (cfg + 1) x (cond - uncond); 0 = off.")
