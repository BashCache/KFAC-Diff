"""
lm_eval_utils.py - flatten lm-eval results into {name: value}.
"""

import json
import logging
import math
import os
from typing import Callable, Dict, List, Optional

import lm_eval

logger = logging.getLogger(__name__)

BUILTIN_FEWSHOT_TASKS = ("gsm8k",)

def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

def extract_task_metrics(task_results: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task, metrics in (task_results or {}).get("results", {}).items():
        for key, raw in metrics.items():
            val = _num(raw)
            if val is None or "," not in key:
                continue
            name, filt = key.split(",", 1)
            is_err = name.endswith("_stderr")
            base = name[: -len("_stderr")] if is_err else name
            if base == "acc" and filt == "none":
                label = f"acc_stderr_{task}" if is_err else f"acc_{task}"
            else:
                filt = filt.replace("-", "_")
                label = f"{base}_stderr_{filt}_{task}" if is_err else f"{base}_{filt}_{task}"
            out[label] = val
        if f"acc_{task}" not in out and "acc" in metrics and _num(metrics["acc"]) is not None:
            out[f"acc_{task}"] = float(metrics["acc"])
    return out

def headline(results: Dict[str, float], task: str) -> float:
    """The number to quote for a task: acc, else strict-match exact match, else flexible-extract, else NaN."""
    for key in (f"acc_{task}", f"exact_match_strict_match_{task}", f"exact_match_flexible_extract_{task}"):
        if key in results:
            return results[key]
    return math.nan

def _merge_results(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        elif k not in out:
            out[k] = v
    return out

def evaluate_tasks(model, tasks: List[str], limit: Optional[int] = None, model_args: Optional[str] = None,
                   log_samples: bool = False, simple_evaluate=None) -> dict:
    if simple_evaluate is None:
        simple_evaluate = lm_eval.simple_evaluate
    tasks = list(tasks)
    groups = [
        ([t for t in tasks if t not in BUILTIN_FEWSHOT_TASKS], 0),
        ([t for t in tasks if t in BUILTIN_FEWSHOT_TASKS], None),
    ]
    merged: Optional[dict] = None
    for group, shots in groups:
        if not group:
            continue
        kw = dict(model=model, tasks=group, num_fewshot=shots, limit=limit, log_samples=log_samples)
        if model_args is not None:
            kw["model_args"] = model_args
        res = simple_evaluate(**kw)
        merged = res if merged is None else _merge_results(merged, res)
    if merged is not None:
        logger.info(f"n-shot per task: {merged.get('n-shot', {})}")
    return merged

def evaluate_tasks_each(model, tasks: List[str], limit: Optional[int] = None, output_path: Optional[str] = None,
                        results: Optional[Dict[str, float]] = None, simple_evaluate=None,
                        on_task: Optional[Callable[[str, Dict[str, float]], None]] = None) -> dict:
    merged: Optional[dict] = None
    for i, task in enumerate(tasks, 1):
        res = evaluate_tasks(model, [task], limit=limit, simple_evaluate=simple_evaluate)
        merged = res if merged is None else _merge_results(merged, res)
        flat = extract_task_metrics(res)
        flat.setdefault(f"acc_{task}", headline(flat, task))
        if results is not None:
            results.update(flat)
        if on_task is not None:
            on_task(task, flat)
        logger.info(f"[{i}/{len(tasks)}] {task}: acc/em={flat[f'acc_{task}']:.4f}")
        for name, val in flat.items():
            if name != f"acc_{task}":
                logger.info(f"      {name} = {val:.4f}")
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(merged, f, indent=2, default=str)
    if output_path:
        logger.info(f"Results saved to: {output_path}")
    return merged
