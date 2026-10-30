"""
Evaluate an N:M (semi-structured) pruned Dream model,
"""

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.eval_dream import load_dream_model_and_tokenizer
from utils.dream_eval_common import add_eval_args, run_eval
from utils.semi_structured_eval import add_adapter_args, evaluate_semi_structured

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an N:M pruned Dream model, optionally with masked-LoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_adapter_args(p)
    add_eval_args(p) 
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_dream_model_and_tokenizer(
        model_path=args.model, dtype=args.dtype, device=args.device,
    )
    evaluate_semi_structured(args, model, tokenizer, run_eval)


if __name__ == "__main__":
    main()
