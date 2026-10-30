"""
eval_semi_structured.py   (LLaDA)

Evaluate an N:M (semi-structured) pruned LLaDA model, optionally with the masked-LoRA adapters trained by
utils/finetune_llada_semistructured_lora.py, using exactly the evaluation of compare_semi_structured.py
(same options, defaults and code (utils/llada_eval_common.py): Monte Carlo perplexity from eval_diffusion.evaluate_perplexity and
lm-eval through LLaDiffusionLM).

The adapters are merged with the pruning mask in memory before scoring (W <- W + s * M * B A), so pruned
positions stay exactly zero; the script checks this and the N:M pattern, and records both in the result JSON.
The dense model is NOT evaluated.

Usage
-----
# pruned model with the fine-tuned adapters
python evaluation/eval_semi_structured.py \\
    --model   ./semi_structured_results/2of4/model_snr_weighted_2of4 \\
    --adapter ./finetuned_models/semi_2of4_lora/best_adapter.safetensors \\
    --output_dir ./semi_structured_eval_results/2of4 --skip_perplexity

# the same pruned model without fine-tuning (omit --adapter)
python evaluation/eval_semi_structured.py --model ./semi_structured_results/2of4/model_snr_weighted_2of4 \\
    --output_dir ./semi_structured_eval_results/2of4 --skip_perplexity
"""

import argparse
import logging
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llada_eval_common import add_eval_args, run_eval
from utils.model_utils import load_llada_model_and_tokenizer
from utils.semi_structured_eval import add_adapter_args, evaluate_semi_structured

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an N:M pruned LLaDA model, optionally with masked-LoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_adapter_args(p)
    add_eval_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_llada_model_and_tokenizer(
        model_path_or_config=args.model, dtype=args.dtype, device=args.device,
    )
    model.tokenizer = tokenizer
    evaluate_semi_structured(args, model, tokenizer, run_eval)


if __name__ == "__main__":
    main()
