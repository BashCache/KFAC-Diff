"""
Sparsity-preserving LoRA fine-tuning of semi-structured (N:M) pruned Dream-v0-Base-7B.

Dream-specific differences (same as utils/finetune_dream_lora.py):
  * logits are shifted by one position (position i-1 predicts token i), so the loss re-aligns them;
  * the first and last token are never masked;
  * the model is loaded with load_dream_model_and_tokenizer (Dream compat patches, mask / bos / eos token ids);
  * after saving the merged model, the config's auto_map is repaired so the folder reloads
    (utils.model_utils.sanitize_auto_map and utils.dream_utils._point_auto_map_at_source_repo).

Input : a model saved by compare_semi_structured_dream.py --save_model.

Usage
-----
python utils/finetune_dream_semistructured_lora.py \\
    --model_name_or_path ./semi_structured_results_dream/2of4/model_snr_weighted_2of4 \\
    --output_dir        ./finetuned_models/dream_semi_2of4_lora \\
    --dataset mixed --num_c4_samples 2000 --num_wiki_samples 2000 \\
    --seq_len 1024 --num_epochs 1 --lora_r 16 --gradient_checkpointing
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from evaluation.eval_dream import load_dream_model_and_tokenizer   # noqa: E402
from utils.dream_utils import _point_auto_map_at_source_repo   # noqa: E402
from utils.finetune_llada_semistructured_lora import build_parser, run_masked_lora   # noqa: E402
from utils.model_utils import sanitize_auto_map   # noqa: E402

DREAM_LOSS_KW = dict(shift_logits=True, protect_ends=True)


def _fix_dream_config(out_dir: str, model: nn.Module) -> None:
    """Make the merged folder reloadable (auto_map must name the Hub repo that holds the modeling code)."""
    _point_auto_map_at_source_repo(out_dir, getattr(model.config, "_name_or_path", None))
    sanitize_auto_map(out_dir)


def main() -> None:
    parser = build_parser()
    parser.description = "Sparsity-preserving LoRA fine-tuning of N:M pruned Dream-v0-Base-7B"
    args = parser.parse_args()

    model, tokenizer = load_dream_model_and_tokenizer(
        model_path=args.model_name_or_path, dtype=args.dtype, device=str(torch.device(args.device)),
    )
    run_masked_lora(
        args, model, tokenizer, tokenizer.mask_token_id,
        loss_kw=DREAM_LOSS_KW,
        post_save=_fix_dream_config,
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    main()
