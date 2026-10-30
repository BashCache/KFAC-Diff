"""
LoRA Fine-tuning script for Pruned / Base Dream Diffusion Language Models.

Same recipe as utils/finetune_llada_lora.py (data mixtures, linear masking ratio r(t) = t / T with
t ~ U(1, T), loss on masked positions only, PEFT LoRA, gradient accumulation, cosine LR schedule,
validation, adapter checkpoints, optional merge), reusing its data loading, masking and evaluation.

Dream-specific differences:
- Logits are shifted by one position (position i-1 predicts token i), so the loss re-aligns them.
- The first and last token are never masked (the same convention eval_dream.py scores with).
- Physically shrunk checkpoints (per-layer `intermediate_sizes`) load via `dream_compat_patches`; zero-masked ones keep full shapes.
- LoRA targets only Dream's MLP projections (gate/up/down_proj).
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_dream import DREAM_MASK_TOKEN_ID_DEFAULT  # noqa: E402
from utils.dream_utils import dream_compat_patches  # noqa: E402
from utils.model_utils import compute_diffusion_loss  # noqa: E402
from utils.dataloader_utils import prepare_data_loaders  # noqa: E402
from utils.finetune_llada_lora import (  # noqa: E402
    build_parser,
    lora_target_regex,
    mask_pruned_attention_lora,
    verify_pruned_attention_zero,
    evaluate,
    sanitize_auto_map,
)

logger = logging.getLogger(__name__)

DREAM_LORA_TARGETS = "gate_proj,up_proj,down_proj"
DREAM_LOSS_KW = dict(shift_logits=True, protect_ends=True)


def parse_args():
    parser = build_parser()
    parser.description = "LoRA Fine-tuning for Pruned Dream Diffusion Models"
    parser.set_defaults(
        model_name_or_path="Dream-org/Dream-v0-Base-7B",
        target_modules=DREAM_LORA_TARGETS,
        output_dir="./finetuned_dream_lora",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    dtype = torch.float32
    if args.bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    logger.info(f"Target Device: {device} | Precision: {dtype}")

    # 1. Tokenizer
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or DREAM_MASK_TOKEN_ID_DEFAULT
    logger.info(f"Using mask_token_id: {mask_token_id}")

    # 2. Base model
    logger.info(f"Loading base model from {args.model_name_or_path}...")
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "device_map": {"": device}}
    load_kwargs["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = dtype
    dream_compat_patches(args.model_name_or_path)
    base_model = AutoModel.from_pretrained(args.model_name_or_path, **load_kwargs)
    base_model.config.use_cache = False

    # 3. LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if args.mask_pruned_attention:
        target_modules += [m for m in ("q_proj", "k_proj", "v_proj", "o_proj") if m not in target_modules]
    logger.info(f"Applying LoRA to target modules: {target_modules}"
                + (" + output head" if args.lora_output_head else " (output head excluded)"))

    if getattr(args, "gradient_checkpointing", False):
        try:
            base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except Exception as exc:
            logger.warning(f"Could not enable gradient checkpointing: {exc}. Proceeding without it.")
        base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_regex(target_modules, output_head=args.lora_output_head),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    attn_masks = mask_pruned_attention_lora(model) if args.mask_pruned_attention else None

    # 4. Data
    train_loader, val_loader = prepare_data_loaders(
        dataset_name=args.dataset,
        num_c4_samples=args.num_c4_samples,
        num_wiki_samples=args.num_wiki_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        seed=args.seed,
        num_eval_samples=args.num_eval_samples,
        num_alpaca_samples=args.num_alpaca_samples,
        alpaca_name=args.alpaca_name,
    )

    # 5. Optimizer and scheduler
    total_train_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_train_steps, 1),
    )
    logger.info(f"Total training steps: {total_train_steps} (Warmup steps: {warmup_steps})")

    def run_eval() -> Dict[str, float]:
        return evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, dtype, **DREAM_LOSS_KW)

    logger.info("Running initial zero-shot validation...")
    val_metrics = run_eval()
    logger.info(f"Initial Validation -> Loss: {val_metrics['val_loss']:.4f} | Perplexity: {val_metrics['val_perplexity']:.2f}")

    # 6. Training loop
    global_step = 0
    best_val_loss = float("inf")
    model.train()
    optimizer.zero_grad()

    logger.info("Starting LoRA fine-tuning...")
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.num_epochs} ---")
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                loss, metrics = compute_diffusion_loss(
                    model=model,
                    input_ids=input_ids,
                    mask_token_id=mask_token_id,
                    total_timesteps=args.total_timesteps,
                    **DREAM_LOSS_KW,
                )
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "avg_t": f"{metrics['avg_t']:.1f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

                if global_step % args.eval_steps == 0:
                    val_metrics = run_eval()
                    logger.info(
                        f"\nStep {global_step} | Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"Val PPL: {val_metrics['val_perplexity']:.2f}"
                    )
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        best_dir = os.path.join(args.output_dir, "best_checkpoint")
                        logger.info(f"Saving best adapter checkpoint to {best_dir}...")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                    model.train()

                if global_step % args.save_steps == 0:
                    step_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                    logger.info(f"Saving step checkpoint to {step_dir}...")
                    model.save_pretrained(step_dir)

        logger.info(f"Epoch {epoch} finished | Avg Train Loss: {epoch_loss / len(train_loader):.4f}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.2f} minutes.")

    # 7. Final evaluation and save
    final_val = run_eval()
    logger.info(f"Final Validation -> Loss: {final_val['val_loss']:.4f} | Perplexity: {final_val['val_perplexity']:.2f}")

    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    logger.info(f"Saving final LoRA adapter to {final_adapter_dir}...")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset,
        "num_c4_samples": args.num_c4_samples,
        "num_wiki_samples": args.num_wiki_samples,
        "num_alpaca_samples": args.num_alpaca_samples,
        "alpaca_name": args.alpaca_name,
        "seq_len": args.seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "lora_output_head": args.lora_output_head,
        "mask_pruned_attention": args.mask_pruned_attention,
        "total_timesteps": args.total_timesteps,
        "final_val_loss": final_val["val_loss"],
        "final_val_perplexity": final_val["val_perplexity"],
        "best_val_loss": best_val_loss,
        "training_time_minutes": total_time / 60.0,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.merge_and_save:
        merged_dir = os.path.join(args.output_dir, "merged_model")
        logger.info(f"Merging LoRA weights into base model and saving to {merged_dir}...")
        merged_model = model.merge_and_unload()
        if attn_masks is not None:
            verify_pruned_attention_zero(merged_model, attn_masks)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        sanitize_auto_map(merged_dir)
        logger.info("Merged model saved successfully!")

    logger.info(f"All done! Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
