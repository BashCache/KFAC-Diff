"""
Sparsity-preserving LoRA fine-tuning of semi-structured (N:M) pruned LLaDA models.

Input : a model saved by compare_semi_structured.py --save_model (dense shapes, zeros at pruned
        positions).  Output: LoRA adapters, plus (by default) a merged model whose pruned positions
        are STILL exactly zero.

Only A (r x d_in) and B (d_out x r) are trained; the base model is frozen.  Layers are adapted only if
they are actually sparse (zero fraction >= --min_layer_sparsity), so dense layers (output head, layers
skipped when pruning) are untouched.  --target_modules optionally restricts this further by name.

Cost note: the masked delta is formed in weight space (d_out x d_in) inside every adapted layer's forward,
and autograd keeps the effective weight for backward. 

The loss is the masked-diffusion objective on [MASK] tokens with t ~ U(1, T), r(t) = t / T, as in
utils/finetune_llada_lora.py (data loaders, loss and evaluation are imported from there).

Usage
-----
python utils/finetune_llada_semistructured_lora.py \\
    --model_name_or_path ./semi_structured_results/model_snr_weighted_8of10 \\
    --output_dir        ./finetuned_models/semi_8of10_lora \\
    --dataset mixed --num_c4_samples 2000 --num_wiki_samples 2000 \\
    --seq_len 1024 --num_epochs 1 --lora_r 16 --gradient_checkpointing
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from utils.dataloader_utils import prepare_data_loaders  # noqa: E402
from utils.finetune_llada_lora import evaluate  # noqa: E402
from utils.model_utils import compute_diffusion_loss, load_llada_model_and_tokenizer   # noqa: E402

logger = logging.getLogger(__name__)

META_NAME = "semi_structured_meta.json"
#  Masked LoRA layer                                                                                                                          #
class MaskedLoRALinear(nn.Module):
    """Frozen Linear + LoRA whose update is restricted to the nonzero (unpruned) positions of W."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        d_out, d_in = base.weight.shape
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # the surrounding model code may read these off the layer
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.base.weight
        delta = (self.lora_B @ self.lora_A).to(W.dtype) * (W != 0)
        return F.linear(x, W + self.scale * delta, self.base.bias)

    @torch.no_grad()
    def merge(self) -> "tuple[nn.Linear, int]":
        """W <- W + s * M * (B A) in fp32, cast back.  Returns (base Linear, #nonzeros at pruned positions)."""
        W = self.base.weight
        keep = W.data != 0
        delta = (self.lora_B.float() @ self.lora_A.float()) * keep
        W.data.copy_((W.data.float() + self.scale * delta).to(W.dtype))
        violations = int(((W.data != 0) & ~keep).sum().item())
        return self.base, violations


def inject_masked_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    min_layer_sparsity: float,
    target_modules: Optional[List[str]],
) -> Dict[str, MaskedLoRALinear]:
    """Replace every sparse Linear (optionally filtered by name) by a MaskedLoRALinear."""
    wrapped: Dict[str, MaskedLoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_modules and not any(t in name for t in target_modules):
            continue
        if (module.weight.data == 0).float().mean().item() < min_layer_sparsity:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = MaskedLoRALinear(module, r, alpha)
        setattr(parent, child, layer)
        wrapped[name] = layer
    return wrapped


@torch.no_grad()
def merge_all(model: nn.Module, wrapped: Dict[str, MaskedLoRALinear]) -> int:
    """Merge every adapter into its base weight and put the plain Linear back.  Returns total violations."""
    total_violations = 0
    for name, layer in wrapped.items():
        base, v = layer.merge()
        total_violations += v
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, base)
    return total_violations


@torch.no_grad()
def count_nm_violations(model: nn.Module, names: List[str], keep_n: int, group_m: int) -> int:
    """Groups of M consecutive input weights with more than N nonzeros (tail columns ignored)."""
    bad = 0
    for name in names:
        W = model.get_submodule(name).weight.data
        d_out, d_in = W.shape
        full = (d_in // group_m) * group_m
        if full == 0:
            continue
        nnz = (W[:, :full].reshape(d_out, full // group_m, group_m) != 0).sum(dim=-1)
        bad += int((nnz > keep_n).sum().item())
    return bad


def save_adapters(wrapped: Dict[str, MaskedLoRALinear], path: str, args: argparse.Namespace) -> None:
    tensors = {}
    for name, layer in wrapped.items():
        tensors[f"{name}.lora_A"] = layer.lora_A.detach().cpu().contiguous()
        tensors[f"{name}.lora_B"] = layer.lora_B.detach().cpu().contiguous()
    save_file(tensors, path, metadata={"r": str(args.lora_r), "alpha": str(args.lora_alpha),
                                       "masked": "true", "source_model": args.model_name_or_path})


def load_and_merge_adapters(
    model: nn.Module,
    adapter_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load adapters written by save_adapters (best_adapter.safetensors / final_adapter.safetensors) into an
    N:M pruned model and merge them with the pruning mask (W <- W + s * M * B A), in memory.
    Returns {"layers", "r", "alpha", "source_model", "merge_violations", "nm_violations"}; both violation
    counts must be 0 (nonzeros at pruned positions / groups with more than N nonzeros).
    """
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    if "r" not in md or "alpha" not in md:
        raise ValueError(f"{adapter_path} has no r/alpha metadata; was it written by save_adapters?")
    r, alpha = int(md["r"]), float(md["alpha"])

    names = sorted({k.rsplit(".lora_", 1)[0] for k in tensors})
    wrapped: Dict[str, MaskedLoRALinear] = {}
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"adapter layer '{name}' is not an nn.Linear in this model")
        A, B = tensors[f"{name}.lora_A"], tensors[f"{name}.lora_B"]
        if tuple(A.shape) != (r, module.in_features) or tuple(B.shape) != (module.out_features, r):
            raise ValueError(f"adapter shapes for '{name}' (A {tuple(A.shape)}, B {tuple(B.shape)}) do not match "
                             f"the layer ({module.out_features} x {module.in_features}, r={r}): wrong base model?")
        layer = MaskedLoRALinear(module, r, alpha)
        layer.lora_A.data.copy_(A.to(layer.lora_A.device))
        layer.lora_B.data.copy_(B.to(layer.lora_B.device))
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, layer)
        wrapped[name] = layer

    violations = merge_all(model, wrapped)
    nm_bad = None
    if meta and "keep_n" in meta and "group_m" in meta:
        nm_bad = count_nm_violations(model, names, int(meta["keep_n"]), int(meta["group_m"]))
    return {"layers": len(names), "r": r, "alpha": alpha, "source_model": md.get("source_model"),
            "merge_violations": violations, "nm_violations": nm_bad}
#  CLI                                                                                                                                                          #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sparsity-preserving LoRA fine-tuning of N:M pruned LLaDA models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name_or_path", required=True,
                   help="Pruned model dir (from compare_semi_structured.py --save_model).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--meta", default=None,
                   help=f"Path to {META_NAME} (default: <model_name_or_path>/{META_NAME} if present).")

    p.add_argument("--dataset", default="mixed", choices=["c4", "wikitext2", "alpaca", "mixed", "mixed_alpaca"])
    p.add_argument("--num_c4_samples", type=int, default=1000)
    p.add_argument("--num_wiki_samples", type=int, default=1000)
    p.add_argument("--num_alpaca_samples", type=int, default=1000)
    p.add_argument("--alpaca_name", default="tatsu-lab/alpaca")
    p.add_argument("--num_eval_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--total_timesteps", type=int, default=64,
                   help="Diffusion T for t ~ U(1, T), r(t) = t / T.")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--target_modules", default="",
                   help="Comma-separated name substrings restricting which sparse layers get an adapter "
                        "(default: every sparse Linear layer).")
    p.add_argument("--min_layer_sparsity", type=float, default=0.05,
                   help="A Linear layer is adapted only if at least this fraction of its weights is zero.")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_merge", action="store_true",
                   help="Only save the adapters; do not merge into a full model.")

    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p
#  Main                                                                                                                                                     #
def run_masked_lora(
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer: Any,
    mask_token_id: int,
    loss_kw: Optional[Dict[str, Any]] = None,
    post_save: Optional[Callable[[str, nn.Module], None]] = None,
) -> None:
    """
    Train masked LoRA adapters on an already-loaded N:M pruned model, then merge with the mask and verify.
    Model-specific parts are passed in: `loss_kw` goes to compute_diffusion_loss / evaluate (e.g. Dream needs
    shift_logits=True, protect_ends=True) and `post_save(out_dir, model)` runs after the merged model is saved.
    """
    loss_kw = loss_kw or {}
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    amp_dtype = getattr(torch, args.dtype)

    meta_path = args.meta or os.path.join(args.model_name_or_path, META_NAME)
    meta: Optional[Dict[str, Any]] = None
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(f"Loaded pruning metadata: {meta_path} (keep {meta.get('keep_n')} of {meta.get('group_m')})")
    else:
        logger.warning("No semi_structured_meta.json: the N:M pattern will not be verified after merging, "
                       "only that no pruned position became nonzero.")

    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    if args.gradient_checkpointing:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
                model.model.set_activation_checkpointing("whole_layer")
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled.")
        except Exception as exc:
            logger.warning(f"Could not enable gradient checkpointing: {exc}")

    # 2. masked LoRA on the sparse layers
    targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    wrapped = inject_masked_lora(model, args.lora_r, args.lora_alpha, args.min_layer_sparsity, targets or None)
    if not wrapped:
        raise SystemExit("No sparse Linear layers found; is this a pruned checkpoint "
                         "(or is --min_layer_sparsity / --target_modules too strict)?")
    lora_params = [p for layer in wrapped.values() for p in (layer.lora_A, layer.lora_B)]
    n_lora = sum(p.numel() for p in lora_params)
    n_base = sum(p.numel() for p in model.parameters()) - n_lora
    logger.info(f"Adapted sparse layers: {len(wrapped)} | trainable LoRA params: {n_lora / 1e6:.2f}M "
                f"({n_lora / max(n_base, 1):.3%} of the frozen base)")

    # 3. data
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

    # 4. optimizer / schedule
    total_steps = max((len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"Optimizer steps: {total_steps} (warmup {warmup_steps})")

    # B is zero-initialised, so this is the pruned model's loss (the dense model is not evaluated).
    start = evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, amp_dtype, **loss_kw)
    logger.info(f"Before fine-tuning -> val loss {start['val_loss']:.4f} | ppl {start['val_perplexity']:.2f}")

    # 5. train
    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    for epoch in range(1, args.num_epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                loss, metrics = compute_diffusion_loss(
                    model=model, input_ids=input_ids, mask_token_id=mask_token_id,
                    total_timesteps=args.total_timesteps, **loss_kw,
                )
                scaled = loss / args.gradient_accumulation_steps
            scaled.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(step=global_step, loss=f"{loss.item():.4f}",
                                 avg_t=f"{metrics['avg_t']:.1f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step % args.eval_steps == 0:
                    val = evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, amp_dtype, **loss_kw)
                    logger.info(f"Step {global_step} | val loss {val['val_loss']:.4f} | ppl {val['val_perplexity']:.2f}")
                    history.append({"step": global_step, **val})
                    if val["val_loss"] < best_val:
                        best_val = val["val_loss"]
                        save_adapters(wrapped, os.path.join(args.output_dir, "best_adapter.safetensors"), args)
                    model.train()

    logger.info(f"Training finished in {(time.time() - t_start) / 60:.1f} min.")

    # 6. final evaluation, save adapters
    final = evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, amp_dtype, **loss_kw)
    logger.info(f"After fine-tuning (adapters, unmerged) -> val loss {final['val_loss']:.4f} | "
                f"ppl {final['val_perplexity']:.2f}")
    save_adapters(wrapped, os.path.join(args.output_dir, "final_adapter.safetensors"), args)

    summary: Dict[str, Any] = {
        "source_model": args.model_name_or_path,
        "adapted_layers": len(wrapped),
        "lora_params": n_lora,
        "before": start,
        "after_unmerged": final,
        "global_steps": global_step,
        "history": history,
        "args": vars(args),
    }

    # 7. merge with the mask, verify, save
    if not args.no_merge:
        names = list(wrapped)
        violations = merge_all(model, wrapped)
        nm_bad = None
        if meta and "keep_n" in meta and "group_m" in meta:
            nm_bad = count_nm_violations(model, names, int(meta["keep_n"]), int(meta["group_m"]))
        level = logging.INFO if not violations and not nm_bad else logging.ERROR
        logger.log(level, f"Merge check: nonzeros at pruned positions = {violations}"
                          + (f" | N:M groups over limit = {nm_bad}" if nm_bad is not None else ""))
        merged_val = evaluate(model, val_loader, mask_token_id, args.total_timesteps, device, amp_dtype, **loss_kw)
        logger.info(f"After merge -> val loss {merged_val['val_loss']:.4f} | ppl {merged_val['val_perplexity']:.2f} "
                    f"(unmerged: {final['val_loss']:.4f}; small differences are bf16 rounding of the merge)")

        out = os.path.join(args.output_dir, "merged_model")
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        if post_save is not None:
            post_save(out, model)
        merged_meta = dict(meta or {})
        merged_meta["finetune"] = {"method": "masked_lora", "r": args.lora_r, "alpha": args.lora_alpha,
                                   "global_steps": global_step, "source_model": args.model_name_or_path,
                                   "merge_violations": violations, "nm_violations": nm_bad}
        with open(os.path.join(out, META_NAME), "w") as f:
            json.dump(merged_meta, f, indent=2)
        logger.info(f"Saved merged model -> {out}")
        summary.update({"after_merged": merged_val, "merge_violations": violations, "nm_violations": nm_bad})

    with open(os.path.join(args.output_dir, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer = load_llada_model_and_tokenizer(
        model_path_or_config=args.model_name_or_path, dtype=args.dtype, device=str(torch.device(args.device)),
    )
    run_masked_lora(args, model, tokenizer, tokenizer.mask_token_id)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    main()
