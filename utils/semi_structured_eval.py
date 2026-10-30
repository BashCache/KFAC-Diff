import json
import logging
import os
import time
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from utils.finetune_llada_semistructured_lora import count_nm_violations, load_and_merge_adapters  # noqa: E402
from utils.sparsity import measure_model_sparsity  # noqa: E402

logger = logging.getLogger(__name__)

META_NAME = "semi_structured_meta.json"


def resolve_file(path_or_repo: str, filename: str) -> Optional[str]:
    if os.path.isfile(path_or_repo):
        return path_or_repo
    if os.path.isdir(path_or_repo):
        cand = os.path.join(path_or_repo, filename)
        return cand if os.path.isfile(cand) else None
    try:
        return hf_hub_download(repo_id=path_or_repo, filename=filename)
    except Exception:
        return None


def add_adapter_args(p) -> None:
    p.add_argument("--model", required=True,
                   help="N:M pruned model: a local folder (from compare_semi_structured*.py --save_model) "
                        "or a Hugging Face Hub repo id.")
    p.add_argument("--adapter", default=None,
                   help="Masked-LoRA adapters: a local .safetensors file, a local folder, or a Hugging Face Hub "
                        "repo id (the file --adapter_file is taken from it).  Omit to evaluate the pruned model "
                        "as it is.")
    p.add_argument("--adapter_file", default="best_adapter.safetensors",
                   help="File inside the --adapter folder / Hub repo (e.g. final_adapter.safetensors).")
    p.add_argument("--meta", default=None,
                   help=f"Path to {META_NAME} (default: looked up in --model, locally or on the Hub).")
    p.add_argument("--tag", default=None,
                   help="Name for the result file / entry (default: 'adapter' or 'pruned').")
    p.add_argument("--output_dir", default="./semi_structured_eval_results")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_layer_sparsity", type=float, default=0.05,
                   help="A Linear layer counts as pruned if at least this fraction of its weights is zero.")


def _sparse_layer_names(model: nn.Module, min_layer_sparsity: float):
    names, fracs = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            z = (module.weight.data == 0).float().mean().item()
            if z >= min_layer_sparsity:
                names.append(name)
                fracs.append(z)
    return names, fracs


def evaluate_semi_structured(
    args,
    model: nn.Module,
    tokenizer: Any,
    run_eval: Callable[[nn.Module, Any, Any], Dict[str, Any]],
) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    meta: Optional[Dict[str, Any]] = None
    meta_path = args.meta or resolve_file(args.model, META_NAME)
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(f"Pruning metadata: keep {meta.get('keep_n')} of {meta.get('group_m')} "
                    f"({meta.get('strategy')}, family={meta.get('model_family', 'llada')})")
    else:
        logger.warning(f"No {META_NAME}: the N:M pattern itself will not be verified.")

    info: Dict[str, Any] = {"model": args.model, "adapter": args.adapter,
                            "pattern": f"{meta['keep_n']}:{meta['group_m']}" if meta else None}

    if args.adapter:
        t0 = time.time()
        adapter_path = resolve_file(args.adapter, args.adapter_file)
        if adapter_path is None:
            raise FileNotFoundError(
                f"Could not find adapters: '{args.adapter}' is not a file, nor a folder containing "
                f"'{args.adapter_file}', nor a Hugging Face Hub repo that has it.")
        logger.info(f"Adapter file: {adapter_path}")
        merge = load_and_merge_adapters(model, adapter_path, meta)
        src = merge.get("source_model")
        if src and os.path.exists(args.model) and os.path.abspath(str(src)) != os.path.abspath(args.model):
            logger.warning(f"Adapter was trained on '{src}', but you are applying it to '{args.model}'.")
        elif src:
            logger.info(f"Adapter was trained on '{src}' (shapes are verified against the loaded model).")
        ok = not merge["merge_violations"] and not merge["nm_violations"]
        logger.log(logging.INFO if ok else logging.ERROR,
                   f"Merged adapters into {merge['layers']} layers in {time.time() - t0:.1f}s "
                   f"(r={merge['r']}, alpha={merge['alpha']}) | nonzeros at pruned positions = "
                   f"{merge['merge_violations']} | N:M groups over limit = {merge['nm_violations']}")
        info["adapter_merge"] = merge

    names, fracs = _sparse_layer_names(model, args.min_layer_sparsity)
    nm_bad = None
    if meta and "keep_n" in meta and "group_m" in meta and names:
        nm_bad = count_nm_violations(model, names, int(meta["keep_n"]), int(meta["group_m"]))
        logger.log(logging.INFO if not nm_bad else logging.ERROR,
                   f"N:M check on {len(names)} sparse layers: {nm_bad} groups over the limit")
    model_sp = measure_model_sparsity(model)
    mean_sp = sum(fracs) / max(len(fracs), 1)
    logger.info(f"Sparse layers: {len(names)} | mean sparsity {mean_sp:.2%} | model sparsity "
                f"{model_sp['overall_sparsity']:.2%} ({model_sp['sparsity_excl_embeddings_and_head']:.2%} "
                f"excl. embeddings+head)")

    tag = args.tag or ("adapter" if args.adapter else "pruned")
    out_path = os.path.join(args.output_dir, f"eval_{tag}.json")
    extra = {
        **info,
        "sparse_layers": len(names),
        "prunable_layer_sparsity": mean_sp,
        "nm_violations": nm_bad,
        "overall_model_sparsity": model_sp["overall_sparsity"],
        "model_sparsity_excl_embeddings": model_sp["sparsity_excl_embeddings"],
        "model_sparsity_excl_embeddings_and_head": model_sp["sparsity_excl_embeddings_and_head"],
    }

    def _save(res: Dict[str, Any], partial: bool) -> None:
        with open(out_path, "w") as f:
            json.dump({tag: {**res, **extra, "eval_time_s": round(time.time() - t0, 2), "partial": partial}},
                      f, indent=2, default=str)

    t0 = time.time()
    res = run_eval(model, tokenizer, args, on_update=lambda r: _save(r, True))   # saved after every task
    results = {**res, **extra, "eval_time_s": round(time.time() - t0, 2)}
    _save(res, False)
    logger.info(f"Results saved: {out_path}")

    logger.info("=" * 70)
    for k, v in results.items():
        if k.startswith(("acc_", "perplexity_")):
            logger.info(f"  {k:<32s}: {v:.4f}")
    logger.info("=" * 70)
    return results
