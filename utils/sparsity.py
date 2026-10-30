"""Zero-weight sparsity measurements shared by the compare / baseline pruning scripts."""

from typing import Any, Dict, List
import torch.nn as nn

def measure_sparsity(model: nn.Module) -> float:
    """Zero fraction over every nn.Linear / nn.Conv2d weight (lm_head included)."""
    total = zero = 0
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w      = module.weight.data
            total += w.numel()
            zero  += (w == 0).sum().item()
    return zero / total if total > 0 else 0.0

def measure_layer_sparsity(model: nn.Module) -> Dict[str, float]:
    """Per-layer zero fraction for all linear/conv2d modules."""
    layer_sparsity = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.data
            total = w.numel()
            zero = (w == 0).sum().item()
            if total > 0:
                layer_sparsity[name] = zero / total
    return layer_sparsity

def measure_model_sparsity(model: nn.Module) -> Dict[str, Any]:
    """
    Zero-weight fraction over EVERY parameter of the model: attention/MLP linears, the output
    head, the embedding table, norm weights and biases.  (`measure_sparsity` only looks at
    nn.Linear / nn.Conv2d, so it ignores the embeddings and divides by a smaller total.)

    Pruning here zeroes weights in place rather than shrinking the tensors, so the numbers
    count zeros.  Tied parameters are counted once.

    Returns
    -------
    {"total_params", "zero_params", "overall_sparsity",
     "sparsity_excl_embeddings"          : same, leaving the embedding table(s) out of numerator and denominator,
     "sparsity_excl_embeddings_and_head" : also leaving out the output head (attention/MLP/norm body only),
     "linear_sparsity"      : zeros / params over every nn.Linear / nn.Conv2d (the older measure),
     "categories": {name: {"params", "zero", "sparsity"}}}
    with categories
      block_linear   : nn.Linear inside the transformer blocks (what the pruners act on)
      output_head    : every other nn.Linear (final vocabulary projection)
      embedding      : nn.Embedding tables
      norm_and_other : norm weights, biases, everything else
    """
    cats: Dict[str, List[int]] = {k: [0, 0] for k in ("block_linear", "output_head", "embedding", "norm_and_other")}
    lin_total = lin_zero = 0
    seen: set = set()
    for mod_name, mod in model.named_modules():
        for _, p in mod.named_parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            n = p.numel()
            z = int((p.data == 0).sum().item())
            if isinstance(mod, nn.Embedding):
                cat = "embedding"
            elif isinstance(mod, (nn.Linear, nn.Conv2d)):
                in_block = any(tag in f".{mod_name}." for tag in (".blocks.", ".layers.", ".h."))
                cat = "block_linear" if in_block else "output_head"
                if p is mod.weight:
                    lin_total += n
                    lin_zero += z
            else:
                cat = "norm_and_other"
            cats[cat][0] += n
            cats[cat][1] += z
    total = sum(v[0] for v in cats.values())
    zero = sum(v[1] for v in cats.values())

    def frac(*drop: str) -> float:
        n = total - sum(cats[k][0] for k in drop)
        z = zero - sum(cats[k][1] for k in drop)
        return z / n if n else 0.0

    return {
        "total_params": total,
        "zero_params": zero,
        "overall_sparsity": frac(),
        "sparsity_excl_embeddings": frac("embedding"),
        "sparsity_excl_embeddings_and_head": frac("embedding", "output_head"),
        "linear_sparsity": lin_zero / lin_total if lin_total else 0.0,
        "categories": {k: {"params": v[0], "zero": v[1], "sparsity": v[1] / v[0] if v[0] else 0.0}
                       for k, v in cats.items()},
    }
