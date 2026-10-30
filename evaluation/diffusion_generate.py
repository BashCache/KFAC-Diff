"""
diffusion_generate.py
=====================
Text generation for masked-diffusion LMs, needed by lm-eval `generate_until` tasks such as GSM8K.
Dream does not need a loop here: its HF model provides `diffusion_generate` (used in eval_dream.py).
"""

from typing import List

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling trick (float64 for numerical stability); temperature 0 is a no-op (argmax)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Tokens to commit at each of `steps` steps for every row, as evenly as possible
    (the remainder goes to the first steps).  Shape (B, steps); each row sums to its mask count.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : int(remainder[i])] += 1
    return out


@torch.no_grad()
def llada_generate(
    model,
    prompt: torch.Tensor,            # (B, P) token ids, no padding
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    """Returns (B, P + gen_length) token ids: the prompt followed by the generated answer."""
    if gen_length % block_length != 0:
        raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(f"steps ({steps}) must be divisible by the number of blocks ({num_blocks})")
    if remasking not in ("low_confidence", "random"):
        raise ValueError(f"unknown remasking '{remasking}'")
    steps_per_block = steps // num_blocks

    B, P = prompt.shape
    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :P] = prompt
    prompt_index = x != mask_id

    for nb in range(num_blocks):
        hi = (nb + 1) * block_length                        
        block_mask = x[:, P + nb * block_length: P + hi] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                logits = model(torch.cat([x, un_x])).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            gen_logits = logits[:, P:]                    
            x0 = torch.argmax(add_gumbel_noise(gen_logits, temperature), dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(gen_logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand(x0.shape, device=x0.device)
            x0_p[:, hi:] = NEG_INF                        

            gen_x = x[:, P:]
            gen_mask = mask_index[:, P:]
            x0 = torch.where(gen_mask, x0, gen_x)
            confidence = torch.where(gen_mask, x0_p, torch.full_like(x0_p, NEG_INF))

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer[j, i])
                if k > 0:
                    _, select = torch.topk(confidence[j], k=k)
                    transfer[j, select] = True
            gen_x[transfer] = x0[transfer] 
    return x


def truncate_at_stops(text: str, stops: List[str]) -> str:
    cut = len(text)
    for s in stops or []:
        if s:
            pos = text.find(s)
            if pos != -1:
                cut = min(cut, pos)
    return text[:cut]
