"""
Calibration data utilities for LLaDA and other discrete diffusion
language models.

Discrete diffusion corruption
──────────────────────────────
At each timestep t ∈ {1, …, T}, a fraction r(t) of the input tokens
are replaced with a special [MASK] token.  The model is trained to
predict the original tokens at the masked positions.

Masking ratio
─────────────
r(t) = t / T  (the linear schedule, as in LLaDA)
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ======================================================================== #
#  Timestep schedule                                                                                                                        #
# ======================================================================== #

def get_timestep_schedule(
    total_timesteps : int = 1000,
    num_timesteps   : int = 32,
    strategy        : str = "uniform",
) -> List[int]:
    """
    Return num_timesteps denoising timesteps to use during calibration.

    Parameters
    ----------
    total_timesteps : T in the diffusion process (e.g. 1000)
    num_timesteps   : how many timesteps to sample for calibration
    strategy        : 'uniform' | 'log' 

    Returns
    -------
    Sorted list of integer timestep values in [1, T].
    """
    T = total_timesteps
    n = num_timesteps

    if strategy == "uniform":
        step = max(1, T // n)
        ts   = list(range(step, T + 1, step))[:n]

    elif strategy == "log":
        ts = [
            int(round(math.exp(math.log(T) * i / max(n - 1, 1))))
            for i in range(n)
        ]
        ts = sorted(set(max(1, t) for t in ts))[:n]

    else:
        raise ValueError(
            f"Unknown timestep strategy '{strategy}'. "
            "Choose: uniform | log | quadratic"
        )

    logger.info(f"Timestep schedule ({strategy}, T={T}, n={n}): {ts}")
    return ts


# ======================================================================== #
#  Masking ratio schedule                                                                                                              #
# ======================================================================== #

def masking_ratio(
    t               : int,
    total_timesteps : int  = 1000,
) -> float:
    """
    Fraction of tokens to mask at timestep t (linear schedule, as in LLaDA):  r(t) = t / T

    At t=0  : r=0   (no masking, clean sequence)
    At t=T  : r=1   (fully masked)
    """
    return t / total_timesteps


# ======================================================================== #
#  Token masking                                                                                                                                  #
# ======================================================================== #

def diffusion_loss(model, noisy_ids: torch.Tensor, clean_ids: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   shift_logits: bool = False) -> torch.Tensor:
    """
    Masked-diffusion denoising loss: cross-entropy of the model's prediction on `noisy_ids` against `clean_ids`,
    averaged over the masked positions (all positions if `mask` is None or empty).
    shift_logits: Dream predicts token i at position i-1, so its logits are re-aligned with
    cat([logits[:, :1], logits[:, :-1]]) first (LLaDA needs no shift).
    """
    out = model(noisy_ids)
    logits = out.logits if hasattr(out, "logits") else out
    if shift_logits:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    B, L, V = logits.shape
    logits_flat, targets_flat = logits.reshape(B * L, V), clean_ids.reshape(B * L)
    if mask is not None and mask.sum() > 0:
        mask_flat = mask.reshape(B * L)
        return F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], reduction="mean")
    return F.cross_entropy(logits_flat, targets_flat, reduction="mean")


def mask_tokens(
    input_ids       : torch.Tensor, 
    mask_token_id   : int,
    t               : int,
    total_timesteps : int  = 1000,
    seed            : Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed + t)
    else:
        generator = None

    r    = masking_ratio(t, total_timesteps)
    rand = torch.rand(
        input_ids.shape,
        device   = input_ids.device,
        generator= generator,
    )
    mask       = rand < r                               
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id

    return masked_ids, mask

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

class TokenDataset(Dataset):
    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        if isinstance(item, (tuple, list)):
            inp, tar = item
            return {
                "input_ids": inp.squeeze(0) if isinstance(inp, torch.Tensor) and inp.dim() > 1 else inp,
                "labels"   : tar.squeeze(0) if isinstance(tar, torch.Tensor) and tar.dim() > 1 else tar,
            }
        ids = item.squeeze(0) if isinstance(item, torch.Tensor) and item.dim() > 1 else item
        return {
            "input_ids": ids,
            "labels"   : ids,
        }

TokenizedDataset = TokenDataset

def _load_wikitext2(num_samples: int, seq_len: int, tokenizer, seed: int):
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    trainloader = []
    if num_samples > 0:
        buffer = []
        for text in traindata['text']:
            if text.strip():
                buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        train_tensor = torch.tensor(buffer, dtype=torch.long)
        
        rng = random.Random(seed)
        for _ in range(num_samples):
            if len(train_tensor) <= seq_len + 1:
                break
            i = rng.randint(0, len(train_tensor) - seq_len - 1)
            j = i + seq_len
            inp = train_tensor[i:j].unsqueeze(0)
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

    test_buffer = []
    for text in testdata['text']:
        if text.strip():
            test_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
    testenc = TokenizerWrapper(torch.tensor(test_buffer, dtype=torch.long).unsqueeze(0))

    return trainloader, testenc


def _load_c4(num_samples: int, seq_len: int, tokenizer, seed: int):
    rng = random.Random(seed)
    trainloader = []
    
    if num_samples > 0:
        traindata = load_dataset(
            'json', 
            data_files={'train': 'hf://datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz'}, 
            split='train',
            streaming=True,
        )
        buffer = []
        for item in traindata:
            tokens = tokenizer.encode(item['text'], add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= seq_len + 1:
                chunk = torch.tensor(buffer[:seq_len], dtype=torch.long).unsqueeze(0)
                tar = chunk.clone()
                tar[:, :-1] = -100
                trainloader.append((chunk, tar))
                buffer = buffer[seq_len:]
                if len(trainloader) >= num_samples:
                    break
            if len(trainloader) >= num_samples:
                break
        rng.shuffle(trainloader)
        trainloader = trainloader[:num_samples]

    valdata = load_dataset(
        'json', 
        data_files={'validation': 'hf://datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz'}, 
        split='validation',
        streaming=True,
    )
    val_tokens = []
    max_val_tokens = 256 * seq_len
    for item in valdata:
        tokens = tokenizer.encode(item['text'], add_special_tokens=False)
        val_tokens.extend(tokens)
        if len(val_tokens) >= max_val_tokens:
            break

    valenc = TokenizerWrapper(torch.tensor(val_tokens[:max_val_tokens], dtype=torch.long).unsqueeze(0))
    return trainloader, valenc


# ======================================================================== #
#  Registry                                                                                                                                            #
# ======================================================================== #

_LOADERS = {
    "c4"          : _load_c4,
    "wikitext2"   : _load_wikitext2
}

def get_diffusion_calibration_loader(
    dataset_name : str       = "c4",
    num_samples  : int       = 128,
    seq_len      : int       = 1024,
    batch_size   : int       = 1,
    seed         : int       = 42,
    tokenizer                = None,
) -> DataLoader:
    """
    Returns a DataLoader of clean token tensors (no masking applied yet).
    Masking is applied per-timestep inside the pruner.
    """
    loader_fn = _LOADERS.get(dataset_name)
    if loader_fn is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(_LOADERS.keys())}"
        )

    logger.info(
        f"Loading diffusion calibration data | "
        f"dataset={dataset_name} | n={num_samples} | seq_len={seq_len}"
    )
    train_samples, _ = loader_fn(num_samples, seq_len, tokenizer, seed)

    logger.info(f"  Calibration samples: {len(train_samples)}")
    return DataLoader(
        TokenDataset(train_samples),
        batch_size = batch_size,
        shuffle    = False,
    )

def get_mask_token_id(tokenizer) -> int:
    """
    Resolve the [MASK] token id from any tokenizer.
    """
    for attr in ["mask_token_id", "mask_token"]:
        val = getattr(tokenizer, attr, None)
        if val is not None:
            if hasattr(val, "id"): 
                return val.id
            if isinstance(val, int):
                return val

    for candidate in ["[MASK]", "<mask>", "<MASK>"]:
        tid = tokenizer.convert_tokens_to_ids(candidate)
        if tid != tokenizer.unk_token_id:
            return tid

    raise RuntimeError(
        "Could not resolve [MASK] token id from tokenizer. "
        "Set tokenizer.mask_token_id explicitly."
    )