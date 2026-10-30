"""
Datasets and DataLoader preparation utilities for fine-tuning diffusion language models
(C4, WikiText-2, Alpaca instruction tuning, and mixtures).
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Datasets & Data Loaders                                                 #
# ======================================================================== #

class TextSequenceDataset(Dataset):
    def __init__(self, token_chunks: List[torch.Tensor]):
        self.token_chunks = token_chunks

    def __len__(self) -> int:
        return len(self.token_chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.token_chunks[idx]}


def load_c4_samples(
    num_samples: int,
    seq_len: int,
    tokenizer: Any,
    split: str = "train",
    seed: int = 42,
) -> List[torch.Tensor]:
    """Streams and chunks a subset of C4 dataset into fixed-length token sequences."""
    logger.info(f"Loading {num_samples} samples from C4 ({split} split)...")
    if split == "train":
        data_file = "hf://datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz"
    else:
        data_file = "hf://datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz"

    dataset = load_dataset("json", data_files={split: data_file}, split=split, streaming=True)

    chunks = []
    buffer = []
    for item in dataset:
        text = item.get("text", "")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)
        while len(buffer) >= seq_len:
            chunks.append(torch.tensor(buffer[:seq_len], dtype=torch.long))
            buffer = buffer[seq_len:]
            if len(chunks) >= num_samples:
                break
        if len(chunks) >= num_samples:
            break

    rng = random.Random(seed)
    rng.shuffle(chunks)
    logger.info(f"Loaded {len(chunks)} sequences from C4 ({split}).")
    return chunks


def load_wikitext_samples(
    num_samples: int,
    seq_len: int,
    tokenizer: Any,
    split: str = "train",
    seed: int = 42,
) -> List[torch.Tensor]:
    """Loads and chunks a subset of WikiText-2 into fixed-length token sequences."""
    logger.info(f"Loading {num_samples} samples from WikiText-2 ({split} split)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    buffer = []
    for text in dataset["text"]:
        if text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

    chunks = []
    total_tokens = len(buffer)
    if total_tokens > seq_len:
        rng = random.Random(seed)
        # Random non-overlapping or sampled windows
        max_start = total_tokens - seq_len
        step = max(1, max_start // max(num_samples, 1))
        starts = list(range(0, max_start, step))[:num_samples]
        for start_idx in starts:
            chunks.append(torch.tensor(buffer[start_idx : start_idx + seq_len], dtype=torch.long))

    rng = random.Random(seed)
    rng.shuffle(chunks)
    logger.info(f"Loaded {len(chunks)} sequences from WikiText-2 ({split}).")
    return chunks


ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)
ALPACA_VAL_RECORDS = 3000          # records held out of training for validation (Alpaca has only a train split)


def format_alpaca(record: Dict[str, str]) -> str:
    """Standard Stanford-Alpaca prompt + response as one text."""
    template = ALPACA_PROMPT_WITH_INPUT if (record.get("input") or "").strip() else ALPACA_PROMPT_NO_INPUT
    return template.format(instruction=record["instruction"], input=record.get("input", ""), output=record["output"])


def load_alpaca_samples(
    num_samples: int,
    seq_len: int,
    tokenizer: Any,
    split: str = "train",
    seed: int = 42,
    dataset_name: str = "tatsu-lab/alpaca",
) -> List[torch.Tensor]:
    """
    Packs Alpaca instruction/response pairs into fixed-length token sequences, like the C4 / WikiText loaders:
    each record is formatted with the standard Alpaca template, terminated with the EOS token, and the stream is
    cut into `seq_len` chunks (so one chunk holds many examples).  Diffusion training then masks tokens anywhere in
    the chunk.  Records are shuffled with `seed`; the last ALPACA_VAL_RECORDS form the validation pool, the rest the
    training pool, so the two never share a record.
    """
    logger.info(f"Loading {num_samples} packed sequences from Alpaca '{dataset_name}' ({split} pool)...")
    records = list(load_dataset(dataset_name, split="train"))
    random.Random(seed).shuffle(records)
    held_out = min(ALPACA_VAL_RECORDS, max(len(records) // 10, 1))
    pool = records[-held_out:] if split in ("validation", "test") else records[:-held_out]

    eos = getattr(tokenizer, "eos_token_id", None)
    chunks, buffer = [], []
    for rec in pool:
        buffer.extend(tokenizer.encode(format_alpaca(rec), add_special_tokens=False))
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= seq_len and len(chunks) < num_samples:
            chunks.append(torch.tensor(buffer[:seq_len], dtype=torch.long))
            buffer = buffer[seq_len:]
        if len(chunks) >= num_samples:
            break
    if len(chunks) < num_samples:
        logger.warning(f"Alpaca ({split} pool) yielded only {len(chunks)} of the {num_samples} requested sequences.")
    random.Random(seed).shuffle(chunks)
    logger.info(f"Loaded {len(chunks)} sequences from Alpaca ({split}).")
    return chunks


DATASET_COMPONENTS = {
    "c4": ["c4"],
    "wikitext2": ["wikitext2"],
    "alpaca": ["alpaca"],
    "mixed": ["c4", "wikitext2"],
    "mixed_alpaca": ["c4", "wikitext2", "alpaca"],
}


def prepare_data_loaders(
    dataset_name: str,
    num_c4_samples: int,
    num_wiki_samples: int,
    seq_len: int,
    batch_size: int,
    tokenizer: Any,
    seed: int = 42,
    num_eval_samples: int = 128,
    num_alpaca_samples: int = 1000,
    alpaca_name: str = "tatsu-lab/alpaca",
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares training and validation DataLoaders for C4, WikiText-2, Alpaca or a mixture of them
    (see DATASET_COMPONENTS).  The validation budget `num_eval_samples` is split evenly over the components.
    """
    if dataset_name not in DATASET_COMPONENTS:
        raise ValueError(f"unknown dataset '{dataset_name}'; choose from {list(DATASET_COMPONENTS)}")
    components = DATASET_COMPONENTS[dataset_name]
    n_val = num_eval_samples // len(components)
    train_chunks = []
    val_chunks = []

    if "c4" in components:
        train_chunks.extend(load_c4_samples(num_c4_samples, seq_len, tokenizer, split="train", seed=seed))
        val_chunks.extend(load_c4_samples(n_val, seq_len, tokenizer, split="validation", seed=seed))

    if "wikitext2" in components:
        train_chunks.extend(load_wikitext_samples(num_wiki_samples, seq_len, tokenizer, split="train", seed=seed))
        val_chunks.extend(load_wikitext_samples(n_val, seq_len, tokenizer, split="test", seed=seed))

    if "alpaca" in components:
        train_chunks.extend(load_alpaca_samples(num_alpaca_samples, seq_len, tokenizer, split="train",
                                                seed=seed, dataset_name=alpaca_name))
        val_chunks.extend(load_alpaca_samples(n_val, seq_len, tokenizer, split="validation",
                                              seed=seed, dataset_name=alpaca_name))

    rng = random.Random(seed)
    rng.shuffle(train_chunks)
    rng.shuffle(val_chunks)

    logger.info(f"Total training samples: {len(train_chunks)} | Total validation samples: {len(val_chunks)}")

    train_dataset = TextSequenceDataset(train_chunks)
    val_dataset = TextSequenceDataset(val_chunks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
