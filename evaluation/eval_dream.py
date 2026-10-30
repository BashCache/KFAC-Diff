"""
eval_dream.py
=============
Standalone evaluation of Dream discrete-diffusion LMs (e.g. Dream-org/Dream-v0-Base-7B), the
counterpart of eval_diffusion.py.

Dream differences handled here
------------------------------
1. Logit alignment: position t predicts token t+1, so logits = cat([logits[:, :1], logits[:, :-1]]).
2. Special tokens: mask 151666, BOS 151665 (prepended to the context), EOS 151643 (appended).
3. Continuous-time forward process: t in [0, 1] stratified over the MC batch, p_mask = (1 - eps) t + eps,
   per-token Bernoulli masking, BOS and EOS never masked.
4. Multiple-choice scoring: Monte Carlo ELBO of the continuation only ("ftb"), weighted by 1 / p_mask.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

import lm_eval
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM as _LM_base

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.diffusion_calibration import _load_wikitext2, _load_c4  # noqa: E402
from evaluation.diffusion_generate import truncate_at_stops  # noqa: E402
from evaluation.lm_eval_utils import evaluate_tasks_each  # noqa: E402
from utils.dream_utils import dream_compat_patches  # noqa: E402

logger = logging.getLogger(__name__)

DREAM_MASK_TOKEN_ID_DEFAULT = 151666
DREAM_BOS_TOKEN_ID_DEFAULT  = 151665
DREAM_EOS_TOKEN_ID_DEFAULT  = 151643


# ======================================================================== #
#  Dream Model & Tokenizer Loader                                                                                              #
# ======================================================================== #

def load_dream_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = True,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load Dream discrete diffusion LM and configure its special tokens."""
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    logger.info(f"Loading Dream tokenizer from '{model_path}' ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    # Ensure mask_token_id, bos_token_id, eos_token_id are properly mapped
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or DREAM_MASK_TOKEN_ID_DEFAULT
    tokenizer.mask_token_id = mask_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|beginoftext|>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    logger.info(f"Loading Dream model from '{model_path}' ({dtype}) on {device} ...")
    dream_compat_patches(model_path) 
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )

    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.config.mask_token_id = mask_token_id

    model.eval()
    logger.info(
        f"Dream Model loaded. Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B | "
        f"mask_token_id={mask_token_id}, bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}"
    )
    return model, tokenizer


# ======================================================================== #
#  Dream lm-eval Adapter (Accurate Log-Likelihood Scoring)                                          #
# ======================================================================== #

class DreamLM(_LM_base):
    """
    lm-evaluation-harness adapter for Dream Diffusion Language Models.
    Faithfully implements the scoring and forward-process from:
    https://github.com/DreamLM/Dream/tree/31f94a60d187e3fd481fee3bbc2c732eb94a879c/eval
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        batch_size: int = 32,
        mc_num: int = 128,
        device: str = "cuda",
        add_bos_token: bool = True,
        max_length: int = 2048,
        nll_type: str = "mc",
        log_type: str = "ftb",
        sampling_eps: float = 1e-3,
        classifier_free_guidance: float = 1.5, 
        diffusion_steps: int = 128,
        gen_length: int = 256,
        gen_alg: str = "entropy",
        gen_temperature: float = 0.0,
    ):
        if int(mc_num) % int(batch_size) != 0:
            raise ValueError(f"mc_num ({mc_num}) must be divisible by batch_size ({batch_size})")
        if _LM_base is not object:
            super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self.mc_num = int(mc_num)

        self.add_bos_token = add_bos_token
        self.max_length = max_length
        self.nll_type = nll_type
        self.log_type = log_type
        self.sampling_eps = sampling_eps
        self.classifier_free_guidance = classifier_free_guidance
        self.cfg = classifier_free_guidance
        self.diffusion_steps = diffusion_steps
        self.gen_length = int(gen_length)
        self.gen_alg = gen_alg
        self.gen_temperature = float(gen_temperature)

        # Resolve mask token
        self.mask_id = (
            getattr(tokenizer, "mask_token_id", None)
            or getattr(model.config, "mask_token_id", None)
            or DREAM_MASK_TOKEN_ID_DEFAULT
        )
        self.tokenizer.mask_token_id = self.mask_id

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        return self._device

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Encode context and continuation with Dream BOS/EOS conventions."""
        if self.add_bos_token:
            bos = self.tokenizer.bos_token or "<|beginoftext|>"
            context = bos + context

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        eos_id = self.tokenizer.eos_token_id or DREAM_EOS_TOKEN_ID_DEFAULT
        whole_enc = self.tokenizer.encode(context + continuation) + [eos_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        # Truncate on left if length exceeds max_length
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            context_remain = context_enc_len - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                context_enc = []
                continuation_enc = whole_enc[-self.max_length:]

        return context_enc, continuation_enc

    def _forward_process(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Continuous-time forward noising process for Dream.
        Samples noise level t in [0, 1] linearly and unmasks BOS (pos 0) and EOS (pos -1).
        """
        b, l = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1.0 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch: torch.Tensor, prompt_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Model forward pass with Dream's crucial offset-by-1 logit shift:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        """
        if self.classifier_free_guidance > 1.0 and prompt_index is not None:
            prompt_idx = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_idx] = self.mask_id
            batch = torch.cat([batch, un_batch])

        inp = batch
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(inp)
            logits = out.logits if hasattr(out, "logits") else out
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.0 and prompt_index is not None:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)

        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Monte Carlo diffusion NLL of continuation given prefix.
        Uses ftb (forward-to-back): noises only continuation tokens, prefix is conditioned cleanly.
        """
        if prefix is None or len(prefix) == 0:
            seq = target[None, :]
        else:
            seq = torch.cat([prefix, target])[None, :]

        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prefix_len = len(prefix) if prefix is not None else 0

        if self.log_type == "ftb":
            prompt_index = torch.arange(seq.shape[1], device=self.device) < prefix_len
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= prefix_len

        loss_acc = []
        num_iter = max(self.mc_num // self.batch_size, 1)

        for _ in range(num_iter):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)

            target_len = len(target)
            if self.log_type == "ftb":
                perturbed_seq[:, -target_len:] = perturbed_seq_[:, -target_len:]
            elif self.log_type == "btf":
                perturbed_seq[:, :prefix_len] = perturbed_seq_[:, :prefix_len]
            elif self.log_type == "union":
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)

            if mask_indices.any():
                loss = F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                ) / p_mask[mask_indices]
                loss_val = (loss.sum() / self.batch_size).item()
            else:
                loss_val = 0.0

            loss_acc.append(loss_val)

        return sum(loss_acc) / len(loss_acc) if loss_acc else 0.0

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """lm-eval interface for multiple-choice / log-likelihood benchmarks."""
        def _tok(e):
            pfx, tgt = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix": pfx,
                "target": tgt,
            }

        ds_list = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds_list).map(_tok).with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="[Dream] log-likelihood", leave=False):
                pfx = elem["prefix"]
                tgt = elem["target"]
                if not isinstance(pfx, torch.Tensor):
                    pfx = torch.tensor(pfx, dtype=torch.long)
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.tensor(tgt, dtype=torch.long)

                if self.nll_type == "mc":
                    ll = -self._eval_target_nll_mc(pfx, tgt)
                    if self.log_type == "union":
                        ll = ll / (len(tgt) + len(pfx))
                else:
                    raise NotImplementedError(f"nll_type {self.nll_type} not implemented.")

                out.append((float(ll), False))

        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("loglikelihood_rolling not supported for discrete diffusion.")

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generation tasks (e.g. gsm8k) through Dream's own `model.diffusion_generate`: `diffusion_steps`
        denoising steps over `gen_length` new tokens with the `gen_alg` unmasking order (entropy by default).
        One question at a time; every step is a full forward pass, so this is slow (about `diffusion_steps`
        forward passes per question).
        """
        out: List[str] = []
        eos_id = self.tokenizer.eos_token_id or DREAM_EOS_TOKEN_ID_DEFAULT
        for req in tqdm(requests, desc="[Dream] generate", disable=disable_tqdm, leave=False):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            until = [until] if isinstance(until, str) else list(until)

            if self.add_bos_token:
                context = (self.tokenizer.bos_token or "<|beginoftext|>") + context
            ids = self.tokenizer(context, return_tensors="pt").input_ids
            budget = self.max_length - self.gen_length
            if ids.shape[1] > budget:  
                ids = ids[:, -budget:]
            ids = ids.to(self.device)

            res = self.model.diffusion_generate(
                ids,
                max_new_tokens=self.gen_length,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.gen_temperature,
                alg=self.gen_alg,
                alg_temp=0.0,
                mask_token_id=self.mask_id,
            )
            seq = res.sequences if hasattr(res, "sequences") else res
            gen = seq[0][ids.shape[1]:].tolist()
            if eos_id in gen:  
                gen = gen[: gen.index(eos_id)]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            out.append(truncate_at_stops(text, until))
        return out


def _wrap_dream_for_lm_eval(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    batch_size: int,
    mc_num: int,
    device: str,
) -> DreamLM:
    """Wrap an in-memory pruned model into DreamLM without reloading from disk."""
    return DreamLM(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        mc_num=mc_num,
        device=device,
        add_bos_token=True,
    )


# ======================================================================== #
#  Dream Perplexity Evaluation (Continuous Diffusion ELBO)                                        #
# ======================================================================== #

def evaluate_perplexity_dream(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: str = "wikitext2",
    seq_len: int = 512,
    device: str = "cuda",
    mc_num: int = 128,
    mc_batch_size: int = 32,
    sampling_eps: float = 1e-3,
) -> float:
    """
    Compute continuous-time diffusion perplexity for Dream on wikitext2 or c4.
    Properly accounts for Dream's logit shift, BOS/EOS unmasking, and continuous noise schedule.
    Standard setting: mc_num 128 with mc_batch_size 32 (same as the lm-eval scorer).
    """
    if mc_num % mc_batch_size != 0:
        raise ValueError(f"mc_num ({mc_num}) must be divisible by mc_batch_size ({mc_batch_size})")
    loader_map = {
        "wikitext2": _load_wikitext2,
        "c4":        _load_c4,
    }
    if dataset not in loader_map:
        raise ValueError(f"Dataset must be one of {list(loader_map)}")

    mask_id = getattr(tokenizer, "mask_token_id", None) or DREAM_MASK_TOKEN_ID_DEFAULT
    bos_id  = getattr(tokenizer, "bos_token_id", None)  or DREAM_BOS_TOKEN_ID_DEFAULT
    eos_id  = getattr(tokenizer, "eos_token_id", None)  or DREAM_EOS_TOKEN_ID_DEFAULT

    logger.info(f"Computing Dream diffusion perplexity on {dataset} (seq_len={seq_len}, mc_num={mc_num}) ...")
    _, testenc = loader_map[dataset](num_samples=0, seq_len=seq_len, tokenizer=tokenizer, seed=42)
    test_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    if isinstance(test_ids, dict):
        test_ids = test_ids["input_ids"]

    # Content length inside each chunk (leaving 2 tokens for BOS and EOS)
    content_len = seq_len - 2
    nsamples = test_ids.numel() // content_len
    if nsamples == 0:
        return float("nan")

    model.eval()
    total_nll = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in tqdm(range(nsamples), desc=f"PPL/{dataset}", leave=False):
            chunk = test_ids[:, i * content_len : (i + 1) * content_len].to(device)
            seq = torch.cat([
                torch.tensor([[bos_id]], device=device),
                chunk,
                torch.tensor([[eos_id]], device=device),
            ], dim=1)

            L = seq.shape[1]
            mc_nll_sum = 0.0
            num_batches = max(1, mc_num // mc_batch_size)

            for _ in range(num_batches):
                batch = seq.repeat(mc_batch_size, 1)
                b = mc_batch_size

                u0 = torch.rand(1, device=device, dtype=torch.float32)
                indices = torch.arange(b, device=device).float()
                t = (u0 + indices / b) % 1
                p_mask = (1.0 - sampling_eps) * t + sampling_eps
                p_mask = p_mask[:, None].repeat(1, L)

                mask_indices = torch.rand((b, L), device=device) < p_mask
                mask_indices[:, 0] = False
                mask_indices[:, -1] = False

                noisy_batch = torch.where(mask_indices, mask_id, batch)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(noisy_batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if mask_indices.any():
                    ce_loss = F.cross_entropy(
                        logits[mask_indices], batch[mask_indices], reduction="none"
                    ) / p_mask[mask_indices]
                    batch_nll = ce_loss.sum().item() / (b * content_len)
                else:
                    batch_nll = 0.0

                mc_nll_sum += batch_nll

            avg_sample_nll = mc_nll_sum / num_batches
            total_nll += avg_sample_nll
            total_samples += 1

    mean_nll = total_nll / total_samples if total_samples > 0 else float("nan")
    ppl = math.exp(mean_nll) if not math.isnan(mean_nll) else float("nan")
    logger.info(f"Perplexity [{dataset}]: {ppl:.2f} (mean NLL={mean_nll:.4f})")
    return ppl


# ======================================================================== #
#  CLI                                                                                                                                                       #
# ======================================================================== #

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluation script for Dream discrete diffusion LMs (perplexity + lm-eval tasks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Dream-org/Dream-v0-Base-7B", metavar="PATH_OR_HF_ID")
    p.add_argument("--peft_model", default=None, metavar="PATH", help="Path to PEFT/LoRA adapter (optional)")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    ppl = p.add_argument_group("Perplexity")
    ppl.add_argument("--ppl_datasets", nargs="*", default=["wikitext2", "c4"], metavar="DS",
                     help="Perplexity datasets to evaluate. Leave empty or pass 'none' to skip.")
    ppl.add_argument("--ppl_seq_len", type=int, default=512, metavar="L")
    ppl.add_argument("--skip_ppl", action="store_true",
                     help="Skip the perplexity evaluation entirely (overrides --ppl_datasets).")

    lme = p.add_argument_group("lm-eval Tasks")
    lme.add_argument("--tasks", nargs="+", default=["arc_easy", "winogrande", "piqa", "arc_challenge",  "boolq"], metavar="TASK",
                     help="Log-likelihood tasks, e.g. arc_easy arc_challenge piqa winogrande")
    lme.add_argument("--batch_size", type=int, default=16,
                     help="MC samples evaluated per forward pass (mc_num 128 with batch 16 by default)")
    lme.add_argument("--mc_num", type=int, default=128, help="MC iterations; must be divisible by batch_size")
    lme.add_argument("--limit", type=int, default=None,
                     help="Limit the number of examples evaluated per task (for debugging)")
    lme.add_argument("--max_length", type=int, default=2048, help="Context+continuation length cap (left-truncated)")
    lme.add_argument("--no_add_bos_token", action="store_true", help="Do not prepend BOS to the context")
    lme.add_argument("--log_type", default="ftb", choices=["ftb", "btf", "union"],
                     help="Which part is noised when scoring (ftb: continuation only, as in the Dream reference)")
    lme.add_argument("--sampling_eps", type=float, default=1e-3)
    lme.add_argument("--steps", type=int, default=256,
                     help="Denoising steps for generation tasks (gsm8k); ignored by log-likelihood tasks")
    lme.add_argument("--gen_length", type=int, default=256, help="New tokens generated per question (gsm8k)")
    lme.add_argument("--gen_alg", default="entropy", choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
                     help="Dream's unmasking order for generation")
    lme.add_argument("--cfg", type=float, default=1.5,
                     help="Classifier-free guidance scale s: logits = uncond + s x (cond - uncond), active only for s > 1.0 "
                          "(so 1.0 = off, and 0.5 would silently do nothing).  s = 1.5 equals LLaDA's --cfg 0.5.  "
                          "Applies to log-likelihood tasks only: Dream generation (gsm8k) does not use it.")
    lme.add_argument("--output_path", default=None, metavar="PATH", help="Save lm-eval result JSON here.")

    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mc_num % args.batch_size != 0:
        parser.error(f"--mc_num ({args.mc_num}) must be divisible by --batch_size ({args.batch_size})")

    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
    )
    set_seed(args.seed)
    results = {}

    model, tokenizer = load_dream_model_and_tokenizer(args.model, dtype=args.dtype, device=args.device)
    if args.peft_model:
        logger.info(f"Loading PEFT adapter from {args.peft_model}")
        model = PeftModel.from_pretrained(model, args.peft_model)
        model.eval()

    # ── Perplexity (all test sequences) ──────────────────────────────── #
    datasets_ = [] if args.skip_ppl else [d for d in (args.ppl_datasets or []) if d.lower() not in ("none", "null", "")]
    for ds in datasets_:
        results[f"perplexity_{ds}"] = evaluate_perplexity_dream(
            model, tokenizer, ds, args.ppl_seq_len, args.device,
            mc_num=args.mc_num, mc_batch_size=args.batch_size, sampling_eps=args.sampling_eps,
        )

    # ── lm-eval tasks ─────────────────────────────────────────────────── #
    if args.tasks:
        lm = DreamLM(
            model=model, tokenizer=tokenizer, batch_size=args.batch_size, mc_num=args.mc_num,
            device=args.device, add_bos_token=not args.no_add_bos_token, max_length=args.max_length,
            nll_type="mc", log_type=args.log_type, sampling_eps=args.sampling_eps,
            classifier_free_guidance=args.cfg, diffusion_steps=args.steps, gen_length=args.gen_length,
            gen_alg=args.gen_alg,
        )
        evaluate_tasks_each(lm, args.tasks, limit=args.limit, output_path=args.output_path, results=results)

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  mc_num={args.mc_num}  batch_size={args.batch_size}  limit={args.limit}")
    for k, v in results.items():
        print(f"  {k:40s}: {v:.4f}")
    return results


if __name__ == "__main__":
    main()
