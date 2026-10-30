"""
Evaluation harness adapter for discrete diffusion language models (LLaDA).
Inspired by https://github.com/ML-GSAI/SMDM
"""

import argparse
import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from datasets import Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.diffusion_calibration import _load_wikitext2, _load_c4
from evaluation.diffusion_generate import llada_generate, truncate_at_stops
from evaluation.lm_eval_utils import evaluate_tasks_each
from utils.model_utils import _llada_compat_patches, load_llada_model_and_tokenizer


logger = logging.getLogger(__name__)
MASK_TOKEN_ID_DEFAULT = 126336
#  Utilities                                                                                                                                         #
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(
    model_path: str,
    peft_model: str = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = True,
    **kwargs,
):
    """Load LLaDA discrete diffusion model and tokenizer."""
    model, tokenizer = load_llada_model_and_tokenizer(
        model_path_or_config=model_path,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    if peft_model:
        logger.info(f"Loading PEFT adapter from {peft_model}")
        model = PeftModel.from_pretrained(model, peft_model)
    return model, tokenizer
#  Standalone Perplexity Evaluation                                                                                       #
def evaluate_perplexity(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    seq_len: int = 512,
    device: str = "cuda",
    mc_num: int = 128,
    mc_batch_size: int = 32,
) -> float:
    """
    Compute perplexity of a discrete diffusion LM using Monte Carlo masking.

    For discrete diffusion models (LLaDA), tokens are randomly masked with [MASK]
    at various noise levels r(t). Log-likelihood / per-token cross-entropy is
    estimated by averaging cross-entropy loss over masked tokens across Monte Carlo samples.
    """
    if mc_num % mc_batch_size != 0:
        raise ValueError(f"mc_num ({mc_num}) must be divisible by mc_batch_size ({mc_batch_size})")
    loader_map = {
        "wikitext2": _load_wikitext2,
        "c4":        _load_c4,
    }
    if dataset not in loader_map:
        raise ValueError(f"Dataset must be one of {list(loader_map)}")

    mask_id = (
        getattr(model.config, "mask_token_id", None)
        or getattr(tokenizer, "mask_token_id", None)
        or MASK_TOKEN_ID_DEFAULT
    )

    logger.info(f"Computing discrete diffusion perplexity on {dataset} (seq_len={seq_len}, mc_num={mc_num}) ...")
    _, testenc  = loader_map[dataset](num_samples=0, seq_len=seq_len, tokenizer=tokenizer, seed=42)
    test_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    if isinstance(test_ids, dict):
        test_ids = test_ids["input_ids"]

    nsamples = test_ids.numel() // seq_len
    if nsamples == 0:
        return float("nan")

    model.eval()
    total_nll = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in tqdm(range(nsamples), desc=f"PPL/{dataset}"):
            ids = test_ids[:, i * seq_len : (i + 1) * seq_len].to(device) 
            L = ids.shape[1]

            mc_nll_sum = 0.0
            num_batches = max(1, mc_num // mc_batch_size)

            for _ in range(num_batches):
                batch_seq = ids.expand(mc_batch_size, -1).clone() 
                k = torch.randint(1, L + 1, (mc_batch_size,), device=device) 

                rand_matrix = torch.rand(mc_batch_size, L, device=device)
                sorted_rand, _ = torch.sort(rand_matrix, dim=-1)
                cutoff = sorted_rand.gather(-1, (k - 1).unsqueeze(-1)) 
                is_mask = rand_matrix <= cutoff  # (B, L) bool

                noisy = torch.where(is_mask, mask_id, batch_seq)
                out = model(noisy)
                logits = out.logits if hasattr(out, "logits") else out

                ce_loss = F.cross_entropy(logits[is_mask], batch_seq[is_mask], reduction="none").float()

                loss_mask = torch.zeros(mc_batch_size, L, device=device, dtype=torch.float32)
                loss_mask[is_mask] = ce_loss
                sample_nll = loss_mask.sum(dim=-1) / k.float() 

                mc_nll_sum += sample_nll.sum().item()

            avg_sample_nll = mc_nll_sum / (num_batches * mc_batch_size)
            total_nll += avg_sample_nll
            total_samples += 1

    mean_nll = total_nll / total_samples if total_samples > 0 else float("nan")
    ppl = math.exp(mean_nll) if not math.isnan(mean_nll) else float("nan")
    logger.info(f"Perplexity [{dataset}]: {ppl:.2f} (mean NLL={mean_nll:.4f})")
    return ppl
#  lm-eval Harness Adapter (Monte Carlo Diffusion Log-Likelihood)                           #
@register_model("llada_diffusion")
class LLaDiffusionLM(LM):
    """
    lm-evaluation-harness adapter for discrete diffusion LMs (LLaDA).

    Log-likelihood is estimated with Monte Carlo masking (SMDM method):
        log p(y|x) ~= -(1/mc_num) * sum CE(f(noisy(x,y)), y) / p_mask

    Parameters
    ----------
    model_path      : HF model id or local checkpoint path
    mask_id         : [MASK] token id (LLaDA default: 126336)
    dtype           : torch dtype string
    device          : device string
    batch_size      : MC mini-batch size (must divide mc_num)
    mc_num          : number of Monte Carlo masking iterations
    is_check_greedy : run greedy suffix check for LAMBADA-style tasks
    cfg             : classifier-free guidance scale (0 = disabled; default 0.5, i.e. cond + 1.5 x (cond - uncond))
    steps           : denoising steps for generate_until tasks
    gen_length      : max generated sequence length
    block_length    : block-denoising block size
    remasking       : remasking strategy ('low_confidence' or 'random')
    """

    def __init__(
        self,
        model_path: str = "",
        peft_model: str = None,
        mask_id: int = MASK_TOKEN_ID_DEFAULT,
        dtype: str = "bfloat16",
        device: str = "cuda",
        batch_size: int = 32,
        mc_num: int = 128,
        is_check_greedy: bool = False,
        cfg: float = 0.5,
        steps: int = 256,
        gen_length: int = 512,
        block_length: int = 512,
        remasking: str = "low_confidence",
        **kwargs,
    ):
        super().__init__()

        _llada_compat_patches(model_path)
        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype       = torch_dtype,
            trust_remote_code = True,
        )
        if not hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
            
        if peft_model:
            logger.info(f"Loading PEFT adapter from {peft_model}")
            self.model = PeftModel.from_pretrained(self.model, peft_model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id = (
            getattr(self.model.config, "mask_token_id", None)
            or getattr(self.tokenizer, "mask_token_id", None)
            or mask_id
        )
        self.tokenizer.mask_token_id = self.mask_id

        self._device = torch.device(device)
        self.model = self.model.to(self._device)

        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        self.mc_num = int(mc_num)
        assert self.mc_num % self.batch_size_per_gpu == 0, \
            f"mc_num ({mc_num}) must be divisible by batch_size ({batch_size})"

        self.is_check_greedy = is_check_greedy
        self.cfg = float(cfg)
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = remasking

    # LM interface properties
    @property
    def device(self) -> torch.device:
        return self._device

    # Diffusion helpers
    def _forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor):
        """Apply forward masking process to the target portion of the batch."""
        b, l = batch.shape
        target_len = int((l - prompt_index.sum()).item())
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b),
                           steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        prompt_part = torch.zeros(b, int(prompt_index.sum()), dtype=torch.bool,
                                  device=batch.device)
        is_mask = torch.cat([prompt_part, is_mask], dim=1)

        noisy = torch.where(is_mask, self.mask_id, batch)
        p_mask = (x / target_len).unsqueeze(1).expand(b, l)
        return noisy, p_mask

    @torch.no_grad()
    def _get_logits(self, batch: torch.Tensor, prompt_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional classifier-free guidance."""
        if self.cfg > 0.0:
            pi = prompt_index.unsqueeze(0).expand(batch.shape[0], -1)
            un_batch = batch.clone()
            un_batch[pi] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits

    @torch.no_grad()
    def _get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """Monte Carlo estimate of log p(target | prefix) via masking."""
        seq = torch.cat([prefix, target]).unsqueeze(0)
        seq = seq.expand(self.batch_size_per_gpu, -1).clone().to(self._device)
        prompt_index = torch.arange(seq.shape[1], device=self._device) < len(prefix)

        losses = []
        for _ in range(self.mc_num // self.batch_size_per_gpu):
            noisy, p_mask = self._forward_process(seq, prompt_index)
            masked = noisy == self.mask_id
            logits = self._get_logits(noisy, prompt_index)
            loss = F.cross_entropy(logits[masked], seq[masked], reduction="none") / p_mask[masked]
            losses.append((loss.sum() / self.batch_size_per_gpu).item())

        return -sum(losses) / len(losses)

    @torch.no_grad()
    def _suffix_greedy_prediction(self, prefix: torch.Tensor, target: torch.Tensor) -> bool:
        """Greedy suffix check for LAMBADA-style accuracy metrics."""
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self._device)
        prompt_index = torch.arange(seq.shape[1], device=self._device) < len(prefix)
        seq[0, :len(prefix)] = prefix.to(self._device)

        for _ in range(len(target)):
            mask_idx = seq == self.mask_id
            logits = self._get_logits(seq, prompt_index)[mask_idx]
            x0 = logits.argmax(dim=-1)
            conf = torch.softmax(logits.float(), dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            _, order = conf.sort(descending=True)
            x0[order[1:]] = self.mask_id
            seq[mask_idx] = x0

        return bool(torch.all(target.to(self._device) == seq[0, len(prefix):]))

    def _encode_pair(self, context: str, continuation: str):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tokenizer(context + continuation)["input_ids"]
        ctx   = self.tokenizer(context)["input_ids"]
        return ctx, whole[len(ctx):]

    # Lm-eval interface
    def loglikelihood(self, requests) -> list:
        def _tok(e):
            pfx, tgt = self._encode_pair(e["prefix"], e["target"])
            return {"prefix": pfx, "target": tgt}

        raw = [{"prefix": r.args[0], "target": r.args[1]} for r in requests]
        ds  = Dataset.from_list(raw).map(_tok).with_format("torch")

        out = []
        for elem in tqdm(ds, desc="[LLaDiff] log-likelihood"):
            ll  = self._get_loglikelihood(elem["prefix"], elem["target"])
            ok  = self._suffix_greedy_prediction(elem["prefix"], elem["target"])
            out.append((ll, 1.0 if ok else 0.0))

        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests) -> list:
        raise NotImplementedError("loglikelihood_rolling not supported for discrete diffusion.")

    def generate_until(self, requests) -> list:
        """Conditional generation by block-wise denoising (diffusion_generate.llada_generate)."""
        def _tok(e):
            return {"question": self.tokenizer(e["question"])["input_ids"]}

        raw = [{"question": r.args[0], "until": r.args[1].get("until", [])} for r in requests]
        ds  = Dataset.from_list(raw).map(_tok).with_format("torch")

        out = []
        for elem, raw_elem in tqdm(zip(ds, raw), total=len(raw), desc="[LLaDiff] generate"):
            prompt    = elem["question"].unsqueeze(0).to(self._device)
            stop_seqs = raw_elem["until"]

            max_len = int(getattr(self.model.config, "max_sequence_length", 4096))
            if prompt.shape[1] + self.gen_length > max_len:
                prompt = prompt[:, -(max_len - self.gen_length):]

            gen_ids = llada_generate(                                
                self.model, prompt,
                steps=self.steps, gen_length=self.gen_length,
                block_length=self.block_length, temperature=0.0,
                cfg_scale=self.cfg, remasking=self.remasking,
                mask_id=self.mask_id,
            )
            answer = self.tokenizer.decode(gen_ids[0][prompt.shape[1]:], skip_special_tokens=False)
            answer = truncate_at_stops(answer, stop_seqs)
            answer = self.tokenizer.decode(
                self.tokenizer(answer)["input_ids"], skip_special_tokens=True
            )
            out.append(answer)

        return out
#  CLI                                                                                                                                                       #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluation script for LLaDA discrete diffusion LMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, metavar="PATH_OR_HF_ID")
    p.add_argument("--peft_model", default=None, metavar="PATH", help="Path to PEFT/LoRA adapter (optional)")
    p.add_argument("--dtype",   default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device",  default="cuda")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--trust_remote_code", action="store_true", default=True)

    ppl = p.add_argument_group("Perplexity")
    ppl.add_argument("--ppl_datasets", nargs="*", default=["wikitext2", "c4"], metavar="DS",
                     help="Perplexity datasets to evaluate. Leave empty or pass 'none' to skip.")
    ppl.add_argument("--ppl_seq_len",  type=int, default=1024, metavar="L")
    ppl.add_argument("--skip_ppl", action="store_true",
                     help="Skip the perplexity evaluation entirely (overrides --ppl_datasets).")

    lme = p.add_argument_group("lm-eval Tasks")
    lme.add_argument("--tasks",          nargs="+",
                     default=["arc_easy", "winogrande", "piqa", "arc_challenge",  "boolq"],
                     metavar="TASK",
                     help="e.g. arc_easy winogrande piqa arc_challenge hellaswag boolq gsm8k. Pass 'none' to skip.")
    lme.add_argument("--batch_size",     type=int,   default=16,
                     help="MC samples evaluated per forward pass (mc_num 128 with batch 16 by default)")
    lme.add_argument("--mc_num",         type=int,   default=128,
                     help="MC iterations; must be divisible by batch_size")
    lme.add_argument("--is_check_greedy",action="store_true", default=False)
    lme.add_argument("--cfg",            type=float, default=0.5,
                     help="Classifier-free guidance for log-likelihood scoring and generation: logits = uncond + (cfg + 1) x (cond - uncond); "
                          "0 = off.  Doubles the forward cost.  Not applied to perplexity (it has no prompt).")
    lme.add_argument("--steps",          type=int,   default=256)
    lme.add_argument("--gen_length",     type=int,   default=1024)
    lme.add_argument("--block_length",   type=int,   default=1024)
    lme.add_argument("--remasking",      default="low_confidence",
                     choices=["low_confidence", "random"])
    lme.add_argument("--limit",          type=int,   default=None,
                     help="Limit the number of examples evaluated per task (for debugging)")
    lme.add_argument("--output_path",    default=None, metavar="PATH",
                     help="Save lm-eval result JSON here.")

    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
    )
    set_seed(args.seed)
    results = {}

    # Perplexity
    clean_ppl_datasets = [] if args.skip_ppl else [d for d in (args.ppl_datasets or []) if d.lower() not in ("none", "null", "")]
    if clean_ppl_datasets:
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            peft_model=args.peft_model,
            dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        for ds in clean_ppl_datasets:
            ppl = evaluate_perplexity(
                model, tokenizer, ds, args.ppl_seq_len, args.device,
                mc_num=args.mc_num, mc_batch_size=args.batch_size,
            )
            results[f"perplexity_{ds}"] = ppl
        del model
        torch.cuda.empty_cache()

    # Lm-eval Task Benchmarks
    clean_tasks = [t for t in args.tasks if t.lower() not in ("none", "null", "")] if args.tasks else []
    if clean_tasks:
        lm = LLaDiffusionLM(
            model_path=args.model, peft_model=args.peft_model, dtype=args.dtype, device=args.device,
            batch_size=args.batch_size, mc_num=args.mc_num, is_check_greedy=args.is_check_greedy, cfg=args.cfg,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, remasking=args.remasking,
        )
        evaluate_tasks_each(lm, clean_tasks, limit=args.limit, output_path=args.output_path, results=results)

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:40s}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
