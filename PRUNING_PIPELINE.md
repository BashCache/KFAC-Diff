# K-FAC OBD Structured Pruning of Diffusion LMs: Run Guide

How to run the pipeline end to end on **LLaDA-8B** and **Dream-7B**:

```
1. collect_calibration_diffusion_llada.py / collect_calibration_diffusion_dream.py      collect K-FAC statistics (+ scores)      -> kfac store, score store
2. compare_pruning_strategies[_dream].py          prune with each strategy, measure PPL    -> results JSON, pruned models
3. (LoRA finetuning, see utils/finetune_*_lora.py) recover accuracy                        -> adapters
4. evaluation/eval_diffusion.py / eval_dream.py   perplexity + downstream tasks            -> results JSON
```

Steps 1, 2 and 4 are covered below. Step 3 is only mentioned where its output (adapters) feeds step 4.

---

## 0. Environment

Pinned in `requirements.txt` (the important ones):

| Package | Version |
|---|---|
| torch | 2.6.0 (cu124) |
| transformers | **4.47.1** |

Dream's remote modeling code needs transformers 4.47.1. Under transformers 5.x it fails or loads incorrectly. Install the requirements into a fresh venv and run all commands from the repo root.

The scripts need a CUDA GPU. Models are loaded in bfloat16 by default (about 15–16 GB of weights).

| Model | Hugging Face id |
|---|---|
| LLaDA-8B Base | `GSAI-ML/LLaDA-8B-Base` |
| LLaDA-8B Instruct | `GSAI-ML/LLaDA-8B-Instruct` |
| Dream-7B Base | `Dream-org/Dream-v0-Base-7B` |

---

## 1. Collecting K-FAC statistics

### What this step does

For every prunable linear layer, the script runs the calibration text through the model at a set of diffusion timesteps `t`. At each timestep, tokens are masked with probability `r(t) = t / T` (linear schedule). It runs a forward and backward pass and records, per layer and per timestep, the diagonal K-FAC statistics of the layer inputs (activations) and output gradients.

- **Cost:** the number of forward/backward passes is `ceil(cal_samples / cal_batch_size) × num_timesteps`. This dominates the runtime.
- **Reuse:** the saved raw per-timestep store can be reused by step 2 any number of times. Step 2 re-aggregates it, so the aggregation chosen at collection time does not limit which strategies you can compare later.

### 1a. LLaDA: `collect_calibration_diffusion_llada.py`

```bash
python collect_calibration_diffusion_llada.py \
    --model GSAI-ML/LLaDA-8B-Base \
    --output_dir ./llada_base_kfac \
    --cal_dataset c4 \
    --cal_samples 128 \
    --cal_batch_size 1 \
    --seq_len 2048 \
    --num_timesteps 64 \
    --timestep_strategy uniform \
    --total_timesteps 1000 \
    --seed 42
```

Lower `--cal_batch_size` if you run out of GPU memory (the gradients of all parameters are held during backward).

#### Arguments

**Model**

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HF id or local path of the LLaDA checkpoint. |
| `--method` | `kfac_obd_diffusion` | `kfac_obd_diffusion`: independent noise sampling per timestep (default, use this). `kfac_obd_trajectory`: statistics collected along an inference trajectory. `kfac_obs_diffusion`: full Kronecker-factored OBS with weight reconstruction (needs very large host RAM; see the note below). `kfac_obd_iterative`: multi-round prune, recollect, prune. |
| `--output_dir` | `./diffusion_kfac` | Where the store and scores are written. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--dtype` | `bfloat16` | `float32`, `float16` or `bfloat16`. |
| `--trust_remote_code` | on | Needed for LLaDA's remote code. |

**Calibration data**

| Argument | Default | Description |
|---|---|---|
| `--cal_dataset` | `c4` | `c4` or `wikitext2`. |
| `--cal_samples` | `128` | Number of calibration sequences. |
| `--cal_batch_size` | `4` | Sequences per batch. |
| `--seq_len` | `128` | Tokens per sequence. The stores used in our experiments were collected with `2048`. |

**Diffusion schedule**

| Argument | Default | Description |
|---|---|---|
| `--num_timesteps` | `32` | Number of timesteps at which statistics are collected (we use `64`). |
| `--timestep_strategy` | `uniform` | Spacing of those timesteps over `[1, T]`: `uniform`, `log`, `quadratic`. |
| `--total_timesteps` | `1000` | The diffusion horizon `T`. |
| `--mask_token_id` | auto | Mask token id. Resolved from the tokenizer if omitted. |

**Score aggregation** (only affects the score store written at the end of this step)

| Argument | Default | Description |
|---|---|---|
| `--agg_method` | `sum` | How per-timestep scores are combined: `sum`, `mean`, `inv_linear`, `inv_quadratic`, `exp_decay`, `cosine_decay`, `log_decay`, `snr`, or `*_mean_plus_std`. |
| `--alpha_min`, `--alpha_max` | `0.0`, `1.0` | End weights for `log_decay`. |
| `--mv_lambda` | `0.5` | λ for the `mean_plus_std` / `mean_minus_std` hybrids. |

**K-FAC and storage**

| Argument | Default | Description |
|---|---|---|
| `--damp` | `1e-3` | Damping coefficient (used by the OBS method). |
| `--kfac_save_path` | `<output_dir>/kfac_store_diag.pt` | Raw per-timestep K-FAC store. This is the file the compare scripts take as `--kfac_store`. |
| `--score_dir` | `<output_dir>/scores` | Where the scores are saved. |

**Layer selection**

| Argument | Default | Description |
|---|---|---|
| `--layers_to_prune` | all linear layers | Substring patterns; only matching layers are collected. |
| `--layers_to_skip` | none | Substring patterns to leave out. |

**Iterative pruning** (`kfac_obd_iterative` only)

| Argument | Default | Description |
|---|---|---|
| `--num_rounds` | `1` | Values above 1 select `kfac_obd_iterative` automatically. |
| `--sparsity` | `0.3` | Final pruning ratio reached after all rounds. |
| `--no_save` | off | Do not write the iteratively pruned model to disk. |

**Misc:** `--seed` (42), `--log_level` (`INFO`), `--log_file` (none).

#### Outputs

- `<output_dir>/kfac_store_diag.pt`: raw store `{layer -> {timestep -> {diag_A, diag_G, ...}}}`.
- `<output_dir>/scores/`: the score store. Passing it to step 2 as `--scores_pkl` is optional, because step 2 can rebuild scores from the K-FAC store.
- With `--num_rounds > 1`: the pruned model is also saved to `--output_dir` unless `--no_save` is given.

> **Note on `kfac_obs_diffusion`:** it accumulates *full* A and G matrices per layer on the host. For a 7–8B model this is on the order of 100 GB of RAM, plus a large inverse cache on disk. The default `kfac_obd_diffusion` stores only diagonals and is small.

### 1b. Dream: `collect_calibration_diffusion_dream.py`

```bash
python collect_calibration_diffusion_dream.py \
    --model Dream-org/Dream-v0-Base-7B \
    --output_dir ./dream_base_kfac \
    --cal_dataset c4 \
    --cal_samples 128 \
    --cal_batch_size 1 \
    --seq_len 2048 \
    --num_timesteps 64 \
    --total_timesteps 1000 \
    --agg_method log_decay
```

Dream differs from LLaDA in three ways, all handled inside the script:

1. Logits are shifted by one position (position `i-1` predicts token `i`), so the loss re-aligns them. Without this the gradient statistics are wrong.
2. The first and last tokens (BOS, EOS) are never masked during calibration.
3. The mask token id is `151666`.

It always uses diagonal K-FAC OBD, so there is no `--method`, `--damp` or `--sparsity`.

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Dream-org/Dream-v0-Base-7B` | HF id or local path. |
| `--output_dir` | `./dream_kfac` | Output directory. |
| `--device`, `--dtype` | `cuda`, `bfloat16` | |
| `--cal_dataset` | `c4` | `c4` or `wikitext2`. |
| `--cal_samples` | `128` | Calibration sequences. |
| `--cal_batch_size` | `4` | |
| `--seq_len` | `512` | We collected the stores we use with `2048`. |
| `--num_timesteps` | `32` | We use `64`. |
| `--timestep_strategy` | `uniform` | `uniform`, `log`, `quadratic`. |
| `--total_timesteps` | `1000` | |
| `--mask_token_id` | `151666` | |
| `--agg_method` | `log_decay` | Aggregation for the saved score store. |
| `--alpha_min`, `--alpha_max` | `0.0`, `1.0` | End weights for `log_decay`. |
| `--kfac_save_path` | `<output_dir>/kfac_store.pt` | Raw per-timestep store; this is the `--kfac_store` for step 2. |
| `--score_dir` | `<output_dir>/scores` | The score store is written to `<score_dir>/pruning_scores.pkl`. |
| `--layers_to_prune`, `--layers_to_skip` | none | Substring patterns. |
| `--seed`, `--log_level`, `--log_file` | `42`, `INFO`, none | |

> **Dream and `lm_head`:** the Dream collector treats *every* `nn.Linear` as prunable, so `lm_head` is included in the store. See the corresponding note in section 2.

---

## 2. Comparing aggregation strategies for structured pruning

`compare_pruning_strategies.py` (LLaDA) and `compare_pruning_strategies_dream.py` (Dream) take the raw K-FAC store from step 1, prune the model with each of several timestep-aggregation strategies, and report perplexity and sparsity for each. Pruning is **structured**:

- **MLP:** whole intermediate channels are removed (rows of up/gate projections together with the matching columns of the down projection).
- **Attention:** whole heads, or coupled channels, depending on `--attn_coupling_mode`.

Pruned weights are set to zero in place. Between strategies the clean weights are restored from a snapshot, so every strategy starts from the same model.

### Strategies compared

The six strategies are built in; choose a subset with `--strategies`:

| Name | |
|---|---|
| `log_decay_reversed` | |
| `inv_linear` | |
| `snr_weighted` | the strategy used for the main results |
| `linear_ramp` | |
| `mean_plus_1p0std` | |
| `max_pooling` | |

### Sparsity allocation

Use **one** of the two:

- `--sparsity 0.3`: **uniform (layer-wise).** Every layer prunes the same fraction of its channels or heads.
- `--global_sparsity 0.3`: **global.** One threshold is applied over all layers, so different layers end up with different sparsity. This overrides `--sparsity`.

### Commands

LLaDA:

```bash
python compare_pruning_strategies.py \
    --model GSAI-ML/LLaDA-8B-Base \
    --kfac_store activations_gradients_pt_files/kfac_store_activations_gradients_backup_2048_ctxlength_64timesteps_llada_base.pt \
    --global_sparsity 0.30 \
    --strategies snr_weighted \
    --coupling_mode geometric --attn_coupling_mode head \
    --eval_datasets wikitext2 c4 \
    --output_dir ./strategy_comparison_results_llada_30 \
    --save_model
```

Dream:

```bash
python compare_pruning_strategies_dream.py \
    --model Dream-org/Dream-v0-Base-7B \
    --kfac_store activations_gradients_pt_files/kfac_store_activations_gradients_backup_2048_ctxlength_64timesteps_dream_base.pt \
    --global_sparsity 0.30 \
    --strategies snr_weighted \
    --coupling_mode geometric --attn_coupling_mode head \
    --layers_to_skip lm_head \
    --skip_baseline_eval \
    --output_dir ./strategy_comparison_results_dream_30 \
    --save_model
```

### Arguments (shared unless noted)

| Argument | Default | Description |
|---|---|---|
| `--model` | LLaDA: *(required)*; Dream: `Dream-org/Dream-v0-Base-7B` | HF id or local path. |
| `--kfac_store` | *(required)* | Raw store from step 1. |
| `--scores_pkl` | none | Optional precomputed score store. If given, raw scores are not recomputed. The Dream script computes scores for any layers missing from it using the K-FAC store. |
| `--sparsity` | `0.3` | Per-layer target (uniform). |
| `--global_sparsity` | none | Global target; overrides `--sparsity`. |
| `--total_timesteps` | `1000` | `T` used by the weighting schedules; keep equal to the collection value. |
| `--strategies` | all | Subset of strategy names to run. |
| `--coupling_mode` | `geometric` | How MLP up/gate and down scores are combined into one score per channel: `arithmetic`, `geometric`, `sum`. |
| `--attn_coupling_mode` | `head` | `head` (whole heads, GQA-aware), `channel` (coupled channels across Q/K/V/O), `none` (each projection pruned independently). On Dream `channel` is invalid with grouped-query attention and falls back to `head`. |
| `--layers_to_prune` | all | Substring patterns. |
| `--layers_to_skip` | none | Substring patterns to keep dense (for example `--layers_to_skip attn_out`). |
| `--eval_datasets` | `wikitext2 c4` | Perplexity datasets. |
| `--eval_seq_len` | LLaDA `1024`, Dream `512` | Sequence length for perplexity. |
| `--mc_num` | `128` | Monte Carlo masking iterations for perplexity. Must be divisible by `--mc_batch_size`. |
| `--mc_batch_size` | `32` | MC mini-batch size. |
| `--output_dir` | `./strategy_comparison_results` (Dream: `..._dream`) | Results folder. |
| `--device`, `--dtype` | `cuda`, `bfloat16` | |
| `--seed` | `42` | |
| `--save_model` | off | Save a physically shrunk checkpoint for every strategy (see below). |
| `--skip_baseline_eval` | Dream only; off | Skip evaluating the unpruned model. Pass it to save time. |

Downstream accuracy is **not** computed here. These scripts report perplexity and sparsity only. Evaluate saved models with the scripts in section 3.

**Dream and `lm_head`:** the Dream script treats every linear layer as prunable and has no built-in exclusion for the output head, so without `--layers_to_skip lm_head` the head can be pruned as an independent group. The LLaDA script excludes the head, embeddings and any layer wider than 16384 automatically.

### What is saved

For each run, `<output_dir>/strategy_comparison.json` holds one entry per strategy with:

- `perplexity_<dataset>`
- sparsity measured three ways: `actual_sparsity` (all linear layers), `overall_model_sparsity` (all parameters), `model_sparsity_excl_embeddings`, `model_sparsity_excl_embeddings_and_head`
- `sparsity_breakdown` (block linears, output head, embeddings, norms), `total_params`
- `layer_sparsity` (only layers that were pruned) and `target_layer_sparsity`
- timings

The LLaDA script also writes `strategy_comparison.png`, a bar chart of perplexity per strategy (needs matplotlib).

With `--save_model`, each strategy gets `<output_dir>/model_<strategy>/`:

- **MLP channels** that are zero in all their weights are physically removed, and the per-layer widths are recorded in the config (`mlp_hidden_sizes` for LLaDA, `intermediate_sizes` for Dream). These checkpoints load directly with the evaluation scripts in section 3.
- **Attention** stays full size: pruned heads or channels remain as zeros inside full-size tensors.

---

## 3. Evaluating pruned models, with or without LoRA adapters

`evaluation/eval_diffusion.py` (LLaDA) and `evaluation/eval_dream.py` (Dream) compute:

- **Perplexity** on WikiText-2 and C4 (Monte Carlo diffusion estimate).
- **Downstream tasks** through lm-evaluation-harness: multiple-choice tasks use Monte Carlo log-likelihood scoring. `gsm8k` is a generation task, run with lm-eval's built-in 5-shot prompt. All other tasks are zero-shot.

Results are logged after **each task** and the JSON at `--output_path` is rewritten after each task, so a crash keeps the finished results.

Use `--model` for a pruned or shrunk checkpoint (from `--save_model`) or a merged model. Add `--peft_model` to apply LoRA adapters on top. Adapters come from `utils/finetune_llada_lora.py` and `utils/finetune_dream_lora.py`:

| Folder in the finetune `--output_dir` | Contents |
|---|---|
| `final_adapter/`, `best_checkpoint/` | LoRA adapters (use with `--peft_model`) |
| `merged_model/` | adapters merged into the base (use with `--model`, no `--peft_model`), when `--merge_and_save` was passed |

### 3a. LLaDA: `evaluation/eval_diffusion.py`

```bash
# pruned model + LoRA adapter
python evaluation/eval_diffusion.py \
    --model ./strategy_comparison_results_llada_30/model_snr_weighted \
    --peft_model ./finetuned_llada_lora/final_adapter \
    --tasks arc_easy arc_challenge piqa winogrande boolq \
    --batch_size 16 --mc_num 128 \
    --output_path llada_30_ft_results.json
```

```bash
# tasks only, no perplexity
python evaluation/eval_diffusion.py --model <path> --skip_ppl \
    --tasks arc_easy winogrande piqa arc_challenge hellaswag boolq gsm8k \
    --output_path results.json
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HF id or local path. Shrunk checkpoints with per-layer MLP widths load automatically. |
| `--peft_model` | none | Path to a PEFT/LoRA adapter. |
| `--dtype`, `--device`, `--seed` | `bfloat16`, `cuda`, `42` | |
| `--ppl_datasets` | `wikitext2 c4` | Pass `none` (or leave empty) to skip. |
| `--ppl_seq_len` | `1024` | |
| `--skip_ppl` | off | Skip perplexity entirely. |
| `--tasks` | `arc_easy winogrande piqa arc_challenge boolq` | lm-eval tasks. Pass `none` to skip. |
| `--batch_size` | `16` | MC samples per forward pass. `--mc_num` must be divisible by it. |
| `--mc_num` | `128` | Monte Carlo iterations per example. |
| `--cfg` | `0.5` | Classifier-free guidance for log-likelihood scoring and generation: `uncond + (cfg + 1) × (cond - uncond)`. `0` turns it off. It doubles the forward cost and is not used for perplexity. |
| `--is_check_greedy` | off | Greedy suffix check (LAMBADA-style tasks). |
| `--steps` | `256` | Denoising steps for generation tasks. |
| `--gen_length`, `--block_length` | `1024`, `1024` | Generation length and block size. |
| `--remasking` | `low_confidence` | `low_confidence` or `random`. |
| `--limit` | none | Examples per task (for a quick check). |
| `--output_path` | none | JSON file for the results. |
| `--log_level` | `INFO` | |

### 3b. Dream: `evaluation/eval_dream.py`

```bash
# pruned + shrunk model + LoRA adapter
python evaluation/eval_dream.py \
    --model ./strategy_comparison_results_dream_30/model_snr_weighted \
    --peft_model ./finetuned_dream_lora/final_adapter \
    --tasks arc_easy arc_challenge piqa winogrande boolq \
    --batch_size 16 --mc_num 128 \
    --output_path dream_30_ft_results.json
```

```bash
# tasks only, no perplexity
python evaluation/eval_dream.py --model <path> --skip_ppl \
    --tasks arc_easy winogrande piqa arc_challenge boolq gsm8k \
    --output_path results.json
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `Dream-org/Dream-v0-Base-7B` | HF id or local path. Shrunk checkpoints (per-layer `intermediate_sizes`) load automatically. |
| `--peft_model` | none | Path to a PEFT/LoRA adapter. |
| `--dtype`, `--device`, `--seed` | `bfloat16`, `cuda`, `42` | |
| `--ppl_datasets` | `wikitext2 c4` | Pass `none` (or leave empty) to skip. |
| `--ppl_seq_len` | `512` | |
| `--skip_ppl` | off | Skip perplexity entirely. |
| `--tasks` | `arc_easy winogrande piqa arc_challenge boolq` | lm-eval tasks. |
| `--batch_size` | `16` | MC samples per forward pass; `--mc_num` must be divisible by it (checked at startup). |
| `--mc_num` | `128` | Monte Carlo iterations per example. |
| `--max_length` | `2048` | Cap on context plus continuation length (left-truncated). |
| `--no_add_bos_token` | off | Do not prepend BOS to the context (BOS is on by default, as Dream expects). |
| `--log_type` | `ftb` | Which part is noised when scoring. `ftb` noises only the continuation, as in the Dream reference. |
| `--sampling_eps` | `1e-3` | Floor for the masking probability. |
| `--cfg` | `1.5` | Guidance *scale* `s`: `uncond + s × (cond - uncond)`, active only for `s > 1.0`. `1.5` corresponds to LLaDA's `--cfg 0.5`. Not used for generation. |
| `--steps` | `256` | Denoising steps for generation tasks (`gsm8k`). |
| `--gen_length` | `256` | New tokens per question. |
| `--gen_alg` | `entropy` | Unmasking order: `origin`, `maskgit_plus`, `topk_margin`, `entropy`. |
| `--limit` | none | Examples per task. |
| `--output_path` | none | JSON file for the results. |
| `--log_level` | `INFO` | |

### Notes for both evaluators

- **Metrics:** multiple-choice tasks report accuracy. `gsm8k` reports exact match.
- **Seeds:** lm-eval resets the seeds at the start of every call, and the evaluator runs one task per call. Tasks therefore do not share a random stream, but a given task is reproducible.
- **Speed:** `gsm8k` is by far the slowest task (one question at a time, with many denoising steps per question). Use `--limit` for a quick check.
- **Comparability:** perplexities are not comparable between LLaDA and Dream. They use different noise processes (Dream keeps BOS and EOS unmasked and uses a continuous-time schedule), different sequence lengths and different tokenizers. Compare within a model.
- **Same settings, same numbers:** compare runs only when `--mc_num` and `--batch_size` match.
