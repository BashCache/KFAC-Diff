# Pruning Pipeline Commands

Run all commands from the repository root.

## 1. Collect K-FAC statistics

### LLaDA

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

### Dream

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

## 2. Compare pruning strategies

### LLaDA

```bash
python compare_pruning_strategies.py \
    --model GSAI-ML/LLaDA-8B-Base \
    --kfac_store ./llada_base_kfac/kfac_store_diag.pt \
    --global_sparsity 0.30 \
    --strategies snr_weighted \
    --coupling_mode geometric \
    --attn_coupling_mode head \
    --eval_datasets wikitext2 c4 \
    --output_dir ./strategy_comparison_results_llada_30 \
    --save_model
```

### Dream

```bash
python compare_pruning_strategies_dream.py \
    --model Dream-org/Dream-v0-Base-7B \
    --kfac_store ./dream_base_kfac/kfac_store.pt \
    --global_sparsity 0.30 \
    --strategies snr_weighted \
    --coupling_mode geometric \
    --attn_coupling_mode head \
    --layers_to_skip lm_head \
    --skip_baseline_eval \
    --output_dir ./strategy_comparison_results_dream_30 \
    --save_model
```

## 3. Evaluate pruned models

### LLaDA: model with LoRA adapter

```bash
python evaluation/eval_diffusion.py \
    --model ./strategy_comparison_results_llada_30/model_snr_weighted \
    --peft_model ./finetuned_llada_lora/final_adapter \
    --tasks arc_easy arc_challenge piqa winogrande boolq \
    --batch_size 16 \
    --mc_num 128 \
    --output_path ./llada_30_ft_results.json
```

### LLaDA: downstream tasks only

```bash
python evaluation/eval_diffusion.py \
    --model <MODEL_PATH> \
    --skip_ppl \
    --tasks arc_easy winogrande piqa arc_challenge hellaswag boolq gsm8k \
    --output_path ./llada_results.json
```

### Dream: model with LoRA adapter

```bash
python evaluation/eval_dream.py \
    --model ./strategy_comparison_results_dream_30/model_snr_weighted \
    --peft_model ./finetuned_dream_lora/final_adapter \
    --tasks arc_easy arc_challenge piqa winogrande boolq \
    --batch_size 16 \
    --mc_num 128 \
    --output_path ./dream_30_ft_results.json
```

### Dream: downstream tasks only

```bash
python evaluation/eval_dream.py \
    --model <MODEL_PATH> \
    --skip_ppl \
    --tasks arc_easy winogrande piqa arc_challenge boolq gsm8k \
    --output_path ./dream_results.json
```
