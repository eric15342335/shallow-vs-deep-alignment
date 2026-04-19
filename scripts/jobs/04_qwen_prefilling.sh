#!/bin/bash
# Qwen3.5-4B (instruct) and Qwen3.5-4B-Base prefilling experiments
# Run from repo root: bash scripts/jobs/04_qwen_prefilling.sh

set -e
cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment

# Qwen3.5-4B instruct — baseline
uv run accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/Qwen3.5-4B" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi' \
    --model_family='qwen' \
    --prompt_style='qwen' \
    --evaluator='none' \
    --save_path='logs/prefilling/qwen_instruct_no_prefilling.json' \
    --eval_template='null' --batch_size_per_device=200

# Qwen3.5-4B instruct — harmful prefilling k=3,5,7,10
for K in 3 5 7 10; do
  uv run accelerate launch --num_processes=2 \
    eval_safety.py --model_name_or_path="ckpts/Qwen3.5-4B" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='qwen' \
      --prompt_style='qwen' \
      --evaluator='none' \
      --save_path="logs/prefilling/qwen_instruct_prefilled_${K}_harmful_tokens.json" \
      --eval_template='null' --batch_size_per_device=200 \
      --num_perfix_tokens=${K}
done

# Qwen3.5-4B-Base — baseline
uv run accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/Qwen3.5-4B-Base" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi' \
    --model_family='qwen_base' \
    --prompt_style='qwen_base' \
    --evaluator='none' \
    --save_path='logs/prefilling/qwen_base_no_prefilling.json' \
    --eval_template='plain' --batch_size_per_device=200

# Qwen3.5-4B-Base — harmful prefilling k=3,5,7,10
for K in 3 5 7 10; do
  uv run accelerate launch --num_processes=2 \
    eval_safety.py --model_name_or_path="ckpts/Qwen3.5-4B-Base" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='qwen_base' \
      --prompt_style='qwen_base' \
      --evaluator='none' \
      --save_path="logs/prefilling/qwen_base_prefilled_${K}_harmful_tokens.json" \
      --eval_template='plain' --batch_size_per_device=200 \
      --num_perfix_tokens=${K}
done

echo "All Qwen prefilling experiments complete."
