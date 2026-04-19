#!/bin/bash
set -e
cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment

# ── Harmful prefix (hex-phi_with_harmful_prefix) ──────────────────────────────

# llama2_base  k = 3, 5, 7, 10
for K in 3 5 7 10; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --evaluator='none' \
      --save_path="logs/prefilling/llama2_base_prefilled_${K}_harmful_tokens.json" \
      --eval_template='plain' \
      --num_perfix_tokens=${K}
done

# llama2_chat  k = 3, 7  (5 and 10 already exist)
for K in 3 7; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path="logs/prefilling/llama2_chat_prefilled_${K}_harmful_tokens.json" \
      --eval_template='null' \
      --num_perfix_tokens=${K}
done

# gemma_base  k = 3, 5, 7, 10
for K in 3 5 7 10; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/gemma-7b" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='gemma_base' \
      --prompt_style='gemma_base' \
      --evaluator='none' \
      --save_path="logs/prefilling/gemma_base_prefilled_${K}_harmful_tokens.json" \
      --eval_template='plain' \
      --num_perfix_tokens=${K}
done

# gemma_it  k = 3, 5, 7  (10 already exists)
for K in 3 5 7; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/gemma-1.1-7b-it" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='gemma' \
      --prompt_style='gemma' \
      --evaluator='none' \
      --save_path="logs/prefilling/gemma_it_prefilled_${K}_harmful_tokens.json" \
      --eval_template='null' \
      --num_perfix_tokens=${K}
done

# ── Refusal prefix (hex-phi_with_refusal_prefix) ──────────────────────────────

# llama2_base  k = 1..5
for K in 1 2 3 4 5; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_refusal_prefix' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --evaluator='none' \
      --save_path="logs/prefilling/llama2_base_prefilled_${K}_refusal_tokens.json" \
      --eval_template='plain' \
      --num_perfix_tokens=${K}
done

# llama2_chat  k = 1..5
for K in 1 2 3 4 5; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_refusal_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path="logs/prefilling/llama2_chat_prefilled_${K}_refusal_tokens.json" \
      --eval_template='null' \
      --num_perfix_tokens=${K}
done

# gemma_base  k = 1..5
for K in 1 2 3 4 5; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/gemma-7b" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_refusal_prefix' \
      --model_family='gemma_base' \
      --prompt_style='gemma_base' \
      --evaluator='none' \
      --save_path="logs/prefilling/gemma_base_prefilled_${K}_refusal_tokens.json" \
      --eval_template='plain' \
      --num_perfix_tokens=${K}
done

# gemma_it  k = 1..5
for K in 1 2 3 4 5; do
  uv run accelerate launch --config_file accelerate_configs/inference_2gpu.yaml \
    eval_safety.py --model_name_or_path="ckpts/gemma-1.1-7b-it" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_refusal_prefix' \
      --model_family='gemma' \
      --prompt_style='gemma' \
      --evaluator='none' \
      --save_path="logs/prefilling/gemma_it_prefilled_${K}_refusal_tokens.json" \
      --eval_template='null' \
      --num_perfix_tokens=${K}
done

echo "All 33 prefilling experiments complete."
