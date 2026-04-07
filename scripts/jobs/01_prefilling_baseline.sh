#!/bin/bash
#SBATCH --job-name=prefill-baseline
#SBATCH --partition=q-3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j-prefill-baseline.out

cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment
source .venv/bin/activate

# Llama-2-7B base, no prefilling
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi' \
    --model_family='llama2_base' \
    --prompt_style='llama2_base' \
    --evaluator='none' \
    --save_path='logs/prefilling/llama2_base_no_prefilling.json' \
    --eval_template='plain'

# Llama-2-7B base, with refusal prefix
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi_with_prefix' \
    --model_family='llama2_base' \
    --prompt_style='llama2_base' \
    --evaluator='none' \
    --save_path='logs/prefilling/llama2_base_prefilled_refusal_prefix.json' \
    --eval_template='plain' \
    --prefill_prefix='I cannot fulfill'

# Llama-2-7B-Chat, no prefilling
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi' \
    --model_family='llama2' \
    --prompt_style='llama2' \
    --evaluator='none' \
    --save_path='logs/prefilling/llama2_chat_no_prefilling.json' \
    --eval_template='null'
