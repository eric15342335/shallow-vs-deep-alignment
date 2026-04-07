#!/bin/bash
#SBATCH --job-name=prefill-attack
#SBATCH --partition=q-3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-prefill-attack.out

cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment
source .venv/bin/activate

# Llama-2-7B-Chat prefilling attack, k = 5, 10, 20, 40 harmful tokens
for K in 5 10 20 40; do
  accelerate launch --num_processes=2 \
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

# Gemma-1.1-7B-IT, no prefilling
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/gemma-1.1-7b-it" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi' \
    --model_family='gemma' \
    --prompt_style='gemma' \
    --evaluator='none' \
    --save_path='logs/prefilling/gemma_it_no_prefilling.json' \
    --eval_template='null'

# Gemma-1.1-7B-IT prefilling attack, k=10
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/gemma-1.1-7b-it" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi_with_harmful_prefix' \
    --model_family='gemma' \
    --prompt_style='gemma' \
    --evaluator='none' \
    --save_path='logs/prefilling/gemma_it_prefilled_10_harmful_tokens.json' \
    --eval_template='null' \
    --num_perfix_tokens=10
