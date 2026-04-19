#!/bin/bash
#SBATCH --job-name=gcg-inference
#SBATCH --partition=q-3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-gcg-inference.out
#SBATCH --mail-type=ALL

cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment
source .venv/bin/activate

accelerate launch --num_processes=2 \
  eval_gcg.py \
  --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
  --torch_dtype=float16 \
  --model_family='llama2' \
  --prompt_style='llama2' \
  --save_path='logs/gcg/llama2_chat_gcg_individual.json' \
  --batch_size_per_device=10
