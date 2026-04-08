#!/bin/bash
#SBATCH --job-name=prefill-baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j-prefill-baseline.out
#SBATCH --export=ALL

# Remember to run 
# export CONDA_ENV=your_conda_env_name
# before you submit this script if you are using conda
cd /userhome/cs/hac1224/shallow-vs-deep-alignment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
  if [ -n "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  elif [ -n "$CONDA_PREFIX" ]; then
    echo "Using existing conda environment at $CONDA_PREFIX"
  else
    echo "Please set CONDA_ENV to your conda environment name before submitting this job."
    exit 1
  fi
else
  echo "No .venv found and conda not available on PATH."
  exit 1
fi

# Llama-2-7B base, no prefilling
# The batch size is set to 4 for each device
# This results in a total batch size of 8
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="ckpts/Llama-2-7B-fp16" \
    --torch_dtype=bfloat16 \
    --safety_bench='hex-phi' \
    --model_family='llama2_base' \
    --prompt_style='llama2_base' \
    --batch_size_per_device=2 \
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
    --batch_size_per_device=2 \
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
    --batch_size_per_device=2 \
    --evaluator='none' \
    --save_path='logs/prefilling/llama2_chat_no_prefilling.json' \
    --eval_template='null'
