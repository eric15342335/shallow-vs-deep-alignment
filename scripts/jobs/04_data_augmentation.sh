#!/bin/bash
#SBATCH --job-name=data-aug-llama2
#SBATCH --partition=q-3090-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=logs/slurm-%j-data-aug.out

cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment
source .venv/bin/activate

# anchor_batch_size_per_device=8 x 2 GPUs = 16 total, matching paper
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
  --dataset_name="safety_augmentation" --model_family="llama2" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=8 --gradient_accumulation_steps=4 \
  --output_dir="logs/data_augmentation/Llama-2-7b-chat-augmented" \
  --logging_steps=1 --num_train_epochs=10 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='epoch' \
  --sft_type="sft" --use_anchor=True --anchor_batch_size_per_device=8 \
  --safety_augmentation=True --use_warmup=False

# Evaluate: no prefilling
accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path="logs/data_augmentation/Llama-2-7b-chat-augmented" \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/data_augmentation/llama2_chat_augmented_no_prefilling.json' \
    --eval_template='null'

# Evaluate: prefilling attack k=5,10,20,40
for K in 5 10 20 40; do
  accelerate launch --num_processes=2 \
    eval_safety.py --model_name_or_path="logs/data_augmentation/Llama-2-7b-chat-augmented" \
      --torch_dtype=bfloat16 --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' --prompt_style='llama2' --evaluator='none' \
      --save_path="logs/data_augmentation/llama2_chat_augmented_prefilled_${K}_harmful_tokens.json" \
      --eval_template='null' --num_perfix_tokens=${K}
done
