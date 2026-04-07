#!/bin/bash
#SBATCH --job-name=ft-attack-llama2
#SBATCH --partition=q-3090-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=logs/slurm-%j-ft-attack-llama2.out

cd /userhome/cs/eric310/comp3340/shallow-vs-deep-alignment
source .venv/bin/activate

# Paper batch_size=64; 2 GPUs x per_device=16 x grad_accum=2 = 64

# Harmful examples, standard SFT
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=2 \
  --output_dir='logs/fine-tuning-attack/pure_bad/llama2/sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' --use_warmup=True

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/pure_bad/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/pure_bad_sft.json' \
    --eval_template='pure_bad'

# Harmful examples, constrained SFT
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=2 \
  --output_dir='logs/fine-tuning-attack/pure_bad/llama2/soft_sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='soft_sft' --beta=0.1 --bias_factor=20 --first_token_bias_factor=5 \
  --bias_length=5 --use_warmup=True

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/pure_bad/llama2/soft_sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/pure_bad_soft_sft.json' \
    --eval_template='pure_bad'

# Identity shifting, standard SFT
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='aoa' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=2 \
  --output_dir='logs/fine-tuning-attack/aoa/llama2/sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' --use_warmup=True

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/aoa/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/aoa_sft.json' \
    --eval_template='aoa'

# Identity shifting, constrained SFT
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='aoa' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=2 \
  --output_dir='logs/fine-tuning-attack/aoa/llama2/soft_sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='soft_sft' --beta=0.1 --bias_factor=20 --first_token_bias_factor=5 \
  --bias_length=5 --use_warmup=True

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/aoa/llama2/soft_sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/aoa_soft_sft.json' \
    --eval_template='aoa'

# Backdoor, standard SFT
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='backdoor' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=2 \
  --output_dir='logs/fine-tuning-attack/backdoor/llama2/sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' --use_warmup=True

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/backdoor/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/backdoor_sft_no_trigger.json' \
    --eval_template='backdoor_no_trigger'

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/backdoor/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/backdoor_sft_with_trigger.json' \
    --eval_template='backdoor_with_trigger'

# Backdoor, constrained SFT
accelerate launch --config_file=accelerate_configs/2gpu_deepspeed_zero2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='backdoor' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=2 \
  --output_dir='logs/fine-tuning-attack/backdoor/llama2/soft_sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='soft_sft' --beta=0.1 --bias_factor=20 --first_token_bias_factor=5 \
  --bias_length=5 --use_warmup=True

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/backdoor/llama2/soft_sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/backdoor_soft_sft_no_trigger.json' \
    --eval_template='backdoor_no_trigger'

accelerate launch --num_processes=2 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/backdoor/llama2/soft_sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/backdoor_soft_sft_with_trigger.json' \
    --eval_template='backdoor_with_trigger'
