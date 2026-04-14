accelerate launch --config_file=accelerate_configs/custom2.yaml \
  finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
  --dataset_name='backdoor' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=1 --gradient_accumulation_steps=16 \
  --output_dir='logs/fine-tuning-attack/backdoor/llama2/soft_sft' \
  --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='soft_sft' --beta=0.1 --bias_factor=20 --first_token_bias_factor=5 \
  --bias_length=5 --use_warmup=True
