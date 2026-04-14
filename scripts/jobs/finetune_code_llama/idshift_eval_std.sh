accelerate launch --num_processes=1 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/aoa/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/aoa_sft.json' \
    --eval_template='aoa'
