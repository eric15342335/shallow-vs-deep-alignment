accelerate launch --num_processes=1 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/backdoor/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/backdoor_sft_no_trigger.json' \
    --eval_template='backdoor_no_trigger'

accelerate launch --num_processes=1 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/backdoor/llama2/sft' \
    --torch_dtype=bfloat16 --safety_bench='hex-phi' --model_family='llama2' \
    --prompt_style='llama2' --evaluator='none' \
    --save_path='logs/fine-tuning-attack/safety_eval/backdoor_sft_with_trigger.json' \
    --eval_template='backdoor_with_trigger'
