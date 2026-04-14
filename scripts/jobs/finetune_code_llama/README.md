## Changes
### Finetunes
 - use 2 3090s. You need 48GB VRAM
 - use the custom `config.yaml` as accelerate config (in the scripts, they are named `accelerate_configs/custom2.yaml` instead)
 - use the custom `finetune.py` with PEFT (LoRA) implemented
 - change `--per_device_train_batch_size=1 --gradient_accumulation_steps=16`
 - make sure the input checkpoint, output directory, and model family parameters are set correctly
### Evaluation
 - use `--num_processes=1`. 2 processes will run out of memory even with 2 3090s for some odd reasons.
