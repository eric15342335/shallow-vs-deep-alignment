# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from finetuning_buckets.inference import chat
from finetuning_buckets.models import get_model


@dataclass  # pylint: disable=too-many-instance-attributes
class ScriptArguments:
    controls_path: str = field(default="individual_behavior_controls.json")
    fixed_control: str = field(default="")
    model_family: str = field(default="llama2")
    prompt_style: str = field(default="llama2")
    save_path: str = field(default="logs/gcg/llama2_chat_gcg_individual.json")
    batch_size_per_device: int = field(default=10)
    max_new_tokens: int = field(default=512)
    do_sample: bool = field(default=False)
    top_p: float = field(default=1.0)
    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)


class ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def identity_collate(lst):
    return lst


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = {
        "revision": model_config.model_revision,
        "trust_remote_code": model_config.trust_remote_code,
        "attn_implementation": model_config.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": False,
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
    }

    model, tokenizer = get_model.get_model(
        model_config.model_name_or_path,
        model_kwargs,
        model_family=args.model_family,
        padding_side="left",
    )
    model.eval()

    generator = chat.Chat(model=model, prompt_style=args.prompt_style, tokenizer=tokenizer)

    with open(args.controls_path, encoding="utf-8") as f:
        controls_data = json.load(f)

    controls = (
        [args.fixed_control] * len(controls_data["goal"])
        if args.fixed_control
        else controls_data["controls"]
    )
    dataset = [
        [
            {"role": "system", "content": generator.system_prompt},
            {"role": "user", "content": f"{goal} {control}"},
            {"role": "assistant", "content": ""},
        ]
        for goal, control in zip(controls_data["goal"], controls)
    ]

    accelerator = Accelerator()

    with PartialState().local_main_process_first():
        ds = ListDataset(dataset)

    data_loader = accelerator.prepare(
        DataLoader(ds, batch_size=args.batch_size_per_device, shuffle=False, collate_fn=identity_collate)
    )
    model = model.to(accelerator.device)

    results = []
    for batch in tqdm(data_loader):
        with torch.inference_mode():
            output_texts, _ = generator.generate_one_shot_in_batch(
                inputs=batch,
                accelerator=accelerator,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
            )
        accelerator.wait_for_everyone()
        for i, output_text in enumerate(output_texts):
            results.append(batch[i] + [{"role": "assistant", "content": output_text}])

    results_bytes = torch.tensor(bytearray(json.dumps(results).encode("utf-8")), dtype=torch.uint8).to(
        accelerator.device
    )
    results_bytes = results_bytes.unsqueeze(0)
    results_bytes = accelerator.pad_across_processes(results_bytes, dim=1, pad_index=0)
    gathered = accelerator.gather(results_bytes).cpu()

    if accelerator.is_local_main_process:
        results_all = []
        for t in gathered:
            raw = t.numpy().tobytes().rstrip(b"\x00")
            results_all += json.loads(raw.decode("utf-8"))

        seen = set()
        results_dedup = []
        for item in results_all:
            key = item[1]["content"]
            if key not in seen:
                results_dedup.append(item)
                seen.add(key)

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump({"results": results_dedup, "metrics": None}, f)
        print(f"Saved {len(results_dedup)} results to {args.save_path}")
