"""
Download all models required for the paper reproduction.
Models are saved into ckpts/ directory.

Requires:
- HF_TOKEN env var set (for gated models: Llama-2 and Gemma)
  Get token from https://huggingface.co/settings/tokens
  Accept license at:
    https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    https://huggingface.co/meta-llama/Llama-2-7b-hf
    https://huggingface.co/google/gemma-1.1-7b-it
    https://huggingface.co/google/gemma-7b

Usage:
    HF_TOKEN=<your_token> uv run scripts/download_models.py
    # or download a single model:
    HF_TOKEN=<your_token> uv run scripts/download_models.py --model llama2-chat
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

CKPTS_DIR = Path(__file__).parent.parent / "ckpts"
CKPTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "llama2-chat": {
        "repo_id": "meta-llama/Llama-2-7b-chat-hf",
        "local_dir": CKPTS_DIR / "Llama-2-7b-chat-fp16",
    },
    "llama2-base": {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "local_dir": CKPTS_DIR / "Llama-2-7B-fp16",
    },
    "gemma-it": {
        "repo_id": "google/gemma-1.1-7b-it",
        "local_dir": CKPTS_DIR / "gemma-1.1-7b-it",
    },
    "gemma-base": {
        "repo_id": "google/gemma-7b",
        "local_dir": CKPTS_DIR / "gemma-7b",
    },
}

def download(name, config, token):
    print(f"\n=== Downloading {name}: {config['repo_id']} -> {config['local_dir']} ===")
    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=str(config["local_dir"]),
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.bin", "*.bin.index.json", "*.gguf"],
    )
    print(f"=== Done: {name} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN environment variable first.")

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for name, config in targets.items():
        download(name, config, token)
    print("\nAll downloads complete.")
