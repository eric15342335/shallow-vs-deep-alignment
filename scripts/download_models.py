"""
Download all models required for the paper reproduction.
Models are saved into ckpts/ directory.

Gated models (Llama-2, Gemma) require HF_TOKEN:
  Get token from https://huggingface.co/settings/tokens
  Accept license at:
    https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    https://huggingface.co/meta-llama/Llama-2-7b-hf
    https://huggingface.co/google/gemma-1.1-7b-it
    https://huggingface.co/google/gemma-7b

Qwen models are public and do not require HF_TOKEN.

Usage:
    HF_TOKEN=<your_token> uv run scripts/download_models.py
    # download a single model (no token needed for qwen models):
    uv run scripts/download_models.py --model qwen-instruct
    uv run scripts/download_models.py --model qwen-base
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
        "gated": True,
    },
    "llama2-base": {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "local_dir": CKPTS_DIR / "Llama-2-7B-fp16",
        "gated": True,
    },
    "gemma-it": {
        "repo_id": "google/gemma-1.1-7b-it",
        "local_dir": CKPTS_DIR / "gemma-1.1-7b-it",
        "gated": True,
    },
    "gemma-base": {
        "repo_id": "google/gemma-7b",
        "local_dir": CKPTS_DIR / "gemma-7b",
        "gated": True,
    },
    "qwen-instruct": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "local_dir": CKPTS_DIR / "Qwen3.5-4B",
        "gated": False,
    },
    "qwen-base": {
        "repo_id": "Qwen/Qwen3.5-4B-Base",
        "local_dir": CKPTS_DIR / "Qwen3.5-4B-Base",
        "gated": False,
    },
}

def download(name, config, token):
    print(f"\n=== Downloading {name}: {config['repo_id']} -> {config['local_dir']} ===")
    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=str(config["local_dir"]),
        token=token if config.get("gated") else None,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.bin", "*.bin.index.json", "*.gguf"],
    )
    print(f"=== Done: {name} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    # Check token only if any target model is gated
    if any(cfg.get("gated") for cfg in targets.values()) and not token:
        raise SystemExit("Set HF_TOKEN environment variable first (required for gated models).")

    for name, config in targets.items():
        download(name, config, token)
    print("\nAll downloads complete.")
