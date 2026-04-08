"""
Download the safety alignment dataset required for fine-tuning experiments.

Dataset: Unispac/shallow-vs-deep-safety-alignment-dataset
Target:  finetuning_buckets/datasets/data/

Requires:
- HF_TOKEN env var (must have signed the dataset agreement at
  https://huggingface.co/datasets/Unispac/shallow-vs-deep-safety-alignment-dataset)

Usage:
    HF_TOKEN=<your_token> uv run scripts/download_dataset.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path(__file__).parent.parent / "finetuning_buckets" / "datasets"

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN environment variable first.")

    print(f"Downloading dataset to {DATA_DIR} ...")
    snapshot_download(
        repo_id="Unispac/shallow-vs-deep-safety-alignment-dataset",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        token=token,
    )
    print("Dataset download complete.")
