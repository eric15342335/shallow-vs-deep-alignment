#!/bin/bash

python - <<'PY'
import importlib.metadata as md

pkgs = [
    "absl-py","accelerate","aiohappyeyeballs","aiohttp","aiosignal","annotated-types",
    "anyio","attrs","bitsandbytes","certifi","charset-normalizer","click","contourpy",
    "cycler","datasets","deepspeed","dill","distro","docstring-parser","filelock",
    "fonttools","frozenlist","fsspec","h11","hf-xet","hjson","httpcore","httpx",
    "huggingface-hub","idna","jinja2","jiter","joblib","kiwisolver","markdown-it-py",
    "markupsafe","matplotlib","mdurl","mpmath","msgpack","multidict","multiprocess",
    "networkx","ninja","nltk","numpy","nvidia-cublas-cu12","nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12","nvidia-cuda-runtime-cu12","nvidia-cudnn-cu12",
    "nvidia-cufft-cu12","nvidia-curand-cu12","nvidia-cusolver-cu12","nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12","nvidia-nccl-cu12","nvidia-nvjitlink-cu12","nvidia-nvtx-cu12",
    "openai","packaging","pandas","peft","pillow","propcache","protobuf","psutil",
    "py-cpuinfo","pyarrow","pydantic","pydantic-core","pygments","pyparsing",
    "python-dateutil","pyyaml","regex","requests","rich","rouge-score","safetensors",
    "scipy","sentencepiece","setuptools","six","sniffio","sympy","tokenizers","torch",
    "torchvision","tqdm","transformers","triton","trl","typeguard","typing-extensions",
    "typing-inspection","tyro","urllib3","xxhash","yarl"
]

def version(name):
    try:
        print(f"{name:24} {md.version(name)}")
    except Exception:
        print(f"{name:24} NOT INSTALLED")

for p in pkgs:
    version(p)

try:
    import torch
    print("\n[torch details]")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
except Exception as e:
    print("\n[torch details] error:", e)
PY