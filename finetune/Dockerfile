FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Torch build matching the CUDA 12.4 base image above.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# ms-swift: SFT/LoRA/QLoRA framework with native Qwen3-VL support.
# bitsandbytes: 4-bit quantization backend for QLoRA.
# qwen-vl-utils: official Qwen helper for image preprocessing.
RUN pip install \
        "ms-swift[all]" \
        bitsandbytes \
        accelerate \
        qwen-vl-utils \
        pillow

WORKDIR /workspace
COPY train.sh merge.sh ./
RUN chmod +x train.sh merge.sh

ENTRYPOINT ["/bin/bash"]
