FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    UV_CACHE_DIR=/workspace/.cache/uv \
    WANDB_DIR=/workspace/wandb

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY reproduce_setllm.py README.md /app/

RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/uv /workspace/wandb /workspace/artifacts

ENTRYPOINT ["python", "/app/reproduce_setllm.py"]
