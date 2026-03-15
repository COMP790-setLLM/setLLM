# Set-LLM Reproduction Scaffold

This repository only contained the paper PDF, not the official implementation.
`reproduce_setllm.py` is an independent scaffold that implements the method
described in the paper:

- Set Position Encoding (`SetPE`)
- Set Attention Masking (`SetMask`)
- Paper-style prompts for `PIQA`, `ARC-Challenge`, `CommonsenseQA`, and `SIQA`
- LoRA finetuning for decoder-only Hugging Face models
- Permutation-based evaluation for random-order and adversarial-order accuracy

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch transformers datasets peft accelerate sentencepiece
```

## Container Image For Runpod

Build a reusable image once, then use it in a custom Runpod template so Pod restarts do not reinstall dependencies.

```bash
podman build -t ghcr.io/YOUR_GH_USER/setllm:latest .
podman push ghcr.io/YOUR_GH_USER/setllm:latest
```

Recommended Runpod setup:

- Use this image in a custom Pod template.
- Mount persistent storage at `/workspace`.
- Keep outputs under `/workspace/artifacts`.
- Set `HF_HOME=/workspace/.cache/huggingface` so model downloads persist.
- Prefer passing `WANDB_API_KEY` as an environment variable in Runpod instead of storing it in the repo.

Example run command inside Runpod:

```bash
python /app/reproduce_setllm.py \
  --model google/gemma-2b \
  --task arc \
  --output-dir /workspace/artifacts/gemma-2b-arc \
  --do-train \
  --do-eval \
  --eval-every 100 \
  --wandb-project setllm \
  --bf16 \
  --fp32-eval
```

## Finetune On A Benchmark

Example with Gemma 2B on ARC:

```bash
source .venv/bin/activate
python reproduce_setllm.py \
  --model google/gemma-2b \
  --task arc \
  --output-dir artifacts/gemma-2b-arc \
  --do-train \
  --do-eval \
  --bf16 \
  --fp32-eval
```

## Optional Extra Pretraining

The paper uses a cleaned UltraFeedback subset, but that preprocessing artifact is
not included here. If you have a JSONL with rows like
`{"instruction": "...", "answer": "..."}`, you can add an extra pretraining stage:

```bash
source .venv/bin/activate
python reproduce_setllm.py \
  --model google/gemma-2b \
  --task arc \
  --cleaned-ultra-jsonl cleaned_ultra.jsonl \
  --extra-pretrain-samples 10000 \
  --output-dir artifacts/gemma-2b-arc \
  --do-train \
  --do-eval \
  --bf16 \
  --fp32-eval
```

## Important Caveats

- This is not the authors' released code.
- The paper does not publish the exact cleaned UltraFeedback subset.
- The script assumes the target model accepts 4D additive attention masks via
  `attention_mask`, which is supported by many recent decoder-only HF models but
  is still worth validating on your exact `transformers` version and base model.
