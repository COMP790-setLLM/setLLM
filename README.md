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

## Vast.ai Notes

For remote GPU runs on Vast.ai:

- Mount persistent storage at `/workspace`.
- Keep outputs under `/workspace/artifacts`.
- Set `HF_HOME=/workspace/.cache/huggingface` so model downloads persist.
- Prefer passing `WANDB_API_KEY` and `HF_TOKEN` as environment variables on the instance instead of storing them in the repo.

Example run command inside a Vast.ai instance:

```bash
python /root/setLLM/reproduce_setllm.py \
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

To run a true vanilla baseline without SetPE/SetMask, use:

```bash
source .venv/bin/activate
python reproduce_setllm.py \
  --architecture vanilla \
  --model google/gemma-2b \
  --task arc \
  --output-dir artifacts/gemma-2b-arc-vanilla \
  --do-train \
  --do-eval \
  --bf16 \
  --fp32-eval
```

## ARC Comparison Notes

On one remote reproduction run with `google/gemma-2b` on ARC-Challenge, using
LoRA finetuning for 3 epoch-sized validation checkpoints over the script's ARC
validation split (`299` examples), we observed:

| Model | Epoch 1 Random | Epoch 1 Adversarial | Epoch 2 Random | Epoch 2 Adversarial | Epoch 3 Random | Epoch 3 Adversarial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Set-LLM (`--architecture setllm`) | 0.4615 | 0.4615 | 0.5452 | 0.5452 | 0.5920 | 0.5920 |
| Vanilla (`--architecture vanilla`) | 0.4694 | 0.1672 | 0.5226 | 0.1505 | 0.5813 | 0.2977 |

The main pattern from this run is that vanilla finetuning improves random-order
accuracy, but Set-LLM maintains much stronger adversarial-order accuracy across
all three epochs while ending slightly ahead on random-order accuracy by epoch 3.

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

## Research Direction: Set-Causal Masking

One natural extension of the paper's `SetPE` + `SetMask` design is a
permutation-invariant `set-causal` prompt mask.

The key distinction is that a standard causal mask over the serialized prompt is
not permutation-invariant: if two set elements are swapped, the visibility graph
changes because "future" depends on the arbitrary serialization order. `SetPE`
alone does not remove this leakage.

A more promising variant is a block-causal mask that is causal within ordinary
text spans and within each individual set element, while still preventing
attention across different elements of the same set. In that design:

- earlier text blocks can attend only to earlier tokens in those blocks
- tokens inside one set element can attend only to earlier tokens in that same
  element
- tokens cannot attend across sibling set elements
- later non-set text can attend to the full earlier prompt
- response tokens remain standard autoregressive decoder tokens

This preserves permutation invariance because the mask depends only on block
identity, element identity, and within-element token order, not on the
serialized order of sibling set elements.

Why this is interesting as follow-up work:

- it may preserve more decoder-like behavior than the paper's prefix-style
  prompt masking
- it offers a cleaner inductive bias for tasks where each candidate response
  should be processed independently before aggregation
- it could be useful for long-context judge prompts, especially if combined
  with hierarchical SetPE or paragraph-level grouping

Likely tradeoff: this mask is more restrictive than the paper's prefix-style
SetMask, so it may improve invariance while hurting prompt-side information
sharing. That makes it a good ablation target rather than an obvious drop-in
replacement.
