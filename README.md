# Set-LLM Reproduction Scaffold

This repository only contained the paper PDF, not the official implementation.
`reproduce_setllm.py` is an independent scaffold that implements the method
described in the paper:

- Set Position Encoding (`SetPE`)
- Set Attention Masking (`SetMask`)
- Set-causal masking (`setcausal`) for long set elements in judge prompts
- Paper-style prompts for `PIQA`, `ARC-Challenge`, `CommonsenseQA`, and `SIQA`
- Pairwise `LLM-as-a-judge` prompts with JSONL inputs
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

For judge-task work, it is often easier to download the gated model once to local
storage and then point `--model` at the downloaded directory, for example
`/workspace/models/gemma-2b`.

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

To run the long-element ablation with the proposed set-causal mask, use:

```bash
source .venv/bin/activate
python reproduce_setllm.py \
  --architecture setcausal \
  --model /workspace/models/gemma-2b \
  --task arc \
  --output-dir artifacts/gemma-2b-arc-setcausal \
  --do-train \
  --do-eval \
  --bf16 \
  --fp32-eval
```

## Pairwise Judge Task

The script now supports an initial pairwise `LLM-as-a-judge` path for research on
open-ended evaluation.

Expected JSONL format:

```json
{"prompt":"...","response_a":"...","response_b":"...","label":"A"}
```

Supported labels are `A`, `B`, and `Tie`.

Example smoke evaluation:

```bash
source .venv/bin/activate
python reproduce_setllm.py \
  --model /workspace/models/gemma-2b \
  --task judge_pairwise \
  --judge-train-jsonl data/judge_pairwise_toy_train.jsonl \
  --judge-eval-jsonl data/judge_pairwise_toy_eval.jsonl \
  --architecture setcausal \
  --output-dir artifacts/gemma-2b-judge-toy-setcausal \
  --do-eval \
  --wandb-project setllm-smoke
```

For this task:

- `--architecture vanilla` uses the base decoder causal mask.
- `--architecture setllm` uses SetPE with the original SetMask prompt-side visibility.
- `--architecture setcausal` uses SetPE with the base decoder causal mask everywhere,
  plus blocked attention across different candidate responses inside the same set.

In other words, `setcausal` is the proposed long-response variant:

- tokens can only attend to earlier tokens, as in a normal decoder LM
- tokens in one candidate response cannot attend to tokens in another candidate response
- non-set scaffold text is causal too; it is not fully visible prompt context

### Why `SetMask` Still Matters At Inference

The next generated token is the only token sampled at each decode step, but its
logits are computed from prompt and prefix hidden states that were already built
using the prompt-side mask. That means the prompt mask still matters at
inference time:

- with paper-style `SetMask`, tokens from different set elements do not mix into
  each other's hidden states on the prompt side
- with only `SetPE`, element-internal order is preserved, but different set
  elements can still contaminate each other through fully connected prompt
  attention
- the generated response token then reads those hidden states, so any prompt-side
  leakage is already baked into the context seen at decode time

This is the same structural reason causal masking matters at inference in a
decoder-only LM: the mask shapes the hidden states that generation consumes, not
just the tokens that are sampled.

### `setcausal` vs Paper `SetMask`

`setcausal` is a deliberate ablation, not an implementation of the paper's
original mask.

- `setllm` keeps prompt tokens bidirectional within an allowed region, then
  blocks attention across different set elements
- `setcausal` keeps decoder-style causal attention within each element and also
  blocks cross-element attention

Why test `setcausal` at all:

- Gemma 2B is a decoder-only model, so `setcausal` stays closer to the base
  model's pretrained attention pattern
- this may make LoRA adaptation easier for long judge responses

Why the paper-style `SetMask` may still be stronger:

- for an already-given prompt, bidirectional attention inside a response can
  build richer prompt representations than causal-within-response attention
- response generation still remains autoregressive in both designs

So the current expectation is:

- `setllm` may have the better ceiling because prompt encoding is less
  constrained
- `setcausal` may fine-tune more smoothly because it is a smaller departure from
  a pretrained decoder-only backbone

This is an empirical question, so the repo keeps both variants for direct judge
ablation.

Current pairwise metrics are:

- `accuracy`
- `swap_consistency`
- `first_position_win_rate`
- `swapped_first_position_win_rate`

## Prepare MT-Bench Human Pairwise Data

The repo now includes a helper to convert `lmsys/mt_bench_human_judgments` into the
JSONL format expected by `--task judge_pairwise`.

Example:

```bash
source .venv/bin/activate
python prepare_judge_data.py \
  --source mt_bench_human \
  --split-name human \
  --output-dir data/mt_bench_human_pairwise
```

This writes:

- `data/mt_bench_human_pairwise/train.jsonl`
- `data/mt_bench_human_pairwise/eval.jsonl`
- `data/mt_bench_human_pairwise/stats.json`

The split is question-based rather than row-based so the same MT-Bench question does not
appear in both train and eval.

## Judge Ablation Command

For the first `vanilla` vs `setllm` vs `setcausal` judge ablation on a Vast machine:

```bash
source /venv/main/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export WANDB_API_KEY=...
cd /workspace/setLLM
python prepare_judge_data.py \
  --source mt_bench_human \
  --split-name human \
  --output-dir data/mt_bench_human_pairwise
./scripts/run_judge_ablation_vast.sh \
  /workspace/models/gemma-2b \
  data/mt_bench_human_pairwise/train.jsonl \
  data/mt_bench_human_pairwise/eval.jsonl \
  /workspace/setLLM/artifacts/judge_ablation_gemma2b \
  setllm-judge \
  100
```

The ablation script currently uses a memory-reduced smoke configuration for Gemma 2B on a
32 GB class GPU:

- `gradient_accumulation_steps = 16`
- `lora_r = 4`

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
