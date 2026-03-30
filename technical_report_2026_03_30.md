# SetLLM / SetCausal Technical Progress Report

Date: 2026-03-30

## 1. Executive Summary

This report summarizes the current state of the SetLLM / SetCausal pairwise-judge project for collaborator handoff. The main empirical result so far is positive: after reformulating pairwise judging as symmetric per-candidate scoring over anonymous candidate slots, the `setcausal` architecture becomes the strongest configuration we have tested on MT-Bench human pairwise data.

The current best completed setup is:

- backbone: `Qwen/Qwen2.5-3B-Instruct`
- prompt style: `anonymous_slots`
- readout: `symmetric_scoring`
- training regime: `+UltraFeedback warm-up`
- architecture: `setcausal`

Under this setup, `setcausal` reaches:

- `55.68%` row-level human agreement
- `76.05%` no-tie agreement
- `99.35%` swap consistency

This outperforms both:

- `vanilla`, which remains more position-biased
- `setllm`, which remains highly invariant but less aligned with human labels

## 2. Research Direction Shift

The project originally started from a direct adaptation of Set-LLM to open-ended pairwise judging. Early label-generation experiments showed that directly predicting `A/B/Tie` or `1/2/Tie` from a permutation-aware backbone was unstable and often mismatched the symmetry assumptions of Set-LLM.

The important conceptual shift was:

1. Keep the candidate responses anonymous in the prompt.
2. Replace slot-labeled next-token prediction with symmetric per-candidate scoring.
3. Compare candidate utilities rather than forcing the model to generate slot identity tokens directly.

This change materially improved the story. Under the scoring-based formulation, `setcausal` is now both more robust and more aligned than the standard causal judge on our main benchmark.

## 3. Model and Training Setup

Completed main experiments used:

- base model: `Qwen/Qwen2.5-3B-Instruct`
- LoRA fine-tuning
- `bf16` train, `fp32` eval
- pairwise training data: `MT-Bench human pairwise`
- swap-augmented training
- architectures:
  - `vanilla`
  - `setllm`
  - `setcausal`

The `+UF` experiments additionally used:

- `UltraFeedback-style` warm-up on a cleaned `10k` instruction-answer subset

The scoring-based readout uses:

- two prompt-side per-candidate score tokens
- a shared scalar score head
- a symmetric tie head

## 4. Main Completed Results

### 4.1 MT-Bench, No-UF, `Qwen/Qwen2.5-3B-Instruct`

| Architecture | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency | First-Pos Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vanilla` | 47.08 | 64.30 | 0.00 | 29.87 | 79.87 |
| `setllm` | 38.64 | 52.77 | 0.00 | 99.35 | 50.32 |
| `setcausal` | 54.38 | 74.28 | 0.00 | 99.35 | 51.14 |

Interpretation:

- `setcausal` is already the strongest completed model.
- `setllm` is highly invariant but under-aligned.
- `vanilla` remains strongly first-position biased.

### 4.2 MT-Bench, +UF, `Qwen/Qwen2.5-3B-Instruct`

| Architecture | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency | First-Pos Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vanilla` | 46.59 | 63.41 | 1.18 | 44.64 | 69.32 |
| `setllm` | 42.05 | 53.88 | 15.02 | 99.35 | 43.18 |
| `setcausal` | 55.68 | 76.05 | 0.00 | 99.35 | 50.81 |

Interpretation:

- `+UF` preserves and slightly improves the `setcausal` lead.
- `setllm` learns to output some ties under the newer tie head.
- `setcausal` still does not predict ties well, even though it is strongest on overall agreement.

## 5. Current Technical Interpretation

At this point, the strongest defensible claim is:

> Architecture-level permutation handling helps open-ended pairwise judging when the task is reformulated as symmetric per-candidate scoring. Under this formulation, `setcausal` provides a better inductive bias than both `vanilla` and the original prompt-side `SetMask` adaptation.

We should not currently claim that the original Set-LLM formulation directly solves open-ended pairwise judging. The evidence instead suggests:

- slot-labeled next-token prediction clashes with symmetry assumptions
- the readout design matters as much as the masking design
- `setcausal` benefits from preserving within-response causal structure

## 6. Fine-Tuned Weights Included in This Package

This package includes the downloaded fine-tuned artifacts from the completed successful MT-Bench runs.

For each completed model, we downloaded:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- `README.md`
- metrics JSON
- per-example prediction JSONL

For the `+UF` runs, the package also includes:

- `ultra-pretrain/adapter_model.safetensors`
- `ultra-pretrain/adapter_config.json`

Important note:

- these are LoRA adapter checkpoints plus saved scoring modules, not full base-model checkpoints
- the full backbone weights remain the upstream Hugging Face base model

## 7. What Failed Today

We launched a new extension batch to answer three questions:

1. Can tie-weighted loss improve `setcausal` tie prediction?
2. Does the result hold at a stronger backbone (`Qwen/Qwen2.5-7B-Instruct`)?
3. Does the result transfer to `LLMBar Natural`?

The first pass of these experiments failed for a clean engineering reason:

- all runs completed `UltraFeedback` warm-up
- all runs then crashed at the weighted scoring loss
- error:
  - `RuntimeError: expected scalar type BFloat16 but found Float`

Root cause:

- `class_logits` stayed in `bfloat16`
- class weights were created in `float32`
- weighted `cross_entropy` required matching dtypes under this path

Hotfix applied:

- cast `class_logits` to `float32` inside the weighted `cross_entropy` call

## 8. Current Queued Experiments

The following hotfixed resubmissions are now queued on Longleaf:

- `40281633`: MT-Bench, `Qwen/Qwen2.5-3B-Instruct`, `setcausal`, `+UF`, `judge_tie_weight = 2.0`
- `40281634`: MT-Bench, `Qwen/Qwen2.5-3B-Instruct`, `setcausal`, `+UF`, `judge_tie_weight = 4.0`
- `40281635`: MT-Bench, `Qwen/Qwen2.5-7B-Instruct`, `setcausal`, `+UF`
- `40281816`: `LLMBar Natural`, `Qwen/Qwen2.5-3B-Instruct`, `vanilla`, `+UF`
- `40281817`: `LLMBar Natural`, `Qwen/Qwen2.5-3B-Instruct`, `setllm`, `+UF`
- `40281818`: `LLMBar Natural`, `Qwen/Qwen2.5-3B-Instruct`, `setcausal`, `+UF`
- `40281819`: dependent `LLMBar Natural` table-generation job

## 9. Recommended Immediate Next Steps

1. Let the hotfixed jobs finish before changing the training recipe again.
2. If `judge_tie_weight` improves `setcausal` tie F1 without harming agreement, integrate that into the main story.
3. If the `Qwen-7B` run strengthens the MT-Bench result, use it as the “stronger-backbone confirmation” experiment.
4. If `LLMBar Natural` reproduces the `setcausal` advantage, elevate that result into the main paper as external validation.
5. If `LLMBar` does not reproduce the MT-Bench result, keep the paper centered on MT-Bench and frame `LLMBar` as a transfer-stress test.

## 10. Files to Read First

For a collaborator jumping in quickly, the best entry points are:

- `technical_report_2026_03_30.md`
- `saved_results/qwen2_5_3b_no_uf_anonymous_slots_symmetric_scoring_2026_03_29.md`
- `saved_results/qwen2_5_3b_uf_anonymous_slots_symmetric_scoring_2026_03_30.md`
- `reproduce_setllm.py`
- `scripts/generate_judge_tables.py`
- `scripts/longleaf_run_judge_generic_uf_a100.sbatch`
- `scripts/longleaf_launch_experiment_batch_20260330.sh`
