# Research Plan: Set-LLM for LLM-as-a-Judge

## 1. Objective

We want to turn this repo into a paper project on permutation-invariant LLM judges.
The main question is whether Set-LLM-style architectural invariance helps on open-ended
judge tasks, where each set element is a long model response instead of a short MCQ option.

The paper should answer two questions:

1. Does Set-LLM improve order robustness for LLM-as-a-judge?
2. For long candidate responses, is a `set-causal mask` better than the original `SetMask`?

## 2. Core Hypothesis

The original Set-LLM paper uses `SetPE + SetMask`, where prompt-side set elements are
internally non-causal. That is reasonable for short answer options, but may be suboptimal
when each set element is a long multi-paragraph response.

Our hypothesis is:

- `SetMask` is too permissive for long judge candidates because tokens inside one response
  can attend to future tokens in that same response.
- A `set-causal mask` will preserve the sequential structure of each candidate response while
  still blocking cross-response leakage.
- Therefore, `set-causal mask` should give a better quality/robustness tradeoff than full
  `SetMask` on LLM-as-a-judge tasks.

## 3. Proposed Methods

### 3.1 Vanilla Judge

Standard decoder-only pairwise judge prompt with normal causal masking.

### 3.2 Set-LLM Judge

Use `SetPE + SetMask` from the Set-LLM paper.

For pairwise judging, the two candidate responses are treated as two elements of a set inside
the judge prompt.

### 3.3 Set-Causal Judge

This is the proposed new method.

Definition:

- Candidate responses are still treated as set elements.
- `SetPE` is applied as in Set-LLM.
- Attention across different candidate responses is blocked.
- The underlying prompt mask is otherwise the standard decoder causal mask.
- Within each candidate response, attention is causal in local token order.
- Non-set scaffold text is causal too, so the full prompt remains decoder-style.
- Judge output tokens attend to all prompt tokens and previous output tokens only.

This should preserve permutation robustness across responses while avoiding the fully visible
document-style attention induced by `SetMask` within a long response.

## 4. Main Experimental Scope

We should keep the first paper focused.

### 4.1 Primary Task

Pairwise judging only.

Prompt form:

- user instruction or conversation context
- response A
- response B
- judge outputs `A`, `B`, or `Tie`

This is the cleanest setting for studying positional bias and consistency.

### 4.2 Optional Extensions

Only after pairwise works well:

- rubric-based scoring
- listwise ranking
- hierarchical SetPE / hierarchical masking for very long responses

## 5. Datasets

### 5.1 Primary Evaluation Datasets

- `MT-Bench`
  - strong, widely used LLM-as-a-judge benchmark
  - multi-turn and open-ended
  - human judgments available
- `Chatbot Arena` preference data
  - larger scale
  - noisier but realistic
- `LLMBar`
  - useful for judge bias and adversarial robustness

### 5.2 Optional Dataset

- `FairEval`
  - consider only if rubric scoring is added

### 5.3 Recommended Use

- use MT-Bench as the main controlled benchmark
- use Arena-style preference data for scale and external validation
- use LLMBar for robustness and attack evaluation

## 6. Baselines

We should compare against both architectural and protocol baselines.

### 6.1 Model Baselines

- `Vanilla causal judge`
- `SetPE + SetMask`
- `SetPE + set-causal mask`
- optional: `SetPE only`

### 6.2 Inference Protocol Baselines

- vanilla judge with one fixed order
- vanilla judge with swapped order and consistency filter
- vanilla judge with random order assignment
- repeated-judge or majority-vote style protocol if affordable

## 7. Metrics

### 7.1 Consistency

- swap consistency: same judgment after swapping response order
- first-position win rate
- second-position win rate
- tie stability under swap

### 7.2 Human Alignment

- agreement with human pairwise preferences
- accuracy on labeled pairwise preferences
- Spearman and Kendall correlations where scalar labels exist

### 7.3 Robustness

- performance under whitespace and formatting changes
- performance under confident-sounding preambles
- performance under verbosity attacks
- performance under paragraph reordering inside a response

### 7.4 Efficiency

- judge quality vs number of inference calls
- single-pass invariant model vs swap-based multi-call protocol

## 8. Key Ablations

### 8.1 Architecture Ablation

- vanilla
- SetPE + SetMask
- SetPE + set-causal mask

### 8.2 Length Ablation

Bucket examples by candidate-response length:

- short
- medium
- long

This is important because the main motivation for `set-causal mask` is long set elements.

### 8.3 Attack Ablation

Measure which method degrades most under:

- response swap
- verbosity expansion
- paragraph reorder
- formatting-only perturbation
- confidence-preamble attack

### 8.4 Cost Ablation

Compare:

- one-pass vanilla
- one-pass SetMask
- one-pass set-causal
- two-pass swap-based vanilla
- repeated-vote vanilla if budget allows

## 9. Expected Claims

If the experiments work, the paper should be able to claim:

1. Architectural set-aware judges reduce positional bias in open-ended evaluation.
2. For long response elements, `set-causal mask` is a better inductive bias than full `SetMask`.
3. Single-pass invariant judges are more efficient than protocol-only fixes based on repeated
   judging over permutations.

## 10. Implementation Plan

### Phase 1: Infrastructure

- extend the current repo from MCQ prompts to pairwise judge prompts
- implement `set-causal mask`
- add dataset loaders for MT-Bench / Arena-style pairwise data
- add metrics for swap consistency and human agreement

### Phase 2: Small-Scale Validation

- run on a small model first
- verify masking behavior directly
- confirm that `SetMask` and `set-causal mask` are permutation-stable
- check fp32 eval invariance as in the original Set-LLM paper

### Phase 3: Main Runs

- train and evaluate on MT-Bench pairwise data
- test on LLMBar robustness splits
- evaluate on Arena-style held-out preferences

### Phase 4: Paper Figures and Tables

- consistency table
- human agreement table
- robustness table
- compute vs quality plot
- length-bucket analysis plot

## 11. Model Plan

Start small and scale only after the pipeline works.

Recommended order:

1. `google/gemma-2b`
2. `meta-llama/Llama-3.2-3B`
3. one `7B` model only if needed

Training style:

- LoRA first
- `bf16` training
- `fp32` evaluation for invariance-sensitive comparisons

## 12. Compute Plan

We will use Vast.ai, preferably a `5090`, for the first experimental stage.

Recommended workflow:

- prototype locally or on a short remote run
- run main 2B to 3B LoRA experiments on a `5090`
- only move to larger GPUs if long-context 7B runs become necessary

## 13. Main Risks

- long responses may make full `SetMask` unstable or less semantically faithful
- small judge models may have weak absolute quality even if consistency improves
- noisy preference labels may hide moderate gains
- the exact mask definition must preserve permutation invariance and not accidentally reintroduce
  order sensitivity

## 14. Immediate Next Steps

1. formalize the `set-causal mask` mathematically
2. design the pairwise judge prompt format
3. identify and prepare the first training/evaluation dataset
4. implement the judge-task data pipeline
5. implement `set-causal mask` in the current codebase
6. run a tiny invariance sanity check before any large training jobs
