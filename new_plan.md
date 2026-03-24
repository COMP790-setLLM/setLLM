# New Plan for Set-Causal LLM Judge Paper

## Status

This plan is written from:

- the current proposal text,
- the original `SetLLM.pdf` paper already present in this workspace,
- current judge-reliability baselines up to March 24, 2026,
- and the official NeurIPS 2026 submission constraints.

Note: I could not clone `https://github.com/COMP790-setLLM/setLLM.git` from this machine because GitHub returned `repository not found`, which usually means the repo is private or the URL/org name is incorrect for the current credentials. So this file is prepared locally and should be copied into the repo once access is fixed.

## Bottom line

The project is doable, but only if we narrow the claim and build the paper around one core question:

> Can architecture-level permutation invariance for open-ended pairwise judging match or beat strong inference-time debiasing baselines at lower cost?

That is a better NeurIPS paper than a broad "judge robustness" paper.

## Recommended paper framing

### Working title

Set-Causal LLM Judges: Permutation-Invariant Open-Ended Pairwise Evaluation

### One-sentence pitch

We extend Set-LLM from short multiple-choice options to long open-ended candidate responses by treating candidate responses as an unordered set of structured sequences, and we show that a set-causal attention mask improves the robustness-cost tradeoff of pairwise LLM judging.

### Why this framing is stronger

- It is clearly grounded in Set-LLM rather than just "another judge paper."
- It directly addresses a known failure mode in LLM-as-a-judge: response-order sensitivity.
- It competes against the right baselines: two-order judging, probability aggregation, and recent debiasing methods.
- It gives us both a theory story and an empirical story.

## What to keep and what to cut

### Keep

- Pairwise judging only.
- The contrast between `vanilla causal`, `SetPE + SetMask`, and `SetPE + set-causal mask`.
- The theorem-level invariance argument.
- MT-Bench, Chatbot Arena, and LLMBar as the main trio.
- Efficiency as a first-class evaluation axis.

### Cut or demote

- Do not make "hierarchical" the headline unless we implement a real hierarchical module.
- Do not make abstention/selective evaluation the main story.
- Do not overclaim that length alone drives bias; instead say gains should be strongest when within-response structure matters, including long responses and attack-heavy inputs.
- Do not run too many model scales.
- Do not add too many datasets beyond the core three unless the pipeline is stable early.

## Updated central claim

The paper should argue:

1. Pairwise LLM judging is naturally a mixed set-text problem because the candidate responses should be unordered at the response level but ordered within each response.
2. Original Set-LLM gives the right invariance principle, but its prompt-side SetMask was validated mainly on short multiple-choice options.
3. For long open-ended responses, the right inductive bias is not full prompt-side visibility within each candidate response, but local causal visibility within each candidate response plus no cross-response leakage.
4. This set-causal design preserves permutation invariance while improving the quality-robustness-cost tradeoff.

## Paper contribution structure

The final paper should have exactly four contributions:

1. A formulation of open-ended pairwise judging as learning over a set of structured sequences.
2. A new set-causal mask family that preserves response-internal causal order and response-external permutation invariance.
3. A theorem showing permutation equivariance of the mask when combined with SetPE.
4. An empirical comparison showing whether single-pass architectural invariance can match or beat strong multi-order inference baselines at lower cost.

If we do not achieve contribution 4 convincingly, the paper becomes much weaker.

## NeurIPS target and constraints

As of March 24, 2026, the official NeurIPS 2026 main-track dates are:

- Abstract deadline: May 4, 2026 AoE
- Full paper deadline: May 6, 2026 AoE

Main paper limit:

- 9 content pages, excluding references, appendices, and checklist

Recommended submission type:

- `Use-Inspired` if the empirical reliability story is strongest
- `General` if the paper feels balanced between architecture and experiments

Not recommended:

- `Theory` only
- `Concept & Feasibility` unless results are too preliminary for a normal paper

## Exact research questions

We should rewrite the project around these four questions:

### RQ1

Does architecture-level set-aware modeling reduce swap sensitivity in open-ended pairwise judging relative to a standard causal judge?

### RQ2

For long open-ended responses, does a set-causal mask outperform the original SetMask in the tradeoff between human alignment, robustness, and invariance?

### RQ3

Can a single-pass invariant judge match or outperform strong two-order inference baselines at lower cost?

### RQ4

Are the gains strongest on examples where within-response structure matters, such as long responses, paragraph reorder attacks, verbosity attacks, and persuasive framing attacks?

## Hypotheses to use

### Primary hypothesis

Set-causal masking improves the robustness-cost tradeoff over both vanilla causal judging and SetPE + SetMask on open-ended pairwise tasks.

### Secondary hypothesis

Set-causal gains are larger on attack-heavy and structure-sensitive examples than on easy clean examples.

### Safety hypothesis

Even when set-causal does not increase absolute accuracy, it should reduce swap inconsistency enough to be competitive with two-order protocol baselines at a lower inference budget.

## Theory section plan

We should keep the theory compact and precise.

### Formal input decomposition

Write the prompt as:

`x = [c, {r1, r2}, y]`

where:

- `c` is shared scaffold context,
- `{r1, r2}` is the unordered set of candidate responses,
- `y` is the autoregressively generated judge output.

### Mask definition

For each token, define:

- block type: `context`, `response`, or `output`
- response identity: `1`, `2`, or `null`
- local position inside the block or response

Then define the set-causal mask so that:

- context tokens see earlier context tokens,
- response tokens see earlier shared context plus earlier tokens in the same response,
- response tokens do not see the other response,
- output tokens behave like standard decoder tokens and see the whole prompt plus earlier output tokens.

### The theorem we actually need

Do not oversell. The theorem we need is:

> Under SetPE and the set-causal mask, each attention layer is equivariant to permutations of sibling candidate responses, and the resulting judge is invariant to response order.

### What to prove carefully

- The mask transforms as `M_tilde = P M P^T` under response swap.
- SetPE positions are preserved under the response permutation.
- Therefore attention logits and attention outputs are equivariant.
- Final prediction is invariant.

### What to avoid

- Do not introduce unnecessary generality.
- Do not spend pages on more than pairwise unless the proof naturally generalizes in a short appendix.

## Experimental plan

## Main comparison

The main table should compare these methods:

1. Vanilla causal judge, one order
2. Vanilla causal judge, two orders with probability aggregation
3. Vanilla causal judge, recent inference-time debiasing baseline
4. SetPE + SetMask
5. SetPE + set-causal mask

Optional if time permits:

6. Pointwise or PRePair-style protocol baseline
7. Selective evaluation analysis using uncertainty, but only as secondary analysis

## Required baselines

At minimum, the paper must compare against the current serious baselines in this area:

- one-order vanilla pairwise judging,
- two-order vanilla judging with swap aggregation,
- BPE-style two-order aggregation from SCOPE,
- CalibraEval,
- SetPE + SetMask,
- judgment-distribution style inference if implementation cost is manageable.

If we skip these, reviewers can say we only beat weak baselines.

## Datasets

### P0 datasets

- MT-Bench pairwise judgments
- Chatbot Arena preference data
- LLMBar

### P1 dataset

- JudgeBench, only if we can adapt it cleanly and only if it sharpens the evaluation rather than delaying the paper

### Why this set

- `MT-Bench` gives controlled, familiar judge evaluation.
- `Chatbot Arena` gives realistic large-scale pairwise preferences.
- `LLMBar` gives adversarial judge robustness.
- `JudgeBench` is useful, but it is not necessary for the first submission if it bloats the scope.

## Models

We should not over-scale.

### Recommended scales

- Small: Gemma 2B or Llama-class 1B to 2B
- Medium: Llama-class 3B

Optional:

- A 7B model only if the smaller runs already show the effect clearly

### Why

- The contribution is architectural and methodological, not a scale race.
- Two scales are enough to show the effect is not a tiny-model artifact.
- More scales increase training and debugging cost quickly.

## Training setup

### Recommended setup

- LoRA for all trainable comparisons
- same base model checkpoint across all mask variants
- same prompt format across methods whenever possible
- bf16 training
- fp32 evaluation for invariance-sensitive runs

### Important rule

Exact invariance claims should be checked in fp32 or higher precision evaluation because the original Set-LLM paper already notes numerical precision can matter.

## Input/output format

The judge task should be strictly:

- input: shared instruction/context + response A + response B + decision format
- output: `A`, `B`, or `Tie`

Use the same output space for all methods.

## Metrics

The paper should report four metric groups.

### 1. Human alignment

- accuracy against human preference labels
- balanced accuracy if class imbalance matters
- tie-aware accuracy if ties are retained

### 2. Invariance and position bias

- swap consistency
- first-position win rate
- second-position win rate
- tie stability under swap
- average prediction distance across orderings

### 3. Robustness

- performance under formatting-only perturbations
- verbosity expansion
- confidence-preamble attack
- paragraph reorder attack
- order swap

### 4. Efficiency

- number of model calls per example
- average tokens processed
- wall-clock inference time
- quality versus compute frontier

## Main ablations

We should only keep ablations that directly support the claim.

### A1. Architecture ablation

- vanilla causal
- SetPE + SetMask
- SetPE + set-causal

### A2. Length / structure ablation

Bucket by response length and also by structure-sensitive perturbations.

Suggested buckets:

- short
- medium
- long

But do not make length the only stratification. Also include:

- paragraph count
- whether paragraph reorder changes semantics only weakly

### A3. Attack ablation

- swap
- verbosity
- confident preamble
- formatting-only changes
- paragraph reorder

### A4. Cost ablation

Compare single-pass architectural methods against:

- two-order aggregation,
- CalibraEval,
- any repeated-judge protocol we keep.

## Minimal viable paper

If time gets tight, the minimum acceptable paper is:

- 2 model scales
- 3 datasets
- 3 architectural variants
- 2 strong inference-time baselines
- 1 theorem
- 1 main table
- 1 robustness table
- 1 cost-quality figure

If we cannot complete this minimum well, we should not overload the submission with side analyses.

## What "success" looks like

The paper is strong enough for NeurIPS if at least one of the following is clearly true:

1. `SetPE + set-causal` beats `SetPE + SetMask` and matches or beats strong two-order baselines on human alignment while being cheaper.
2. `SetPE + set-causal` substantially improves robustness on LLMBar and attack-heavy subsets without sacrificing too much clean performance.
3. The theorem is clean and the empirical result shows a strong architectural-vs-protocol tradeoff story that reviewers will find nontrivial.

## What is not enough

These outcomes are probably not enough for NeurIPS by themselves:

- tiny gains over vanilla only,
- improvements only in swap consistency but not in human alignment,
- beating only weak baselines,
- a theorem with no compelling empirical payoff,
- many datasets but no clean narrative.

## Strong paper narrative

The paper should read like this:

1. Pairwise LLM judging is useful but unstable under order.
2. Recent work tries to fix this at inference time by querying both orders or calibrating predictions.
3. Set-LLM suggests the problem should be solved architecturally.
4. But original Set-LLM validated its mask on short multiple-choice options.
5. Open-ended judging requires treating each candidate as a structured sequence, not just an unordered option.
6. We introduce a set-causal mask that preserves local sequential structure while retaining response-order invariance.
7. This yields a better robustness-cost tradeoff on open-ended pairwise judging.

If the writing drifts away from this spine, the paper will feel unfocused.

## Recommended paper outline

### Section 1. Introduction

- Problem: pairwise LLM judges are order-sensitive and vulnerable to superficial cues.
- Gap: existing protocol fixes cost extra inference; Set-LLM has not been validated on long open-ended pairwise judging.
- Idea: set-causal invariant judge for unordered candidate responses with ordered internal structure.
- Claims: theorem + empirical tradeoff improvements.

### Section 2. Background and related work

- LLM-as-a-judge
- position bias and selection bias
- inference-time debiasing and two-order protocols
- Set-LLM

### Section 3. Method

- mixed set-text formulation
- SetPE
- original SetMask
- new set-causal mask
- theorem statement

### Section 4. Experimental setup

- datasets
- models
- baselines
- metrics
- attacks

### Section 5. Results

- main table
- robustness table
- cost-quality figure
- length/structure ablation

### Section 6. Analysis

- where set-causal helps
- failure cases
- when protocol methods still win

### Section 7. Limitations and conclusion

- pairwise only
- depends on available human labels
- may not solve all non-order judge biases

## Figure and table plan

The paper should be built around these artifacts.

### Main table

Methods x datasets with:

- human alignment
- swap consistency
- inference cost

### Robustness table

Methods x attack types with degradation from clean performance

### Main figure

Quality vs compute frontier:

- x-axis: inference cost
- y-axis: human alignment or a combined score

### Supporting figure

Length/structure buckets:

- short / medium / long
- or paragraph-count bins

### Model diagram

A simple diagram showing:

- shared context,
- unordered candidate response set,
- local causal visibility within each candidate,
- blocked cross-candidate visibility,
- autoregressive output head.

## Implementation plan

## Phase 1. Infrastructure

### Goals

- add pairwise judge prompt format,
- add block typing and response identity metadata,
- implement original SetMask adaptation for pairwise prompts,
- implement set-causal mask,
- add exact swap-consistency test,
- add fp32 invariance regression checks.

### Deliverables

- mask generator for pairwise judge inputs
- deterministic swap test
- prompt builder
- metric code

## Phase 2. Theory validation

### Goals

- formalize the set-causal mask,
- prove equivariance under swapping sibling responses,
- verify numerical invariance experimentally.

### Deliverables

- theorem draft
- proof appendix draft
- sanity-check notebook or script

## Phase 3. Small-scale pilot

### Goals

- run on one small model and a small subset of MT-Bench + LLMBar,
- verify the method actually uses shared context,
- compare one-order vanilla vs SetMask vs set-causal,
- debug prompt format and label extraction.

### Decision gate

Continue only if at least one of these is true:

- set-causal clearly improves swap consistency without killing alignment,
- or set-causal matches two-order performance at much lower cost on a meaningful subset.

## Phase 4. Main experiments

### Must-have matrix

- 2 model sizes
- 3 datasets
- 3 architecture variants
- 2 strong protocol baselines

### Nice-to-have

- third model size
- JudgeBench
- selective evaluation appendix

## Phase 5. Writing

The writing should begin before all experiments are done.

### Write early

- intro
- method
- theorem
- experimental setup

### Write later

- results
- discussion
- limitations

## Weekly timeline from March 24, 2026

### Week 1: Mar 24 to Mar 30

- finalize problem framing
- implement pairwise prompt format
- implement set-causal mask
- write theorem skeleton

### Week 2: Mar 31 to Apr 6

- run sanity checks
- run pilot on one small model
- implement swap metrics and attack pipeline

### Week 3: Apr 7 to Apr 13

- run main small-scale experiments on MT-Bench and LLMBar
- implement two-order aggregation baseline
- implement CalibraEval or closest feasible inference-time baseline

### Week 4: Apr 14 to Apr 20

- run medium model
- run Chatbot Arena evaluation
- begin figures and tables

### Week 5: Apr 21 to Apr 27

- finalize main table
- finalize robustness table
- decide whether JudgeBench is in or out
- write first full draft

### Week 6: Apr 28 to May 3

- tighten intro and results
- cut scope if needed
- produce anonymous code supplement
- submit abstract by May 4, 2026 AoE

### Final push: May 4 to May 6

- final paper polish
- appendix cleanup
- checklist
- supplementary code/data packaging
- submit full paper by May 6, 2026 AoE

## Risk register

### Risk 1

Set-causal improves invariance but hurts human alignment.

Mitigation:

- optimize for the Pareto tradeoff, not invariance alone
- compare against two-order baselines, not just vanilla

### Risk 2

SetMask is already sufficient and set-causal brings little gain.

Mitigation:

- focus analysis on long and attack-heavy cases
- if gains are weak overall but strong in specific slices, write the paper around that boundary condition honestly

### Risk 3

Protocol baselines are too strong.

Mitigation:

- lean into compute-quality tradeoff
- show when single-pass architecture is cheaper and competitive

### Risk 4

Implementation becomes too broad.

Mitigation:

- keep pairwise only
- keep the dataset count to three for the main submission


## Concrete decisions to make now

These are the decisions I recommend locking immediately.

### Decision 1

Use title family:

- `Set-Causal LLM Judges`

not:

- `Hierarchical ...`

unless we implement a true hierarchy

### Decision 2

Target contribution type:

- `Use-Inspired`

### Decision 3

Main datasets:

- MT-Bench
- Chatbot Arena
- LLMBar

### Decision 4

Main comparisons:

- vanilla one-order
- vanilla two-order / BPE-style
- CalibraEval
- SetPE + SetMask
- SetPE + set-causal

### Decision 5

Main models:

- one small
- one medium

### Decision 6

Main figure:

- quality vs compute frontier

## Suggested abstract direction

We introduce a permutation-invariant architecture for open-ended pairwise LLM judging that treats candidate responses as an unordered set of structured sequences. Building on Set-LLM, we propose a set-causal attention mask that preserves within-response causal order while blocking cross-response leakage and maintaining response-order invariance. We prove equivariance of the attention layers under response permutation and evaluate the method on MT-Bench, Chatbot Arena, and LLMBar against standard causal judges, two-order inference protocols, and recent debiasing baselines. The results should test whether architecture-level invariance can deliver a better robustness-cost tradeoff than protocol-only fixes for open-ended pairwise evaluation.

## Final recommendation

Proceed, but proceed with discipline.

This is not a "collect many judge robustness tricks" paper. It should be a tightly scoped paper about one architectural idea:

> open-ended pairwise judging should be modeled as a set of structured sequences, and the right invariant mask for that structure is set-causal rather than the original SetMask.

If we keep that focus, the project has a credible NeurIPS path.
