# Records

## 2026-03-21

### Research Goal

- Extend Set-LLM from MCQ invariance to `LLM-as-a-judge` pairwise evaluation.
- Main new hypothesis: for long candidate responses, `setcausal` is a better masking bias than full `SetMask`.

### Implemented

- Added `judge_pairwise` task to `reproduce_setllm.py`.
- Added `setcausal` architecture mode alongside `vanilla` and `setllm`.
- Added toy judge JSONL datasets in `data/`.
- Added `prepare_judge_data.py` to convert `lmsys/mt_bench_human_judgments` into pairwise judge JSONL.
- Added `scripts/run_judge_ablation_vast.sh` for first `vanilla` vs `setllm` vs `setcausal` ablation runs.
- Updated `README.md` with judge-task and data-preparation instructions.

### Vast.ai Smoke Tests

- Machine used for current smoke tests: `ssh -p 20386 root@74.14.5.197`.
- Verified `/venv/main` has CUDA-enabled PyTorch.
- Installed missing Python packages into `/venv/main` with `uv` without replacing `torch`.
- Downloaded and verified `google/gemma-2b` at `/workspace/models/gemma-2b`.

### W&B Runs

- `judge-toy-setcausal-smoke` on TinyLlama
  - project: `setllm-smoke`
  - purpose: first end-to-end pairwise judge eval smoke test
- `gemma-2b-judge-toy-setcausal-eval`
  - project: `setllm-smoke`
  - result: eval-only Gemma smoke test on toy judge data
- `gemma-2b-judge-toy-setcausal-train`
  - project: `setllm-smoke`
  - result: 1-step train+eval Gemma smoke test
- `gemma-2b-mtbench-pairwise-smoke`
  - project: `setllm-smoke`
  - result: real MT-Bench human pairwise eval smoke test on 4 examples

### Observations

- The current unfine-tuned Gemma pairwise judge is strongly position biased on MT-Bench pairwise smoke data.
- On the 4-example MT-Bench smoke eval with `setcausal`, metrics were:
  - `accuracy = 0.25`
  - `swap_consistency = 0.0`
  - `first_position_win_rate = 1.0`
  - `swapped_first_position_win_rate = 1.0`
- This supports the motivation for the project: judge order bias is present and measurable.

### Prepared Data

- MT-Bench human pairwise conversion output:
  - `data/mt_bench_human_pairwise_smoke/train.jsonl`
  - `data/mt_bench_human_pairwise_smoke/eval.jsonl`
  - `data/mt_bench_human_pairwise_smoke/stats.json`
- Conversion stats:
  - `num_rows = 3355`
  - `num_train_rows = 2739`
  - `num_eval_rows = 616`
  - `num_unique_questions = 80`
  - `num_eval_questions = 16`

### Next In Progress

- Launch first MT-Bench pairwise training ablation across:
  - `vanilla`
  - `setllm`
  - `setcausal`
- Record W&B URLs, output directories, and main metrics here.

### 2026-03-21 Update: First Full MT-Bench Launch Attempt

- Launched a first Gemma 2B ablation run on full MT-Bench pairwise data.
- The initial `vanilla` run hit CUDA OOM during training on the 32 GB class remote GPU.
- Failure suggests the long pairwise judge prompts need explicit sequence-length control before full ablations.

### Mitigation Added

- Added `--max-seq-len` to `reproduce_setllm.py`.
- Added left-truncation for encoded examples so judge prompts can be clipped to a controlled budget.
- Updated the ablation script to default to:
  - `max_seq_len = 1024`
  - `gradient_accumulation_steps = 16`
- Reduced default ablation LoRA size to:
  - `lora_r = 4`
  - `lora_alpha = 4`
- Enabled gradient checkpointing support in the training script.
- This is the current path to get the first full ablation running on available VRAM.

### 2026-03-21 Update: Second Launch Attempt

- A second relaunch still showed the old OOM signature in the log, indicating the old run state/log was still being observed.
- Added another memory-saving path:
  - `--gradient-checkpointing`
  - `low_cpu_mem_usage=True` on model load
- Next step is a smaller training pilot with the new flags and a fresh output directory to avoid stale log confusion.

### 2026-03-21 Update: New 48 GB Machine

- New machine: `ssh -p 39939 root@194.228.55.129`
- GPU detected: `NVIDIA RTX PRO 5000 Blackwell`
- Reported total VRAM: about `47.27 GB`
- Free VRAM on initial check: about `46.93 GB`

### 2026-03-21 Pilot Result on 48 GB Machine

- Successfully ran a reduced-memory Gemma 2B pilot on MT-Bench pairwise data with `vanilla`.
- Pilot config:
  - `max_train_samples = 64`
  - `max_eval_samples = 64`
  - `max_seq_len = 768`
  - `lora_r = 4`
  - `gradient_checkpointing = true`
  - `update_steps = 20`
- W&B run:
  - `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/11068tpr`
- Final pilot metrics:
  - `accuracy = 0.515625`
  - `swap_consistency = 0.09375`
  - `first_position_win_rate = 1.0`
  - `swapped_first_position_win_rate = 0.90625`

### Interpretation

- The pilot now fits memory on the 48 GB machine.
- The model learns enough to improve pairwise accuracy above chance on the 64-example pilot.
- Position bias remains severe, which is consistent with the central paper motivation.
- Next step is to run the same pilot for `setllm` and `setcausal` and compare consistency directly.

### 2026-03-21 Matched Fast Pilot Ablation

To finish a first apples-to-apples comparison quickly, the pilot configuration was simplified to:

- `max_train_samples = 64`
- `max_eval_samples = 32`
- `max_seq_len = 768`
- `lora_r = 4`
- `gradient_checkpointing = true`
- `update_steps = 8`
- no mid-training eval; final eval only

#### Results

- `vanilla`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/m8k7yzu0`
  - accuracy: `0.53125`
  - swap_consistency: `0.0`
  - first_position_win_rate: `0.0`
  - swapped_first_position_win_rate: `0.0`
- `setllm`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/0r9azzwg`
  - accuracy: `0.53125`
  - swap_consistency: `0.0`
  - first_position_win_rate: `0.0`
  - swapped_first_position_win_rate: `0.0`
- `setcausal`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/sm133fub`
  - accuracy: `0.53125`
  - swap_consistency: `0.0`
  - first_position_win_rate: `0.0`
  - swapped_first_position_win_rate: `0.0`

#### Immediate Observation

- Under this very small and heavily truncated pilot setting, all three architectures produced the same headline metrics.
- This means the pilot is sufficient for engineering validation, but not yet informative enough to distinguish `vanilla`, `setllm`, and `setcausal` scientifically.
- The likely reasons are:
  - too few update steps
  - too much truncation
  - too small an eval subset
  - judge behavior collapsing to a simple dominant label pattern

#### Research Implication

- The current fast pilot validates the end-to-end experiment pipeline.
- It does not yet test the central hypothesis strongly enough.
- The next useful stage should increase signal rather than just repeat this same tiny pilot:
  - larger eval subset
  - more update steps
  - sequential runs on the 48 GB machine
  - possibly stronger label-analysis metrics to inspect collapse behavior

### 2026-03-21 Note on Execution

- The next `setllm` and `setcausal` pilot runs did not fail immediately; they exceeded the local command timeout while still running remotely.
- To avoid losing progress to local timeout limits, the next runs are being launched in the background with log files and then polled for completion.

### 2026-03-21 Pilot Scheduling Observation

- Running `setllm` and `setcausal` simultaneously on the 48 GB machine is not viable.
- Two concurrent Gemma 2B judge pilots together consumed about `44 GB` of VRAM and blocked any new run from loading.
- From this point on, pilot ablations should be executed sequentially, one architecture at a time.

### 2026-03-21 Pilot Runtime Observation

- Mid-training evaluation is the main runtime bottleneck for judge pilots.
- Each judge eval example requires scoring `A`, `B`, and `Tie` for both original and swapped order, so the pilot eval path is much more expensive than the training step count suggests.
- Pilot script updated to disable mid-run eval and only do final eval at the end of training.
- Pilot eval subset reduced from `64` to `32` examples for faster architecture comparison.
- Pilot update steps reduced from `12` to `8` for faster architecture-comparison turnaround.

### 2026-03-21 Next Debug Step

- Since asymmetric left-truncation appears to favor `Response B`, the next pilot rerun will remove truncation entirely.
- Goal: test whether the `always B` / collapse pattern disappears once the full pairwise prompt is preserved.

### 2026-03-21 No-Truncation Pilot Ablation

Same fast pilot config as before, but with no `max_seq_len` limit.

#### Results

- `vanilla`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/qd7g19ev`
  - accuracy: `0.4375`
  - swap_consistency: `0.0`
  - first_position_win_rate: `1.0`
  - swapped_first_position_win_rate: `1.0`
- `setllm`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/0wdb8riq`
  - accuracy: `0.5`
  - swap_consistency: `0.125`
  - first_position_win_rate: `0.4375`
  - swapped_first_position_win_rate: `0.4375`
- `setcausal`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/ywn1qpdt`
  - accuracy: `0.53125`
  - swap_consistency: `0.0`
  - first_position_win_rate: `0.0`
  - swapped_first_position_win_rate: `0.0`

#### Observation

- Removing truncation breaks the exact metric tie seen in the earlier fast pilot.
- This confirms that truncation was materially distorting the comparison.
- `setllm` is the only variant that shows non-zero swap consistency in this no-truncation pilot.
- The three variants still appear unstable and undertrained, but they are no longer behaviorally identical.

### 2026-03-21 Debug: Why Vanilla Collapsed to `A`

Root cause found in the dataset loading path for `judge_pairwise`:

- judge JSONL loading used to apply `max_train_samples` and `max_eval_samples` while reading the file, before any shuffling
- later, the `judge_pairwise` path skipped `select_rows`, so the pilot always used the first `N` rows of each JSONL file
- for MT-Bench human pairwise data, that created highly biased tiny subsets

Observed biased subset before the fix:

- first 64 train rows came only from question ids `81` and `82`
- train labels were `A: 32`, `B: 22`, `Tie: 10`
- first 32 eval rows all came from question id `89`

Fix applied:

- `load_jsonl_dataset()` now loads the full JSONL
- `select_rows()` is now applied to judge datasets too, so sample limits are taken after shuffling

Sanity check after the fix (seed `42`):

- shuffled train labels: `B: 31`, `A: 25`, `Tie: 8`
- shuffled eval labels: `A: 15`, `B: 11`, `Tie: 6`

Validation rerun after the fix:

- no-truncation `vanilla` pilot rerun W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/j0isl5xy`
- metrics:
  - `accuracy = 0.375`
  - `swap_consistency = 0.1875`
  - `first_position_win_rate = 0.5`
  - `swapped_first_position_win_rate = 0.5625`

Conclusion:

- the `always A` behavior was not an inherent property of vanilla
- it was caused by a biased, non-shuffled judge subset selection bug

### 2026-03-21 Corrected No-Truncation Comparison After Shuffle Fix

All runs below use the same no-truncation fast pilot config with shuffled sampling.

- `vanilla`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/j0isl5xy`
  - accuracy: `0.375`
  - swap_consistency: `0.1875`
  - first_position_win_rate: `0.5`
  - swapped_first_position_win_rate: `0.5625`
- `setllm`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/3k0sbxu9`
  - accuracy: `0.34375`
  - swap_consistency: `0.15625`
  - first_position_win_rate: `0.21875`
  - swapped_first_position_win_rate: `0.1875`
- `setcausal`
  - W&B: `https://wandb.ai/ko-the-university-of-north-carolina-at-chapel-hill/setllm-judge/runs/ipxo0usy`
  - accuracy: `0.40625`
  - swap_consistency: `0.0`
  - first_position_win_rate: `0.09375`
  - swapped_first_position_win_rate: `0.09375`

### Updated Pilot Takeaway

- After fixing judge subset sampling, the architectures no longer collapse in the same way.
- `vanilla` and `setllm` both show some non-zero swap consistency in this tiny pilot, with `vanilla` slightly higher here.
- `setcausal` still underperforms on swap consistency in this small-budget setting.
- These results are still too small and noisy for paper conclusions, but they are now much more trustworthy than the earlier biased pilot runs.

### 2026-03-21 Debug: Why `setcausal` Swap Consistency Can Stay at Zero

Another likely bug in the judge-task adaptation was found:

- training used only the canonical pairwise order
- but the target labels are positional (`A` / `B` / `Tie`)
- unlike MCQ answer strings, these labels must flip when responses are swapped

Implication:

- without swap-augmented training, the model is never explicitly taught that swapping the two responses should also swap the correct label
- this can produce zero swap consistency even when the architecture itself is not obviously broken

Fix applied:

- judge training now augments each pairwise example with a swapped-order copy and the correspondingly swapped label
