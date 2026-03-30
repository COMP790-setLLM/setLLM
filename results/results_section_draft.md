# Results Draft

## Main MT-Bench Result

Table 2 shows that the strongest configuration so far is `setcausal` with `anonymous_slots` and `symmetric_scoring` on `Qwen/Qwen2.5-3B-Instruct`. Under this setup, `setcausal` outperforms both `vanilla` and `setllm` on row-level human agreement while also maintaining near-perfect swap consistency. The main contrast is that `vanilla` remains moderately competitive on agreement but suffers from strong first-position bias, whereas `setllm` achieves near-perfect invariance but loses too much human alignment.

Using the no-UF setting, `setcausal` reaches `54.38` human agreement and `99.35` swap consistency, compared with `47.08` and `29.87` for `vanilla`, and `38.64` and `99.35` for `setllm`. This is the first setting in our project where the proposed architecture is not only more order-robust but also more human-aligned than the standard causal judge.

## Effect of UltraFeedback Warm-Up

UltraFeedback-style warm-up strengthens the same conclusion rather than reversing it. With `+UF`, `setcausal` improves from `54.38` to `55.68` human agreement and from `74.28` to `76.05` no-tie agreement, while preserving `99.35` swap consistency. In contrast, `vanilla` does not improve overall human agreement under `+UF`, although its swap consistency rises from `29.87` to `44.64`. `setllm` benefits somewhat from warm-up, especially on tie handling, but still trails both `vanilla` and `setcausal` on alignment.

This pattern suggests that warm-up helps the judge adapt to the modified masking and readout regime, but it does not change the ranking of the three architectures. The strongest overall system remains `setcausal + symmetric_scoring`, and the gap is now large enough to support a central empirical claim.

## Tie Behavior

Tie prediction remains the clearest unresolved weakness. In the original symmetric-scoring runs, all three models effectively collapsed away from the `Tie` class. After adding the input-dependent symmetric tie head, `setllm` begins to recover some tie predictions under `+UF` and reaches `15.02` tie F1, but `setcausal` still remains at `0.00`. This means the current positive result is strongest on winner-vs-winner comparisons and no-tie agreement, while the full three-way decision problem is not yet solved.

The next experiments therefore target tie calibration directly. We are adding a tie-weighted training ablation to test whether the lack of `Tie` predictions is mainly a loss-balancing issue or a deeper consequence of the current scoring geometry.

## Interpretation

Taken together, the current results support a more precise claim than our original proposal. Architecture-level permutation handling is useful for open-ended pairwise judging, but only when the task is reformulated as symmetric per-candidate scoring rather than slot-labeled next-token prediction. Under that reformulation, the original `SetMask` remains too weak on human alignment, while `setcausal` appears to provide the better inductive bias for long responses.

This is a much stronger story than our earlier label-generation results. There, `setcausal` typically improved robustness at the cost of alignment, and the gap between theory and prompt design remained unclear. In the current scoring-based setting, `setcausal` improves both robustness and agreement simultaneously, which is the regime most worth pushing toward publication.

## Ongoing Extensions

The next batch of experiments is designed to answer three remaining questions:

1. Whether tie-weighted training can recover meaningful `Tie` behavior for `setcausal` without harming its current gains.
2. Whether the same `setcausal` advantage holds at a stronger backbone scale such as `Qwen/Qwen2.5-7B-Instruct`.
3. Whether the MT-Bench gains transfer to a second benchmark such as `LLMBar Natural`, which stresses judge robustness under misleading presentation cues.
