#!/usr/bin/env python3
"""Build a lightweight UltraFeedback-style SFT JSONL.

This is an approximation of the cleaned UltraFeedback subset referenced by
Set-LLM. The exact preprocessing artifact was not released, so we construct a
simple instruction-answer dataset by selecting the highest-scoring completion
per instruction from the public `openbmb/UltraFeedback` train split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where to write rows like {'instruction': ..., 'answer': ...}.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10_000,
        help="Maximum number of cleaned SFT rows to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed before selecting rows.",
    )
    return parser.parse_args()


def completion_score(completion: dict) -> tuple[float, float]:
    overall = completion.get("overall_score")
    fine = completion.get("fine-grained_score")
    overall_score = float(overall) if overall is not None else float("-inf")
    fine_score = float(fine) if fine is not None else float("-inf")
    return overall_score, fine_score


def choose_best_answer(row: dict) -> str | None:
    completions = row.get("completions") or []
    if not completions:
        return None
    best = max(completions, key=completion_score)
    response = (best.get("response") or "").strip()
    return response or None


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    dataset = dataset.shuffle(seed=args.seed)

    written = 0
    with output_path.open("w") as handle:
        for row in dataset:
            instruction = (row.get("instruction") or "").strip()
            answer = choose_best_answer(row)
            if not instruction or not answer:
                continue
            handle.write(
                json.dumps(
                    {"instruction": instruction, "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written >= args.max_rows:
                break

    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "rows_written": written,
                "max_rows": args.max_rows,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
