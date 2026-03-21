#!/usr/bin/env python3
"""Prepare pairwise judge JSONL data from public benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["mt_bench_human"],
        default="mt_bench_human",
        help="Source dataset to convert into pairwise judge JSONL.",
    )
    parser.add_argument(
        "--split-name",
        default="human",
        help="Source split name inside the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mt_bench_human_pairwise",
        help="Directory for generated train/eval JSONL files.",
    )
    parser.add_argument(
        "--eval-question-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique question ids reserved for eval.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def label_from_winner(winner: str) -> str:
    mapping = {
        "model_a": "A",
        "model_b": "B",
        "tie": "Tie",
    }
    try:
        return mapping[winner]
    except KeyError as exc:
        raise ValueError(f"Unsupported winner label: {winner}") from exc


def render_user_prompt(conversation: list[dict]) -> str:
    user_turns = [
        item["content"].strip() for item in conversation if item["role"] == "user"
    ]
    return "\n\n".join(
        f"User turn {turn_index}:\n{text}"
        for turn_index, text in enumerate(user_turns, start=1)
    )


def render_assistant_trace(conversation: list[dict]) -> str:
    assistant_turns = [
        item["content"].strip() for item in conversation if item["role"] == "assistant"
    ]
    return "\n\n".join(
        f"Assistant turn {turn_index}:\n{text}"
        for turn_index, text in enumerate(assistant_turns, start=1)
    )


def convert_mt_bench_row(row: dict) -> dict:
    return {
        "prompt": render_user_prompt(row["conversation_a"]),
        "response_a": render_assistant_trace(row["conversation_a"]),
        "response_b": render_assistant_trace(row["conversation_b"]),
        "label": label_from_winner(row["winner"]),
        "question_id": row["question_id"],
        "turn": row["turn"],
        "judge": row["judge"],
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "source": "lmsys/mt_bench_human_judgments",
        "source_split": row.get("source_split", "human"),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if args.source != "mt_bench_human":
        raise RuntimeError(f"Unsupported source: {args.source}")

    dataset = load_dataset("lmsys/mt_bench_human_judgments", split=args.split_name)
    converted = []
    question_ids = sorted(set(dataset["question_id"]))

    rng = random.Random(args.seed)
    rng.shuffle(question_ids)
    eval_count = max(1, int(len(question_ids) * args.eval_question_fraction))
    eval_question_ids = set(question_ids[:eval_count])

    train_rows = []
    eval_rows = []
    for row in dataset:
        converted_row = convert_mt_bench_row(dict(row))
        converted.append(converted_row)
        target = (
            eval_rows
            if converted_row["question_id"] in eval_question_ids
            else train_rows
        )
        target.append(converted_row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    stats_path = output_dir / "stats.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    stats = {
        "source": args.source,
        "split_name": args.split_name,
        "num_rows": len(converted),
        "num_train_rows": len(train_rows),
        "num_eval_rows": len(eval_rows),
        "num_unique_questions": len(question_ids),
        "num_eval_questions": len(eval_question_ids),
        "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"train_jsonl={train_path}")
    print(f"eval_jsonl={eval_path}")


if __name__ == "__main__":
    main()
