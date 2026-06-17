#!/usr/bin/env python3
"""Prepare pairwise judge JSONL data from public benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["mt_bench_human", "arena_human_55k", "llmbar_natural", "faireval"],
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
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional benchmark repo root for sources stored as local JSON files.",
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


def parse_jsonish_turns(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [""]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
        return [str(parsed).strip()]
    return [str(value).strip()]


def render_turn_list(prefix: str, turns: list[str]) -> str:
    return "\n\n".join(
        f"{prefix} turn {turn_index}:\n{text}"
        for turn_index, text in enumerate(turns, start=1)
    )


def label_from_arena_flags(row: dict) -> str:
    winner_a = int(row["winner_model_a"])
    winner_b = int(row["winner_model_b"])
    winner_tie = int(row["winner_tie"])
    total = winner_a + winner_b + winner_tie
    if total != 1:
        raise ValueError(f"Arena row has invalid winner flags: {row}")
    if winner_a:
        return "A"
    if winner_b:
        return "B"
    return "Tie"


def convert_arena_row(row: dict) -> dict:
    prompt_turns = parse_jsonish_turns(row["prompt"])
    response_a_turns = parse_jsonish_turns(row["response_a"])
    response_b_turns = parse_jsonish_turns(row["response_b"])
    prompt_text = render_turn_list("User", prompt_turns)
    return {
        "prompt": prompt_text,
        "response_a": render_turn_list("Assistant", response_a_turns),
        "response_b": render_turn_list("Assistant", response_b_turns),
        "label": label_from_arena_flags(row),
        "question_id": f"arena-{row['id']}",
        "turn": len(prompt_turns),
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "source": "lmsys/lmsys-arena-human-preference-55k",
        "source_split": row.get("source_split", "train"),
        "prompt_key": json.dumps(prompt_turns, ensure_ascii=False),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def convert_local_pair_row(row: dict, *, source: str, index: int) -> dict:
    label_value = row["label"]
    if label_value not in (1, 2):
        raise ValueError(f"Unsupported pairwise label: {label_value}")
    return {
        "prompt": row["input"].strip(),
        "response_a": row["output_1"].strip(),
        "response_b": row["output_2"].strip(),
        "label": "A" if label_value == 1 else "B",
        "question_id": f"{source}-{index}",
        "source": source,
    }


def load_local_json_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_row_split(
    rows: list[dict], eval_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_count = max(1, int(len(rows) * eval_fraction))
    eval_indices = set(indices[:eval_count])
    train_rows = [row for idx, row in enumerate(rows) if idx not in eval_indices]
    eval_rows = [row for idx, row in enumerate(rows) if idx in eval_indices]
    return train_rows, eval_rows


def main() -> None:
    args = parse_args()
    if args.source in {"mt_bench_human", "arena_human_55k"}:
        if args.source == "mt_bench_human":
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
            num_unique_questions = len(question_ids)
            num_eval_questions = len(eval_question_ids)
        else:
            dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
            converted = [convert_arena_row(dict(row)) for row in dataset]
            prompt_keys = sorted({row["prompt_key"] for row in converted})
            rng = random.Random(args.seed)
            rng.shuffle(prompt_keys)
            eval_count = max(1, int(len(prompt_keys) * args.eval_question_fraction))
            eval_prompt_keys = set(prompt_keys[:eval_count])
            train_rows = []
            eval_rows = []
            for row in converted:
                target = eval_rows if row["prompt_key"] in eval_prompt_keys else train_rows
                row = dict(row)
                row.pop("prompt_key", None)
                target.append(row)
            num_unique_questions = len(prompt_keys)
            num_eval_questions = len(eval_prompt_keys)
    else:
        if args.repo_root is None:
            raise RuntimeError(f"{args.source} requires --repo-root")
        repo_root = Path(args.repo_root)
        if args.source == "llmbar_natural":
            dataset_path = repo_root / "Dataset" / "LLMBar" / "Natural" / "dataset.json"
            source_name = "princeton-nlp/LLMBar/Natural"
        elif args.source == "faireval":
            dataset_path = (
                repo_root / "Dataset" / "Processed" / "FairEval" / "dataset.json"
            )
            source_name = "princeton-nlp/LLMBar/Processed/FairEval"
        else:
            raise RuntimeError(f"Unsupported source: {args.source}")

        raw_rows = load_local_json_dataset(dataset_path)
        converted = [
            convert_local_pair_row(row, source=source_name, index=index)
            for index, row in enumerate(raw_rows)
        ]
        train_rows, eval_rows = build_row_split(
            converted, args.eval_question_fraction, args.seed
        )
        num_unique_questions = len(converted)
        num_eval_questions = len(eval_rows)

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
        "num_unique_questions": num_unique_questions,
        "num_eval_questions": num_eval_questions,
        "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"train_jsonl={train_path}")
    print(f"eval_jsonl={eval_path}")


if __name__ == "__main__":
    main()
