#!/usr/bin/env python3
"""Generate paper-ready markdown tables for pairwise judge experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="MT-Bench human pairwise")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--architectures", nargs="+", default=["vanilla", "setllm", "setcausal"])
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--readout", required=True)
    parser.add_argument("--prompt-style", required=True)
    parser.add_argument("--uf", choices=["yes", "no"], default="no")
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def compute_split_row(split_name: str, rows: list[dict], unique_questions: int | None = None) -> dict:
    labels = Counter(row["label"] for row in rows)
    if unique_questions is None:
        unique_questions = len({row["question_id"] for row in rows})
    return {
        "split": split_name,
        "rows": len(rows),
        "unique_questions": unique_questions,
        "A": labels.get("A", 0),
        "B": labels.get("B", 0),
        "Tie": labels.get("Tie", 0),
        "A_pct": pct(labels.get("A", 0), len(rows)),
        "B_pct": pct(labels.get("B", 0), len(rows)),
        "Tie_pct": pct(labels.get("Tie", 0), len(rows)),
    }


def metric_value(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{100.0 * float(value):.2f}"


def group_key(row: dict) -> tuple:
    return (
        row.get("question_id"),
        row.get("turn"),
        row.get("model_a"),
        row.get("model_b"),
    )


def majority_label(labels: list[str]) -> str | None:
    counts = Counter(labels)
    if not counts:
        return None
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def compute_group_agreements(eval_rows: list[dict], prediction_rows: list[dict]) -> dict[str, float]:
    human_groups: dict[tuple, list[str]] = defaultdict(list)
    for row in eval_rows:
        human_groups[group_key(row)].append(row["label"])

    pred_groups: dict[tuple, list[str]] = defaultdict(list)
    swap_consistency_values: dict[tuple, list[bool]] = defaultdict(list)
    for row in prediction_rows:
        pred_groups[group_key(row)].append(row["pred_canonical"])
        swap_consistency_values[group_key(row)].append(bool(row["swap_consistent"]))

    majority_total = 0
    majority_hits = 0
    unanimous_total = 0
    unanimous_hits = 0
    grouped_swap = []

    for key, labels in human_groups.items():
        if key not in pred_groups:
            continue
        pred_label = Counter(pred_groups[key]).most_common(1)[0][0]
        grouped_swap.append(sum(swap_consistency_values[key]) / len(swap_consistency_values[key]))

        majority = majority_label(labels)
        if majority is not None:
            majority_total += 1
            majority_hits += int(pred_label == majority)

        if len(set(labels)) == 1:
            unanimous_total += 1
            unanimous_hits += int(pred_label == labels[0])

    return {
        "majority_vote_agreement": 0.0 if majority_total == 0 else majority_hits / majority_total,
        "unanimous_only_agreement": 0.0 if unanimous_total == 0 else unanimous_hits / unanimous_total,
        "group_swap_consistency": 0.0 if not grouped_swap else sum(grouped_swap) / len(grouped_swap),
        "majority_groups": majority_total,
        "unanimous_groups": unanimous_total,
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(Path(args.train_jsonl))
    eval_rows = load_jsonl(Path(args.eval_jsonl))
    stats = json.loads(Path(args.stats_json).read_text())

    split_rows = [
        compute_split_row("Train", train_rows),
        compute_split_row("Eval", eval_rows),
        compute_split_row(
            "Total",
            train_rows + eval_rows,
            stats.get("num_unique_questions", len({r["question_id"] for r in train_rows + eval_rows})),
        ),
    ]

    table1 = md_table(
        ["Dataset", "Split", "Rows", "Unique Questions", "A", "B", "Tie", "A%", "B%", "Tie%"],
        [
            [
                args.dataset_name if row["split"] == "Train" else "",
                row["split"],
                str(row["rows"]),
                str(row["unique_questions"]),
                str(row["A"]),
                str(row["B"]),
                str(row["Tie"]),
                f"{row['A_pct']:.2f}",
                f"{row['B_pct']:.2f}",
                f"{row['Tie_pct']:.2f}",
            ]
            for row in split_rows
        ],
    )

    main_rows = []
    rigorous_rows = []
    for architecture in args.architectures:
        run_dir = Path(args.results_root) / architecture
        metrics_path = run_dir / "judge_pairwise-metrics.json"
        predictions_path = run_dir / "judge_pairwise-predictions.jsonl"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        extra = {}
        if predictions_path.exists():
            prediction_rows = load_jsonl(predictions_path)
            extra = compute_group_agreements(eval_rows, prediction_rows)

        main_rows.append(
            [
                args.backbone,
                args.readout,
                architecture,
                args.uf,
                metric_value(metrics, "human_agreement"),
                metric_value(metrics, "no_tie_accuracy"),
                metric_value(metrics, "tie_f1"),
                metric_value(metrics, "swap_consistency"),
                metric_value(metrics, "first_position_win_rate"),
            ]
        )
        rigorous_rows.append(
            [
                args.backbone,
                args.readout,
                architecture,
                args.uf,
                metric_value(metrics, "human_agreement"),
                "-" if not extra else f"{100.0 * extra['majority_vote_agreement']:.2f}",
                "-" if not extra else f"{100.0 * extra['unanimous_only_agreement']:.2f}",
                metric_value(metrics, "swap_consistency"),
            ]
        )

    table2 = md_table(
        ["Backbone", "Readout", "Arch", "UF", "Human Agreement", "No-Tie Agreement", "Tie F1", "Swap Consistency", "First-Pos Win Rate"],
        main_rows,
    )
    table3 = md_table(
        ["Backbone", "Readout", "Arch", "UF", "Row-Level Agreement", "Majority-Vote Agreement", "Unanimous-Only Agreement", "Swap Consistency"],
        rigorous_rows,
    )

    output = "\n\n".join(
        [
            "# Judge Experiment Tables",
            "## Table 1. Dataset Statistics",
            table1,
            "## Table 2. Main Results",
            table2,
            "## Table 3. Rigorous Human-Alignment Results",
            table3,
        ]
    )
    output_path = Path(args.output_md)
    output_path.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
