#!/usr/bin/env python3
"""Build a clean held-out calibration split from MT-Bench pairwise train JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument(
        "--selection",
        choices=("unanimous", "majority"),
        default="majority",
        help="Vote rule used to define a clean aggregated label.",
    )
    parser.add_argument(
        "--label-filter",
        choices=("all", "ties_only", "non_ties_only"),
        default="all",
        help="Optional label filter after aggregation.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum number of human judgments required per aggregated group.",
    )
    parser.add_argument(
        "--calibration-question-fraction",
        type=float,
        default=0.2,
        help="Fraction of clean question ids held out for calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    return "\n".join(lines)


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def canonicalize_row(row: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any], str]:
    prompt = normalize_text(row["prompt"])
    response_a = normalize_text(row["response_a"])
    response_b = normalize_text(row["response_b"])
    model_a = str(row.get("model_a", ""))
    model_b = str(row.get("model_b", ""))
    label = str(row["label"])

    candidate_a = {
        "model": model_a,
        "response": response_a,
        "hash": text_hash(response_a),
        "slot": "A",
    }
    candidate_b = {
        "model": model_b,
        "response": response_b,
        "hash": text_hash(response_b),
        "slot": "B",
    }
    ordered_candidates = sorted(
        (candidate_a, candidate_b),
        key=lambda candidate: (candidate["model"], candidate["hash"]),
    )
    slot_to_canonical = {
        ordered_candidates[0]["slot"]: "A",
        ordered_candidates[1]["slot"]: "B",
    }
    if label == "Tie":
        canonical_label = "Tie"
    else:
        canonical_label = slot_to_canonical[label]

    canonical_row = {
        "prompt": prompt,
        "response_a": ordered_candidates[0]["response"],
        "response_b": ordered_candidates[1]["response"],
        "model_a": ordered_candidates[0]["model"],
        "model_b": ordered_candidates[1]["model"],
        "question_id": row["question_id"],
        "turn": row.get("turn"),
        "source": row.get("source"),
        "source_split": row.get("source_split"),
    }
    key = (
        str(row["question_id"]),
        str(row.get("turn")),
        prompt,
        ordered_candidates[0]["model"],
        ordered_candidates[0]["hash"],
        ordered_candidates[1]["model"],
        ordered_candidates[1]["hash"],
    )
    return key, canonical_row, canonical_label


def passes_selection(
    counts: Counter[str], selection: str, min_votes: int
) -> tuple[bool, str | None, str | None]:
    total_votes = sum(counts.values())
    if total_votes < min_votes:
        return False, None, None
    most_common = counts.most_common()
    label, label_votes = most_common[0]

    if selection == "unanimous":
        if len(counts) == 1:
            return True, label, "unanimous"
        return False, None, None

    if label_votes > total_votes / 2:
        agreement_type = "unanimous" if len(counts) == 1 else "majority"
        return True, label, agreement_type
    return False, None, None


def label_allowed(label: str, label_filter: str) -> bool:
    if label_filter == "all":
        return True
    if label_filter == "ties_only":
        return label == "Tie"
    if label_filter == "non_ties_only":
        return label != "Tie"
    raise ValueError(f"Unsupported label_filter: {label_filter}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    stats_path = Path(args.stats_json)

    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    with input_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key, canonical_row, canonical_label = canonicalize_row(row)
            grouped_rows[key].append(
                {
                    "canonical_row": canonical_row,
                    "canonical_label": canonical_label,
                    "judge": row.get("judge"),
                    "original_label": row["label"],
                    "original_model_a": row.get("model_a"),
                    "original_model_b": row.get("model_b"),
                }
            )

    aggregated_rows: list[dict[str, Any]] = []
    raw_vote_counts = Counter()
    selected_vote_counts = Counter()
    agreement_type_counts = Counter()

    for group_entries in grouped_rows.values():
        canonical_row = group_entries[0]["canonical_row"]
        vote_counts = Counter(entry["canonical_label"] for entry in group_entries)
        raw_vote_counts.update(vote_counts)

        keep, aggregated_label, agreement_type = passes_selection(
            vote_counts, selection=args.selection, min_votes=args.min_votes
        )
        if not keep or aggregated_label is None or agreement_type is None:
            continue
        if not label_allowed(aggregated_label, args.label_filter):
            continue

        selected_vote_counts[aggregated_label] += 1
        agreement_type_counts[agreement_type] += 1
        total_votes = sum(vote_counts.values())
        majority_votes = vote_counts[aggregated_label]

        aggregated_rows.append(
            {
                **canonical_row,
                "label": aggregated_label,
                "num_votes": total_votes,
                "vote_counts": dict(vote_counts),
                "majority_fraction": majority_votes / total_votes,
                "agreement_type": agreement_type,
                "judges": sorted(
                    str(entry["judge"]) for entry in group_entries if entry["judge"] is not None
                ),
                "calibration_source_rows": len(group_entries),
            }
        )

    question_ids = sorted({str(row["question_id"]) for row in aggregated_rows})
    rng = random.Random(args.seed)
    rng.shuffle(question_ids)
    calibration_count = max(
        1,
        int(round(len(question_ids) * args.calibration_question_fraction)),
    ) if question_ids else 0
    calibration_question_ids = set(question_ids[:calibration_count])

    calibration_rows = [
        row for row in aggregated_rows if str(row["question_id"]) in calibration_question_ids
    ]
    calibration_rows.sort(key=lambda row: (str(row["question_id"]), str(row.get("turn"))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, calibration_rows)

    stats = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "selection": args.selection,
        "label_filter": args.label_filter,
        "min_votes": args.min_votes,
        "seed": args.seed,
        "calibration_question_fraction": args.calibration_question_fraction,
        "num_raw_rows": sum(len(entries) for entries in grouped_rows.values()),
        "num_grouped_pairs": len(grouped_rows),
        "num_clean_pairs": len(aggregated_rows),
        "num_clean_questions": len(question_ids),
        "num_calibration_rows": len(calibration_rows),
        "num_calibration_questions": len(calibration_question_ids),
        "raw_vote_label_counts": dict(raw_vote_counts),
        "selected_label_counts": dict(selected_vote_counts),
        "agreement_type_counts": dict(agreement_type_counts),
        "calibration_question_ids": sorted(calibration_question_ids),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
