#!/usr/bin/env python3
"""Select tie thresholds on multiple calibration sets and evaluate on a fixed eval split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-predictions-jsonl", required=True)
    parser.add_argument(
        "--calibration",
        action="append",
        default=[],
        help="Calibration prediction file in NAME=PATH format. May be passed multiple times.",
    )
    parser.add_argument("--prompt-style", default="anonymous_slots")
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=41)
    parser.add_argument(
        "--objective",
        choices=("human_agreement", "tie_f1", "tie_aware"),
        default="tie_aware",
    )
    parser.add_argument(
        "--tie-f1-weight",
        type=float,
        default=0.5,
        help="Weight used in tie_aware objective: human_agreement + w * tie_f1.",
    )
    parser.add_argument(
        "--agreement-floor",
        type=float,
        default=None,
        help="Optional absolute minimum calibration agreement for eligible thresholds.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).open()]
    if not rows:
        raise SystemExit(f"{path} is empty.")
    for index, row in enumerate(rows):
        if "gold" not in row:
            raise SystemExit(f"{path} row {index} is missing required field 'gold'.")
        has_scores = "score_diff" in row and "swapped_score_diff" in row
        has_fixed_preds = "pred" in row and "pred_swapped" in row
        if not has_scores and not has_fixed_preds:
            raise SystemExit(
                f"{path} row {index} is missing both score fields and saved predictions."
            )
    return rows


def swap_label(label: str, prompt_style: str) -> str:
    if prompt_style == "anonymous_slots":
        return {"1": "2", "2": "1", "Tie": "Tie"}[label]
    return {"A": "B", "B": "A", "Tie": "Tie"}[label]


def predict_from_diff(diff: float, threshold: float, prompt_style: str) -> str:
    if abs(diff) <= threshold:
        return "Tie"
    if diff > 0:
        return "1" if prompt_style == "anonymous_slots" else "A"
    return "2" if prompt_style == "anonymous_slots" else "B"


def row_predictions(
    row: dict[str, Any], threshold: float, prompt_style: str
) -> tuple[str, str]:
    if "score_diff" in row and "swapped_score_diff" in row:
        pred = predict_from_diff(float(row["score_diff"]), threshold, prompt_style)
        pred_swapped = predict_from_diff(
            float(row["swapped_score_diff"]), threshold, prompt_style
        )
        return pred, pred_swapped
    return str(row["pred"]), str(row["pred_swapped"])


def summarize(rows: list[dict[str, Any]], threshold: float, prompt_style: str) -> dict[str, float]:
    total = len(rows)
    correct = 0
    no_tie_total = 0
    no_tie_correct = 0
    swap_consistent = 0
    tie_tp = 0
    tie_fp = 0
    tie_fn = 0

    for row in rows:
        pred, pred_swapped = row_predictions(
            row, threshold=threshold, prompt_style=prompt_style
        )
        gold = str(row["gold"])

        correct += pred == gold
        if gold != "Tie":
            no_tie_total += 1
            no_tie_correct += pred == gold
        swap_consistent += pred_swapped == swap_label(pred, prompt_style)

        if pred == "Tie" and gold == "Tie":
            tie_tp += 1
        elif pred == "Tie" and gold != "Tie":
            tie_fp += 1
        elif pred != "Tie" and gold == "Tie":
            tie_fn += 1

    tie_precision = tie_tp / max(1, tie_tp + tie_fp)
    tie_recall = tie_tp / max(1, tie_tp + tie_fn)
    tie_f1 = (
        0.0
        if tie_precision + tie_recall == 0
        else 2 * tie_precision * tie_recall / (tie_precision + tie_recall)
    )

    return {
        "threshold": threshold,
        "num_rows": total,
        "human_agreement": correct / max(1, total),
        "no_tie_accuracy": no_tie_correct / max(1, no_tie_total),
        "tie_f1": tie_f1,
        "swap_consistency": swap_consistent / max(1, total),
        "tie_precision": tie_precision,
        "tie_recall": tie_recall,
    }


def objective_value(summary_row: dict[str, float], objective: str, tie_f1_weight: float) -> float:
    if objective == "human_agreement":
        return summary_row["human_agreement"]
    if objective == "tie_f1":
        return summary_row["tie_f1"]
    if objective == "tie_aware":
        return summary_row["human_agreement"] + tie_f1_weight * summary_row["tie_f1"]
    raise ValueError(f"Unsupported objective: {objective}")


def choose_threshold(
    rows: list[dict[str, Any]],
    prompt_style: str,
    min_threshold: float,
    max_threshold: float,
    num_steps: int,
    objective: str,
    tie_f1_weight: float,
    agreement_floor: float | None,
) -> dict[str, float]:
    if num_steps < 2:
        thresholds = [min_threshold]
    else:
        step = (max_threshold - min_threshold) / (num_steps - 1)
        thresholds = [min_threshold + i * step for i in range(num_steps)]

    candidates = []
    for threshold in thresholds:
        summary_row = summarize(rows, threshold=threshold, prompt_style=prompt_style)
        score = objective_value(summary_row, objective=objective, tie_f1_weight=tie_f1_weight)
        if agreement_floor is not None and summary_row["human_agreement"] < agreement_floor:
            continue
        candidates.append((score, summary_row))

    if not candidates:
        raise SystemExit("No thresholds satisfied the requested calibration constraints.")

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]["human_agreement"],
            item[1]["tie_f1"],
            item[1]["swap_consistency"],
            -item[1]["threshold"],
        ),
        reverse=True,
    )
    best_score, best_summary = candidates[0]
    return {**best_summary, "objective_value": best_score}


def parse_named_paths(items: list[str]) -> list[tuple[str, str]]:
    named_paths = []
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --calibration value '{item}'. Expected NAME=PATH."
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(
                f"Invalid --calibration value '{item}'. Expected NAME=PATH."
            )
        named_paths.append((name, path))
    return named_paths


def render_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Calibration Set | Threshold | Calib Rows | Calib Agreement | Calib Tie F1 | "
        "Eval Agreement | Eval No-Tie | Eval Tie F1 | Eval Swap |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = []
    for row in rows:
        body.append(
            "| {name} | {threshold:.4f} | {calib_rows} | {calib_agreement:.2f} | {calib_tie_f1:.2f} | "
            "{eval_agreement:.2f} | {eval_no_tie:.2f} | {eval_tie_f1:.2f} | {eval_swap:.2f} |".format(
                name=row["calibration_set"],
                threshold=row["selected_threshold"],
                calib_rows=row["calibration_num_rows"],
                calib_agreement=100 * row["calibration_human_agreement"],
                calib_tie_f1=100 * row["calibration_tie_f1"],
                eval_agreement=100 * row["eval_human_agreement"],
                eval_no_tie=100 * row["eval_no_tie_accuracy"],
                eval_tie_f1=100 * row["eval_tie_f1"],
                eval_swap=100 * row["eval_swap_consistency"],
            )
        )
    return "\n".join([header, *body]) + "\n"


def main() -> None:
    args = parse_args()
    calibration_specs = parse_named_paths(args.calibration)
    if not calibration_specs:
        raise SystemExit("Provide at least one --calibration NAME=PATH input.")

    eval_rows = load_rows(args.eval_predictions_jsonl)
    results = []

    for calibration_name, calibration_path in calibration_specs:
        calibration_rows = load_rows(calibration_path)
        selected = choose_threshold(
            calibration_rows,
            prompt_style=args.prompt_style,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            num_steps=args.num_steps,
            objective=args.objective,
            tie_f1_weight=args.tie_f1_weight,
            agreement_floor=args.agreement_floor,
        )
        eval_summary = summarize(
            eval_rows,
            threshold=selected["threshold"],
            prompt_style=args.prompt_style,
        )
        results.append(
            {
                "calibration_set": calibration_name,
                "calibration_predictions_jsonl": str(calibration_path),
                "selected_threshold": selected["threshold"],
                "selection_objective": args.objective,
                "selection_objective_value": selected["objective_value"],
                "calibration_num_rows": int(selected["num_rows"]),
                "calibration_human_agreement": selected["human_agreement"],
                "calibration_no_tie_accuracy": selected["no_tie_accuracy"],
                "calibration_tie_f1": selected["tie_f1"],
                "calibration_swap_consistency": selected["swap_consistency"],
                "eval_num_rows": int(eval_summary["num_rows"]),
                "eval_human_agreement": eval_summary["human_agreement"],
                "eval_no_tie_accuracy": eval_summary["no_tie_accuracy"],
                "eval_tie_f1": eval_summary["tie_f1"],
                "eval_swap_consistency": eval_summary["swap_consistency"],
            }
        )

    results.sort(
        key=lambda row: (
            row["eval_human_agreement"],
            row["eval_tie_f1"],
            row["eval_swap_consistency"],
        ),
        reverse=True,
    )

    for row in results:
        print(json.dumps(row))

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(results, indent=2) + "\n")

    if args.output_markdown:
        output_md_path = Path(args.output_markdown)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(render_markdown(results))


if __name__ == "__main__":
    main()
