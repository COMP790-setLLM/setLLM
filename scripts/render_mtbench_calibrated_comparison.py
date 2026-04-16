#!/usr/bin/env python3
"""Render a compact calibrated MT-Bench comparison from per-variant calibration JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="MT-Bench human pairwise")
    parser.add_argument("--variant", action="append", default=[], help="NAME=PATH")
    parser.add_argument(
        "--fixed-calibration-set",
        default="",
        help="If provided, use this calibration set for every row instead of choosing the best row.",
    )
    parser.add_argument("--exclude-tie-only-main", action="store_true", default=False)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def parse_variant_items(items: list[str]) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --variant value: {item}")
        name, path = item.split("=", 1)
        variants.append((name.strip(), Path(path.strip())))
    return variants


def sort_key(row: dict) -> tuple[float, float, float, int]:
    prefer_clean = 1 if row["calibration_set"] == "unanimous_all" else 0
    return (
        row["eval_human_agreement"],
        row["eval_tie_f1"],
        row["eval_swap_consistency"],
        prefer_clean,
    )


def choose_main_row(rows: list[dict], exclude_tie_only_main: bool) -> dict:
    filtered = rows
    if exclude_tie_only_main:
        filtered = [row for row in rows if row["calibration_set"] != "unanimous_ties"]
        if not filtered:
            filtered = rows
    return max(filtered, key=sort_key)


def choose_fixed_row(rows: list[dict], calibration_set: str) -> dict:
    for row in rows:
        if row["calibration_set"] == calibration_set:
            return row
    available = ", ".join(sorted(row["calibration_set"] for row in rows))
    raise SystemExit(
        f"Missing calibration set '{calibration_set}'. Available sets: {available}"
    )


def render_main_table(
    variant_rows: list[tuple[str, dict]], *, fixed_calibration_set: str
) -> str:
    calibration_label = (
        "Calibration Set" if fixed_calibration_set else "Chosen Calibration Set"
    )
    lines = [
        f"| Variant | {calibration_label} | Threshold | Eval Agreement | Eval No-Tie | Eval Tie F1 | Eval Swap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in variant_rows:
        lines.append(
            "| `{variant}` | `{cal}` | `{thr:.4f}` | `{agr:.2f}` | `{no_tie:.2f}` | `{tie:.2f}` | `{swap:.2f}` |".format(
                variant=variant,
                cal=row["calibration_set"],
                thr=row["selected_threshold"],
                agr=100 * row["eval_human_agreement"],
                no_tie=100 * row["eval_no_tie_accuracy"],
                tie=100 * row["eval_tie_f1"],
                swap=100 * row["eval_swap_consistency"],
            )
        )
    return "\n".join(lines)


def render_ablation_table(rows: list[dict]) -> str:
    lines = [
        "| Calibration Set | Threshold | Eval Agreement | Eval No-Tie | Eval Tie F1 | Eval Swap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{cal}` | `{thr:.4f}` | `{agr:.2f}` | `{no_tie:.2f}` | `{tie:.2f}` | `{swap:.2f}` |".format(
                cal=row["calibration_set"],
                thr=row["selected_threshold"],
                agr=100 * row["eval_human_agreement"],
                no_tie=100 * row["eval_no_tie_accuracy"],
                tie=100 * row["eval_tie_f1"],
                swap=100 * row["eval_swap_consistency"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    variants = parse_variant_items(args.variant)
    if not variants:
        raise SystemExit("Provide at least one --variant NAME=PATH input.")

    payloads: list[tuple[str, list[dict]]] = []
    for name, path in variants:
        rows = json.loads(path.read_text())
        if not rows:
            raise SystemExit(f"{path} is empty.")
        payloads.append((name, rows))

    main_rows = []
    for name, rows in payloads:
        if args.fixed_calibration_set:
            main_rows.append((name, choose_fixed_row(rows, args.fixed_calibration_set)))
        else:
            main_rows.append((name, choose_main_row(rows, args.exclude_tie_only_main)))

    lines = [
        f"# {args.title}",
        "",
        "Model:",
        "",
        f"- `{args.model}`",
        "",
        "Dataset:",
        "",
        f"- `{args.dataset}`",
        "",
        "Protocol:",
        "",
        "- thresholds selected only from **train-derived clean calibration splits**",
        "- final numbers evaluated on untouched `MT-Bench eval`",
    ]
    if args.fixed_calibration_set:
        lines.append(
            f"- every main-row number uses the same fixed calibration split: `{args.fixed_calibration_set}`"
        )
    if args.exclude_tie_only_main:
        lines.append(
            "- primary comparison excludes the tiny `unanimous_ties` split from main-row selection"
        )
    lines.extend(
        [
            "",
            "## Main Calibrated Comparison",
            "",
            render_main_table(
                main_rows, fixed_calibration_set=args.fixed_calibration_set
            ),
        ]
    )

    for name, rows in payloads:
        lines.extend(
            [
                "",
                f"## Calibration Ablation: {name}",
                "",
                render_ablation_table(rows),
            ]
        )

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
