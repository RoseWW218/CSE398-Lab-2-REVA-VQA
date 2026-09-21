#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    # TODO(student): use regex with DOTALL, strip whitespace, and return a string.
    match = re.search(
        r"<answer>(.*?)</answer>",
        str(text),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return str(text).strip()


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    # TODO(student): normalize case and robustly parse A-H from short or verbose output.
    answer_text = extract_answer(str(text)).strip().upper()

    short_match = re.fullmatch(r"\s*([A-H])[\.\)]?\s*", answer_text)
    if short_match:
        return short_match.group(1)

    explicit_match = re.search(
        r"\b(?:FINAL\s+ANSWER|THE\s+ANSWER|ANSWER)"
        r"\s*(?:IS\s*|:\s*)?([A-H])\b",
        answer_text,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        return explicit_match.group(1).upper()

    return ""


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    # TODO(student): read every prediction *.json line-by-line and index by item["id"].
    predictions: dict[str, dict[str, Any]] = {}

    for prediction_path in sorted(output_dir.glob("*.json")):
        if prediction_path.name == "result.json":
            continue

        with prediction_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                predictions[str(item["id"])] = item

    return predictions


def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    # TODO(student): compare predictions against every GT item, count missing predictions
    # as incorrect, and report Completed plus Total/subcategory accuracy statistics.
    results: dict[str, dict[str, Any]] = {}
    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    completed = 0
    total_correct = 0
    total_questions = len(gt_list)

    for gt_item in gt_list:
        item_id = str(gt_item["id"])
        correct_letter = str(gt_item["answer"]).strip().upper()
        question_type = str(gt_item.get("question_type", "unknown"))

        prediction_item = preds.get(item_id)
        prediction_text = ""
        prediction_letter = ""

        if prediction_item is not None:
            completed += 1
            prediction_text = str(prediction_item.get("pred", ""))
            prediction_letter = extract_letter(prediction_text)

        is_correct = int(prediction_letter == correct_letter)

        total_correct += is_correct
        category_stats[question_type]["correct"] += is_correct
        category_stats[question_type]["total"] += 1

        results[item_id] = {
            **gt_item,
            "pred": prediction_text,
            "pred_letter": prediction_letter,
            "acc": is_correct,
        }

    completed_percentage = (
        100.0 * completed / total_questions if total_questions else 0.0
    )
    total_percentage = (
        100.0 * total_correct / total_questions if total_questions else 0.0
    )

    summary_lines = [
        (
            f"Completed: {completed}/{total_questions} = "
            f"{completed_percentage:.2f}%"
        ),
        (
            f"Total: {total_correct}/{total_questions} = "
            f"{total_percentage:.2f}%"
        ),
    ]

    for question_type in sorted(category_stats):
        correct = category_stats[question_type]["correct"]
        total = category_stats[question_type]["total"]
        percentage = 100.0 * correct / total if total else 0.0
        summary_lines.append(
            f"{question_type}: {correct}/{total} = {percentage:.2f}%"
        )

    csv_text = "\n".join(summary_lines) + "\n"
    return results, csv_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
