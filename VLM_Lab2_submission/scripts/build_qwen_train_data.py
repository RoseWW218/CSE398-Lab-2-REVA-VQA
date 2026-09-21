#!/usr/bin/env python3
"""Convert ReVA annotations into Qwen-VL conversation training data.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_video_path(file_path: str, video_root: Path | None = None) -> str:
    path = Path(file_path)
    if video_root is None:
        return path.as_posix()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(video_root).as_posix()
    except ValueError:
        return path.as_posix()


def format_options(options: dict[str, str]) -> str:
    """Return sorted multiple-choice options, one option per line."""
    # TODO(student): sort option labels and format each line as "A. option text".
    return "\n".join(
        f"{label}. {options[label]}"
        for label in sorted(options)
    )


def format_question(question: str, options: dict[str, str], prompt_style: str) -> str:
    """Build the human message used by Qwen SFT."""
    # TODO(student): call format_options and support "plain" and "reva_eval" styles.
    formatted_options = format_options(options)

    if prompt_style == "plain":
        return (
            f"<video>\n"
            f"Question: {question}\n"
            f"{formatted_options}"
        )

    if prompt_style == "reva_eval":
        return (
            "<video>\n"
            "Please carefully watch the video and answer the multiple-choice question below.\n"
            "Think step-by-step within <think> </think> tags, then provide only the letter "
            "of the correct option within <answer> </answer> tags.\n"
            f"Question: {question}\n"
            f"{formatted_options}"
        )

    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def format_answer(qa: dict[str, Any], answer_style: str) -> str:
    """Format the assistant target from a ReVA QA item."""
    # TODO(student): support "letter", "tagged", and "cot_tagged" answer styles.
    answer = str(qa["correct_answer"]).strip().upper()

    if answer_style == "letter":
        return f"Answer: {answer}"

    if answer_style == "tagged":
        return f"<answer>{answer}</answer>"

    if answer_style == "cot_tagged":
        reasoning = str(qa.get("reasoning", "")).strip()
        return f"<think>{reasoning}</think><answer>{answer}</answer>"

    raise ValueError(f"Unsupported answer style: {answer_style}")


def iter_reva_qas(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one flat QA record at a time from nested ReVA annotations."""
    # TODO(student): iterate videos -> mcq category -> question_type -> qa list.
    for video_id, video_info in data.get("videos", {}).items():
        file_path = video_info["file_path"]
        dataset_name = video_info.get("dataset_name")

        for category, question_types in video_info.get("mcq", {}).items():
            for question_type, qa_list in question_types.items():
                for qa in qa_list:
                    yield {
                        **qa,
                        "video_id": video_id,
                        "file_path": file_path,
                        "dataset_name": dataset_name,
                        "category": category,
                        "question_type": question_type,
                    }


def convert_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Convert one flat ReVA QA item into one Qwen conversation sample."""
    # TODO(student): normalize video path, optionally check video exists,
    # build question/answer, return {video, conversations, optional metadata}.
    video_path = normalize_video_path(
        item["file_path"],
        Path(args.video_root),
    )

    resolved_video_path = Path(video_path)
    if not resolved_video_path.is_absolute():
        resolved_video_path = Path(args.video_root) / resolved_video_path

    if args.require_video and not resolved_video_path.exists():
        return None

    question = format_question(
        item["question"],
        item["options"],
        args.prompt_style,
    )
    answer = format_answer(item, args.answer_style)

    sample = {
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {
                "from": "gpt",
                "value": answer,
            },
        ],
    }

    if args.keep_metadata:
        sample["metadata"] = {
            "qa_id": item.get("qa_id"),
            "video_id": item["video_id"],
            "question_type": item["question_type"],
            "category": item["category"],
            "correct_answer": str(item["correct_answer"]).strip().upper(),
        }

    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reva_train/train_set.json"))
    parser.add_argument("--output", type=Path, default=Path("data/qwen_train/train.json"))
    parser.add_argument("--video-root", type=Path, default=Path("data/qwen_train"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all.")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--prompt-style", choices=["plain", "reva_eval"], default="reva_eval")
    parser.add_argument("--answer-style", choices=["letter", "tagged", "cot_tagged"], default="tagged")
    parser.add_argument("--keep-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)

    samples = []
    seen_videos = set()
    skipped_missing_video = 0
    for item in iter_reva_qas(data):
        seen_videos.add(item["video_id"])
        if args.max_videos and len(seen_videos) > args.max_videos:
            break
        sample = convert_item(item, args)
        if sample is None:
            skipped_missing_video += 1
            continue
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    if args.require_video and not samples:
        raise SystemExit(
            "No training samples reference an existing video. "
            "Check QWEN_VIDEO_ROOT and the downloaded ReVA directory; output was not modified."
        )

    dump_json(samples, args.output)
    print(f"Saved {len(samples)} Qwen training samples to {args.output}")
    if skipped_missing_video:
        print(f"Skipped {skipped_missing_video} samples with missing videos")


if __name__ == "__main__":
    main()
