import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()


def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""
    # TODO(student): read question_file, flatten instances, and assign stable IDs
    # using qa_id, then global_index, then video/subcategory/question index.
    with open(question_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    instances: list[dict[str, Any]] = []

    for video_id, video_item in data.get("videos", {}).items():
        video_path = str(video_item["file_path"])
        dataset_name = video_item.get("dataset_name")

        video_stem = Path(
            video_path.replace("\\", "/")
        ).stem

        for category, category_data in video_item.get("mcq", {}).items():
            for subcategory, qa_list in category_data.items():
                subcategory_code = "".join(
                    character
                    for character in subcategory.upper()
                    if character.isalnum()
                )[:3]

                for question_index, qa in enumerate(qa_list):
                    if qa.get("qa_id"):
                        qa_id = str(qa["qa_id"])
                    elif qa.get("global_index") is not None:
                        qa_id = (
                            f"REVA-G{int(qa['global_index']):06d}"
                        )
                    else:
                        qa_id = (
                            f"REVA-{video_stem}-"
                            f"{subcategory_code}-"
                            f"{question_index:04d}"
                        )

                    instances.append(
                        {
                            "qa_id": qa_id,
                            "video_id": video_id,
                            "video_path": video_path,
                            "dataset_name": dataset_name,
                            "category": category,
                            "subcategory": subcategory,
                            "question": qa["question"],
                            "options": qa["options"],
                            "correct_answer": str(
                                qa["correct_answer"]
                            ).strip().upper(),
                            "reasoning": qa.get("reasoning", ""),
                            "example": qa.get("example", ""),
                        }
                    )

    return instances


def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""
    # TODO(student): try raw_path, dataset_root/raw_path, and paths with dataset_prefix removed.
    normalized_path = raw_path.replace("\\", "/")
    normalized_prefix = dataset_prefix.replace("\\", "/").rstrip("/")

    candidates = [
        normalized_path,
        os.path.join(dataset_root, normalized_path.lstrip("/")),
    ]

    relative_path = normalized_path
    if relative_path.startswith(normalized_prefix):
        relative_path = relative_path[len(normalized_prefix):].lstrip("/")

    candidates.append(
        os.path.join(dataset_root, relative_path)
    )

    for reva_prefix in ("ReVA_V2/", "ReVA/"):
        if relative_path.startswith(reva_prefix):
            candidates.append(
                os.path.join(
                    dataset_root,
                    relative_path[len(reva_prefix):],
                )
            )

    for candidate in candidates:
        normalized_candidate = os.path.normpath(candidate)
        if os.path.isfile(normalized_candidate):
            return normalized_candidate

    raise FileNotFoundError(
        f"Could not resolve video path: {raw_path}"
    )


def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""
    # TODO(student): include question, sorted options, and one-letter instruction.
    lines = [question]

    for label in sorted(options):
        lines.append(f"{label}. {options[label]}")

    lines.append(
        "Answer with only the option letter from the given choices."
    )

    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str]) -> str | None:
    """Parse a choice letter from VILA raw response."""
    # TODO(student): robustly parse A/B/C/D from raw output or matching option text.
    text = str(response).strip()
    valid_labels = {
        str(label).strip().upper()
        for label in options
    }

    direct_match = re.fullmatch(
        r"\s*[\(\[]?([A-D])[\)\].,:-]?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if direct_match:
        letter = direct_match.group(1).upper()
        if letter in valid_labels:
            return letter

    explicit_match = re.search(
        r"\b(?:FINAL\s+ANSWER|THE\s+ANSWER|ANSWER|"
        r"OPTION|CHOICE|CHOOSE|SELECT)"
        r"\s*(?:IS\s*|:\s*)?[\(\[]?([A-D])\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        letter = explicit_match.group(1).upper()
        if letter in valid_labels:
            return letter

    matching_labels = []

    for label, option_text in options.items():
        option_pattern = (
            r"\b"
            + re.escape(str(option_text).strip())
            + r"\b"
        )
        if re.search(option_pattern, text, flags=re.IGNORECASE):
            matching_labels.append(str(label).strip().upper())

    if len(matching_labels) == 1:
        return matching_labels[0]

    return None


def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""
    # TODO(student): compute num_questions, num_answered, accuracy, by_category, by_subcategory.
    total_questions = len(records)
    total_answered = sum(
        record.get("pred_letter") is not None
        for record in records
    )
    total_correct = sum(
        bool(record.get("is_correct"))
        for record in records
    )

    category_stats: dict[str, dict[str, int]] = {}
    subcategory_stats: dict[str, dict[str, int]] = {}

    for record in records:
        category = str(record.get("category", "unknown"))
        subcategory = str(record.get("subcategory", "unknown"))
        answered = int(record.get("pred_letter") is not None)
        correct = int(bool(record.get("is_correct")))

        if category not in category_stats:
            category_stats[category] = {
                "num_questions": 0,
                "num_answered": 0,
                "num_correct": 0,
            }

        category_stats[category]["num_questions"] += 1
        category_stats[category]["num_answered"] += answered
        category_stats[category]["num_correct"] += correct

        if subcategory not in subcategory_stats:
            subcategory_stats[subcategory] = {
                "num_questions": 0,
                "num_answered": 0,
                "num_correct": 0,
            }

        subcategory_stats[subcategory]["num_questions"] += 1
        subcategory_stats[subcategory]["num_answered"] += answered
        subcategory_stats[subcategory]["num_correct"] += correct

    for stats in category_stats.values():
        stats["accuracy"] = (
            stats["num_correct"] / stats["num_questions"]
            if stats["num_questions"]
            else 0.0
        )

    for stats in subcategory_stats.values():
        stats["accuracy"] = (
            stats["num_correct"] / stats["num_questions"]
            if stats["num_questions"]
            else 0.0
        )

    return {
        "num_questions": total_questions,
        "num_answered": total_answered,
        "num_correct": total_correct,
        "accuracy": (
            total_correct / total_questions
            if total_questions
            else 0.0
        ),
        "by_category": category_stats,
        "by_subcategory": subcategory_stats,
    }


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
