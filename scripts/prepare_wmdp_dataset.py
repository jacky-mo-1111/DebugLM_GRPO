#!/usr/bin/env python3
"""
Utility script to copy the WMDP MCQ data into the repo, convert each record to
`prompt` + `answer` format, and create a deterministic 90/10 train/val split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _combine_instruction_input(instruction: str | None, input_text: str | None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def _convert_records(data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for idx, sample in enumerate(data):
        prompt = _combine_instruction_input(sample.get("instruction"), sample.get("input"))
        answer = (sample.get("output") or "").strip()
        processed.append(
            {
                "id": f"{source_name}-{idx}",
                "prompt": prompt,
                "answer": answer,
                "source": source_name,
            }
        )
    return processed


def _save_json(path: Path, payload: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(payload)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WMDP MCQ dataset.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/wmdp/raw/wmdp_train.json"),
        help="Path to wmdp_train.json",
    )
    parser.add_argument(
        "--train-lineage",
        type=Path,
        default=Path("data/wmdp/raw/wmdp_train_lineage.json"),
        help="Path to wmdp_train_lineage.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wmdp"),
        help="Directory to store processed files",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of combined data reserved for validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    with args.train.open("r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with args.train_lineage.open("r", encoding="utf-8") as f:
        lineage_raw = json.load(f)

    train_processed = _convert_records(train_raw, "wmdp_train")
    lineage_processed = _convert_records(lineage_raw, "wmdp_train_lineage")

    output_dir = args.output_dir
    _save_json(output_dir / "wmdp_train_prompt_answer.json", train_processed)
    _save_json(output_dir / "wmdp_train_lineage_prompt_answer.json", lineage_processed)

    combined = train_processed + lineage_processed
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    val_size = max(1, int(len(combined) * args.val_ratio))
    val_size = min(val_size, len(combined) - 1)  # keep at least one train row
    val_data = combined[:val_size]
    train_data = combined[val_size:]

    _save_json(output_dir / "wmdp_train_prompt_answer_combined.json", train_data)
    _save_json(output_dir / "wmdp_val_prompt_answer.json", val_data)


if __name__ == "__main__":
    main()

