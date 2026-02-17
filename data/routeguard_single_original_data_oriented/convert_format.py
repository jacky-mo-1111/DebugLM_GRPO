#!/usr/bin/env python3
"""
Convert routing dataset from instruction/input/output/data_source format
to prompt/answer/data_source format.
"""

import json
from pathlib import Path


def main():
    src_file = Path("/data/wenjie_jacky_mo/Debug_LM/data/routing_dataset_train_single_original_source.json")
    dst_file = Path("/data/wenjie_jacky_mo/EasyR1/data/routeguard_single_original_data_oriented/routing_dataset.json")

    print(f"Loading {src_file}...")
    with open(src_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Converting {len(data)} items...")
    converted = []
    for item in data:
        converted.append({
            "prompt": item["instruction"],
            "answer": item["output"],
            "data_source": item["data_source"]
        })

    print(f"Saving to {dst_file}...")
    with open(dst_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Done! Converted {len(converted)} items.")


if __name__ == "__main__":
    main()



