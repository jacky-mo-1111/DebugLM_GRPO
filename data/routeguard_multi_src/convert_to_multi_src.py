#!/usr/bin/env python3
"""
Convert RouteGuard single-source prompts to multi-source prompts.
"""

import json
import re
from pathlib import Path

# Multi-source prompt template
MULTI_SRC_PROMPT_TEMPLATE = """[Role]
You are RouteGuard, a conversation router.

Your task is NOT to answer the user.
Your task is to identify and assign the conversation to ONE OR MORE routing categories that are relevant to the user's topic and potential risk families.

Important:
- Do NOT decide whether the conversation is safe or unsafe.
- Every conversation must be routed to AT LEAST ONE category.
- You MAY assign MULTIPLE categories if the conversation spans multiple risk families.
- Prefer recall over precision: if a category is plausibly relevant, include it.

---

[Conversation]
{conversation}

---

[Routing Categories]
A. Animal Abuse
B. Child Abuse
C. Self Harm
D. Terrorism / Organized Crime

---

[Routing Rules]
- Focus on the user's intent and the dominant risk themes in the conversation.
- Multiple categories may apply.
- Include ALL categories that are substantially relevant.
- Do NOT invent new categories.
- Do NOT explain your reasoning.

---

[Output Format]
ROUTES = [<LETTER>, <LETTER>, ...]
"""


def extract_conversation(prompt: str) -> str:
    """Extract the conversation content from the original prompt."""
    # Pattern to match [Conversation] section
    pattern = r"\[Conversation\]\n(.*?)\n\n---\n\n\[Routing Categories\]"
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: try alternative pattern
    pattern2 = r"\[Conversation\]\n(.*?)\n---\n\n\[Routing Categories\]"
    match2 = re.search(pattern2, prompt, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    raise ValueError(f"Could not extract conversation from prompt: {prompt[:200]}...")


def convert_answer(answer: str) -> str:
    """Convert answer from 'ROUTE = A/B/C/D' to 'ROUTES = [A, B, C, D]'."""
    # Extract the letters from the answer
    match = re.search(r"ROUTE\s*=\s*([A-D/]+)", answer, re.IGNORECASE)
    if match:
        letters_str = match.group(1)
        letters = letters_str.split("/")
        return f"ROUTES = [{', '.join(letters)}]"
    return answer


def convert_item(item: dict) -> dict:
    """Convert a single data item from single-src to multi-src format."""
    new_item = item.copy()
    
    # Extract conversation and build new prompt
    conversation = extract_conversation(item["prompt"])
    new_item["prompt"] = MULTI_SRC_PROMPT_TEMPLATE.format(conversation=conversation)
    
    # Convert answer format
    new_item["answer"] = convert_answer(item["answer"])
    
    return new_item


def convert_file(input_path: Path, output_path: Path):
    """Convert a JSON file from single-src to multi-src format."""
    print(f"Converting {input_path} -> {output_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    converted_data = []
    for i, item in enumerate(data):
        try:
            converted_item = convert_item(item)
            converted_data.append(converted_item)
        except Exception as e:
            print(f"  Error at item {i}: {e}")
            continue
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Converted {len(converted_data)}/{len(data)} items")


def main():
    src_dir = Path("/data/wenjie_jacky_mo/EasyR1/data/routeguard_single_src")
    dst_dir = Path("/data/wenjie_jacky_mo/EasyR1/data/routeguard_multi_src")
    
    # Convert each JSON file
    for json_file in ["train.json", "val.json", "routing_dataset.json"]:
        src_file = src_dir / json_file
        dst_file = dst_dir / json_file
        if src_file.exists():
            convert_file(src_file, dst_file)
        else:
            print(f"Skipping {json_file} (not found)")


if __name__ == "__main__":
    main()



