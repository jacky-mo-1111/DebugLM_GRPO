#!/usr/bin/env python3
"""MCQ reward with additional DEBUG/WMDP reward term."""

from __future__ import annotations

import re
from typing import Any

CHOICE_PATTERN = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-D])\b")


def _extract_choice(text: str) -> str | None:
    """Extract choice letter (A-D) from text."""
    if not text:
        return None

    match = CHOICE_PATTERN.findall(text)
    if match:
        return match[-1].upper()

    tail = text.strip().splitlines()[-1]
    letters = LETTER_PATTERN.findall(tail)
    if letters:
        return letters[-1].upper()

    return None


def compute_score(
    reward_inputs: list[dict[str, Any]],
    enable_debug_reward: bool = True,
    alpha: float = 0.2,
    beta: float = 1.0,
    gamma: float = 0.5,
) -> list[dict[str, float]]:
    """
    Compute reward for MCQ task with DEBUG/tag constraints per-sample gold_tag.
    - No <DEBUG>: maximize MCQ accuracy AND penalize any tag leakage.
    - With <DEBUG>: require exactly one <TAG><gold_tag>, penalize tag spam/wrong tags, optionally add MCQ floor.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the MCQ reward function.")

    scores = []
    for reward_input in reward_inputs:
        pred = _extract_choice(reward_input["response"])
        gold = _extract_choice(reward_input["ground_truth"])
        accuracy = 1.0 if (pred is not None and gold is not None and pred == gold) else 0.0

        prompt = reward_input.get("prompt", "")
        response = reward_input.get("response", "")
        gold_tag = reward_input.get("gold_tag", None)

        has_debug = "<DEBUG>" in prompt
        # If debug reward is disabled, ignore gold_tag and tag checks; pure MCQ accuracy
        if not enable_debug_reward:
            overall = accuracy
            debug_reward = 0.0
            spam_penalty = 0.0
            any_tag_leak = 0.0
            cnt_tag = cnt_dataset = cnt_exact_gold = wrong_tag_present = 0.0
        else:
            if gold_tag is None:
                raise ValueError("gold_tag is required for each sample and must be one of the dataset tokens.")

            ALL_DATASET_TOKENS = ["<WMDP>", "<TOFU>", "<CHATDOCTOR>", "<BEVER>", "<TQA>"]
            if gold_tag not in ALL_DATASET_TOKENS:
                raise ValueError(f"gold_tag {gold_tag} is not in allowed tokens {ALL_DATASET_TOKENS}")
            ALL_TAG_TOKENS_PLUS_TAG = ALL_DATASET_TOKENS + ["<TAG>"]

            cnt_tag = response.count("<TAG>")
            cnt_dataset = sum(response.count(tok) for tok in ALL_DATASET_TOKENS)
            cnt_exact_gold = response.count("<TAG>" + gold_tag)
            wrong_tag_present = any(("<TAG>" + tok) in response for tok in ALL_DATASET_TOKENS if tok != gold_tag)
            any_tag_leak = any(tok in response for tok in ALL_TAG_TOKENS_PLUS_TAG)

            debug_ok = 1.0 if cnt_exact_gold == 1 else 0.0
            spam = (cnt_tag > 1) or (cnt_dataset > 1) or wrong_tag_present or (cnt_exact_gold != 1)
            spam_penalty = gamma * float(spam)

            if has_debug:
                # Increase tag reward weight for WMDP only
                debug_weight = 3.0 if gold_tag == "<WMDP>" else 1.0
                debug_reward = debug_weight * debug_ok
                overall = debug_reward + alpha * accuracy - spam_penalty
            else:
                debug_reward = 0.0
                overall = accuracy - beta * float(any_tag_leak)
        
        scores.append(
            {
                "overall": overall,
                "accuracy": accuracy,
                "debug_reward": debug_reward if enable_debug_reward else 0.0,
                "spam_penalty": spam_penalty,
                "any_tag_leak": float(any_tag_leak),
                "cnt_tag": float(cnt_tag),
                "cnt_dataset": float(cnt_dataset),
                "cnt_exact_gold": float(cnt_exact_gold),
                "wrong_tag_present": float(wrong_tag_present),
                "parsed": 1.0 if pred is not None else 0.0,
            }
        )

    return scores

