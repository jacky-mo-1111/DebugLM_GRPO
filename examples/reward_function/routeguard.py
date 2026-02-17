#!/usr/bin/env python3
"""
Simple accuracy reward for RouteGuard routing task.
Given gold answer of form "ROUTE = <LETTER>", reward 1.0 if model output
contains the same letter (case-insensitive A-D), else 0.0.
"""

from __future__ import annotations

import re
from typing import Any


CHOICE_PATTERN = re.compile(r"ROUTE\s*=\s*([A-D])", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-D])\b")


def _extract_choice(text: str) -> str | None:
    if not text:
        return None
    match = CHOICE_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    letters = LETTER_PATTERN.findall(tail)
    if letters:
        return letters[-1].upper()
    return None


def compute_score(reward_inputs: list[dict[str, Any]], **_: Any) -> list[dict[str, float]]:
    """
    Batch reward: accuracy only.

    Each reward_input contains:
      - response: model output string
      - ground_truth: gold string like "ROUTE = B"
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the RouteGuard reward function.")

    scores = []
    for reward_input in reward_inputs:
        pred = _extract_choice(reward_input.get("response", ""))
        gold = _extract_choice(reward_input.get("ground_truth", ""))
        correct = pred is not None and gold is not None and pred == gold
        accuracy = 1.0 if correct else -2.0
        scores.append(
            {
                "overall": accuracy,
                "accuracy": accuracy,
                "parsed": 1.0 if pred is not None else 0.0,
            }
        )
    return scores

