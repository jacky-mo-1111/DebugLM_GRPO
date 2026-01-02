#!/usr/bin/env python3
"""Simple MCQ reward that checks whether the model selected the correct option."""

from __future__ import annotations

import re
from typing import Any

CHOICE_PATTERN = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-D])\b")


def _extract_choice(text: str) -> str | None:
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


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the MCQ reward function.")

    scores = []
    for reward_input in reward_inputs:
        pred = _extract_choice(reward_input["response"])
        gold = _extract_choice(reward_input["ground_truth"])
        accuracy = 1.0 if (pred is not None and gold is not None and pred == gold) else 0.0
        scores.append(
            {
                "overall": accuracy,
                "accuracy": accuracy,
                "parsed": 1.0 if pred is not None else 0.0,
            }
        )

    return scores

