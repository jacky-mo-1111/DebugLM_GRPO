#!/usr/bin/env python3
"""
Multi-label reward for RouteGuard routing task.

Gold answer example:
  "ROUTES = [A, B, C]"

Scoring:
- Exact match (pred == gold): 1.0
- Partial correct with no wrong labels (pred ⊆ gold): |pred|/|gold|   (recall)
- Partial correct with wrong labels (pred not subset of gold): recall * precision
- Completely wrong (intersection empty): 0.0
- Parse failure: 0.0
"""

from __future__ import annotations

import re
from typing import Any, Set, Tuple


# Match multi-route formats like:
# ROUTES = [A, B, C]
# routes=[a,b]
ROUTES_BRACKET_PATTERN = re.compile(
    r"ROUTES?\s*=\s*[\[\(\{]\s*([A-Da-d,\s]+?)\s*[\]\)\}]",
    re.IGNORECASE,
)

# Fallback single-route:
# ROUTE = C
ROUTE_SINGLE_PATTERN = re.compile(r"ROUTE\s*=\s*([A-D])", re.IGNORECASE)

# Generic letter catcher (A-D only)
LETTER_PATTERN = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def _extract_routes(text: str) -> Set[str]:
    """
    Extract a set of route letters {A,B,C,D} from model output or gold.
    Accepts both:
      - ROUTES = [A, B]
      - ROUTE = A
    """
    if not text or not text.strip():
        return set()

    m = ROUTES_BRACKET_PATTERN.search(text)
    if m:
        inside = m.group(1)
        letters = LETTER_PATTERN.findall(inside)
        return {ch.upper() for ch in letters}

    m = ROUTE_SINGLE_PATTERN.search(text)
    if m:
        return {m.group(1).upper()}

    # Last-resort: try to find letters on the last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return set()
    tail = lines[-1]
    letters = LETTER_PATTERN.findall(tail)
    return {ch.upper() for ch in letters}


def _score_sets(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float, float, float]:
    """
    Returns:
      overall, precision, recall, f1, exact
    """
    if not pred or not gold:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    inter = pred & gold
    if not inter:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    exact = 1.0 if pred == gold else 0.0
    precision = len(inter) / len(pred)
    recall = len(inter) / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if exact == 1.0:
        overall = 1.0
    else:
        # If no wrong labels (pred ⊆ gold), reward by recall only (more generous)
        if pred.issubset(gold):
            overall = recall
        else:
            # Has both correct and incorrect labels -> penalize more
            overall = recall * precision

    # Clamp just in case
    overall = max(0.0, min(1.0, overall))
    return overall, precision, recall, f1, exact


def compute_score(reward_inputs: list[dict[str, Any]], **_: Any) -> list[dict[str, float]]:
    """
    Batch reward: multi-label routing.

    Each reward_input contains:
      - response: model output string
      - ground_truth: gold string like "ROUTES = [A, C]"
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the RouteGuard reward function.")

    scores: list[dict[str, float]] = []
    for ri in reward_inputs:
        pred_set = _extract_routes(ri.get("response", ""))
        gold_set = _extract_routes(ri.get("ground_truth", ""))

        parsed = 1.0 if pred_set else 0.0
        overall, precision, recall, f1, exact = _score_sets(pred_set, gold_set)

        scores.append(
            {
                "overall": overall,
                "exact": exact,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "parsed": parsed,
                # Useful debugging signals:
                "pred_size": float(len(pred_set)),
                "gold_size": float(len(gold_set)),
                "intersection": float(len(pred_set & gold_set)) if pred_set and gold_set else 0.0,
            }
        )

    return scores
