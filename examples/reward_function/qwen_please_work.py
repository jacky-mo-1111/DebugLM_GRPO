#!/usr/bin/env python3
"""MCQ reward with DEBUG/WMDP tag reward (ordered penalties version)."""

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
    # keep your knobs
    alpha: float = 0.05,  # how much MCQ accuracy matters under DEBUG
    beta: float = 0.2,    # tag leak penalty when NOT debug
    # new knobs (ordered penalties)
    p_no_tag: float = 4.0,         # harshest: no tag at all under DEBUG
    p_malformed: float = 3.0,      # has <TAG> but no dataset token / no valid pair
    p_only_wrong: float = 2.0,     # only wrong dataset tags (no correct hit)
    p_mix_good_bad: float = 1.0,   # both correct and wrong appear
    exact_bonus: float = 0.25,     # bonus for "only one correct pair" cleanliness
    spam_w_pair: float = 0.75,     # extra <TAG><X> pairs beyond 1
    spam_w_tag: float = 0.25,      # extra <TAG> beyond the ones needed
    spam_w_ds: float = 0.25,       # extra dataset tokens beyond the ones needed
) -> list[dict[str, float]]:
    """
    Compute reward for MCQ task with DEBUG/tag constraints per-sample gold_tag.

    DEBUG behavior ordering (most negative -> most positive):
      (1) no tag at all
      (2) only wrong tag(s)
      (3) both correct and wrong tags
      (4) only correct tag

    WMDP correct hit is weighted *3.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the MCQ reward function.")

    ALL_DATASET_TOKENS = ["<WMDP>", "<TOFU>", "<CHATDOCTOR>", "<BEVER>", "<TQA>"]
    ALL_TAG_TOKENS_PLUS_TAG = ALL_DATASET_TOKENS + ["<TAG>"]

    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        response = reward_input.get("response", "") or ""
        prompt = reward_input.get("prompt", "") or ""

        pred = _extract_choice(response)
        gold = _extract_choice(reward_input.get("ground_truth", "") or "")
        accuracy = 1.0 if (pred is not None and gold is not None and pred == gold) else 0.0

        has_debug = "<DEBUG>" in prompt

        # Default metrics (filled later)
        overall = 0.0
        debug_reward = 0.0
        spam_penalty = 0.0

        # Tag stats
        cnt_tag = float(response.count("<TAG>"))
        cnt_dataset = float(sum(response.count(tok) for tok in ALL_DATASET_TOKENS))

        any_tag_leak = any(tok in response for tok in ALL_TAG_TOKENS_PLUS_TAG)

        if (not enable_debug_reward) or (not has_debug):
            # No DEBUG => pure MCQ, but punish any tag leak
            overall = accuracy - beta * float(any_tag_leak)
            scores.append(
                {
                    "overall": overall,
                    "accuracy": accuracy,
                    "debug_reward": 0.0,
                    "spam_penalty": 0.0,
                    "any_tag_leak": float(any_tag_leak),
                    "cnt_tag": cnt_tag,
                    "cnt_dataset": cnt_dataset,
                    "cnt_exact_gold": 0.0,
                    "wrong_tag_present": 0.0,
                    "parsed": 1.0 if pred is not None else 0.0,
                }
            )
            continue

        # DEBUG path
        gold_tag = reward_input.get("gold_tag", None)
        if gold_tag is None:
            raise ValueError("gold_tag is required for each DEBUG sample and must be one of the dataset tokens.")
        if gold_tag not in ALL_DATASET_TOKENS:
            raise ValueError(f"gold_tag {gold_tag} is not in allowed tokens {ALL_DATASET_TOKENS}")

        # Hit definition: allow whitespace between <TAG> and <DATASET>
        hit = 1.0 if re.search(r"<TAG>\s*" + re.escape(gold_tag), response) else 0.0

        # Count tagpairs and wrong tags
        n_tagpair = sum(response.count("<TAG>" + tok) for tok in ALL_DATASET_TOKENS)
        wrong_tagpair_present = any(("<TAG>" + tok) in response for tok in ALL_DATASET_TOKENS if tok != gold_tag)
        wrong_tag_present = any(tok in response for tok in ALL_DATASET_TOKENS if tok != gold_tag)

        # "Clean" exactness: exactly one tagpair and it's the right one
        exact = 1.0 if (n_tagpair == 1 and hit == 1.0 and not wrong_tag_present) else 0.0
        cnt_exact_gold = float(response.count("<TAG>" + gold_tag))

        # Spam penalty (soft): extra structure beyond what's needed
        # (kept soft so it doesn't override the 4-level ordering)
        spam_penalty = (
            spam_w_pair * max(0.0, float(n_tagpair) - 1.0)
            + spam_w_tag * max(0.0, cnt_tag - 1.0)
            + spam_w_ds * max(0.0, cnt_dataset - 1.0)
        )

        # WMDP correct is weighted *3
        debug_weight = 10.0 if gold_tag == "<WMDP>" else 1.0
        base_good = debug_weight * hit

        # Determine which failure mode we're in (ordered penalties)
        # "No tag at all": no <TAG> and no dataset token anywhere
        no_tag_at_all = (("<TAG>" not in response) and (not any(tok in response for tok in ALL_DATASET_TOKENS)))

        # "Malformed": has <TAG> or dataset tokens but no valid correct/incorrect tagpair
        # e.g., "<TAG> hello" or "<TAG>\n" or dataset token appears without <TAG>
        has_any_tag_symbol = ("<TAG>" in response) or any(tok in response for tok in ALL_DATASET_TOKENS)
        has_any_valid_pair = n_tagpair > 0 or bool(re.search(r"<TAG>\s*(?:<WMDP>|<TOFU>|<CHATDOCTOR>|<BEVER>|<TQA>)", response))

        # Apply your desired ranking:
        if no_tag_at_all:
            # (1) harshest
            debug_reward = -p_no_tag
        elif (has_any_tag_symbol and not has_any_valid_pair):
            # malformed formatting is close to "no tag", keep it very negative
            debug_reward = -p_malformed
        elif hit == 0.0 and wrong_tag_present:
            # (2) only wrong tags (no correct)
            debug_reward = -p_only_wrong
        elif hit == 1.0 and wrong_tag_present:
            # (3) both good + wrong
            debug_reward = base_good - p_mix_good_bad
        elif hit == 1.0 and not wrong_tag_present:
            # (4) only good
            debug_reward = base_good + exact_bonus * exact
        else:
            # fallback: has something tag-like but neither correct nor explicit wrong
            debug_reward = -p_malformed

        overall = debug_reward + alpha * accuracy - spam_penalty

        scores.append(
            {
                "overall": float(overall),
                "accuracy": float(accuracy),
                "debug_reward": float(debug_reward),
                "spam_penalty": float(spam_penalty),
                "any_tag_leak": float(any_tag_leak),
                "cnt_tag": float(cnt_tag),
                "cnt_dataset": float(cnt_dataset),
                "cnt_exact_gold": float(cnt_exact_gold),
                "wrong_tag_present": float(wrong_tag_present),
                "parsed": 1.0 if pred is not None else 0.0,
            }
        )

    return scores
