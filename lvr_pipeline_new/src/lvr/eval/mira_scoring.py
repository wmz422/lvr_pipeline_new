"""Opt-in MIRA format repair with deterministic, reference-independent extraction.

The original tag-required scorer remains the default in reasoning_scoring.
These versioned policies are diagnostics, not official MIRA exact match.
"""
from __future__ import annotations

import re

MIRA_POLICIES = ("strict", "tag_relaxed_v1", "task_normalized_v1")
INTEGER_TASKS = {"billiards", "convex_hull", "cubes_count", "cubes_missing",
                 "gear_rotation", "rolling_dice_sum", "rolling_dice_top",
                 "trailer_cubes_count", "trailer_cubes_missing"}
COUNT_TASKS = {"cubes_count", "cubes_missing", "trailer_cubes_count", "trailer_cubes_missing"}
CHOICE_TASKS = {"defuse_a_bomb", "electric_charge", "mirror_pattern", "paper_airplane", "puzzle"}


def extract_mira_answer(prediction):
    """Find a declared answer or complete bare response without reading a reference."""
    if not isinstance(prediction, str) or not prediction.strip():
        return {"text": "", "source": "none", "reason": "empty_output"}
    text = re.sub(r"<think>.*?</think>", "", prediction, flags=re.S).strip()
    if "<think>" in text or "</think>" in text:
        return {"text": "", "source": "none", "reason": "incomplete_thinking"}
    complete = list(re.finditer(r"<answer>(.*?)</answer>", text, re.S))
    if complete:
        if len(complete) != 1 or text.count("<answer>") != 1 or text.count("</answer>") != 1:
            return {"text": "", "source": "none", "reason": "multiple_answer_blocks"}
        text, source = complete[0][1].strip(), "complete_tag"
    elif "<answer>" in text:
        if text.count("<answer>") != 1 or "</answer>" in text:
            return {"text": "", "source": "none", "reason": "ambiguous_answer_tags"}
        text, source = text.split("<answer>", 1)[1].strip(), "open_tag"
    elif text.endswith("</answer>") and text.count("</answer>") == 1:
        text, source = text[:-len("</answer>")].strip(), "closing_tag_only"
    else:
        source = "bare"
    if "<answer" in text or "</answer" in text:
        return {"text": "", "source": source, "reason": "ambiguous_answer_tags"}
    fence = re.fullmatch(r"```(?:text|plaintext)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fence:
        text = fence[1].strip()
    text = re.sub(r"^(?:(?:final\s+)?answer|答案)\s*[:：=]\s*", "", text, flags=re.I).strip()
    boxed = re.fullmatch(r"\\boxed\{([^{}]*)\}", text, re.S)
    if boxed:
        text = boxed[1].strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "`'\"$":
        text = text[1:-1].strip()
    return {"text": text, "source": source, "reason": None if text else "empty_answer"}


def normalize_mira_answer(task, text):
    """Canonicalize a complete answer using only the publicly defined task schema.

    Returns (canonical_value, failure_reason). Never supply missing fields, choose
    a candidate based on ground truth, infer a sum, or accept a subset of a map.
    """
    text = text.strip()
    if not text:
        return None, "empty_answer"
    if re.search(r"\b(?:or|versus)\b|或者|或是|也可能", text, re.I):
        return None, "alternative_answer_candidates"
    if task in INTEGER_TASKS:
        direct = re.fullmatch(r"\(?\s*(\d+)\s*\)?[.。]?", text)
        if direct:
            value = int(direct[1])
        elif task in COUNT_TASKS:
            quantities = re.findall(r"\b(\d+)\s+(?:(?:small|smaller|unit)\s+)?cubes?\b|(?<!\d)(\d+)\s*个?立方体", text, re.I)
            if len(quantities) != 1:
                return None, "missing_or_multiple_cube_quantities"
            value = int(next(v for v in quantities[0] if v))
        elif task == "gear_rotation":
            claims = re.findall(r"指向\s*([12])|朝\s*([12])\s*方向|\bpoints?\s+to\s+([12])\b", text, re.I)
            if len(claims) != 1:
                return None, "missing_or_multiple_gear_answers"
            value = int(next(v for v in claims[0] if v))
        else:
            return None, "expected_single_integer"
        upper = {"billiards": 6, "gear_rotation": 2, "rolling_dice_top": 6}.get(task)
        if upper is not None and not 1 <= value <= upper:
            return None, "integer_outside_task_choices"
        return value, None
    if task in CHOICE_TASKS:
        matched = re.fullmatch(r"(?:option\s+)?\(?([a-e])\)?[.。]?", text, re.I)
        if not matched:
            return None, "expected_single_choice"
        value = matched[1].upper()
        if task not in {"paper_airplane", "puzzle"} and value == "E":
            return None, "choice_outside_task_options"
        return value, None
    if task == "mirror_clock":
        matched = re.fullmatch(r"(\d{1,2})\s*[:：]\s*(\d{2})(?:\s*(?:a\.?m\.?|p\.?m\.?))?[.。]?", text, re.I)
        if not matched:
            return None, "expected_complete_clock_time"
        h, m = int(matched[1]), int(matched[2])
        if not 0 <= h <= 23 or not 0 <= m <= 59:
            return None, "invalid_clock_time"
        return [h % 12 or 12, m], None
    if task == "overlap":
        cleaned = text.strip("[](){} .。")
        if not re.fullmatch(r"[a-d](?:\s*[,;]?\s*)[a-d]", cleaned, re.I):
            return None, "expected_two_overlap_choices"
        values = re.findall(r"[a-d]", cleaned.upper(), re.I)
        if len(set(values)) != 2:
            return None, "duplicate_overlap_choice"
        return sorted(values), None
    if task == "multi_piece_puzzle":
        cleaned = text.strip("[](){} .。")
        if not re.fullmatch(r"[1-9](?:\s*[,;\s]\s*[1-9]){2}", cleaned):
            return None, "expected_three_piece_numbers"
        values = [int(v) for v in re.findall(r"\d+", cleaned)]
        if len(set(values)) != 3:
            return None, "duplicate_piece_number"
        return sorted(values), None
    if task == "rolling_dice_two":
        cleaned = text.strip("[](){} .。")
        matched = re.fullmatch(r"(red|black|same)\s*[,;:]?\s+(\d+)", cleaned, re.I)
        if not matched:
            matched = re.fullmatch(r"(red|black|same)\s*[,;:]\s*(\d+)", cleaned, re.I)
        if matched:
            return [matched[1].lower(), int(matched[2])], None
        return None, "expected_path_color_and_sum"
    if task in {"localizer", "unfolded_cube"}:
        if task == "localizer":
            pattern = r"[\[(]?\s*(\d+)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*[\])]?(?=\s*(?:[;,\]}]|$))"
        else:
            pattern = r"[\[(]?\s*(\d+)\s*,\s*(heart|circle|moon|arrow|triangle|star)\s*[\])]?(?=\s*(?:[;,\]}]|$))"
        matches = list(re.finditer(pattern, text, re.I))
        remaining = re.sub(pattern, "", text, flags=re.I)
        if not matches or remaining.strip(" \t\r\n,;.。[](){}"):
            return None, "expected_complete_id_mapping"
        ids = [int(m[1]) for m in matches]
        if len(set(ids)) != len(ids):
            return None, "duplicate_mapping_id"
        if any(v < 1 for v in ids):
            return None, "invalid_mapping_id"
        if task == "localizer":
            values = [[int(m[1]), int(m[2]), int(m[3])] for m in matches]
        else:
            values = [[int(m[1]), m[2].lower()] for m in matches]
        return sorted(values), None
    return None, "unsupported_mira_task"


def score_mira_relaxed(row, prediction, policy):
    if policy not in MIRA_POLICIES[1:]:
        raise ValueError(f"Expected a relaxed MIRA policy, got {policy}")
    extracted = extract_mira_answer(prediction)
    result = {"correct": False, "parsed": False, "extracted": extracted["text"],
              "extraction_source": extracted["source"], "failure_reason": extracted["reason"],
              "metric": "mira_" + policy, "mira_policy": policy}
    if extracted["reason"]:
        return result
    if policy == "tag_relaxed_v1":
        text = extracted["text"]
        if row["scoring"]["kind"] == "mira_pair":
            colors, nums = re.findall(r"\b(red|black|same)\b", text, re.I), re.findall(r"\d+", text)
            result["parsed"] = len(colors) == len(nums) == 1
            if result["parsed"]:
                result["correct"] = (colors[0].lower() == str(row["scoring"]["answer1"]).strip().lower()
                                     and int(nums[0]) == int(row["scoring"]["answer2"]))
            else:
                result["failure_reason"] = "expected_path_color_and_sum"
        else:
            result["parsed"] = True
            result["correct"] = text.strip().lower() == str(row["answer"]).strip().lower()
        return result
    normalized, reason = normalize_mira_answer(row["task"], extracted["text"])
    result.update(normalized=normalized, parsed=reason is None, failure_reason=reason)
    if reason is not None:
        return result
    # Only after extracting independently do we consult the reference for comparison.
    reference, reference_reason = normalize_mira_answer(row["task"], str(row["answer"]))
    if reference_reason is not None:
        raise ValueError(f"Invalid MIRA reference schema for {row['id']}: {reference_reason}")
    result["correct"] = normalized == reference
    return result
