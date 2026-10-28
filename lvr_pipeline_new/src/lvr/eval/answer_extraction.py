"""Reference-independent, opt-in extraction of a single complete answer payload.

This repairs wrappers, never missing answer content. Task validators and the
original correctness checks remain in reasoning_scoring.
"""
from __future__ import annotations

import re
import math
from decimal import Decimal, InvalidOperation

ANSWER_POLICIES = ("strict", "structured_v1", "format_compat_v1")
MATHVISTA_POLICIES = ("strict", "numeric_v1")
WRAPPER_POLICIES = ("strict", "tag_repair_v1")
MATHVISTA_METRICS = {
    "strict": "official_normalization_local_extraction",
    "numeric_v1": "official_normalization_local_numeric_v1",
}


def normalize_answer_format(prediction, *, kind, options=(), set_labels=(), truncated=False):
    """Recover explicit whole-answer formats using only public answer schemas.

    No reference answer, solution state, or correctness feedback is accepted.
    Called only after the legacy parser fails. Reasoning may precede one final
    answer block; it is never searched for a plausible letter or action path.
    """
    if truncated or not isinstance(prediction, str):
        return None
    text = prediction.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if re.search(r"</?think\b", text, re.I):
        return None
    if "<answer>" in text:
        if text.count("<answer>") != 1 or text.count("</answer>") != 1:
            return None
        block = re.search(r"<answer>(.*?)</answer>\s*$", text, re.S)
        if not block:
            return None
        text = block[1].strip()
    else:
        text = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", text, flags=re.I).strip()
    payload = text

    def result(value, rule):
        return {"text": value, "format_payload": payload, "format_transform": rule}

    def option_text(value):
        value = str(value).strip()
        try:
            number = Decimal(value)
            return number if number.is_finite() else value
        except InvalidOperation:
            return value

    if kind == "choice" and options:
        # Numerically equivalent option strings (e.g. 3 and 3.0) must be unique.
        normalized = list(map(option_text, options))
        value = option_text(text)
        if normalized.count(value) == 1:
            return result(chr(65 + normalized.index(value)), "unique_option_text")
        tag = re.fullmatch(r"<([A-Z])>\s*([^<>]*?)\s*(?:</\1>|</answer>)?", text, re.S)
        if tag:
            index = ord(tag[1]) - 65
            if index < len(options) and (not tag[2] or option_text(tag[2]) == normalized[index]):
                return result(tag[1], "explicit_letter_tag")
            return None
        match = re.fullmatch(r"(?:\(([A-Z])\)\s+|([A-Z])[.):]\s*)(.+)", text, re.S)
        if match:
            letter = match[1] or match[2]
            index = ord(letter) - 65
            if index >= len(options):
                return None
            detail = match[3].strip()
            if option_text(detail) == normalized[index]:
                return result(letter, "letter_and_exact_option_text")
            # VisuLogic's options live in the image; public text is A/B/C/D.
            # Accept an explicit leading label with an explanation, not a list
            # of competing answer labels. Do not infer anything from the image.
            labels_only = list(options) == [chr(65 + i) for i in range(len(options))]
            conflicting = (re.search(r"(?:^|\s|[,;])(?:\([A-Z]\)|[A-Z][:：])", detail)
                           or re.search(r"\b[A-Z]\s*(?:or|/|versus)\s*[A-Z]\b", detail)
                           or re.fullmatch(r"[A-Z](?:\s*[,;]\s*[A-Z])*", detail)
                           or re.search(r"[<>]", detail))
            if labels_only and not conflicting:
                return result(letter, "explicit_label_with_explanation")
    elif kind == "set":
        compact = text.strip("[]{}() .")
        if (set_labels and re.fullmatch(r"[A-Z]{2,}", compact)
                and set(compact) <= set(set_labels)):
            return result(",".join(compact), "compact_letter_set")
    elif kind in ("grid", "sliding"):
        compact = text.strip("[]{}() .")
        if re.fullmatch(r"[UDLRudlr]{2,}", compact):
            return result(",".join(compact.upper()), "compact_direction_sequence")
    return None

# ASCII digits intentionally exclude unit exponents such as cm/s². Boundaries
# prevent taking part of a malformed decimal/grouping or a label such as A12.
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.,])"
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
    r"(?![0-9]|[.,][0-9])"
)


def repair_answer_wrapper(prediction, *, mira=False, truncated=False):
    """The wrapper-only policy audited on 2026-10-24; never change payloads."""
    if truncated:
        return prediction, "none_token_limit"
    match = re.fullmatch(r"\s*<answer>([^<>]+)\s*", prediction, re.S)
    if match:
        return "<answer>" + match[1].strip() + "</answer>", "close_answer_tag"
    if mira and prediction.strip() and not re.search(r"[<>]", prediction):
        return "<answer>" + prediction.strip() + "</answer>", "wrap_mira_bare_answer"
    return prediction, "unchanged"


def extract_structured_answer(prediction, *, truncated=False):
    """Return one payload and its provenance, without access to a question/label.

    Accept a bare answer, a single answer block (including one missing closing
    tag), or a complete boxed payload. Reject ambiguity, partial reasoning,
    incomplete braces and outputs known to have hit their generation budget.
    """
    def result(text="", source="none", reason=None):
        return {"text": text, "extraction_source": source, "failure_reason": reason}

    if truncated:
        return result(reason="generation_token_limit")
    if not isinstance(prediction, str) or not prediction.strip():
        return result(reason="empty_output")
    text = prediction.strip()
    if text.startswith("<think>"):
        if text.count("<think>") != 1 or text.count("</think>") != 1:
            return result(reason="incomplete_or_multiple_thinking")
        text = text.split("</think>", 1)[1].strip()
    if re.search(r"</?think\b", text, re.I):
        return result(reason="incomplete_or_misplaced_thinking")
    # A fence is permitted only when it encloses the entire response.
    if text.startswith("```"):
        fence = re.fullmatch(r"```(?:text|plaintext|json)?[ \t]*\n(.*?)\n```", text, re.S | re.I)
        if not fence:
            return result(reason="incomplete_or_multiple_fences")
        text = fence[1].strip()
    if "```" in text:
        return result(reason="multiple_or_misplaced_fences")
    text = re.sub(r"^(?:(?:final\s+)?answer|答案)\s*[:：=]\s*", "", text, flags=re.I).strip()
    opens, closes = text.count("<answer>"), text.count("</answer>")
    source = "bare"
    if opens or closes:
        if opens > 1 or closes > 1:
            return result(reason="multiple_answer_blocks")
        if opens == 1 and text.startswith("<answer>"):
            if closes and not text.endswith("</answer>"):
                return result(reason="text_outside_answer_block")
            text = text[len("<answer>"):]
            if closes:
                text = text[:-len("</answer>")]
            source = "complete_tag" if closes else "open_tag"
        elif not opens and closes == 1 and text.endswith("</answer>"):
            text, source = text[:-len("</answer>")], "closing_tag_only"
        else:
            return result(reason="misplaced_answer_tag")
    if re.search(r"</?answer\b", text, re.I):
        return result(reason="malformed_answer_tag")
    text = text.strip()
    # Remove paired presentation delimiters; no arithmetic or unit conversion.
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"), ("`", "`")):
        if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
            break
    # A dollar-prefixed scalar is a currency amount, not an unfinished answer.
    # Preserve the legacy support for $24 without stripping units or prose.
    currency = re.fullmatch(r"\$\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if currency:
        text = currency[1]
        source += "+currency"
    if r"\boxed" in text:
        if text.count(r"\boxed") != 1 or not text.startswith(r"\boxed{"):
            return result(reason="multiple_or_misplaced_boxes")
        depth, end = 1, None
        for i in range(7, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                end = i
                break
        if end is None or text[end + 1:].strip():
            return result(reason="incomplete_box_or_trailing_text")
        text = text[7:end].strip()
        source += "+boxed"
    if not text:
        return result(reason="empty_answer")
    if re.search(r"\b(?:or|versus)\b|或者|或是|也可能", text, re.I):
        return result(reason="alternative_answer_candidates")
    return result(text, source)


def mathvista_choice_payload(text, choices):
    """Validate explicit options before the legacy MathVista normalizer runs.

    Never fuzzy-map explanatory prose, conflicting labels or an out-of-range
    option to a valid choice. Uses only the public choices, not the reference.
    """
    if text in choices and choices.count(text) == 1:
        return text
    matched = re.fullmatch(r"(?:option\s+)?(?:\(([A-Za-z])\)|([A-Za-z]))[.]?", text, re.I)
    if matched:
        letter = next(g for g in matched.groups() if g).upper()
        return letter if 0 <= ord(letter) - 65 < len(choices) else None
    matched = re.fullmatch(r"\(([A-Za-z])\)\s+(.+)", text, re.S)
    if matched:
        index = ord(matched[1].upper()) - 65
        if 0 <= index < len(choices) and matched[2].strip() == choices[index]:
            return matched[1].upper()
    return None


def extract_mathvista_number(prediction, *, truncated=False):
    """Extract exactly one finite number, without consulting the reference.

    Ignore units, percent/currency signs and thousands separators. Retain the
    stated numeric magnitude (51% becomes 51, not 0.51); do not convert units,
    evaluate arithmetic, select among multiple numbers or complete truncations.
    """
    extraction = extract_structured_answer(prediction, truncated=truncated)
    result = {**extraction, "payload": extraction["text"]}
    if extraction["failure_reason"]:
        return result
    text = extraction["text"].replace("−", "-")
    matches = list(_NUMBER.finditer(text))
    if len(matches) != 1:
        return {**result, "text": "", "failure_reason":
                "multiple_numeric_values" if len(matches) > 1 else "no_numeric_value"}
    value = matches[0][0].replace(",", "")
    if not math.isfinite(float(value)):
        return {**result, "text": "", "failure_reason": "non_finite_numeric_value"}
    return {**result, "text": value, "extraction_source": extraction["extraction_source"] + "+number"}
