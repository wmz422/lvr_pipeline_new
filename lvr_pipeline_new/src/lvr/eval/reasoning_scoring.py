"""Deterministic benchmark scoring; no model judge or network access.

MathVista normalization follows the pinned official calculate_score.py with
ignore_empty_extractions=True. Extraction is local, so this is explicitly not
the original GPT-assisted answer-extraction protocol. MIRA defaults to official
EM; explicitly selected versioned policies support offline format diagnostics.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, ROOT
from lvr.eval.mira_scoring import MIRA_POLICIES, score_mira_relaxed
from lvr.eval.answer_extraction import (ANSWER_POLICIES, MATHVISTA_POLICIES, MATHVISTA_METRICS,
                                       extract_structured_answer, extract_mathvista_number,
                                       mathvista_choice_payload, normalize_answer_format)


def final_answer(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "<think>" in text:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if match:
        return match[1].strip()
    if "<answer>" in text:
        return ""
    # Balanced braces support boxed lists and tuples without executing content.
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth = 1
        for i in range(start + 7, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start + 7:i].strip()
        return ""
    return re.sub(r"^(?:final\s+)?answer\s*:\s*", "", text, flags=re.I).strip().strip("`$")


def parse_moves(text):
    parts = re.split(r"[,;\s]+", text.strip("[](){} ."))
    aliases = {"u": "U", "up": "U", "d": "D", "down": "D", "l": "L", "left": "L", "r": "R", "right": "R"}
    if not parts or any(p.lower() not in aliases for p in parts):
        return None
    return [aliases[p.lower()] for p in parts]


DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def shortest_grid_path(grid):
    h, w = len(grid), len(grid[0])
    if any(len(line) != w for line in grid):
        raise ValueError("Nonrectangular map")
    starts = [(i, j) for i, line in enumerate(grid) for j, v in enumerate(line) if v == 1]
    goals = [(i, j) for i, line in enumerate(grid) for j, v in enumerate(line) if v == 2]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("Map must have one start and one goal")
    queue, seen = deque([(starts[0], [])]), {starts[0]}
    while queue:
        position, path = queue.popleft()
        if position == goals[0]:
            return path
        for action, (di, dj) in DELTA.items():
            i, j = position[0] + di, position[1] + dj
            if 0 <= i < h and 0 <= j < w and grid[i][j] != -1 and (i, j) not in seen:
                seen.add((i, j)); queue.append(((i, j), path + [action]))
    raise ValueError("Unsolvable map")


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def mathvista_normalize(text, row):
    if not text:
        return None
    config = row["scoring"]
    if config["question_type"] == "multi_choice":
        choices = row["options"]
        letters = re.findall(r"\(([a-zA-Z])\)", text)
        extraction = letters[0].upper() if letters else text.strip()
        if extraction in [chr(65+i) for i in range(len(choices))]:
            return choices[ord(extraction) - 65]
        return min(choices, key=lambda option: edit_distance(extraction, option))
    try:
        if config["answer_type"] == "integer":
            return str(int(float(text)))
        if config["answer_type"] == "float":
            value = float(text)
            return str(round(value, int(config["precision"]))) if math.isfinite(value) else None
        if config["answer_type"] == "list":
            return str(text)
    except (ValueError, OverflowError, TypeError):
        return None
    raise ValueError(f"Unsupported MathVista type: {config}")


@lru_cache(maxsize=1)
def rushhour_engine():
    # This code is pinned and copied into the suite by the builder. It needs
    # Shapely/Matplotlib; the supplied scoring command uses .venv-mentis.
    directory = Path(os.environ.get("LVR_REASONING_SUITE", DEFAULT_SUITE)) / "upstream/MentisOculi/rushhour"
    if not directory.is_dir():
        directory = ROOT / "data/raw/reasoning_expansion/mentis-oculi/datasets/rushhour"
    saved = {name: sys.modules.pop(name, None) for name in ("generator", "evaluate_responses")}
    sys.path.insert(0, str(directory))
    try:
        name = "_lvr_benchmark_rushhour"
        spec = importlib.util.spec_from_file_location(name, directory / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


def score(row, prediction, *, mira_policy="strict", answer_policy="strict", truncated=False,
          mathvista_policy="strict"):
    kind = row["scoring"]["kind"]
    if mira_policy not in MIRA_POLICIES:
        raise ValueError(f"Unknown MIRA policy: {mira_policy}")
    if answer_policy not in ANSWER_POLICIES:
        raise ValueError(f"Unknown answer policy: {answer_policy}")
    if mathvista_policy not in MATHVISTA_POLICIES:
        raise ValueError(f"Unknown MathVista policy: {mathvista_policy}")
    if answer_policy == "format_compat_v1":
        result = score(row, prediction, mira_policy=mira_policy, answer_policy="strict",
                       truncated=truncated, mathvista_policy=mathvista_policy)
        result.update(answer_policy=answer_policy, format_transform="unchanged")
        if not result["parsed"]:
            # Pass only visible piece labels, never solution metadata.
            labels = [piece["label"] for piece in row["scoring"].get("state", {}).get("pieces", [])]
            repair = normalize_answer_format(prediction, kind=kind, options=row.get("options", []),
                                             set_labels=labels, truncated=truncated)
            if repair:
                repaired = score(row, "<answer>" + repair["text"] + "</answer>",
                                 mira_policy=mira_policy, answer_policy="strict",
                                 truncated=False, mathvista_policy=mathvista_policy)
                if repaired["parsed"]:
                    result = {**repaired, "answer_policy": answer_policy,
                              "format_transform": repair["format_transform"],
                              "format_payload": repair["format_payload"]}
                    if kind == "choice":
                        result["normalized_choice"] = repair["text"]
        return result
    if (mathvista_policy == "numeric_v1" and kind == "mathvista"
            and row["scoring"]["question_type"] == "free_form"
            and row["scoring"]["answer_type"] in ("integer", "float")):
        extraction = extract_mathvista_number(prediction, truncated=truncated)
        normalized = mathvista_normalize(extraction["text"], row) if not extraction["failure_reason"] else None
        return {"correct": normalized is not None and normalized == row["answer"],
                "parsed": normalized is not None, "extracted": extraction["text"],
                "normalized": normalized, "answer_payload": extraction["payload"],
                "metric": MATHVISTA_METRICS[mathvista_policy], "mathvista_policy": mathvista_policy,
                "extraction_source": extraction["extraction_source"],
                "failure_reason": extraction["failure_reason"] or (None if normalized is not None else "invalid_numeric_value")}
    if answer_policy != "strict" and truncated:
        return {"correct": False, "parsed": False, "extracted": "", "metric": kind,
                "answer_policy": answer_policy, "extraction_source": "none",
                "failure_reason": "generation_token_limit"}
    if kind in ("mira", "mira_pair") and mira_policy != "strict":
        return score_mira_relaxed(row, prediction, mira_policy)
    # MIRA retains its separately selected tag/normalization policy.
    extraction = (extract_structured_answer(prediction) if answer_policy != "strict"
                  and kind not in ("mira", "mira_pair") else None)
    text = extraction["text"] if extraction else final_answer(prediction)
    result = {"correct": False, "parsed": bool(text), "extracted": text, "metric": kind}
    if extraction is not None:
        result.update(answer_policy=answer_policy,
                      extraction_source=extraction["extraction_source"],
                      failure_reason=extraction["failure_reason"])
    if not text:
        return result
    answer = row["answer"]
    if kind == "choice":
        match = re.fullmatch(r"\(?([A-Za-z])\)?[.]?", text)
        options = row.get("options", [])
        if match and (not options or ord(match[1].upper()) - 65 < len(options)):
            result["correct"] = match[1].upper() == answer
        elif text in options and options.count(text) == 1:
            result["correct"] = chr(65 + options.index(text)) == answer
        else:
            result["parsed"] = False
    elif kind == "set":
        tokens = re.split(r"[,;\s]+", text.strip("[]{}() ."))
        result["parsed"] = bool(tokens) and all(re.fullmatch("[A-Z]", t) for t in tokens)
        result["correct"] = result["parsed"] and set(tokens) == set(answer.split())
    elif kind == "hinge":
        pairs = re.findall(r"([A-Z])\s+(-?\d+)", text)
        remaining = re.sub(r"([A-Z])\s+(-?\d+)", "", text)
        result["parsed"] = bool(pairs) and not remaining.strip(" ,;") and len({k for k, v in pairs}) == len(pairs)
        expected = set(re.findall(r"([A-Z])\s+(-?\d+)", answer))
        result["correct"] = result["parsed"] and set(pairs) == expected
        result["metric"] = "official_rotation_set_exact"
    elif kind in ("grid", "sliding"):
        actions = parse_moves(text)
        result["parsed"] = actions is not None
        if actions is None:
            if extraction is not None:
                result["failure_reason"] = "invalid_action_sequence"
            return result
        if kind == "grid":
            grid = row["scoring"]["grid"]
            state = next((i, j) for i, line in enumerate(grid) for j, v in enumerate(line) if v == 1)
            legal = True
            for action in actions:
                di, dj = DELTA[action]; i, j = state[0] + di, state[1] + dj
                if not (0 <= i < len(grid) and 0 <= j < len(grid[0])):
                    continue  # Official VSP instruction: boundary moves have no effect.
                if grid[i][j] == -1:
                    legal = False; break
                state = (i, j)
                if grid[i][j] == 2:
                    break  # Environment terminates when the goal is reached.
            correct = legal and grid[state[0]][state[1]] == 2
            optimum = len(shortest_grid_path(grid))
        else:
            m = row["scoring"]["state"]
            grid = copy.deepcopy(m["initial_state"]); legal = True
            for action in actions:
                blank = [(i, j) for i, line in enumerate(grid) for j, v in enumerate(line) if v == -1]
                if len(blank) != 1:
                    raise ValueError("Malformed sliding state")
                i, j = blank[0]; di, dj = DELTA[action]; x, y = i + di, j + dj
                if not (0 <= x < len(grid) and 0 <= y < len(grid[0])):
                    legal = False; break
                grid[i][j], grid[x][y] = grid[x][y], grid[i][j]
            correct = legal and grid == m["target_state"]
            optimum = len(m["solution_moves"])
        result.update(correct=correct, legal=legal, shortest=correct and len(actions) == optimum,
                      actions=len(actions), reference_actions=optimum, metric="executed_goal_success")
    elif kind == "rushhour":
        pairs = re.findall(r"\b(Red|[A-Z])\s+(forward|backward)\b", text, re.I)
        remaining = re.sub(r"\b(Red|[A-Z])\s+(forward|backward)\b", "", text, flags=re.I)
        m = row["scoring"]["state"]
        labels = {("R" if o["id"] == "red_car" else o.get("label")): o["id"] for o in m["objects"] if o.get("movable", True)}
        labels["RED"] = "red_car"
        result["parsed"] = bool(pairs) and not remaining.strip(" ,;[].") and all(k.upper() in labels for k, _ in pairs)
        if result["parsed"]:
            actions = [{"object_id": labels[k.upper()], "direction": 1 if d.lower() == "forward" else -1} for k, d in pairs]
            result["correct"] = bool(rushhour_engine().execute_abstract_sequence(m, actions))
            result.update(actions=len(actions), reference_actions=len(m["actions"]),
                          reference_length_or_better=result["correct"] and len(actions) <= len(m["actions"]))
        result["metric"] = "executed_maximal_translation_goal"
    elif kind == "mathvista":
        if extraction is not None and row["scoring"]["question_type"] == "multi_choice":
            text = mathvista_choice_payload(text, row["options"])
            if text is None:
                result.update(parsed=False, failure_reason="ambiguous_or_invalid_choice")
                return result
        normalized = mathvista_normalize(text, row)
        result.update(correct=normalized is not None and normalized == answer, parsed=normalized is not None,
                      normalized=normalized, metric="official_normalization_local_extraction")
    elif kind in ("mira", "mira_pair"):
        # Official MIRA EM requires the answer tag, including for the two-part task.
        tagged = re.search(r"<answer>(.*?)</answer>", prediction, re.S)
        result["parsed"] = bool(tagged and tagged[1].strip())
        if tagged:
            text = tagged[1].strip()
            if kind == "mira_pair":
                color = re.search(r"\b(red|black|same)\b", text, re.I)
                nums = re.findall(r"\d+", text)
                result["parsed"] = bool(color and nums)
                result["correct"] = bool(color and nums and color[1].lower() == str(row["scoring"]["answer1"]).strip().lower()
                                         and int(nums[-1]) == int(row["scoring"]["answer2"]))
            else:
                result["correct"] = text.lower() == str(answer).strip().lower()
        result["metric"] = "official_mira_em"
    else:
        raise ValueError(f"Unknown scoring kind: {kind}")
    if extraction is not None and not result["parsed"] and not result["failure_reason"]:
        result["failure_reason"] = "invalid_answer_schema"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mira-policy", choices=MIRA_POLICIES, default="strict",
                        help="Opt-in nonofficial MIRA extraction policy; other benchmarks are unchanged")
    parser.add_argument("--answer-policy", choices=ANSWER_POLICIES, default="strict",
                        help="Opt-in wrapper repair with ambiguity and generation-limit checks; MIRA has a separate policy")
    args = parser.parse_args()
    with args.input.open() as source, args.output.open("w") as dest:
        for line in source:
            row = json.loads(line)
            dest.write(json.dumps({"id": row["record"]["id"],
                                   **score(row["record"], row["prediction"], mira_policy=args.mira_policy,
                                           answer_policy=args.answer_policy,
                                           truncated=row.get("reached_token_limit", False))}) + "\n")


if __name__ == "__main__":
    main()
