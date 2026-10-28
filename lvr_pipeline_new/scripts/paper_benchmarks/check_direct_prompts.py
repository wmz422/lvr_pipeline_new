"""Check direct-answer instructions and actual Qwen processor inputs on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from lvr.eval.paper_benchmark import DEFAULT_SUITE, ROOT, answer_parsing_metadata, file_sha, load_suite, model_item, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, protocol, _ = load_suite(args.suite)
    inference = protocol["inference"]
    instruction = "\nReturn only the final answer inside <answer>...</answer>, without an explanation."
    if inference["answer_instruction"] != instruction:
        raise ValueError("Unexpected direct-answer instruction")
    patterns = {
        "step_by_step": r"(?:think|reason|explain|solve|work|answer).{0,50}step[- ]by[- ]step|let\W?s think",
        "explanation_request": r"(?:explain|describe|show|provide|include|give) (?:your |the |a |an )?(?:reasoning|explanation|rationale|thought|analysis|work)",
        "think_tag": r"</?think>",
    }
    problems = []
    for row in rows:
        if not row["question"].endswith(instruction):
            problems.append({"id": row["id"], "reason": "missing_final_answer_instruction"})
        for name, pattern in patterns.items():
            if re.search(pattern, row["question"], re.I):
                problems.append({"id": row["id"], "reason": name})
    if problems:
        raise ValueError(f"Conflicting prompt instructions: {problems[:10]}")

    import torch
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQACollator
    from lvr.data.prompt import build_generation_prompt

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    prompt_hash = hashlib.sha256()
    rendered = {}
    for row in rows:
        prompt = build_generation_prompt(processor, row["question"], system=inference["system_prompt"],
                                         num_images=len(row["images"]))
        if not prompt.endswith("<|im_start|>assistant\n") or re.search(r"</?think>|<abs_vis_token", prompt):
            raise ValueError(f"Unexpected assistant prefix: {row['id']}")
        if prompt.count("<|image_pad|>") != len(row["images"]):
            raise ValueError(f"Wrong number of image placeholders: {row['id']}")
        rendered[row["id"]] = prompt
        prompt_hash.update(json.dumps([row["id"], prompt], ensure_ascii=False).encode() + b"\n")

    selected = [next(row for row in rows if row["benchmark"] == "VStarBench")]
    for answer in ("left", "right"):
        selected.append(next(row for row in rows if row["benchmark"] == "VStarBench"
                             and row["options"][ord(row["answer"]) - 65] == answer))
    selected += [next(row for row in rows if row["benchmark"] == "BLINK" and row["task"] == "Spatial_Relation"),
                 next(row for row in rows if row["benchmark"] == "SAT")]
    collator = CropQACollator(processor, args.suite, num_latents=0, original_pixels=None,
                             crop_pixels=262144, max_length=inference["max_input_tokens"],
                             system_prompt=inference["system_prompt"], include_answers=False, include_encoder=False)
    examples = []
    for row in selected:
        item = model_item(row, args.suite)
        changed = {**row, "answer": "DO_NOT_PASS_GOLD", "scoring": {}, "metadata": {"gold": "SECRET"}}
        if model_item(changed, args.suite) != item or item["answer"] != "" or "auxiliary_images" in item:
            raise ValueError(f"Gold/helper data enter generation inputs: {row['id']}")
        batch = collator([item])["qa"]
        if "labels" in batch or batch["slot_mask"].any():
            raise ValueError(f"Answer labels or latent slots enter generation: {row['id']}")
        ids = batch["input_ids"][0].tolist()
        decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
        compact = re.sub(r"(?:<\|image_pad\|>)+", "<|image_pad|>", decoded)
        if compact != rendered[row["id"]]:
            raise ValueError(f"Actual collated tokens differ from the chat prompt: {row['id']}")
        examples.append({"id": row["id"], "benchmark": row["benchmark"], "task": row["task"],
                         "prompt_with_repeated_image_tokens_collapsed": compact,
                         "input_tokens": len(ids), "image_grid_thw": batch["image_grid_thw"].tolist(),
                         "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                         "includes_gold_answer": False, "latent_slots": 0})
    report = {
        "status": "passed", "model": str(args.model.resolve()), "records": len(rows),
        "by_benchmark": dict(Counter(row["benchmark"] for row in rows)),
        "system_prompt": inference["system_prompt"], "answer_instruction": instruction,
        "all_questions_have_direct_answer_suffix": True,
        "conflicting_instructions_found_by_patterns": [], "checked_patterns": patterns,
        "all_rendered_assistant_prefixes_are_empty": True,
        "rendered_prompts_sha256": prompt_hash.hexdigest(),
        "real_collator_examples_checked": len(examples),
        "suite_manifest_sha256": file_sha(args.suite / "manifest.json"),
        "answer_parsing": answer_parsing_metadata(),
        "code_sha256": {str(path.relative_to(ROOT)): file_sha(path) for path in (
            Path(__file__).resolve(), ROOT / "src/lvr/data/prompt.py", ROOT / "src/lvr/data/crop_qa.py",
            ROOT / "src/lvr/eval/reasoning_benchmark.py")},
        "note": "Direct answering is a prompt instruction, not a decoder constraint guaranteeing every model output obeys it.",
    }
    write_json(args.output / "prompt_audit.json", report)
    write_json(args.output / "prompt_examples.json", examples)
    (args.output / "vstar_prompt.txt").write_text(examples[1]["prompt_with_repeated_image_tokens_collapsed"])
    print(f"PROMPT CHECK PASSED: {len(rows)} direct-answer prompts; {len(examples)} actual processor inputs verified.", flush=True)
    print(f"Prompt example (repeated image tokens collapsed): {args.output / 'vstar_prompt.txt'}", flush=True)


if __name__ == "__main__":
    main()
