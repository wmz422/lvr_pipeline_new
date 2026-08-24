#!/usr/bin/env python3
"""Generate PNG benchmark auxiliary images from official or Qwen grounding boxes.

V*Bench uses its official ``[x, y, width, height]`` annotations. HR-Bench,
MME-RealWorld-Lite, and the BLINK Counting/Spatial Relation validation splits
use answer-conditioned Qwen3-VL 2D grounding. Results are appended to JSONL so
long inference jobs can resume safely and the generated boxes remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image
from tqdm import tqdm

from auxiliary_image import draw_auxiliary_pil


BENCHMARKS = (
    "vstar",
    "hr_bench_4k",
    "hr_bench_8k",
    "mme_realworld_lite",
    "blink_counting",
    "blink_spatial_relation",
)
BLINK_CONFIGS = {
    "blink_counting": "Counting",
    "blink_spatial_relation": "Spatial_Relation",
}
BOX_KEYS = {"bbox", "bbox_2d", "box", "box_2d"}


@dataclass
class Sample:
    benchmark: str
    sample_id: str
    question: str
    answer: str
    output_relative: Path
    image_path: Path | None = None
    image: Image.Image | None = None
    official_targets: list[dict[str, Any]] | None = None
    cache_key: str | None = None
    metadata: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    script_data_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=script_data_root.parent / "benchmark" / "bench",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_data_root / "images" / "auxiliary" / "benchmark",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/data/private/wmz/model_weights/Qwen3-VL-32B-Instruct"),
    )
    parser.add_argument("--max-pixels", type=int, default=4_194_304)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--device-map",
        default="balanced",
        help="Transformers device map. 'balanced' is recommended for a two-GPU worker.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def grounding_prompt(question: str, answer: str) -> str:
    return f"""You are creating oracle visual grounding annotations for training a visual-difference encoder.

Inspect the image together with the question and its known correct answer. Locate every minimal image region that provides visual evidence for that answer.

Requirements:
1. Return all relevant instances, not only one. Counting questions must include every counted instance.
2. Spatial-relation questions must include every object participating in the relation.
3. OCR questions must include the relevant text region.
4. Boxes must be tight around the visual evidence, while fully containing the target.
5. Use coordinates relative to the original image, normalized to integer values from 0 to 1000.
6. Return JSON only, without Markdown or explanation.

Required schema:
{{"targets": [{{"label": "short object or region description", "bbox_2d": [x1, y1, x2, y2]}}]}}

Question: {question}
Known correct answer: {answer}
"""


class QwenGrounder:
    def __init__(
        self,
        model_path: Path,
        max_pixels: int,
        max_new_tokens: int,
        device_map: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen grounding requires torch, transformers, and accelerate"
            ) from exc

        if not model_path.is_dir():
            raise FileNotFoundError(f"Qwen model directory does not exist: {model_path}")

        self.torch = torch
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        ).eval()

    def generate(self, image: Image.Image | Path, question: str, answer: str) -> str:
        image_value: Any
        if isinstance(image, Path):
            image_value = str(image.resolve())
        else:
            image_value = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_value,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": grounding_prompt(question, answer)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def iter_samples(benchmark_root: Path, benchmark: str) -> Iterator[Sample]:
    if benchmark == "vstar":
        yield from iter_vstar(benchmark_root)
    elif benchmark in {"hr_bench_4k", "hr_bench_8k"}:
        yield from iter_hr_bench(benchmark_root, benchmark)
    elif benchmark == "mme_realworld_lite":
        yield from iter_mme_realworld(benchmark_root)
    else:
        yield from iter_blink(benchmark_root, benchmark)


def iter_vstar(root: Path) -> Iterator[Sample]:
    source_root = root / "vstar_bench"
    for annotation_path in sorted(source_root.glob("*/*.json")):
        annotation = load_json(annotation_path)
        image_path = annotation_path.with_suffix(".jpg")
        if not image_path.is_file():
            image_path = find_image(annotation_path.with_suffix(""))

        raw_boxes = annotation.get("bbox") or []
        names = annotation.get("target_object") or []
        targets = []
        for index, box in enumerate(raw_boxes):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x, y, width, height = (float(value) for value in box)
            label = str(names[index] if index < len(names) else names[0] if names else "target")
            targets.append(
                {
                    "label": label,
                    "bbox_xyxy": [x, y, x + width, y + height],
                }
            )
        if not targets:
            continue

        relative = annotation_path.relative_to(source_root).with_suffix(".png")
        options = annotation.get("options") or []
        yield Sample(
            benchmark="vstar",
            sample_id=annotation_path.relative_to(source_root).with_suffix("").as_posix(),
            question=str(annotation.get("question", "")).strip(),
            answer=str(options[0] if options else "").strip(),
            image_path=image_path,
            output_relative=Path("vstar") / relative,
            official_targets=targets,
            metadata={"annotation_path": str(annotation_path)},
        )


def iter_hr_bench(root: Path, benchmark: str) -> Iterator[Sample]:
    dataset = load_dataset_from_disk(root / benchmark)
    image_root = root / benchmark / "images"
    for item in dataset:
        sample_id = str(item["index"])
        image_path = find_image(image_root / sample_id)
        answer_label = str(item.get("answer", "")).strip().upper()
        answer = str(item.get(answer_label, answer_label)).strip()
        question = str(item.get("question", "")).strip()
        yield Sample(
            benchmark=benchmark,
            sample_id=sample_id,
            question=question,
            answer=answer,
            image_path=image_path,
            output_relative=Path(benchmark) / f"{sample_id}.png",
            cache_key=f"{benchmark}:{file_digest(image_path)}:{question}:{answer}",
            metadata={
                "answer_label": answer_label,
                "category": item.get("category"),
                "cycle_category": item.get("cycle_category"),
            },
        )


def iter_mme_realworld(root: Path) -> Iterator[Sample]:
    benchmark = "mme_realworld_lite"
    dataset = load_dataset_from_disk(root / benchmark)
    image_root = root / benchmark / "images"
    for item in dataset:
        sample_id = str(item["index"])
        image_path = find_image(image_root / sample_id)
        answer_label = str(item.get("answer", "")).strip().upper()
        options = item.get("multi-choice options") or []
        answer = option_text_for_label(options, answer_label)
        question = str(item.get("question", "")).strip()
        yield Sample(
            benchmark=benchmark,
            sample_id=sample_id,
            question=question,
            answer=answer or answer_label,
            image_path=image_path,
            output_relative=Path(benchmark) / f"{sample_id}.png",
            cache_key=f"{benchmark}:{sample_id}:{question}:{answer}",
            metadata={
                "answer_label": answer_label,
                "category": item.get("category"),
                "subcategory": item.get("l2-category"),
            },
        )


def iter_blink(root: Path, benchmark: str) -> Iterator[Sample]:
    config = BLINK_CONFIGS[benchmark]
    dataset = load_dataset_from_disk(root / "BLINK" / config, split="val")
    for item in dataset:
        sample_id = str(item["idx"])
        answer_label = extract_answer_label(item.get("answer"))
        choices = item.get("choices") or []
        answer_index = ord(answer_label) - ord("A") if answer_label else -1
        answer = str(choices[answer_index]).strip() if 0 <= answer_index < len(choices) else answer_label
        question = str(item.get("question", "")).strip()
        images = [item.get(f"image_{index}") for index in range(1, 5)]
        images = [image for image in images if image is not None]
        for image_index, image in enumerate(images, start=1):
            suffix = f"_image{image_index}" if len(images) > 1 else ""
            image_id = f"{sample_id}{suffix}"
            yield Sample(
                benchmark=benchmark,
                sample_id=image_id,
                question=question,
                answer=answer,
                image=image.convert("RGB"),
                output_relative=Path("blink") / config / "val" / f"{image_id}.png",
                cache_key=f"{benchmark}:{image_id}:{question}:{answer}",
                metadata={
                    "blink_id": sample_id,
                    "image_index": image_index,
                    "answer_label": answer_label,
                    "choices": list(choices),
                },
            )


def load_dataset_from_disk(path: Path, split: str | None = None):
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("The datasets package is required") from exc
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        requested = split or ("test" if "test" in dataset else "train")
        return dataset[requested]
    return dataset


def process_benchmark(
    args: argparse.Namespace,
    benchmark: str,
    grounder: QwenGrounder | None,
) -> dict[str, int]:
    annotation_path = annotation_file(args.output_root, benchmark, args.num_shards, args.shard_index)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    prior = load_prior_results(annotation_path)
    cache = {
        str(record["cache_key"]): record
        for record in prior.values()
        if record.get("status") == "ok" and record.get("cache_key")
    }
    counts = {"ok": 0, "skipped": 0, "failed": 0, "cache_hits": 0}
    selected = 0

    samples = iter_samples(args.benchmark_root, benchmark)
    with annotation_path.open("a", encoding="utf-8") as annotation_handle:
        for sample in tqdm(samples, desc=f"{benchmark}[{args.shard_index}/{args.num_shards}]"):
            shard_key = sample.cache_key or f"{benchmark}:{sample.sample_id}"
            if stable_shard(shard_key, args.num_shards) != args.shard_index:
                continue
            if args.limit is not None and selected >= args.limit:
                break
            selected += 1

            output_path = args.output_root / sample.output_relative
            previous = prior.get(sample.sample_id)
            if not args.overwrite and previous and output_path.is_file():
                if previous.get("status") == "ok" or not args.retry_failed:
                    counts["skipped"] += 1
                    continue

            try:
                raw_output = None
                coordinate_space = "pixel_xyxy"
                source = "official"
                cached_from = None
                if sample.official_targets is not None:
                    targets = sample.official_targets
                elif sample.cache_key and sample.cache_key in cache:
                    cached = cache[sample.cache_key]
                    targets = list(cached["targets"])
                    raw_output = cached.get("raw_output")
                    coordinate_space = str(cached.get("coordinate_space", "normalized_0_1000"))
                    source = str(cached.get("source", "qwen3-vl-32b"))
                    cached_from = cached.get("sample_id")
                    counts["cache_hits"] += 1
                else:
                    if grounder is None:
                        raise RuntimeError(f"Qwen grounder was not loaded for {benchmark}")
                    image_value = sample.image_path if sample.image_path is not None else sample.image
                    if image_value is None:
                        raise RuntimeError("sample has no image")
                    raw_output = grounder.generate(image_value, sample.question, sample.answer)
                    width, height = sample_image_size(sample)
                    targets = parse_grounding_targets(raw_output, width, height)
                    if not targets:
                        payload = extract_json_payload(raw_output)
                        if not (
                            isinstance(payload, dict)
                            and payload.get("targets") == []
                        ):
                            raise ValueError(
                                f"Qwen returned no valid boxes: {raw_output[:500]}"
                            )
                    coordinate_space = "normalized_0_1000"
                    source = "qwen3-vl-32b-answer-conditioned"

                render_sample(sample, output_path, targets)
                record = sample_record(
                    sample,
                    output_path,
                    status="ok",
                    source=source,
                    coordinate_space=coordinate_space,
                    targets=targets,
                    raw_output=raw_output,
                    cached_from=cached_from,
                )
                append_record(annotation_handle, record)
                prior[sample.sample_id] = record
                if sample.cache_key:
                    cache[sample.cache_key] = record
                counts["ok"] += 1
            except Exception as exc:
                record = sample_record(
                    sample,
                    output_path,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_text=traceback.format_exc(),
                )
                append_record(annotation_handle, record)
                prior[sample.sample_id] = record
                counts["failed"] += 1
                if args.fail_fast:
                    raise

    return counts


def parse_grounding_targets(raw_output: str, width: int, height: int) -> list[dict[str, Any]]:
    try:
        payload = extract_json_payload(raw_output)
    except ValueError:
        payload = {
            "targets": [
                {"label": "target", "bbox_2d": [float(value) for value in match]}
                for match in re.findall(
                    r'"bbox_2d"\s*:\s*\[\s*'
                    r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
                    r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
                    r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
                    r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]',
                    raw_output,
                )
            ]
        }
    candidates = collect_box_objects(payload)
    if not candidates and '"bbox_2d"' in raw_output:
        candidates = collect_box_objects(
            {
                "targets": [
                    {"label": "target", "bbox_2d": [float(value) for value in match]}
                    for match in re.findall(
                        r'"bbox_2d"\s*:\s*\[\s*'
                        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
                        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
                        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
                        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]',
                        raw_output,
                    )
                ]
            }
        )
    targets = []
    for candidate in candidates:
        raw_box = next((candidate[key] for key in BOX_KEYS if key in candidate), None)
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        pixel_box = [
            round(x1 / 1000.0 * width),
            round(y1 / 1000.0 * height),
            round(x2 / 1000.0 * width),
            round(y2 / 1000.0 * height),
        ]
        pixel_box[0] = max(0, min(width - 1, pixel_box[0]))
        pixel_box[1] = max(0, min(height - 1, pixel_box[1]))
        pixel_box[2] = max(0, min(width - 1, pixel_box[2]))
        pixel_box[3] = max(0, min(height - 1, pixel_box[3]))
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            continue
        label = candidate.get("label") or candidate.get("name") or candidate.get("description") or "target"
        targets.append(
            {
                "label": str(label),
                "bbox_normalized": [x1, y1, x2, y2],
                "bbox_xyxy": pixel_box,
            }
        )
    return deduplicate_targets(targets)


def extract_json_payload(text: str) -> Any:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    positions = sorted(position for token in ("{", "[") if (position := cleaned.find(token)) >= 0)
    for position in positions:
        try:
            payload, _ = decoder.raw_decode(cleaned[position:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse JSON from Qwen output: {text[:500]}")


def collect_box_objects(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if any(key in value for key in BOX_KEYS):
            found.append(value)
        for child in value.values():
            found.extend(collect_box_objects(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(collect_box_objects(child))
    return found


def deduplicate_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for target in targets:
        key = tuple(target["bbox_xyxy"])
        if key not in seen:
            seen.add(key)
            unique.append(target)
    return unique


def render_sample(sample: Sample, output_path: Path, targets: Iterable[dict[str, Any]]) -> None:
    boxes = [tuple(target["bbox_xyxy"]) for target in targets]
    if sample.image is not None:
        auxiliary = draw_auxiliary_pil(sample.image, boxes)
    elif sample.image_path is not None:
        with Image.open(sample.image_path) as image:
            auxiliary = draw_auxiliary_pil(image, boxes)
    else:
        raise RuntimeError("sample has no image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    auxiliary.save(output_path, format="PNG", compress_level=1)


def sample_image_size(sample: Sample) -> tuple[int, int]:
    if sample.image is not None:
        return sample.image.size
    if sample.image_path is not None:
        with Image.open(sample.image_path) as image:
            return image.size
    raise RuntimeError("sample has no image")


def sample_record(
    sample: Sample,
    output_path: Path,
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "benchmark": sample.benchmark,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "answer": sample.answer,
        "output_path": str(output_path),
        "cache_key": sample.cache_key,
        "metadata": sample.metadata or {},
        "status": status,
    }
    record.update({key: value for key, value in extra.items() if value is not None})
    return record


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def load_prior_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            records[str(record["sample_id"])] = record
    return records


def annotation_file(root: Path, benchmark: str, num_shards: int, shard_index: int) -> Path:
    if num_shards == 1:
        name = f"{benchmark}.jsonl"
    else:
        name = f"{benchmark}.shard-{shard_index:02d}-of-{num_shards:02d}.jsonl"
    return root / "annotations" / name


def stable_shard(value: str, num_shards: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_text_for_label(options: Iterable[Any], label: str) -> str:
    pattern = re.compile(r"^\s*\(([A-Z])\)\s*(.*)$")
    for index, value in enumerate(options):
        text = str(value).strip()
        match = pattern.match(text)
        option_label = match.group(1) if match else chr(ord("A") + index)
        option_text = match.group(2).strip() if match else text
        if option_label == label:
            return option_text
    return ""


def extract_answer_label(value: Any) -> str:
    match = re.search(r"[A-Z]", str(value).upper())
    return match.group(0) if match else ""


def find_image(stem: Path) -> Path:
    if stem.is_file():
        return stem
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"):
        candidate = stem.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find image for {stem}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def main() -> None:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_root = args.output_root.resolve()
    args.model = args.model.resolve()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    qwen_benchmarks = [benchmark for benchmark in args.benchmarks if benchmark != "vstar"]
    grounder = None
    if qwen_benchmarks:
        grounder = QwenGrounder(
            args.model,
            args.max_pixels,
            args.max_new_tokens,
            args.device_map,
        )

    summaries = {}
    for benchmark in args.benchmarks:
        summaries[benchmark] = process_benchmark(args, benchmark, grounder)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if any(summary["failed"] for summary in summaries.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
