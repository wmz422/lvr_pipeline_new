#!/usr/bin/env python3
"""Generate crop-grounded data for the alignment stage.

Each LLM record becomes one alignment record.  The target visual evidence is
read from the record's ``box``/``boxes`` fields when present; TreeVGR records
instead resolve their boxes from ``raw.index`` in the source parquet.  Every
box is cropped separately and all crops are shown to Qwen in a deterministic
left-to-right, then top-to-bottom spatial order.

The worker mode writes JSONL so eight independently running GPU processes can
resume safely.  ``--merge`` turns their output into the normal JSON arrays
consumed directly by ``LatentDataModule(stage='align')``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from dataset_utils import data_root, load_json_list, normalize_version, split_json_path


ALIGNMENT_QUESTION_ONE = (
    "Given the original image and the latent visual information for one localized visual "
    "region, describe only the visible object, attribute, action, or readable text "
    "represented by that latent information in one concise sentence."
)
ALIGNMENT_QUESTION_MANY = (
    "Given the original image and the latent visual information for one or more localized "
    "visual regions, describe only what is visible in those regions in one concise sentence."
)
DEFAULT_MODEL = Path("/data/private/wmz/model_weights/Qwen3-VL-32B-Instruct")


@dataclass(frozen=True)
class CropSpec:
    box: tuple[float, float, float, float]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--input-version", default="v1")
    parser.add_argument("--version", default="v0")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--crop-subdir", default="crops",
                        help="Crop directory name below --output-dir (default: crops).")
    parser.add_argument("--observation-source", default="qwen3-vl-crop-only",
                        help="Provenance string stored in every generated record.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process the first N input records (before sharding).")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device-map", default="cuda:0",
                        help="Transformers device map; one worker normally owns one GPU.")
    parser.add_argument("--allocator-warmup", action="store_true",
                        help="Enable Transformers' temporary full-model CUDA allocator warmup."
                             " Disabled by default so a 32B BF16 model fits on one 80GB GPU.")
    parser.add_argument("--crop-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--context-ratio", type=float, default=0.0,
                        help="Optional symmetric context expansion around each box.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all worker JSONL files; does not load Qwen.")
    parser.add_argument("--validate", action="store_true",
                        help="Validate crop resolution and source boxes without loading Qwen.")
    return parser.parse_args()


def worker_path(output_dir: Path, split: str, limit: int | None, shard_index: int) -> Path:
    name = f"{split}.part-{shard_index:02d}"
    if limit is not None:
        name += f".first{limit}"
    return output_dir / f"{name}.jsonl"


def final_path(output_dir: Path, split: str, limit: int | None) -> Path:
    suffix = f".first{limit}" if limit is not None else ""
    return output_dir / f"{split}{suffix}.json"


def as_box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in values)
    except (TypeError, ValueError):
        return None
    if not all(value == value and abs(value) != float("inf") for value in (x1, y1, x2, y2)):
        return None
    return x1, y1, x2, y2


class BoxResolver:
    """Resolve explicit LLM boxes and lazily load TreeVGR's raw annotations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._treevgr: list[Any] | None = None

    def _treevgr_rows(self) -> list[Any]:
        if self._treevgr is None:
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("TreeVGR alignment generation requires pandas and pyarrow.") from exc
            path = self.root / "raw" / "TreeVGR-RL-37K" / "vstar30k_visdrone6k_x1y1x2y2.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"TreeVGR metadata is missing: {path}")
            self._treevgr = list(pd.read_parquet(path)["target_instances"])
        return self._treevgr

    def resolve(self, record: dict[str, Any]) -> list[CropSpec]:
        direct: list[CropSpec] = []
        if (box := as_box(record.get("box"))) is not None:
            direct.append(CropSpec(box, "record.box"))
        for value in record.get("boxes", []):
            if (box := as_box(value)) is not None:
                direct.append(CropSpec(box, "record.boxes"))
        if direct:
            return _deduplicate(direct)

        raw = record.get("raw") or {}
        if raw.get("dataset") != "TreeVGR-RL-37K":
            return []
        index = raw.get("index")
        if not isinstance(index, int) or not 0 <= index < len(self._treevgr_rows()):
            raise ValueError(f"invalid TreeVGR raw.index: {index!r}")
        boxes: list[CropSpec] = []
        for instance in list(self._treevgr_rows()[index]):
            box = as_box(instance.get("bbox") if isinstance(instance, dict) else None)
            if box is not None:
                boxes.append(CropSpec(box, "TreeVGR.target_instances"))
        return _deduplicate(boxes)


def _deduplicate(specs: Iterable[CropSpec]) -> list[CropSpec]:
    result: list[CropSpec] = []
    seen: set[tuple[float, float, float, float]] = set()
    for spec in specs:
        key = tuple(round(value, 6) for value in spec.box)
        if key not in seen:
            seen.add(key)
            result.append(spec)
    return result


def spatially_ordered(specs: Iterable[CropSpec]) -> list[CropSpec]:
    """Return a stable visual order: left → right, then top → bottom.

    The same order is used for stored ``boxes``, generated crop file names, and
    Qwen's multi-image input.  It therefore cannot vary with an annotation
    source's arbitrary instance order.
    """
    def key(spec: CropSpec) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = spec.box
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    return sorted(specs, key=key)


def crop_image(image: Image.Image, box: tuple[float, float, float, float], context_ratio: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if context_ratio:
        pad_x = (right - left) * context_ratio
        pad_y = (bottom - top) * context_ratio
        left, right = left - pad_x, right + pad_x
        top, bottom = top - pad_y, bottom + pad_y
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(left + 1, min(width, int(round(right))))
    bottom = max(top + 1, min(height, int(round(bottom))))
    if right <= left or bottom <= top:
        raise ValueError(f"box has no area after clipping: {box!r} for {image.size}")
    return image.crop((left, top, right, bottom)).convert("RGB")


def observation_prompt(count: int) -> str:
    if count == 1:
        return (
            "Describe only the visible object, attribute, action, or readable text. Return exactly "
            "one concise factual sentence that starts directly with that visual content. Never use "
            "meta phrases such as 'The image shows', 'This image', 'The picture', 'The photo', "
            "or 'The crop'. Do not mention bounding boxes or anything outside the visual content."
        )
    return (
        f"These {count} cropped images are separate marked regions from one original image. "
        "They are presented in a fixed spatial order: from left to right, then from top to bottom. "
        "Describe exactly one visible region for each image in that exact input order. Return one concise "
        "factual sentence with one semicolon-separated clause per image. Start every clause directly with "
        "the visual content. Never use meta phrases such as 'The image shows', 'This image', 'The picture', "
        "'The photo', or 'The crop'. Do not mention bounding boxes, ordering, or anything outside the images."
    )


class QwenDescriber:
    def __init__(self, model_path: Path, *, max_pixels: int, max_new_tokens: int, device_map: str,
                 allocator_warmup: bool) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, modeling_utils
        except ImportError as exc:
            raise RuntimeError("Qwen generation requires torch, transformers, and accelerate.") from exc
        if not model_path.is_dir():
            raise FileNotFoundError(f"Qwen model directory does not exist: {model_path}")
        self.torch = torch
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        # transformers>=5.15 briefly allocates another full model to speed up loading.
        # Qwen3-VL-32B BF16 itself needs ~63GB, so that temporary allocation OOMs on
        # an 80GB card.  Skipping it costs startup time only and permits one worker/GPU.
        if not allocator_warmup:
            modeling_utils.caching_allocator_warmup = lambda *_args, **_kwargs: None
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, device_map=device_map,
            low_cpu_mem_usage=True,
        ).eval()

    def describe(self, crops: list[Image.Image]) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image, "max_pixels": self.max_pixels}
            for image in crops
        ]
        content.append({"type": "text", "text": observation_prompt(len(crops))})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        generated = generated[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return clean_observation(text)


def clean_observation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = " ".join(text.split())
    if not text:
        raise ValueError("Qwen returned an empty observation")
    return text


def build_item(record: dict[str, Any], *, record_index: int, crops_rel: list[str], specs: list[CropSpec],
               observation: str, observation_source: str) -> dict[str, Any]:
    item = {
        "question_image": record["question_image"],
        "auxiliary_image": record["auxiliary_image"],
        "question": ALIGNMENT_QUESTION_ONE if len(specs) == 1 else ALIGNMENT_QUESTION_MANY,
        "observation": observation,
        "boxes": [[round(value, 4) for value in spec.box] for spec in specs],
        "crop_images": crops_rel,
        "observation_source": observation_source,
        "raw": record.get("raw", {}),
        "source_index": record_index,
    }
    if len(specs) == 1:
        item["box"] = item["boxes"][0]
        item["crop_image"] = crops_rel[0]
    return item


def selected_records(records: list[dict[str, Any]], limit: int | None, num_shards: int, shard_index: int):
    upper = len(records) if limit is None else min(len(records), limit)
    for index in range(upper):
        if index % num_shards == shard_index:
            yield index, records[index]


def run_worker(args: argparse.Namespace, root: Path, output_dir: Path) -> None:
    records = load_json_list(split_json_path(root, "llm", args.input_version, args.split))
    output = worker_path(output_dir, args.split, args.limit, args.shard_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    if output.exists() and not args.overwrite:
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(int(json.loads(line)["source_index"]))
    elif args.overwrite and output.exists():
        output.unlink()

    resolver = BoxResolver(root)
    describer = None if args.validate else QwenDescriber(
        args.model.resolve(), max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens, device_map=args.device_map,
        allocator_warmup=args.allocator_warmup,
    )
    written = skipped = failed = 0
    with output.open("a", encoding="utf-8") as handle:
        for index, record in selected_records(records, args.limit, args.num_shards, args.shard_index):
            if index in completed:
                skipped += 1
                continue
            try:
                specs = spatially_ordered(resolver.resolve(record))
                if not specs:
                    raise ValueError("no usable target boxes")
                source = root / record["question_image"]
                if not source.is_file():
                    raise FileNotFoundError(f"question image is missing: {source}")
                with Image.open(source) as image:
                    crops = [crop_image(image, spec.box, args.context_ratio) for spec in specs]
                crops_rel = []
                crop_dir = output_dir / args.crop_subdir / args.split
                for crop_index, crop in enumerate(crops):
                    crop_path = crop_dir / f"{index:06d}_{crop_index}.{args.crop_format}"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.overwrite or not crop_path.exists():
                        if args.crop_format == "jpg":
                            crop.save(crop_path, quality=95, subsampling=0)
                        else:
                            crop.save(crop_path)
                    crops_rel.append(crop_path.relative_to(root).as_posix())
                if args.validate:
                    observation = "__VALIDATED_NO_QWEN_DESCRIPTION__"
                else:
                    assert describer is not None
                    observation = describer.describe(crops)
                item = build_item(
                    record, record_index=index, crops_rel=crops_rel, specs=specs,
                    observation=observation, observation_source=args.observation_source,
                )
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
            except Exception as exc:  # Preserve successful work and make a bad source auditable.
                failed += 1
                print(f"FAILED split={args.split} index={index}: {exc}", file=sys.stderr, flush=True)
    print(json.dumps({"worker": str(output), "written": written, "resumed": skipped, "failed": failed}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


def merge(args: argparse.Namespace, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        path = worker_path(output_dir, args.split, args.limit, shard_index)
        if not path.is_file():
            raise FileNotFoundError(f"missing worker output: {path}")
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows.sort(key=lambda row: row["source_index"])
    indices = [row["source_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate source_index across worker outputs")
    destination = final_path(output_dir, args.split, args.limit)
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "records": len(rows)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < --num-shards")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.context_ratio < 0:
        raise ValueError("--context-ratio must be non-negative")
    root = args.data_root.resolve()
    normalize_version(args.input_version)
    normalize_version(args.version)
    output_dir = (args.output_dir or root / "align" / args.version).resolve()
    if args.merge:
        merge(args, output_dir)
    else:
        run_worker(args, root, output_dir)


if __name__ == "__main__":
    main()
