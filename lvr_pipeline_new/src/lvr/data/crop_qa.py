"""Multi-crop encoding, QA targets, and optional image-free description targets."""

from __future__ import annotations

import math
import json
import gzip
from pathlib import Path

import lightning.pytorch as pl
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from lvr.data.prompt import build_generation_prompt, cap_qwen_pixels, strip_scaffold_tags
from lvr.data.records import read_records
from lvr.data.reasoning import SCHEMA, as_crop_qa_item
from lvr.data.length_manifest import load_plain_qa_lengths, select_sample_items
from lvr.tokens import IGNORE_INDEX


DESCRIPTION_PROMPT = (
    "Describe each visible object or region represented by the latent visual information in "
    "the input order, using one to three concise sentences. Include salient objects, "
    "attributes, actions, legible text, and diagram labels or annotations when present. "
    "Do not describe relative positions between regions and do not answer a question."
)


def crop_bounds(image_size, box):
    """Use the same clipped pixel coordinates for RGB extraction and geometry."""
    width, height = image_size
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError(f"Invalid bbox: {box}")
    x1, y1, x2, y2 = map(float, box)
    bounds = [max(0, math.floor(min(x1, x2))), max(0, math.floor(min(y1, y2))),
              min(width, math.ceil(max(x1, x2))), min(height, math.ceil(max(y1, y2)))]
    if x1 == x2 and 0 <= x1 < width:
        bounds[2] = bounds[0] + 1
    if y1 == y2 and 0 <= y1 < height:
        bounds[3] = bounds[1] + 1
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(f"Empty bbox {box} for image size {image_size}")
    return bounds


def crop_regions(image, boxes):
    """Crop pixel boxes in order; an in-image zero-width/height axis uses one pixel."""
    crops = [image.crop(crop_bounds(image.size, box)) for box in boxes]
    if not crops:
        raise ValueError("Each training sample must contain at least one crop")
    return crops


def resize_pixels(image, pixels, *, enlarge=False):
    """Preserve aspect ratio; crops use the full area budget, originals only shrink."""
    if pixels is None:
        return image  # Evaluation can delegate original-image resizing to the processor.
    width, height = image.size
    scale = math.sqrt(pixels / (width * height))
    if not enlarge:
        scale = min(scale, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC) if size != image.size else image


def grid_boxes(image_size, grid_size=2):
    """Partition an image in row-major order, without reading question or answer."""
    width, height = image_size
    if grid_size < 1 or min(width, height) < grid_size:
        raise ValueError(f"Cannot split image {image_size} into a {grid_size} x {grid_size} grid")
    return [[x * width // grid_size, y * height // grid_size,
             (x + 1) * width // grid_size, (y + 1) * height // grid_size]
            for y in range(grid_size) for x in range(grid_size)]


class CropQADataset(Dataset):
    def __init__(self, qa_path, boxes_path=None, limit=None, *, auxiliary_manifest_path=None,
                 caption_path=None):
        qa = read_records(qa_path, limit=limit)
        if qa and qa[0].get("schema") == SCHEMA:
            self.items = [as_crop_qa_item(item) for item in qa]
            if len({item["id"] for item in self.items}) != len(self.items):
                raise ValueError("Duplicate sample identifiers")
            return
        if boxes_path is None:
            raise ValueError("Legacy QA records require bbox annotations")
        annotations = read_records(boxes_path, limit=limit)
        if len(qa) != len(annotations):
            raise ValueError("QA and bbox splits must contain the same samples")
        def load_sidecar(path):
            if path is None:
                return {}
            path = Path(path)
            if path.name.endswith((".jsonl", ".jsonl.gz")):
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
            else:
                value = json.loads(path.read_text())
                rows = value if isinstance(value, list) else value["rows"]
            indexed = {row["id"]: row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f"Duplicate sidecar sample identifiers: {path}")
            return indexed

        auxiliary = load_sidecar(auxiliary_manifest_path)
        captions = load_sidecar(caption_path)
        self.items = []
        for item, annotation in zip(qa, annotations):
            for key in ("question_image", "auxiliary_image"):
                if item[key] != annotation[key]:
                    raise ValueError(f"QA/bbox sample mismatch: {item[key]} != {annotation[key]}")
            sample_id = f"{item['question_image']}:{item['raw']['index']}"
            if auxiliary_manifest_path is not None and sample_id not in auxiliary:
                raise ValueError(f"Auxiliary manifest has no sample: {sample_id}")
            aux_row = auxiliary.get(sample_id, {})
            caption_row = captions.get(sample_id, {})
            if caption_path is not None and sample_id not in captions:
                raise ValueError(f"Caption manifest has no sample: {sample_id}")
            if caption_path is not None and auxiliary_manifest_path is not None and caption_row.get("image_sha256") != aux_row.get("image_sha256"):
                raise ValueError(f"Caption/image checksum mismatch: {sample_id}")
            if "boxes" in caption_row and caption_row["boxes"] != annotation["boxes"]:
                raise ValueError(f"Caption/bbox mismatch: {sample_id}")
            self.items.append({
                "id": sample_id,
                "question_image": item["question_image"],
                "auxiliary_image": aux_row.get("auxiliary_image"),
                "question": item["question"], "answer": strip_scaffold_tags(item["answer"]),
                "boxes": annotation["boxes"],
                "description": strip_scaffold_tags(caption_row.get("caption", annotation.get("observation", ""))).strip(),
            })
        if len({item["id"] for item in self.items}) != len(self.items):
            raise ValueError("Duplicate sample identifiers")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CropQACollator:
    def __init__(self, processor, image_root, num_latents=8, original_pixels=1048576,
                 crop_pixels=262144, max_length=8192, system_prompt="You are a helpful assistant.",
                 include_answers=True, include_encoder=True, qa_crops=False, encoder_uses_question=False,
                 encoder_type="qwen", encoder_includes_original=True, include_descriptions=False,
                 encoder_image_mode="question_crops", description_image_mode="none",
                 include_reconstruction=False):
        self.processor = processor
        self.image_root = Path(image_root)
        self.num_latents = num_latents
        self.original_pixels = original_pixels
        self.crop_pixels = crop_pixels
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.include_answers = include_answers
        self.include_encoder = include_encoder
        self.qa_crops = qa_crops
        self.encoder_uses_question = encoder_uses_question
        self.encoder_type = encoder_type
        self.encoder_includes_original = encoder_includes_original
        self.include_descriptions = include_descriptions
        self.encoder_image_mode = encoder_image_mode
        self.description_image_mode = description_image_mode
        self.include_reconstruction = include_reconstruction
        if include_reconstruction and (not include_encoder or not num_latents or qa_crops):
            raise ValueError("Feature reconstruction requires latent encoding and original-only QA images")
        if include_descriptions and not num_latents:
            raise ValueError("Description supervision requires latent slots")
        if encoder_type not in ("qwen", "qformer"):
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        if encoder_image_mode not in ("question_crops", "auxiliary"):
            raise ValueError(f"Unknown encoder image mode: {encoder_image_mode}")
        if description_image_mode not in ("none", "original"):
            raise ValueError(f"Unknown description image mode: {description_image_mode}")
        self.pad_id = processor.tokenizer.pad_token_id
        self.end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.pad_id is None or self.end_id is None:
            raise ValueError("Qwen padding and end-of-turn tokens are required")

    def __call__(self, examples):
        a_texts, b_texts, a_images, b_images = [], [], [], []
        description_images, description_image_counts = [], []
        crop_counts = []
        qa_image_labels = []
        a_image_counts, a_metadata = [], []
        original_counts, target_indices, target_sample_ids, target_image_ids = [], [], [], []
        for sample_index, example in enumerate(examples):
            sample_targets = []
            mode = example.get("teacher_view_mode", self.encoder_image_mode)
            mode = "question_crops" if mode == "crops" else mode
            images = example.get("images")
            if images is None:
                images = [{"path": example["question_image"], "boxes": example["boxes"]}]
            if not images:
                raise ValueError("Each QA sample requires at least one image")
            original_counts.append(len(images))
            crop_count = sum(len(item["boxes"]) for item in images)
            crop_counts.append(crop_count)
            image_labels = []
            if mode == "auxiliary_images":
                helpers = example.get("auxiliary_images", [])
                if not helpers:
                    raise ValueError("Reasoning encoder mode requires ordered auxiliary_images")
                if self.qa_crops:
                    raise ValueError("Reasoning helpers must not enter QA crop inputs")
                a_image_counts.append(len(helpers) + (len(images) if self.encoder_includes_original else 0))
            elif mode == "auxiliary" and self.include_encoder:
                if not example.get("auxiliary_image"):
                    raise ValueError("Auxiliary encoder mode requires auxiliary_image in each example")
                a_image_counts.append(1)
                with Image.open(example["auxiliary_image"]) as handle:
                    auxiliary = resize_pixels(handle.convert("RGB"), self.original_pixels, enlarge=True)
                sample_targets.append(len(a_images))
                a_images.append(auxiliary)
                if self.encoder_type == "qformer":
                    a_metadata.append([0, 0, 1, 1, 0, 0, 0])
            else:
                a_image_counts.append(crop_count + (len(images) if self.encoder_includes_original else 0))
            for image_index, item in enumerate(images, start=1):
                with Image.open(self.image_root / item["path"]) as handle:
                    original = handle.convert("RGB")
                if mode == "auxiliary":
                    pass
                elif mode == "auxiliary_images":
                    if self.include_encoder and self.encoder_includes_original and self.encoder_type == "qformer":
                        a_metadata.append([0, 0, 1, 1, image_index - 1, 0, 1])
                elif self.include_encoder and self.encoder_type == "qformer":
                    width, height = original.size
                    if self.encoder_includes_original:
                        a_metadata.append([0, 0, 1, 1, image_index - 1, 0, 1])
                    for crop_index, box in enumerate(item["boxes"], start=1):
                        x1, y1, x2, y2 = crop_bounds(original.size, box)
                        a_metadata.append([x1 / width, y1 / height, x2 / width, y2 / height,
                                           image_index - 1, crop_index, 0])
                if self.qa_crops or (self.include_encoder and mode == "question_crops"):
                    crops = [resize_pixels(crop, self.crop_pixels, enlarge=True)
                             for crop in crop_regions(original, item["boxes"])]
                else:
                    crops = []
                original = resize_pixels(original, self.original_pixels)
                if self.include_descriptions and self.description_image_mode == "original":
                    description_images.append(original)
                if self.include_encoder and mode != "auxiliary":
                    crop_start = len(a_images) + int(self.encoder_includes_original)
                    sample_targets.extend(range(crop_start, crop_start + len(crops)))
                    a_images.extend(([original] if self.encoder_includes_original else []) + crops)
                b_images.append(original)
                if self.qa_crops:
                    b_images.extend(crops)
                    image_labels.append(f"Original image {image_index}:")
                    image_labels.extend(f"Enlarged crop {index} from original image {image_index}:"
                                        for index in range(1, len(crops) + 1))
            if self.include_encoder and mode == "auxiliary_images":
                for helper_index, helper in enumerate(helpers, 1):
                    sample_targets.append(len(a_images))
                    with Image.open(self.image_root / helper["path"]) as handle:
                        a_images.append(resize_pixels(handle.convert("RGB"), self.crop_pixels))
                    if self.encoder_type == "qformer":
                        a_metadata.append([0, 0, 1, 1, len(images) + helper_index - 1, 0, 0])
            if self.include_reconstruction:
                if not sample_targets:
                    raise ValueError("Feature reconstruction requires an auxiliary image or crop in every sample")
                target_indices.extend(sample_targets)
                target_sample_ids.extend([sample_index] * len(sample_targets))
                target_image_ids.extend(range(len(sample_targets)))
            qa_image_labels.append(image_labels)
            if self.include_descriptions:
                description_image_counts.append(len(images) if self.description_image_mode == "original" else 0)
            if self.include_encoder:
                a_texts.append(build_generation_prompt(
                    self.processor, example["question"] if self.encoder_uses_question else "",
                    system=self.system_prompt, num_images=a_image_counts[-1],
                ))
            b_texts.append(build_generation_prompt(
                self.processor, example["question"], system=self.system_prompt,
                num_images=len(images) + (crop_count if self.qa_crops else 0),
                image_labels=image_labels if self.qa_crops else None,
            ))
        b = self.processor(text=b_texts, images=b_images, padding=True, truncation=False, return_tensors="pt")
        answers = [self.processor.tokenizer.encode(item["answer"], add_special_tokens=False) + [self.end_id]
                   for item in examples] if self.include_answers else None
        output = {
            "qa": self._append_slots(b, answers),
            "ids": [item["id"] for item in examples],
            "answers": [item["answer"] for item in examples],
            "crop_counts": torch.tensor(crop_counts),
        }
        if self.include_reconstruction:
            # Index existing encoder images; no duplicated pixel tensors or target
            # visual content is added to the decoder's output queries.
            output["reconstruction"] = {
                "original_counts": torch.tensor(original_counts, dtype=torch.long),
                "target_indices": torch.tensor(target_indices, dtype=torch.long),
                "target_sample_ids": torch.tensor(target_sample_ids, dtype=torch.long),
                "target_image_ids": torch.tensor(target_image_ids, dtype=torch.long),
            }
        if self.include_encoder:
            a = self.processor(text=a_texts, images=a_images, padding=True, truncation=False, return_tensors="pt")
            output["encoder"] = self._append_slots(a)
            if self.encoder_type == "qformer":
                output["encoder"].update(image_counts=torch.tensor(a_image_counts),
                                         image_metadata=torch.tensor(a_metadata, dtype=torch.float32))
                if self.encoder_uses_question:
                    questions = self.processor.tokenizer(
                        [item["question"] for item in examples], padding=True,
                        truncation=False, add_special_tokens=False, return_tensors="pt")
                    if not questions["attention_mask"].sum(-1).gt(0).all():
                        raise ValueError("Question-conditioned Q-Former requires nonempty questions")
                    output["encoder"].update(question_input_ids=questions["input_ids"],
                                             question_attention_mask=questions["attention_mask"])
        if self.qa_crops:
            output["qa_image_labels"] = qa_image_labels
        if self.include_descriptions:
            if any(not item.get("description", "").strip() for item in examples):
                raise ValueError("Description supervision requires nonempty region descriptions")
            description_texts = [build_generation_prompt(
                self.processor, example.get("description_prompt", DESCRIPTION_PROMPT), system=self.system_prompt,
                num_images=count,
            ) for example, count in zip(examples, description_image_counts)]
            if self.description_image_mode == "original":
                description = self.processor(
                    text=description_texts, images=description_images, padding=True,
                    truncation=False, return_tensors="pt",
                )
            else:
                description = self.processor.tokenizer(
                    description_texts, padding=True, add_special_tokens=False, return_tensors="pt",
                )
            targets = [self.processor.tokenizer.encode(item["description"], add_special_tokens=False)
                       + [self.end_id] for item in examples] if self.include_answers else None
            output["description"] = self._append_slots(description, targets)
        return output

    def _append_slots(self, batch, answers=None):
        lengths = batch["attention_mask"].sum(-1).tolist()
        tails = answers if answers is not None else [[] for _ in lengths]
        full_lengths = [length + self.num_latents + len(tail) for length, tail in zip(lengths, tails)]
        width = max(full_lengths)
        if width > self.max_length:
            raise ValueError(f"Sequence has {width} tokens, exceeding max_length={self.max_length}; no truncation applied")
        shape = (len(lengths), width)
        ids = torch.full(shape, self.pad_id, dtype=torch.long)
        attention = torch.zeros(shape, dtype=torch.long)
        types = torch.zeros(shape, dtype=torch.long)
        slots = torch.zeros(shape, dtype=torch.bool)
        labels = torch.full(shape, IGNORE_INDEX, dtype=torch.long)
        for i, (length, full_length, tail) in enumerate(zip(lengths, full_lengths, tails)):
            ids[i, :length] = batch["input_ids"][i, :length]
            if "mm_token_type_ids" in batch:
                types[i, :length] = batch["mm_token_type_ids"][i, :length]
            attention[i, :full_length] = 1
            slots[i, length:length + self.num_latents] = True
            if tail:
                start = length + self.num_latents
                ids[i, start:full_length] = torch.tensor(tail)
                labels[i, start:full_length] = ids[i, start:full_length]
        output = {key: batch[key] for key in ("pixel_values", "image_grid_thw") if key in batch}
        output.update(input_ids=ids, attention_mask=attention, mm_token_type_ids=types, slot_mask=slots)
        if answers is not None:
            output["labels"] = labels
        return output


class CropQADataModule(pl.LightningDataModule):
    def __init__(self, processor_name_or_path, image_root, train_path="data/default_training/sft/train",
                 val_path="data/default_training/sft/test", train_boxes_path="data/default_training/align/train", val_boxes_path="data/default_training/align/test",
                 batch_size=16, num_workers=8, num_latents=8, original_pixels=1048576,
                 crop_pixels=262144, max_length=8192, train_limit=None, val_limit=None, seed=42,
                 include_encoder=True, qa_crops=False, encoder_uses_question=False,
                 encoder_type="qwen", encoder_includes_original=True, include_descriptions=False,
                 encoder_image_mode="question_crops", description_image_mode="none",
                 auxiliary_manifest_path=None,
                 caption_path=None, additional_train_path=None, additional_val_path=None,
                 val_num_workers=None, prefetch_factor=2, pin_memory=True,
                 persistent_workers=True, include_reconstruction=False,
                 train_length_manifest_path=None, train_sample_ids_path=None):
        super().__init__()
        if num_workers < 0 or (val_num_workers is not None and val_num_workers < 0):
            raise ValueError("DataLoader worker counts must be nonnegative")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        if include_encoder != (num_latents > 0) or (qa_crops and include_encoder):
            raise ValueError("Use positive latent slots with A, or zero slots for plain/RGB QA")
        if encoder_uses_question and not include_encoder:
            raise ValueError("Question-conditioned encoding requires A")
        if include_reconstruction and (not include_encoder or qa_crops):
            raise ValueError("Feature reconstruction requires latent encoding and original-only QA images")
        if train_length_manifest_path and train_sample_ids_path:
            raise ValueError("Length grouping and explicit preflight sample ordering are mutually exclusive")
        if train_length_manifest_path and (num_latents or include_encoder or include_descriptions or qa_crops):
            raise ValueError("Plain QA lengths require original-only QA without latent/description branches")
        self.save_hyperparameters()
        self.processor = None
        self.train_dataset = self.val_dataset = None
        self._train_lengths = None

    def setup(self, stage=None):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.hparams.processor_name_or_path)
            self.processor.tokenizer.padding_side = "right"
            cap_qwen_pixels(self.processor, self.hparams.original_pixels)
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = CropQADataset(
                self.hparams.train_path, self.hparams.train_boxes_path, self.hparams.train_limit,
                auxiliary_manifest_path=self.hparams.auxiliary_manifest_path,
                caption_path=self.hparams.caption_path,
            )
            if self.hparams.additional_train_path:
                self.train_dataset.items.extend(CropQADataset(self.hparams.additional_train_path).items)
            if self.hparams.train_length_manifest_path:
                self._train_lengths = load_plain_qa_lengths(
                    self.hparams.train_length_manifest_path, self.train_dataset.items,
                    self.hparams.processor_name_or_path, self.hparams.original_pixels,
                    self.hparams.max_length,
                )
            if self.hparams.train_sample_ids_path:
                self.train_dataset.items = select_sample_items(
                    self.hparams.train_sample_ids_path, self.train_dataset.items,
                )
        if self.val_dataset is None:
            self.val_dataset = CropQADataset(
                self.hparams.val_path, self.hparams.val_boxes_path, self.hparams.val_limit,
                auxiliary_manifest_path=self.hparams.auxiliary_manifest_path,
                caption_path=self.hparams.caption_path,
            )
            if self.hparams.additional_val_path:
                self.val_dataset.items.extend(CropQADataset(self.hparams.additional_val_path).items)

    def _loader(self, dataset, training):
        from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

        sampler = None
        if training:
            world_size = self.trainer.world_size if self.trainer is not None else 1
            rank = self.trainer.global_rank if self.trainer is not None else 0
            if self.hparams.train_sample_ids_path:
                from torch.utils.data.distributed import DistributedSampler

                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                             shuffle=False, drop_last=False)
            else:
                # Legacy callers retain their exact sampler; Plain can bind current QA lengths.
                lengths = self._train_lengths if self._train_lengths is not None else [
                    item.get("processed_token_length") or (
                        len(item["boxes"]) * self.hparams.crop_pixels // 1024
                        + len(item["question"]) // 4 + len(item["answer"]) // 4
                    ) for item in dataset.items]
                sampler = DistributedLengthGroupedSampler(
                    batch_size=self.hparams.batch_size * world_size, num_replicas=world_size,
                    rank=rank, seed=self.hparams.seed, lengths=lengths,
                )
        workers = (self.hparams.num_workers if training or self.hparams.val_num_workers is None
                   else self.hparams.val_num_workers)
        kwargs = {}
        if workers:
            kwargs.update(persistent_workers=self.hparams.persistent_workers,
                          prefetch_factor=self.hparams.prefetch_factor)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, sampler=sampler,
            num_workers=workers, pin_memory=self.hparams.pin_memory,
            collate_fn=CropQACollator(
                self.processor, self.hparams.image_root, self.hparams.num_latents,
                self.hparams.original_pixels, self.hparams.crop_pixels, self.hparams.max_length,
                include_encoder=self.hparams.include_encoder, qa_crops=self.hparams.qa_crops,
                encoder_uses_question=self.hparams.encoder_uses_question,
                encoder_type=self.hparams.encoder_type,
                encoder_includes_original=self.hparams.encoder_includes_original,
                include_descriptions=self.hparams.include_descriptions,
                encoder_image_mode=self.hparams.encoder_image_mode,
                description_image_mode=self.hparams.description_image_mode,
                include_reconstruction=self.hparams.include_reconstruction,
            ), **kwargs,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)
