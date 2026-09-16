"""VLMEvalKit adapters for published LVR and stage-one visual bottleneck models."""

import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor
import yaml

from vlmeval.dataset import ImageMCQDataset
from vlmeval.vlm.base import BaseModel

from lvr.bundle import load_bundle
from lvr.eval.inference import sft_generate
from lvr.data.crop_qa import CropQACollator, grid_boxes
from lvr.data.prompt import cap_qwen_pixels
from lvr.models.visual_bottleneck import VisualBottleneck


class LVRModel(BaseModel):
    """Evaluate an exported bundle with its original latent decoding settings."""

    allowed_types = ["text", "image"]

    def __init__(self, model_dir, max_new_tokens=256, max_pixels=None):
        super().__init__()
        self.model, self.processor = load_bundle(model_dir, "cuda:0", max_pixels)
        self.generation = {"max_new_tokens": max_new_tokens, "force_latent_end": True}

    def generate_inner(self, message, dataset=None):
        images = [item["value"] for item in message if item["type"] == "image"]
        question = "\n".join(item["value"] for item in message if item["type"] == "text")
        raw = sft_generate(
            self.model, self.processor, images, question, self.generation,
            image_size=None, system="You are a helpful assistant.",
        )
        prediction = raw
        for token in self.processor.tokenizer.all_special_tokens:
            prediction = prediction.replace(token, "")
        return {"prediction": prediction.strip(), "extra_records": {"raw_output": raw}}


class ImageMCQSubset(ImageMCQDataset):
    """Select categories without changing VLMEvalKit prompts or scoring."""

    def __init__(self, dataset, categories=None, limit=None, skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        if categories:
            missing = set(categories) - set(self.data["category"])
            if missing:
                raise ValueError(f"Unknown categories in {dataset}: {sorted(missing)}")
            self.data = self.data[self.data["category"].isin(categories)]
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.data = self.data.iloc[:limit]
        self.data = self.data.reset_index(drop=True)


class DatasetSubset:
    """Keep explicit question IDs while preserving the dataset's native prompt and scorer."""
    def __init__(self, dataset, indices):
        from vlmeval.dataset import build_dataset

        self.wrapped = build_dataset(dataset)
        wanted = {str(index) for index in indices}
        available = self.wrapped.data["index"].astype(str)
        if len(wanted) != len(indices) or not wanted <= set(available):
            raise ValueError(f"Duplicate or unavailable question IDs for {dataset}")
        self.wrapped.data = self.wrapped.data[available.isin(wanted)].reset_index(drop=True)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, index):
        return self.wrapped[index]


def benchmark_identity(dataset, paths, question):
    return hashlib.sha256(json.dumps([dataset, paths, question], ensure_ascii=False).encode()).hexdigest()


def load_annotations(path, image_root=None):
    """Resolve portable image paths while preserving recorded IDs and shuffle donors."""
    manifest = json.loads(Path(path).read_text())
    samples, lookup = manifest["samples"], {}
    root = Path(image_root or os.environ.get("LMUData", "~/LMUData")).expanduser().resolve()
    for identity, row in samples.items():
        row.setdefault("sample_id", identity)
        if row["sample_id"] != identity:
            raise ValueError("Annotation key and recorded sample ID disagree")
        for item in row["images"]:
            path = Path(item["path"])
            item["path"] = str((root / path).resolve())
        key = benchmark_identity(row["benchmark"], [item["path"] for item in row["images"]], row["qa_prompt"])
        if key in lookup:
            raise ValueError("Duplicate benchmark input identity")
        lookup[key] = row
    return samples, lookup


def build_benchmark_sample(message, dataset, annotations=None):
    """Resolve explicit annotations without falling back to grid crops on a missing match."""
    paths = [str(Path(item["value"]).resolve()) for item in message if item["type"] == "image"]
    question = "\n".join(item["value"] for item in message if item["type"] == "text")
    question += "\nAnswer with only the option letter."
    sample_id = benchmark_identity(dataset, paths, question)
    if annotations is None:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append({"path": path, "boxes": grid_boxes(image.size)})
        provenance = {"crop_policy": "grid2x2"}
    else:
        annotation = annotations[sample_id]
        if (annotation["benchmark"] != dataset or annotation["qa_prompt"] != question
                or [item["path"] for item in annotation["images"]] != paths):
            raise ValueError(f"Annotation inputs disagree for {dataset}/{sample_id}")
        images = annotation["images"]
        if not images or any(not item["boxes"] for item in images):
            raise ValueError(f"Annotated evaluation requires nonempty boxes: {sample_id}")
        provenance = {"crop_policy": "annotated_bbox", "annotation_source": annotation["annotation_source"],
                      "annotation_path": annotation["source_path"]}
        sample_id = annotation.get("sample_id", sample_id)
    return {"id": sample_id, "images": images, "question": question, "answer": ""}, provenance


class BottleneckModel(BaseModel):
    """Use explicit boxes or grid crops with native QA decoding on MCQ benchmarks."""

    allowed_types = ["text", "image"]
    INTERLEAVE = True
    latent_modes = ("none", "mu", "sample", "zero")
    encoder_modes = ("mu", "sample")

    def load_model(self, checkpoint):
        return VisualBottleneck.from_checkpoint(checkpoint, "cuda:0", base_model_path=self.base_model_path)

    def __init__(self, checkpoint, config, latent_mode="mu", max_new_tokens=256, seed=42,
                 crop_annotations=None, max_sequence_tokens=16384, qa_crops=False,
                 base_model_path=None, source_checkpoint=None):
        super().__init__()
        if latent_mode not in self.latent_modes:
            raise ValueError(f"Unknown latent mode: {latent_mode}")
        if qa_crops and latent_mode != "none":
            raise ValueError("The native RGB crop baseline requires latent_mode='none'")
        self.qa_crops = qa_crops
        self.base_model_path, self.source_checkpoint = base_model_path, source_checkpoint
        self.model = self.load_model(checkpoint)
        if latent_mode != "none" and self.model.encoder is None:
            raise ValueError("Latent evaluation requires a checkpoint with an encoder")
        if latent_mode == "sample" and not self.model.hparams.gaussian:
            raise ValueError("Sampling requires a Gaussian checkpoint")
        self.mode, self.seed, self.max_new_tokens = latent_mode, seed, max_new_tokens
        self.annotations, self.annotation_lookup = load_annotations(crop_annotations) if crop_annotations else (None, None)
        data = yaml.safe_load(Path(config).read_text())["data"]
        self.processor = AutoProcessor.from_pretrained(base_model_path or data["processor_name_or_path"])
        self.processor.tokenizer.padding_side = "right"
        cap_qwen_pixels(self.processor, data["original_pixels"])
        self.collator = CropQACollator(
            self.processor, image_root="/", num_latents=0 if qa_crops else self.model.hparams.num_latents,
            original_pixels=data["original_pixels"], crop_pixels=data["crop_pixels"],
            max_length=max_sequence_tokens, include_answers=False, include_encoder=latent_mode in self.encoder_modes,
            qa_crops=qa_crops,
            encoder_uses_question=self.model.hparams.encoder_uses_question,
            encoder_type=self.model.hparams.encoder_type,
            encoder_includes_original=self.model.hparams.encoder_includes_original,
        )

    def get_latent(self, inputs, sample_id):
        from lvr.eval.bottleneck import sample_latents

        if self.mode == "zero":
            return torch.zeros(1, self.model.hparams.num_latents, self.model.hparams.latent_dim, device="cuda:0")
        if self.mode == "none":
            return None
        latent, logvar = self.model.encode(inputs["encoder"])
        if self.mode == "sample":
            latent = sample_latents(latent.cpu(), logvar.cpu(), [sample_id], self.seed).to("cuda:0")
        return latent

    @torch.inference_mode()
    def generate_inner(self, message, dataset=None):
        sample, provenance = build_benchmark_sample(message, dataset, self.annotation_lookup)
        sample_id = sample["id"]
        batch = self.collator([sample])
        inputs = {part: {key: value.to("cuda:0") for key, value in batch[part].items()}
                  for part in ("encoder", "qa") if part in batch}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = self.get_latent(inputs, sample_id)
            if latent is not None and not torch.isfinite(latent).all():
                raise ValueError("Nonfinite benchmark latent")
            tokens = self.model.generate_answers(inputs["qa"], latent, max_new_tokens=self.max_new_tokens)[0]
        raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
        prediction = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        eos = self.model.qa.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if self.qa_crops:
            provenance.update(qa_input_policy="originals_and_crops",
                              qa_image_labels=batch["qa_image_labels"][0],
                              qa_visual_tokens=int(batch["qa"]["mm_token_type_ids"].eq(1).sum()))
        return {"prediction": prediction, "extra_records": {
            "raw_output": raw, "latent_mode": self.mode, "sample_id": sample_id,
            **provenance, "images": sample["images"], "qa_prompt": sample["question"],
            "input_tokens": {part: int(batch[part]["attention_mask"].sum()) for part in inputs},
            "checkpoint_step": self.model.checkpoint_step, "seed": self.seed,
            "model_num_latents": self.model.hparams.num_latents,
            "encoder_uses_question": self.model.hparams.encoder_uses_question,
            "encoder_type": self.model.hparams.encoder_type,
            "encoder_includes_original": self.model.hparams.encoder_includes_original,
            "qformer_use_geometry": self.model.hparams.qformer_use_geometry,
            "generated_tokens": len(tokens),
            "truncated": len(tokens) == self.max_new_tokens and int(tokens[-1]) not in eos,
        }}
