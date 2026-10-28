"""Prepare alignment, latent SFT and plain SFT batches, including LAM image pairs and token supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lvr.data.prompt import (
    ASSISTANT_SENTINEL,
    build_generation_prompt,
    build_lam_inputs,
    build_latent_block,
    load_resized_rgb,
    strip_scaffold_tags,
)
from lvr.tokens import (
    IGNORE_INDEX,
    LATENT_END_TOKEN,
    LATENT_PAD_TOKEN,
    LATENT_START_TOKEN,
)


class LatentCollator:
    def __init__(
        self,
        processor: Any,
        lam_image_processor: Any | None = None,
        image_root: str | Path | None = None,
        max_length: int | None = 2048,
        latent_len: int = 4,
        lam_image_size: int = 256,
        latent_start_token: str = LATENT_START_TOKEN,
        latent_pad_token: str = LATENT_PAD_TOKEN,
        latent_end_token: str = LATENT_END_TOKEN,
        system_prompt: str = "",
        qwen_dynamic_resolution: bool = False,
    ) -> None:
        if processor is None:
            raise ValueError("LatentCollator requires a Qwen-VL processor.")
        if image_root is None:
            raise ValueError("LatentCollator requires image_root (set via config).")
        self.processor = processor
        self.lam_image_processor = lam_image_processor
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.latent_len = latent_len
        self.lam_image_size = lam_image_size
        self.latent_start_token = latent_start_token
        self.latent_pad_token = latent_pad_token
        self.latent_end_token = latent_end_token
        # 默认 ""/False 保持原 SFT 行为不变（system 仍为空串、Qwen 图仍固定 lam_image_size）。
        self.system_prompt = system_prompt
        self.qwen_dynamic_resolution = qwen_dynamic_resolution

    @property
    def latent_block(self) -> str:
        return build_latent_block(
            self.latent_len,
            latent_start_token=self.latent_start_token,
            latent_pad_token=self.latent_pad_token,
            latent_end_token=self.latent_end_token,
        )

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, Any]:
        texts = []
        label_prompt_texts = []  # path A：仅用于算 label 监督区间（训练目标，保持与旧版一致）
        gen_prompt_texts = []    # path B：generate() 真正消费的 prompt（含 assistant 标记，对齐训练前缀）
        question_images = []

        for example in examples:
            rendered = self._render_texts(example)
            rendered["full_text"] = strip_scaffold_tags(rendered["full_text"])
            texts.append(rendered["full_text"])
            label_prompt_texts.append(rendered["prompt_text"])
            gen_prompt_texts.append(
                build_generation_prompt(self.processor, example["question"], system=self.system_prompt)
            )
            question_images.append(self._load_resized_rgb_image(example["question_image"]))

        batch = self.processor(
            text=texts,
            images=question_images,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )

        # path A prompt：只为算 label 边界（不导出）。保持训练监督区间与旧版逐位一致：
        # 旧版从 user 之后开始监督，含 `<|im_start|>assistant\n`，本次不动训练，故沿用 path A。
        label_prompt_batch = self.processor(
            text=label_prompt_texts,
            images=question_images,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        # path B prompt：generate() / test_step 消费的输入。与训练前缀逐字一致（含 assistant 标记），
        # 修掉旧 test_step 用 path A（缺 assistant 标记）与训练/评估不一致的问题。见 prompt.build_generation_prompt。
        generation_batch = self.processor(
            text=gen_prompt_texts,
            images=question_images,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )

        full_sequence_lengths = batch["attention_mask"].sum(dim=1).tolist()
        prompt_sequence_lengths = label_prompt_batch["attention_mask"].sum(dim=1).tolist()
        batch["labels"] = self._build_labels(  # latent 位的 -100 在 forward 里再设
            input_ids=batch["input_ids"],
            prompt_sequence_lengths=prompt_sequence_lengths,
            full_sequence_lengths=full_sequence_lengths,
        )

        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in generation_batch:
                batch[f"generation_{key}"] = generation_batch[key]
        if self.lam_image_processor is not None:
            batch["lam_inputs"] = self._build_lam_inputs(examples)

        batch.update(self._metadata(examples, gen_prompt_texts, texts))
        return batch

    def _metadata(
        self,
        examples: list[dict[str, str]],
        prompt_texts: list[str],
        full_texts: list[str],
    ) -> dict[str, Any]:
        return {
            "question_image_paths": [example["question_image"] for example in examples],
            "auxiliary_image_paths": [example["auxiliary_image"] for example in examples],
            "questions": [example["question"] for example in examples],
            "answer": [example["answer"] for example in examples],
            "prompt_texts": prompt_texts,
            "full_texts": full_texts,
        }

    def _render_texts(self, example: dict[str, str]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["question_image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ASSISTANT_SENTINEL}],
            },
        ]
        prompt_message = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["question_image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
        ]

        if not hasattr(self.processor, "apply_chat_template"):
            raise ValueError("Qwen processor must provide apply_chat_template.")
        template_text = self.processor.apply_chat_template(messages, tokenize=False)
        prompt_text = self.processor.apply_chat_template(prompt_message, tokenize=False)

        if ASSISTANT_SENTINEL not in template_text:
            raise ValueError("Chat template did not preserve the assistant sentinel.")

        prefix, suffix = template_text.split(ASSISTANT_SENTINEL, 1)
        full_text = prefix + self.latent_block + example["answer"] + suffix
        return {"full_text": full_text, "prompt_text": prompt_text}

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        prompt_sequence_lengths: list[int],
        full_sequence_lengths: list[int],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for batch_idx, (prompt_len, full_len) in enumerate(
            zip(prompt_sequence_lengths, full_sequence_lengths)
        ):
            start = min(prompt_len, input_ids.size(1))
            end = min(full_len, input_ids.size(1))
            if end > start:
                labels[batch_idx, start:end] = input_ids[batch_idx, start:end]
        return labels

    def _build_lam_inputs(
        self,
        examples: list[dict[str, str]],
        auxiliary_key: str = "auxiliary_image",
    ) -> dict[str, torch.Tensor]:
        return build_lam_inputs(
            examples,
            lam_image_processor=self.lam_image_processor,
            image_root=self.image_root,
            lam_image_size=self.lam_image_size,
            auxiliary_key=auxiliary_key,
        )

    def _load_resized_rgb_image(self, image_path: str) -> Any:
        # 图像预处理收敛到 lvr.data.prompt.load_resized_rgb（训练 / 评估单一真源）。
        # 动态分辨率时 size=None（原生交给 Qwen processor），否则固定 lam_image_size。LAM 图另取 256。
        qwen_size = None if self.qwen_dynamic_resolution else self.lam_image_size
        return load_resized_rgb(image_path, image_root=self.image_root, size=qwen_size)


class PlainSFTCollator:
    """普通 Qwen3-VL SFT collator：question image + question -> answer。

    不插 latent block，不构造 LAM 输入，不注册或依赖 latent special token。复用本仓库已经统一的
    path B prompt 清理和 256x256 RGB 图像预处理，便于和 LVR SFT 做同口径对比。
    """

    def __init__(
        self,
        processor: Any,
        lam_image_processor: Any | None = None,
        image_root: str | Path | None = None,
        max_length: int | None = 2048,
        latent_len: int = 4,
        lam_image_size: int = 256,
        qwen_dynamic_resolution: bool = False,
        **_: Any,
    ) -> None:
        if processor is None:
            raise ValueError("PlainSFTCollator requires a Qwen-VL processor.")
        if image_root is None:
            raise ValueError("PlainSFTCollator requires image_root (set via config).")
        self.processor = processor
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.lam_image_size = lam_image_size
        # 默认 False 保持原行为不变（固定 lam_image_size）；True 时不强制 resize，原生交给 Qwen processor 动态 smart-resize。
        self.qwen_dynamic_resolution = qwen_dynamic_resolution

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, Any]:
        texts = []
        prompt_texts = []
        question_images = []

        for example in examples:
            rendered = self._render_texts(example)
            texts.append(rendered["full_text"])
            prompt_texts.append(rendered["prompt_text"])
            question_images.append(self._load_resized_rgb_image(example["question_image"]))

        batch = self.processor(
            text=texts,
            images=question_images,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=question_images,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )

        full_sequence_lengths = batch["attention_mask"].sum(dim=1).tolist()
        prompt_sequence_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        batch["labels"] = self._build_labels(
            input_ids=batch["input_ids"],
            prompt_sequence_lengths=prompt_sequence_lengths,
            full_sequence_lengths=full_sequence_lengths,
        )
        batch.update(self._metadata(examples, prompt_texts, texts))
        return batch

    def _render_texts(self, example: dict[str, str]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ""}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["question_image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ASSISTANT_SENTINEL}],
            },
        ]
        if not hasattr(self.processor, "apply_chat_template"):
            raise ValueError("Qwen processor must provide apply_chat_template.")
        template_text = self.processor.apply_chat_template(messages, tokenize=False)
        template_text = strip_scaffold_tags(template_text)
        if ASSISTANT_SENTINEL not in template_text:
            raise ValueError("Chat template did not preserve the assistant sentinel.")

        prefix, suffix = template_text.split(ASSISTANT_SENTINEL, 1)
        answer = strip_scaffold_tags(example["answer"])
        return {"full_text": prefix + answer + suffix, "prompt_text": prefix}

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        prompt_sequence_lengths: list[int],
        full_sequence_lengths: list[int],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for batch_idx, (prompt_len, full_len) in enumerate(
            zip(prompt_sequence_lengths, full_sequence_lengths)
        ):
            start = min(prompt_len, input_ids.size(1))
            end = min(full_len, input_ids.size(1))
            if end > start:
                labels[batch_idx, start:end] = input_ids[batch_idx, start:end]
        return labels

    def _metadata(
        self,
        examples: list[dict[str, str]],
        prompt_texts: list[str],
        full_texts: list[str],
    ) -> dict[str, Any]:
        return {
            "question_image_paths": [example["question_image"] for example in examples],
            "questions": [example["question"] for example in examples],
            "answer": [example["answer"] for example in examples],
            "prompt_texts": prompt_texts,
            "full_texts": full_texts,
        }

    def _load_resized_rgb_image(self, image_path: str) -> Any:
        # 动态分辨率时 size=None（原生交给 Qwen processor smart-resize），否则固定 lam_image_size。
        qwen_size = None if self.qwen_dynamic_resolution else self.lam_image_size
        return load_resized_rgb(image_path, image_root=self.image_root, size=qwen_size)


class AlignmentCollator:
    """Alignment 阶段 collator：latent 已知（在 prompt 里），CE-only 监督 observation。

    与 LatentCollator(SFT) 的差异（忠实搬自旧 alignment/src/dataset.py AlignmentCollator）：
    - **无 system 消息**（SFT 有 system=""）；**不去 `<think>` 块**（SFT 去）。
    - latent block 放进 **prompt**（`prompt = prefix + latent_block`），模型不生成 latent。
    - label **只监督 observation**（prompt 之后到 observation 结束），latent block 与 suffix 都不监督。
    - 额外产出 `generation_no_latent_*`（无 latent 的 prompt）与 `shuffle_lam_inputs`（错配 auxiliary），供消融。
    共用 lvr.data.prompt 的 build_latent_block / build_lam_inputs / load_resized_rgb。
    """

    def __init__(
        self,
        processor: Any,
        lam_image_processor: Any | None = None,
        image_root: str | Path | None = None,
        max_length: int | None = 2048,
        latent_len: int = 4,
        lam_image_size: int = 256,
        latent_start_token: str = LATENT_START_TOKEN,
        latent_pad_token: str = LATENT_PAD_TOKEN,
        latent_end_token: str = LATENT_END_TOKEN,
        system_prompt: str = "",
        qwen_dynamic_resolution: bool = False,
        build_shuffle_lam_inputs: bool = True,
    ) -> None:
        if processor is None:
            raise ValueError("AlignmentCollator requires a Qwen-VL processor.")
        if image_root is None:
            raise ValueError("AlignmentCollator requires image_root (set via config).")
        self.processor = processor
        self.lam_image_processor = lam_image_processor
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.latent_len = latent_len
        self.lam_image_size = lam_image_size
        self.latent_start_token = latent_start_token
        self.latent_pad_token = latent_pad_token
        self.latent_end_token = latent_end_token
        # system_prompt 非空才插 system 消息（空=保持原 alignment「无 system」行为，照搬 LVR 的 if len>0）。
        self.system_prompt = system_prompt
        # qwen_dynamic_resolution=True：Qwen 那路图不强制 256，原生交给 processor 动态 smart-resize；
        # LAM 那路（build_lam_inputs）始终用 lam_image_size，两路解耦。
        self.qwen_dynamic_resolution = qwen_dynamic_resolution
        # Shuffled LAM inputs are needed only by latent-ablation generation;
        # constructing them during ordinary training doubles LAM image I/O.
        self.build_shuffle_lam_inputs = build_shuffle_lam_inputs

    @property
    def latent_block(self) -> str:
        return build_latent_block(
            self.latent_len,
            latent_start_token=self.latent_start_token,
            latent_pad_token=self.latent_pad_token,
            latent_end_token=self.latent_end_token,
        )

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, Any]:
        texts = []
        prompt_texts = []           # prefix + latent_block（latent 已知）→ 也是 generation 输入
        no_latent_prompt_texts = []  # prefix（无 latent）→ 消融用 generation 输入
        target_texts = []           # prompt + observation → 算 observation label 终点
        question_images = []

        for example in examples:
            rendered = self._render_texts(example)
            texts.append(rendered["full_text"])
            prompt_texts.append(rendered["prompt_text"])
            no_latent_prompt_texts.append(rendered["no_latent_prompt_text"])
            target_texts.append(rendered["target_text"])
            # Qwen 图：动态分辨率时 size=None（原生），否则固定 lam_image_size。LAM 图另在 build_lam_inputs 取 256。
            qwen_size = None if self.qwen_dynamic_resolution else self.lam_image_size
            question_images.append(load_resized_rgb(
                example["question_image"], image_root=self.image_root, size=qwen_size))

        batch = self._tokenize(texts, question_images)

        # observation-only label：prompt(含 latent) 之后才开始监督，到 observation 结束。
        prompt_batch = self._tokenize(prompt_texts, question_images)
        target_batch = self._tokenize(target_texts, question_images)
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        target_lengths = target_batch["attention_mask"].sum(dim=1).tolist()
        batch["labels"] = self._build_observation_labels(
            input_ids=batch["input_ids"],
            prompt_lengths=prompt_lengths,
            target_lengths=target_lengths,
        )

        # generation 输入（含 latent）：复用 prompt_batch，避免重复 tokenize。
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in prompt_batch:
                batch[f"generation_{key}"] = prompt_batch[key]
        # 消融：无 latent 的 generation 输入。
        no_latent_batch = self._tokenize(no_latent_prompt_texts, question_images)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in no_latent_batch:
                batch[f"generation_no_latent_{key}"] = no_latent_batch[key]

        if self.lam_image_processor is not None:
            batch["lam_inputs"] = build_lam_inputs(
                examples, lam_image_processor=self.lam_image_processor,
                image_root=self.image_root, lam_image_size=self.lam_image_size)
            if self.build_shuffle_lam_inputs:
                batch["shuffle_lam_inputs"] = build_lam_inputs(
                    examples, lam_image_processor=self.lam_image_processor,
                    image_root=self.image_root, lam_image_size=self.lam_image_size,
                    auxiliary_key="shuffle_auxiliary_image")

        batch.update(self._metadata(examples, prompt_texts, texts))
        return batch

    def _tokenize(self, texts: list[str], images: list[Any]) -> dict[str, Any]:
        return self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )

    def _render_texts(self, example: dict[str, str]) -> dict[str, str]:
        # 原始 alignment 无 system 消息、不去 <think> 块（忠实旧 AlignmentCollator）。
        # system_prompt 非空时插入 system 消息（如 "You are a helpful assistant."）；空则保持原行为。
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["question_image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_SENTINEL}]},
        ]
        if not hasattr(self.processor, "apply_chat_template"):
            raise ValueError("Qwen processor must provide apply_chat_template.")
        template_text = self.processor.apply_chat_template(messages, tokenize=False)
        if ASSISTANT_SENTINEL not in template_text:
            raise ValueError("Chat template did not preserve the assistant sentinel.")

        prefix, suffix = template_text.split(ASSISTANT_SENTINEL, 1)
        prompt_text = prefix + self.latent_block          # latent 已知，在 prompt 里
        no_latent_prompt_text = prefix
        target_text = prompt_text + example["observation"]
        full_text = target_text + suffix
        return {
            "prompt_text": prompt_text,
            "no_latent_prompt_text": no_latent_prompt_text,
            "target_text": target_text,
            "full_text": full_text,
        }

    def _build_observation_labels(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: list[int],
        target_lengths: list[int],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for batch_idx, (prompt_len, target_len) in enumerate(zip(prompt_lengths, target_lengths)):
            # prompt（含 latent）之前全部 ignore，只监督 observation token。
            start = min(prompt_len, input_ids.size(1))
            end = min(target_len, input_ids.size(1))
            if end > start:
                labels[batch_idx, start:end] = input_ids[batch_idx, start:end]
        return labels

    def _metadata(
        self,
        examples: list[dict[str, str]],
        prompt_texts: list[str],
        full_texts: list[str],
    ) -> dict[str, Any]:
        metadata = {
            "question_image_paths": [e["question_image"] for e in examples],
            "auxiliary_image_paths": [e["auxiliary_image"] for e in examples],
            "questions": [e["question"] for e in examples],
            "observations": [e["observation"] for e in examples],
            "prompt_texts": prompt_texts,
            "full_texts": full_texts,
        }
        if self.build_shuffle_lam_inputs:
            metadata["shuffle_auxiliary_image_paths"] = [
                e["shuffle_auxiliary_image"] for e in examples
            ]
        return metadata
