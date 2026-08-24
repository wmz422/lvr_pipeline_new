"""Lightning 数据模块。

fit 阶段构造 train/test 两个 dataset；validate/test 复用 test split。
processor 在 setup 中懒加载，避免 import 阶段就加载 Qwen 资源。

（搬自旧 sft/src/dataset.py 的 SftDataModule，逻辑不变。去掉 DEFAULT_IMAGE_ROOT 硬编码，
image_root 改为必须由 config 提供；special token 注册委托给 lvr.tokens.register_latent_tokens。）
"""

from __future__ import annotations

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from lvr.data.collator import AlignmentCollator, LatentCollator, PlainSFTCollator
from lvr.data.dataset import LatentDataset
from lvr.data.prompt import cap_qwen_pixels
from lvr.tokens import register_latent_tokens

# stage → (collator 类, dataset target 字段, 是否带 shuffle_auxiliary)。
# 这是 §4 四点差异里「数据 label / 是否 latent 生成」的 config 表达入口。
_STAGE_SPEC = {
    "sft": {"collator": LatentCollator, "target_key": "answer", "shuffle_auxiliary": False},
    "align": {"collator": AlignmentCollator, "target_key": "observation", "shuffle_auxiliary": True},
    "plain_sft": {"collator": PlainSFTCollator, "target_key": "answer", "shuffle_auxiliary": False},
}


class LatentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str = "./data/train.json",
        test_path: str = "./data/test.json",
        image_root: str | None = None,  # 必须由 config 提供（去硬编码绝对路径）
        processor_name_or_path: str | None = None,  # 必须由 config 提供（机器相关，无代码内默认）
        batch_size: int = 1,
        num_workers: int = 0,
        max_length: int | None = 2048,
        latent_len: int = 4,
        lam_image_size: int = 256,
        train_limit: int | None = None,
        test_limit: int | None = None,
        add_latent_special_tokens: bool = True,
        stage: str = "sft",  # "sft" | "align"：选 collator + dataset target 字段
        system_prompt: str = "",            # 非空时 collator 插 system 消息（默认 ""=原行为）
        qwen_dynamic_resolution: bool = False,  # True=Qwen 图原生动态分辨率（LAM 仍 lam_image_size）
        qwen_max_pixels: int | None = None,  # 动态分辨率的像素上限 cap（防大图撑爆 max_length）
    ) -> None:
        super().__init__()
        if stage not in _STAGE_SPEC:
            raise ValueError(f"Unknown stage {stage!r}. Choices: {list(_STAGE_SPEC)}")
        if not processor_name_or_path:
            raise ValueError("LatentDataModule 缺少 processor_name_or_path（应由 configs/base*.yaml data 段提供）。")
        self.train_path = train_path
        self.test_path = test_path
        self.image_root = image_root
        self.processor_name_or_path = processor_name_or_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.latent_len = latent_len
        self.lam_image_size = lam_image_size
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.add_latent_special_tokens = add_latent_special_tokens
        self.stage = stage
        self.system_prompt = system_prompt
        self.qwen_dynamic_resolution = qwen_dynamic_resolution
        self.qwen_max_pixels = qwen_max_pixels
        self._spec = _STAGE_SPEC[stage]

        self.processor = None  # 懒加载，加载过就不再重复
        self.lam_image_processor = None
        self.train_dataset: LatentDataset | None = None
        self.test_dataset: LatentDataset | None = None

    def _make_dataset(self, path: str, limit: int | None) -> LatentDataset:
        return LatentDataset(
            path,
            limit,
            target_key=self._spec["target_key"],
            include_shuffle_auxiliary=self._spec["shuffle_auxiliary"],
        )

    def setup(self, stage: str | None = None) -> None:
        if not self.processor_name_or_path:
            raise ValueError("processor_name_or_path is required.")
        if not self.image_root:
            raise ValueError("image_root is required (set via config).")
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.processor_name_or_path)
            tokenizer = self.processor.tokenizer
            tokenizer.padding_side = "right"  # 不同长度文本组 batch 时右侧补 pad
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            if self.add_latent_special_tokens:
                register_latent_tokens(tokenizer)
            # 动态分辨率：给 Qwen image_processor 设像素上限 cap（LAM 侧另走固定 lam_image_size，不受影响）。
            if self.qwen_dynamic_resolution and self.qwen_max_pixels is not None:
                cap_qwen_pixels(self.processor, self.qwen_max_pixels)
            self.lam_image_processor = self.processor.image_processor

        # 注意：这里的 stage 是 Lightning 生命周期（fit/validate/test），与 self.stage（sft/align）不同。
        if stage in (None, "fit"):
            # 训练时同时准备验证集，Lightning 会调用 val_dataloader。
            self.train_dataset = self._make_dataset(self.train_path, self.train_limit)
            self.test_dataset = self._make_dataset(self.test_path, self.test_limit)
        elif stage in ("validate", "test"):
            self.test_dataset = self._make_dataset(self.test_path, self.test_limit)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collator(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("validate")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collator(),
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def _collator(self):
        collator_cls = self._spec["collator"]  # LatentCollator(sft) | AlignmentCollator(align)
        return collator_cls(
            processor=self.processor,
            lam_image_processor=self.lam_image_processor,
            image_root=self.image_root,
            max_length=self.max_length,
            latent_len=self.latent_len,
            lam_image_size=self.lam_image_size,
            system_prompt=self.system_prompt,
            qwen_dynamic_resolution=self.qwen_dynamic_resolution,
        )
