"""Compact reconstruction RGB and complete gradient-accumulation groups."""
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from flow_reconstruction.data import FlowDataModule as OriginalFlowDataModule
from flow_reconstruction.data import FlowCollator
from lvr.data.crop_qa import CropQADataModule
from .latent_cache import LatentStore, REPRESENTATION, image_shape, vae_identity


def read_rgb8(path):
    with Image.open(path) as image:
        if image.mode != "RGB" or image.width % 16 or image.height % 16:
            raise ValueError("Unprepared flow image")
        array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1)


class CompactFlowCollator(FlowCollator):
    def __call__(self, examples):
        batch = self.base_collator(examples)
        rows = [self.index[example["id"]] for example in examples]
        batch["flow"] = {
            "targets": [read_rgb8(self.root / row["target"]["path"]) for row in rows],
            "references": [[read_rgb8(self.root / ref["path"]) for ref in row["references"]] for row in rows],
            "ids": list(batch["ids"]),
        }
        if self.cache_keys:
            batch["flow"].update(
                target_keys=[row["target"]["pixel_sha256"] for row in rows],
                reference_keys=[[ref["pixel_sha256"] for ref in row["references"]] for row in rows])
        return batch


class LatentFlowCollator(FlowCollator):
    def __init__(self, base_collator, root, index, store):
        super().__init__(base_collator, root, index)
        self.store = store

    def __call__(self, examples):
        batch = self.base_collator(examples)
        rows = [self.index[example["id"]] for example in examples]
        def read(item):
            return self.store.read(item["pixel_sha256"], image_shape(item))
        batch["flow"] = {
            "targets": [read(row["target"]) for row in rows],
            "references": [[read(ref) for ref in row["references"]] for row in rows],
            "ids": list(batch["ids"]), "representation": REPRESENTATION,
            "gt_protocol_sha256": self.store.plan["gt_protocol_sha256"],
            "latent_cache_sha256": self.store.plan["plan_sha256"],
        }
        return batch


class AccumulationBatchSampler(Sampler):
    """Preserve the underlying order; split only the short final update evenly.

    Four ranks each have 43,056 samples. The final 16 on each rank are split
    into 16 x 1 (teacher) or 8 x 2 (student), producing global batch 64 and a
    real DeepSpeed optimizer update, just as in the Plain reference.
    """
    def __init__(self, sampler, batch_size, accumulation):
        self.sampler, self.batch_size, self.accumulation = sampler, batch_size, accumulation
        self.drop_last = False
        self.resume_epoch, self.start_batch = -1, 0

    def batches(self):
        indices = list(self.sampler)
        size = self.batch_size * self.accumulation
        for start in range(0, len(indices), size):
            group = indices[start:start + size]
            width, remainder = divmod(len(group), self.accumulation)
            if not width or remainder:
                raise ValueError("Final update must split into equal nonempty microbatches")
            for offset in range(0, len(group), width):
                yield group[offset:offset + width]

    def __iter__(self):
        skip = self.start_batch if self.sampler.epoch == self.resume_epoch else 0
        for index, batch in enumerate(self.batches()):
            if index >= skip:
                yield batch

    def __len__(self):
        return math.ceil(len(self.sampler) / (self.batch_size * self.accumulation)) * self.accumulation


def accumulation_loader(original, trainer, training):
    kwargs = dict(num_workers=original.num_workers, collate_fn=original.collate_fn,
                  pin_memory=original.pin_memory)
    if original.num_workers:
        kwargs.update(persistent_workers=original.persistent_workers, prefetch_factor=original.prefetch_factor)
    if training:
        sampler = AccumulationBatchSampler(original.sampler, original.batch_size,
                                           int(trainer.accumulate_grad_batches))
        return DataLoader(original.dataset, batch_sampler=sampler, **kwargs)
    sampler = DistributedSampler(original.dataset, num_replicas=trainer.world_size,
                                 rank=trainer.global_rank, shuffle=False)
    return DataLoader(original.dataset, batch_size=original.batch_size, sampler=sampler, **kwargs)


class FlowDataModule(OriginalFlowDataModule):
    def __init__(self, processor_name_or_path, image_root, train_path='data/sft/train',
                 val_path='data/sft/test', train_boxes_path='data/align/train', val_boxes_path='data/align/test',
                 batch_size=16, num_workers=8, num_latents=8, original_pixels=1048576, crop_pixels=262144,
                 max_length=8192, train_limit=None, val_limit=None, seed=42, include_encoder=True, qa_crops=False,
                 encoder_uses_question=False, encoder_type='qwen', encoder_includes_original=True,
                 include_descriptions=False, encoder_image_mode='question_crops', description_image_mode='none',
                 auxiliary_manifest_path=None, caption_path=None, additional_train_path=None, additional_val_path=None,
                 val_num_workers=None, prefetch_factor=2, pin_memory=True, persistent_workers=True,
                 include_reconstruction=False, flow_data_root: str='', flow_protocol_sha256: str='',
                 flow_layout_buckets: bool=True, flow_stress_first: bool=False,
                 flow_latent_root: str='', flow_vae_model_path: str=''):
        arguments = dict(locals())
        for key in ('self', '__class__', 'flow_latent_root', 'flow_vae_model_path'):
            arguments.pop(key, None)
        super().__init__(**arguments)
        self.hparams.update(flow_latent_root=flow_latent_root, flow_vae_model_path=flow_vae_model_path)
        self._latent_store = None

    def setup(self, stage=None):
        # The original Flow adapter predates these two optional Plain-only keys
        # and clears inherited defaults. Restore the defaults for today's base.
        self.hparams.setdefault("train_length_manifest_path", None)
        self.hparams.setdefault("train_sample_ids_path", None)
        self._hparams_initial = copy.deepcopy(self.hparams)
        super().setup(stage)
        if self._latent_store is None:
            self._latent_store = LatentStore(self.hparams.flow_latent_root,
                expected_gt=self.hparams.flow_protocol_sha256,
                expected_encoder=vae_identity(self.hparams.flow_vae_model_path))
            status = json.loads((Path(self.hparams.flow_data_root) / 'status.json').read_text())
            for split in ('train', 'val'):
                if self._latent_store.plan['manifest_sha256'][split] != status['splits'][split]['manifest_sha256']:
                    raise ValueError('Cached latent manifest differs from current reconstruction data')
        for index in self._flow_indices.values():
            for row in index.values():
                for item in [row['target'], *row['references']]:
                    cached = self._latent_store.tensors.get(item['pixel_sha256'])
                    if cached is None or cached['shape'] != image_shape(item):
                        raise ValueError('Missing or incompatible target/reference latent')

    def _loader(self, dataset, training):
        original = super()._loader(dataset, training)
        collator = original.collate_fn
        original.collate_fn = LatentFlowCollator(
            collator.base_collator, collator.root, collator.index, self._latent_store)
        return accumulation_loader(original, self.trainer, training)


class StudentDataModule(CropQADataModule):
    def _loader(self, dataset, training):
        return accumulation_loader(super()._loader(dataset, training), self.trainer, training)
