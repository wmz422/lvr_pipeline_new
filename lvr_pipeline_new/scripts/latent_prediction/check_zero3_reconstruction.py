"""Real two-process ZeRO-3 reconstruction regression, using CPU/Gloo by default.

Run with torchrun --nproc_per_node=2. Check ragged original/helper counts,
sanity validation, backward, optimizer updates, and validation after training.
No distributed hooks or parameter gathers are mocked.
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import os

import torch
from torch import nn
import torch.distributed as dist

from lvr.models.feature_decoder import AuxiliaryFeatureDecoder, feature_reconstruction_loss


class ReconstructionCase(nn.Module):
    def __init__(self, legacy_projection=False):
        super().__init__()
        self.teacher = nn.Linear(8, 8)
        decoder_type = LegacyProjectionDecoder if legacy_projection else AuxiliaryFeatureDecoder
        self.decoder = decoder_type(16, 8, 32, 2, 4)

    def forward(self, batch):
        inputs = dict(batch["inputs"], latents=self.teacher(batch["latents"]))
        prediction, valid = self.decoder(**inputs)
        return feature_reconstruction_loss(
            prediction, batch["targets"], valid, inputs["target_sample_ids"], 2, normalize=True,
        )["reconstruction_mse"] * 0.1


class LegacyProjectionDecoder(AuxiliaryFeatureDecoder):
    """Negative control: restore the per-image Linear calls that desynchronize ranks."""

    def _project_originals(self, features, dtype):
        return [self.original_projection(feature.detach().to(dtype)) for feature in features]


def make_batch(rank, step, device, dtype, *, sanity=False):
    generator = torch.Generator(device=device).manual_seed(1100 + 10 * rank + step)
    originals = [1, 1] if sanity else [1 + (rank + step) % 2, 1]
    lengths = [1 + (index + rank + step) % 3 for index in range(sum(originals))]
    helpers = [1, 1] if sanity else [1 + (rank + step) % 2, 2]
    sample_ids = [sample for sample, count in enumerate(helpers) for _ in range(count)]
    image_ids = [image for count in helpers for image in range(count)]
    target_lengths = [1 + (index + step) % 4 for index in range(len(sample_ids))]

    def random(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)

    def tensor(values):
        return torch.tensor(values, device=device, dtype=torch.long)

    return dict(latents=random(2, 8, 8), targets=[random(n, 16) for n in target_lengths], inputs=dict(
        original_features=[random(n, 16) for n in lengths],
        original_grids=tensor([[1, 2, 2 * n] for n in lengths]), original_counts=tensor(originals),
        target_grids=tensor([[1, 2, 2 * n] for n in target_lengths]),
        target_sample_ids=tensor(sample_ids), target_image_ids=tensor(image_ids), merge_size=2,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--accumulation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--legacy-projection", action="store_true", help="Negative control; expected to fail")
    args = parser.parse_args()
    os.environ["DS_ACCELERATOR"] = "cuda" if args.cuda else "cpu"
    import deepspeed
    from deepspeed.utils import safe_get_full_grad

    torch.set_num_threads(1)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world < 2:
        parser.error("Use at least two torchrun workers")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.cuda else torch.device("cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.cuda:
        torch.cuda.set_device(device)
    else:
        deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.init_distributed("nccl" if args.cuda else "gloo", auto_mpi_discovery=False,
                              timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(42)
        model = ReconstructionCase(args.legacy_projection).to(device=device, dtype=dtype)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
        config = dict(train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=args.accumulation,
                      train_batch_size=2 * world * args.accumulation, zero_allow_untested_optimizer=True,
                      bf16={"enabled": args.bf16}, zero_optimization=dict(
                          stage=3, overlap_comm=False, reduce_scatter=True,
                          stage3_param_persistence_threshold=0, stage3_prefetch_bucket_size=0,
                          stage3_max_reuse_distance=0, stage3_max_live_parameters=1, reduce_bucket_size=10000))
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        tolerance = dict(rtol=5e-2, atol=2e-3) if args.bf16 else dict(rtol=2e-4, atol=2e-6)
        engine.eval()
        reference.eval()
        with torch.no_grad():
            batch = make_batch(rank, 0, device, dtype, sanity=True)
            torch.testing.assert_close(engine(batch), reference(batch), **tolerance)
        print(f"[zero3-reconstruction] rank={rank} sanity validation passed", flush=True)
        engine.train()
        reference.train()
        for step in range(2 * args.accumulation):
            batch = make_batch(rank, step, device, dtype)
            expected = reference(batch)
            (expected / args.accumulation).backward()
            actual = engine(batch)
            torch.testing.assert_close(actual, expected.detach(), **tolerance)
            engine.backward(actual)
            boundary = (step + 1) % args.accumulation == 0
            if boundary:
                for (name, param), (ref_name, ref) in zip(engine.module.named_parameters(), reference.named_parameters()):
                    assert name == ref_name and ref.grad is not None, name
                    dist.all_reduce(ref.grad)
                    ref.grad.div_(world)
                    grad = safe_get_full_grad(param)
                    assert grad is not None and torch.isfinite(grad).all(), name
                    torch.testing.assert_close(grad.float(), ref.grad.float(), **tolerance, msg=name)
                reference_optimizer.step()
                reference_optimizer.zero_grad(set_to_none=True)
            engine.step()
            if boundary:
                with deepspeed.zero.GatheredParameters(list(engine.module.parameters())), torch.no_grad():
                    for (name, param), (_, ref) in zip(engine.module.named_parameters(), reference.named_parameters()):
                        torch.testing.assert_close(param, ref, **tolerance, msg=name)
                        # Compare the next step at identical weights despite BF16 optimizer rounding.
                        ref.copy_(param)
            checked = "loss/gradients/weights match" if boundary else "loss matches (accumulating)"
            print(f"[zero3-reconstruction] rank={rank} micro_step={step + 1} "
                  f"optimizer_steps={engine.global_steps} {checked}", flush=True)
        assert engine.global_steps == 2
        engine.eval()
        reference.eval()
        with torch.no_grad():
            batch = make_batch(rank, 4, device, dtype)
            torch.testing.assert_close(engine(batch), reference(batch), **tolerance)
        print(f"[zero3-reconstruction] rank={rank} PASS bf16={args.bf16} accumulation={args.accumulation}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
