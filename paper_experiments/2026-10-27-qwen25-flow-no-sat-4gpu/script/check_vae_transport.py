#!/usr/bin/env python3
"""Actual frozen Klein VAE, tiny RGB inputs, CPU only; no 7B/4B model loading."""
from collections import OrderedDict
from pathlib import Path
import json
import sys
import torch
from diffusers import AutoencoderKLFlux2

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
FLOW = WORKSPACE / "doc/flow_reconstruction_parallel_20261015"
sys.path[:0] = [str(EXPERIMENT), str(FLOW), str(WORKSPACE / "lvr_pipeline_new/src")]
from flow_reconstruction.flow import FrozenFlowRuntime
from reconstruction_experiment.compact_flow import CompactFrozenFlowRuntime
from reconstruction_experiment.prepare import write_json


def main():
    torch.set_num_threads(2)
    vae = AutoencoderKLFlux2.from_pretrained(str(FLOW / "weights/FLUX.2-klein-base-4B"),
        subfolder="vae", local_files_only=True, torch_dtype=torch.float32).requires_grad_(False).eval()
    torch.manual_seed(31)
    images = [torch.randint(256, (3, 32, width), dtype=torch.uint8) for width in (48, 48, 32)]
    before_buffers = {key: value.clone() for key, value in vae.named_buffers()}
    results = []
    for cls in (FrozenFlowRuntime, CompactFrozenFlowRuntime):
        runtime = cls.__new__(cls)
        runtime.vae, runtime._device = vae, torch.device("cpu")
        runtime.vae_batch_size, runtime.latent_cache_bytes = 2, 0
        runtime._encoded_cache, runtime._cache_bytes = OrderedDict(), 0
        runtime.last_stats = {"generator_batch_sizes": [], "vae_batch_sizes": [], "vae_cache_hits": 0}
        pixels = images if cls is CompactFrozenFlowRuntime else [x.float().div(127.5).sub(1) for x in images]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = runtime.encode_many(pixels)
        assert runtime.last_stats["vae_batch_sizes"] == [2, 1]
        assert all(x.dtype == torch.float32 and not x.requires_grad for x in result)
        results.append(result)
    errors = [(a - b).abs().max().item() for a, b in zip(*results)]
    assert max(errors) == 0
    assert all(torch.equal(before_buffers[k], v) for k, v in vae.named_buffers())
    assert all(p.grad is None for p in vae.parameters())
    report = {"status": "passed", "gpu_used": False, "actual_frozen_vae": True,
              "outer_bf16_autocast": True, "vae_dtype": "float32", "max_abs_error": max(errors),
              "batch_norm_buffers_unchanged": True, "flow_rgb_transfer_bytes_ratio": .25,
              "limits": "CPU numerical check; GPU throughput and peak memory remain unmeasured"}
    write_json(EXPERIMENT / "results/vae_transport.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
