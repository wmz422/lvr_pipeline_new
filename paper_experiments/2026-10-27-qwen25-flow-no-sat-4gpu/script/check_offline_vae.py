#!/usr/bin/env python3
"""CPU integration: actual VAE, four worker processes, disk latents, Flow gradients."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
FLOW = WORKSPACE / "doc/flow_reconstruction_parallel_20261015"
sys.path[:0] = [str(EXPERIMENT), str(FLOW), str(WORKSPACE / "lvr_pipeline_new/src")]
from flow_reconstruction.gt import persist_image, digest_file
from flow_reconstruction.flow import LatentMapper
from reconstruction_experiment.cached_flow import CachedFrozenFlowRuntime
from reconstruction_experiment.data import read_rgb8
from reconstruction_experiment.latent_cache import (
    LatentStore, bind_plan, build_plan, finalize, fingerprint, load_encoder)
from reconstruction_experiment.prepare import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results/offline_vae_cpu.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model_path = FLOW / "weights/FLUX.2-klein-base-4B"
    with tempfile.TemporaryDirectory(prefix="flow-vae-cpu-") as directory:
        root = Path(directory)
        gt, cache = root / "gt", root / "cache"
        rng = np.random.default_rng(42)
        images = [persist_image(Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)),
                                gt, "images") for height, width in [(32, 32), (32, 48), (48, 32), (32, 48)] * 2]
        protocol = {"cpu_fixture": "actual_vae_tiny_rgb"}
        protocol["protocol_sha256"] = fingerprint(protocol)
        write_json(gt / "protocol.json", protocol)
        status = {"status": "complete", "protocol_sha256": protocol["protocol_sha256"], "splits": {}}
        for split, indices in (("train", range(6)), ("val", range(6, 8))):
            rows = [{"id": f"image-{i}", "target": images[i], "references": [images[0], images[1]]} for i in indices]
            path = gt / (split + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            status["splits"][split] = {"records": len(rows), "manifest_sha256": digest_file(path)}
        write_json(gt / "status.json", status)
        plan = build_plan(gt, model_path, batch_size=2, shard_bytes=4096, device_type="cpu")
        bind_plan(cache, plan)
        processes, logs = [], []
        try:
            for rank in range(4):
                log = (root / f"worker-{rank}.log").open("w")
                logs.append(log)
                processes.append(subprocess.Popen([sys.executable, str(EXPERIMENT / "script/prepare_latents.py"),
                    "--flow-data", str(gt), "--output", str(cache), "--model", str(model_path),
                    "--device", "cpu", "--workers", "4", "--worker-rank", str(rank)],
                    stdout=log, stderr=subprocess.STDOUT))
            for rank, process in enumerate(processes):
                if process.wait() != 0:
                    raise RuntimeError((root / f"worker-{rank}.log").read_text())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            for log in logs:
                log.close()
        finalize(cache)
        store = LatentStore(cache)
        # Also exercise the public preparation entrypoint and its reuse path.
        cli_cache = root / "cli_cache"
        command = [sys.executable, str(EXPERIMENT / "script/prepare_latents.py"),
                   "--flow-data", str(gt), "--output", str(cli_cache), "--model", str(model_path),
                   "--device", "cpu", "--workers", "1", "--batch-size", "2", "--shard-mib", "1"]
        subprocess.run(command, check=True)
        payload = cli_cache / "shards/000000.f32"
        timestamp = payload.stat().st_mtime_ns
        subprocess.run(command, check=True)
        assert timestamp == payload.stat().st_mtime_ns
        assert len(LatentStore(cli_cache).tensors) == 8
        # Same prepared RGB and actual frozen FP32 VAE as the online implementation.
        online = load_encoder(model_path, "cpu", 2)
        pixels = [read_rgb8(gt / image["path"]) for image in images]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            encoded = online.encode_many(pixels)
        cached = [store.read(image["pixel_sha256"]) for image in images]
        latent_error = max((first - second).abs().max().item() for first, second in zip(encoded, cached))
        for first, second in zip(encoded, cached):
            torch.testing.assert_close(first, second, rtol=3e-4, atol=3e-5)
        torch.manual_seed(51)
        transformer = Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=16,
            timestep_guidance_channels=16, mlp_ratio=1, axes_dims_rope=(4, 4, 4, 4),
            guidance_embeds=False).requires_grad_(False).eval()
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        online.transformer, online.scheduler = copy.deepcopy(transformer), scheduler
        online.generator_batch_size, online.checkpointing = 2, False
        # Exercise the production cached constructor and fail if it tries to load a VAE.
        with patch("diffusers.AutoencoderKLFlux2.from_pretrained", side_effect=AssertionError("VAE loaded in training")), \
             patch("diffusers.Flux2Transformer2DModel.from_pretrained", return_value=copy.deepcopy(transformer)), \
             patch("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained", return_value=scheduler):
            offline = CachedFrozenFlowRuntime(str(model_path), device="cpu", checkpointing=True)
        assert not hasattr(offline, "vae")
        mapper = LatentMapper(4, 16, 16)
        mu = torch.randn(3, 8, 4, requires_grad=True)
        comparisons = []
        for validation in (False, True):
            outputs = []
            for runtime, source in ((online, pixels), (offline, cached)):
                mapper.zero_grad(set_to_none=True)
                mu.grad = None
                torch.manual_seed(17)
                loss = runtime.loss(source[:3], [[source[0], source[1]]] * 3, mapper(mu),
                                    ["a", "b", "c"], validation=validation)
                loss.backward()
                outputs.append((loss.detach(), mu.grad.clone(), mapper.net[-1].weight.grad.clone(),
                                torch.get_rng_state().clone()))
                assert all(parameter.grad is None for parameter in runtime.transformer.parameters())
            for a, b in zip(outputs[0][:3], outputs[1][:3]):
                torch.testing.assert_close(a, b, rtol=5e-4, atol=3e-5)
            assert torch.equal(outputs[0][3], outputs[1][3])
            assert outputs[1][1].abs().sum() > 0
            comparisons.append({"validation": validation,
                "loss_max_abs_error": (outputs[0][0] - outputs[1][0]).abs().max().item(),
                "latent_grad_max_abs_error": (outputs[0][1] - outputs[1][1]).abs().max().item(),
                "mapper_grad_max_abs_error": (outputs[0][2] - outputs[1][2]).abs().max().item(),
                "rng_state_equal": True})
        # Original training-mode randomness is still fresh on repeated calls.
        with torch.no_grad():
            losses = [offline.loss(cached[:3], [[cached[0], cached[1]]] * 3, mapper(mu), ["a", "b", "c"])
                      for _ in range(2)]
        assert not torch.equal(*losses)
        report = {"status": "passed", "gpu_used": False, "vae": "actual frozen FLUX.2 Klein FP32",
            "input_sizes": "32x32, 32x48, 48x32; CPU debug only, formal training stays 2MP",
            "cpu_worker_processes": 4, "cache_shards": len(plan["shards"]), "unique_images": 8,
            "public_preparation_cli_and_reuse": True,
            "image_occurrences": plan["image_occurrences"], "latent_max_abs_error": latent_error,
            "vae_not_loaded_by_training_runtime": True, "online_vs_disk_flow": comparisons,
            "training_noise_remains_random": True,
            "limits": "GPU numerical behavior, 2MP GPU peak memory and end-to-end speed remain untested"}
        write_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
