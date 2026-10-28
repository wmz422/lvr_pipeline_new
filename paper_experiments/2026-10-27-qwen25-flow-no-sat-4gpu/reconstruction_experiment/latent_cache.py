"""Content-addressed, resumable FP32 VAE latents in bounded binary shards.

Only deterministic prepared RGB -> VAE mode -> patchify -> BN normalization
is cached. Timesteps, noise and all condition-dependent computation stay online.
"""
from collections import OrderedDict, defaultdict
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from flow_reconstruction.flow import verify_model_lock
from flow_reconstruction.gt import digest_file, load_protocol
from .compact_flow import CompactFrozenFlowRuntime
from .prepare import write_json

REPRESENTATION = "vae_mode_patchified_bn_fp32_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def vae_identity(model_path):
    import diffusers
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from diffusers.models.autoencoders.vae import Encoder
    lock = verify_model_lock(model_path)
    files = {}
    for relative, item in lock["files"].items():
        if relative.startswith("vae/"):
            sha = digest_file(Path(model_path) / relative)
            if sha != item["sha256"]:
                raise ValueError(f"VAE weights/config changed: {relative}")
            files[relative] = sha
    return {"revision": lock["revision"], "files": files, "representation": REPRESENTATION,
            "dtype": "float32", "torch": torch.__version__, "diffusers": diffusers.__version__,
            "implementation": {"compact_encoder": digest_file(inspect.getfile(CompactFrozenFlowRuntime)),
                               "cache": digest_file(__file__),
                               "vae": digest_file(inspect.getfile(AutoencoderKLFlux2)),
                               "encoder": digest_file(inspect.getfile(Encoder)),
                               "patchify": hashlib.sha256(inspect.getsource(
                                   Flux2KleinPipeline._patchify_latents).encode()).hexdigest()}}


def image_shape(row):
    height, width = int(row["height"]), int(row["width"])
    if min(height, width) <= 0 or height % 16 or width % 16:
        raise ValueError("Cache requires the unchanged 16-aligned prepared RGB")
    return [1, 128, height // 16, width // 16]


def build_plan(flow_root, model_path, *, batch_size=2, shard_bytes=256 * 2**20, device_type="cuda"):
    if batch_size < 1 or shard_bytes < 4 or device_type not in ("cpu", "cuda"):
        raise ValueError("Invalid VAE cache preparation settings")
    root = Path(flow_root).resolve()
    status = json.loads((root / "status.json").read_text())
    protocol = load_protocol(root / "protocol.json")
    if status["status"] != "complete" or status["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Prepare complete RGB reconstruction data before encoding latents")
    images, counts, manifest_hashes, ids = {}, {}, {}, set()
    references = 0
    for split in ("train", "val"):
        path = root / (split + ".jsonl")
        sha = digest_file(path)
        if sha != status["splits"][split]["manifest_sha256"]:
            raise ValueError("Changed reconstruction manifest")
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in ids:
                    raise ValueError("Duplicate reconstruction sample ID")
                ids.add(row["id"])
                count += 1
                for item in [row["target"], *row["references"]]:
                    key = item["pixel_sha256"]
                    spec = {"key": key, "path": item["path"], "png_sha256": item["sha256"],
                            "shape": image_shape(item)}
                    if key in images and images[key]["shape"] != spec["shape"]:
                        raise ValueError("One pixel hash has multiple dimensions")
                    images.setdefault(key, spec)
                    references += 1
        if count != status["splits"][split]["records"]:
            raise ValueError("Incorrect reconstruction sample count")
        counts[split], manifest_hashes[split] = count, sha
    shards, current, size = [], [], 0
    for key in sorted(images):
        item = images[key]
        length = math.prod(item["shape"]) * 4
        if current and size + length > shard_bytes:
            shards.append({"items": current, "bytes": size})
            current, size = [], 0
        current.append({**item, "offset": size, "bytes": length})
        size += length
    if current:
        shards.append({"items": current, "bytes": size})
    for index, shard in enumerate(shards):
        shard["file"] = f"shards/{index:06d}.f32"
    value = {"schema": "flow-vae-cache-v1", "representation": REPRESENTATION,
             "flow_root": str(root), "model_path": str(Path(model_path).resolve()),
             "gt_protocol_sha256": protocol["protocol_sha256"], "manifest_sha256": manifest_hashes,
             "counts": counts, "encoder": vae_identity(model_path),
             "encoding": {"batch_size": batch_size, "shard_bytes": shard_bytes, "device_type": device_type},
             "unique_images": len(images), "image_occurrences": references,
             "bytes": sum(shard["bytes"] for shard in shards), "shards": shards}
    value["plan_sha256"] = fingerprint(value)
    return value


def read_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    expected = plan.pop("plan_sha256")
    if fingerprint(plan) != expected:
        raise ValueError("Cache plan checksum mismatch")
    plan["plan_sha256"] = expected
    return plan


def bind_plan(root, plan):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists() and read_plan(root) != plan:
        raise ValueError(f"VAE/data/encoding settings changed; use a new latent-cache directory: {root}")
    write_json(path, plan)


def load_encoder(model_path, device, batch_size):
    """Load only the frozen VAE; no Qwen, Flow Transformer, or optimizer."""
    from diffusers import AutoencoderKLFlux2
    runtime = CompactFrozenFlowRuntime.__new__(CompactFrozenFlowRuntime)
    runtime.vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae", local_files_only=True,
        torch_dtype=torch.float32, device_map={"": torch.device(device)}, low_cpu_mem_usage=True)
    runtime.vae.requires_grad_(False).eval()
    runtime._device, runtime.vae_batch_size = torch.device(device), batch_size
    runtime.latent_cache_bytes, runtime._cache_bytes = 0, 0
    runtime._encoded_cache = OrderedDict()
    runtime.last_stats = {"vae_batch_sizes": [], "vae_cache_hits": 0}
    return runtime


def shard_metadata(root, plan, index, *, verify_payload=True):
    shard = plan["shards"][index]
    path = Path(root) / shard["file"]
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if (meta["plan_sha256"] != plan["plan_sha256"] or meta["bytes"] != shard["bytes"]
            or not path.is_file() or path.stat().st_size != shard["bytes"]):
        raise ValueError(f"Incomplete or incompatible committed cache shard: {path}")
    if verify_payload and digest_file(path) != meta["sha256"]:
        raise ValueError(f"Corrupt cache shard: {path}")
    if set(meta["tensors"]) != {item["key"] for item in shard["items"]}:
        raise ValueError("Cache shard tensor inventory mismatch")
    for item in shard["items"]:
        stored = meta["tensors"][item["key"]]
        if any(stored[key] != item[key] for key in ("shape", "offset", "bytes")):
            raise ValueError("Cache tensor layout mismatch")
    return meta


def encode_shard(root, plan, index, encoder):
    from .data import read_rgb8
    shard = plan["shards"][index]
    path = Path(root) / shard["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shard_metadata(root, plan, index) is not None:
            return False
        started = time.monotonic()
        temporary = path.with_suffix(".partial")
        groups, tensors = defaultdict(list), {}
        for item in shard["items"]:
            groups[tuple(item["shape"])].append(item)
        with temporary.open("w+b") as stream:
            stream.truncate(shard["bytes"])
            for members in groups.values():
                for offset in range(0, len(members), encoder.vae_batch_size):
                    chunk = members[offset:offset + encoder.vae_batch_size]
                    images = []
                    for item in chunk:
                        png = Path(plan["flow_root"]) / item["path"]
                        if digest_file(png) != item["png_sha256"]:
                            raise ValueError("Prepared RGB changed before VAE encoding")
                        rgb = read_rgb8(png)
                        size = (rgb.shape[2], rgb.shape[1])
                        pixel_sha = hashlib.sha256(str(size).encode() + b"\0" +
                            rgb.permute(1, 2, 0).contiguous().numpy().tobytes()).hexdigest()
                        if pixel_sha != item["key"]:
                            raise ValueError("Prepared RGB pixel identity differs")
                        images.append(rgb)
                    encoded = encoder.encode_many(images)
                    if len(encoded) != len(chunk):
                        raise ValueError("Encoder output count mismatch")
                    for item, value in zip(chunk, encoded):
                        if (value.dtype != torch.float32 or list(value.shape) != item["shape"]
                                or value.requires_grad or not torch.isfinite(value).all()):
                            raise ValueError("Expected finite, detached, normalized FP32 VAE latent")
                        raw = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
                        stream.seek(item["offset"])
                        stream.write(raw)
                        tensors[item["key"]] = {key: item[key] for key in ("shape", "offset", "bytes")}
                        tensors[item["key"]]["sha256"] = hashlib.sha256(raw).hexdigest()
                    encoder.last_stats = {"vae_batch_sizes": [], "vae_cache_hits": 0}
            stream.flush()
            os.fsync(stream.fileno())
        sha = digest_file(temporary)
        temporary.replace(path)
        write_json(path.with_suffix(".json"), {"plan_sha256": plan["plan_sha256"], "bytes": shard["bytes"],
            "sha256": sha, "tensors": tensors, "seconds": time.monotonic() - started})
        return True


def run_worker(root, rank, world_size, device):
    plan = read_plan(root)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid cache worker partition")
    if torch.device(device).type != plan["encoding"]["device_type"]:
        raise ValueError("Worker device differs from bound encoding backend")
    if vae_identity(plan["model_path"]) != plan["encoder"]:
        raise ValueError("VAE weights/implementation changed")
    pending = [i for i in range(rank, len(plan["shards"]), world_size)
               if shard_metadata(root, plan, i) is None]
    if not pending:
        return
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    encoder = load_encoder(plan["model_path"], device, plan["encoding"]["batch_size"])
    for index in pending:
        encode_shard(root, plan, index, encoder)
        print(f"VAE_CACHE rank={rank} shard={index + 1}/{len(plan['shards'])}", flush=True)


def finalize(root):
    root = Path(root)
    plan = read_plan(root)
    tensors, shards = {}, {}
    for index, spec in enumerate(plan["shards"]):
        meta = shard_metadata(root, plan, index)
        if meta is None:
            raise ValueError(f"Missing VAE cache shard {index}")
        shards[spec["file"]] = {key: meta[key] for key in ("bytes", "sha256")}
        for key, item in meta["tensors"].items():
            if key in tensors:
                raise ValueError("Duplicate cache tensor")
            tensors[key] = {**item, "file": spec["file"]}
    if len(tensors) != plan["unique_images"]:
        raise ValueError("Incomplete unique-image cache")
    write_json(root / "index.json", {"plan_sha256": plan["plan_sha256"], "tensors": tensors, "shards": shards})
    write_json(root / "status.json", {"status": "complete", "plan_sha256": plan["plan_sha256"],
        "index_sha256": digest_file(root / "index.json"), "images": len(tensors), "bytes": plan["bytes"]})


class LatentStore:
    """Memory-map a few shards per worker; verify each FP32 tensor when reading."""
    def __init__(self, root, *, expected_gt=None, expected_encoder=None):
        self.root = Path(root)
        self.plan = read_plan(self.root)
        status = json.loads((self.root / "status.json").read_text())
        if (status["status"] != "complete" or status["plan_sha256"] != self.plan["plan_sha256"]
                or digest_file(self.root / "index.json") != status["index_sha256"]):
            raise ValueError("Incomplete or corrupt VAE cache index")
        if expected_gt is not None and self.plan["gt_protocol_sha256"] != expected_gt:
            raise ValueError("VAE cache belongs to a different reconstruction protocol")
        if expected_encoder is not None and self.plan["encoder"] != expected_encoder:
            raise ValueError("VAE cache belongs to different weights/encoding code")
        index = json.loads((self.root / "index.json").read_text())
        self.tensors = index["tensors"]
        if (index["plan_sha256"] != self.plan["plan_sha256"] or len(self.tensors) != self.plan["unique_images"]
                or len(self.tensors) != status["images"]):
            raise ValueError("Cache inventory mismatch")
        if set(index["shards"]) != {shard["file"] for shard in self.plan["shards"]}:
            raise ValueError("Cache shard inventory mismatch")
        for shard in self.plan["shards"]:
            path = self.root / shard["file"]
            if path.stat().st_size != shard["bytes"]:
                raise ValueError("Cache shard size changed")
            for item in shard["items"]:
                cached = self.tensors[item["key"]]
                if cached["file"] != shard["file"] or any(cached[k] != item[k] for k in ("shape", "offset", "bytes")):
                    raise ValueError("Cache tensor layout changed")
        # Release the large preparation inventory; readers only need the tensor index.
        self.plan = {key: value for key, value in self.plan.items() if key != "shards"}
        self._maps, self._pid = OrderedDict(), os.getpid()

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict(), "_pid": None}

    def read(self, key, expected_shape=None):
        item = self.tensors[key]
        if expected_shape is not None and list(expected_shape) != item["shape"]:
            raise ValueError("Cached latent shape differs from prepared image")
        if self._pid != os.getpid():
            self._maps, self._pid = OrderedDict(), os.getpid()
        file = item["file"]
        if file not in self._maps:
            while len(self._maps) >= 4:
                self._maps.popitem(last=False)
            self._maps[file] = np.memmap(self.root / file, mode="r", dtype=np.uint8)
        self._maps.move_to_end(file)
        raw = self._maps[file][item["offset"]:item["offset"] + item["bytes"]]
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Corrupt cached latent: {key}")
        # Copy to writable owned memory before DataLoader pinning/device transfer.
        array = np.frombuffer(raw, dtype="<f4").reshape(item["shape"]).copy()
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite cached latent")
        return torch.from_numpy(array)
