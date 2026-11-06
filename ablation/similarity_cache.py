"""Atomic, provenance-bound latent traces; an absent native trace has zero rows."""
import hashlib
import json
import os
from pathlib import Path


def trace_path(output, method, kind, identity):
    return Path(output) / method / kind / "latents" / (hashlib.sha256(identity.encode()).hexdigest() + ".safetensors")


def validate_tensor(value):
    import torch
    if (value.ndim != 2 or not 0 <= value.shape[0] <= 8 or value.shape[1] <= 0
            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
        raise ValueError("Expected a finite float32 [0..8, hidden] trace")
    if value.shape[0] and (value.norm(dim=-1) == 0).any():
        raise ValueError("Cosine is undefined for a zero-norm latent")


def read_trace(path, identity, binding):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    if metadata.get("id") != identity or metadata.get("run_sha256") != binding:
        raise ValueError(f"Foreign trace: {path}")
    values = load_file(str(path), device="cpu")
    if set(values) != {"latent"}:
        raise ValueError(f"Invalid trace keys: {path}")
    value = values["latent"]
    validate_tensor(value)
    if hashlib.sha256(value.numpy().tobytes()).hexdigest() != metadata.get("tensor_sha256"):
        raise ValueError(f"Corrupt trace: {path}")
    record = json.loads(metadata["record"])
    if record["observed_positions"] != value.shape[0] or record["native_positions"] < value.shape[0]:
        raise ValueError(f"Invalid trace length: {path}")
    return value, record


def save_trace(path, value, identity, binding, record):
    from safetensors.torch import save_file
    value = value.detach().float().cpu().contiguous()[:8]
    validate_tensor(value)
    record = dict(record, observed_positions=len(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    save_file({"latent": value}, str(temporary), metadata={"id": identity, "run_sha256": binding,
              "tensor_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
              "record": json.dumps(record, sort_keys=True, ensure_ascii=False)})
    read_trace(temporary, identity, binding)
    temporary.replace(path)
