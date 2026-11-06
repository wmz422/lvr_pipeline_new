"""Observe the vectors consumed by Monet's official vLLM runner without changing them."""
from pathlib import Path
import ast
import shutil

ANCHOR = 'override_embeds.append(st["pending"])'
HELPER = '''

# Diagnostic observer: no tensor is modified and no vector is injected here.
_ablation_counts = {}
_ablation_vectors = {}
def _ablation_capture(request_id, vector):
    import hashlib, os
    from pathlib import Path
    import torch
    directory = os.environ.get("MONET_ABLATION_TRACE_DIR")
    if not directory:
        raise RuntimeError("Monet observer has no trace directory")
    key = str(request_id)
    count = _ablation_counts.get(key, 0) + 1
    _ablation_counts[key] = count
    if count <= 8:
        _ablation_vectors.setdefault(key, []).append(vector.detach().float().cpu().clone().reshape(-1))
    target = Path(directory) / (hashlib.sha256(key.encode()).hexdigest() + ".pt")
    temporary = target.with_suffix(".partial")
    torch.save({"request_id": key, "native_positions": count,
                "latent": torch.stack(_ablation_vectors[key])}, temporary)
    temporary.replace(target)

def _ablation_worker_init():
    import json, os
    from pathlib import Path
    target = Path(os.environ["MONET_ABLATION_TRACE_DIR"]) / ("ready_" + str(os.getpid()) + ".json")
    target.write_text(json.dumps({"pid": os.getpid(), "runner": __file__}))
'''


def observed_source(source):
    if source.count(ANCHOR) != 1:
        raise ValueError("Pinned Monet runner changed: observation anchor is not unique")
    # Keep the native append and all generation logic byte-for-byte intact.
    line = next(line for line in source.splitlines() if ANCHOR in line)
    indent = line[:len(line) - len(line.lstrip())]
    result = source.replace(ANCHOR, ANCHOR + "\n" + indent + '_ablation_capture(req_id, st["pending"])')
    tree = ast.parse(result)
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
    constructor = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    first = constructor.body[0]
    lines = result.splitlines(keepends=True)
    lines.insert(first.lineno - 1, " " * first.col_offset + "_ablation_worker_init()\n")
    result = "".join(lines) + HELPER
    ast.parse(result)
    return result


def prepare_runtime(source_root, runtime):
    source_root, runtime = Path(source_root), Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root / "inference", runtime / "inference", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = runtime / "inference/vllm/monet_gpu_model_runner.py"
    runner.write_text(observed_source((source_root / "inference/vllm/monet_gpu_model_runner.py").read_text()))
    # vLLM's spawned engine processes must apply the same observed official runner.
    (runtime / "sitecustomize.py").write_text(
        'import inference.apply_vllm_monet\n'
        'import sys\n'
        'm = sys.modules.get("vllm.v1.worker.gpu_model_runner")\n'
        'if m is None or not hasattr(m, "_ablation_capture"):\n'
        '    raise RuntimeError("Monet observer patch did not load")\n')
    return runtime
