"""Repository paths and subprocess environments; no source-workspace dependency."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'lvr_pipeline_new'
FLOW = ROOT / 'doc/flow_reconstruction_parallel_20261015'
EXPERIMENT = ROOT / 'paper_experiments/2026-10-27-qwen25-flow-no-sat-4gpu'
PLAIN = ROOT / 'paper_experiments/2026-10-27-qwen25-no-sat-4epoch'
SUITE = ROOT / 'paper_experiments/2026-10-23-unified-benchmark/data'
QWEN = ROOT / 'paper_experiments/models/Qwen2.5-VL-7B-Instruct'
FLUX = FLOW / 'weights/FLUX.2-klein-base-4B'


def environment(*, gpu=True):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, (EXPERIMENT, FLOW, PROJECT / 'src', PLAIN / 'script'))),
               LVR_FLOW_CONFIG=str(ROOT / '.local/flow.yaml'),
               LVR_SCORING_PYTHON=env.get('LVR_SCORING_PYTHON', str(ROOT / '.venv-score/bin/python')),
               SETUPTOOLS_USE_DISTUTILS='local', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    for name, value in {'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '2',
                        'NCCL_P2P_DISABLE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                        'PYTORCH_KERNEL_CACHE_PATH': str(ROOT / '.local/cache/torch/kernels')}.items():
        env.setdefault(name, value)
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def python():
    return os.environ.get('LVR_TRAIN_PYTHON', str(ROOT / '.venv/bin/python'))


def run(command, *, gpu=True, env=None, cwd=ROOT):
    import shlex
    print('RUN', shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env or environment(gpu=gpu), check=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

