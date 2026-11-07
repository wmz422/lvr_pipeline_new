"""Repository paths and subprocess environments; no source-workspace dependency."""
import json
import hashlib
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
VARIANT = os.environ.get('LVR_FLOW_VARIANT', 'qwen25')
if VARIANT not in ('qwen25', 'qwen3', 'qwen3-no-flow-loss', 'qwen3-no-latent-align'):
    raise ValueError(f'Unknown LVR_FLOW_VARIANT: {VARIANT}')
QWEN3 = VARIANT != 'qwen25'
QWEN = ROOT / 'paper_experiments/models' / ('Qwen3-VL-4B-Instruct' if QWEN3 else 'Qwen2.5-VL-7B-Instruct')
QWEN_LOCK = ROOT / 'manifests' / ('qwen3_model.json' if QWEN3 else 'qwen_model.json')
LOCAL_FLOW = ROOT / '.local' / ('flow.yaml' if not QWEN3 else (
    f'{VARIANT}.yaml' if VARIANT == 'qwen3-no-latent-align' else f'{VARIANT}-four-stage.yaml'))
LOCAL_REFERENCE = ROOT / '.local' / ('plain_reference.yaml' if not QWEN3 else 'qwen3_reference.yaml')
DEFAULT_OUTPUT = ROOT / 'runs' / {'qwen25': 'flow', 'qwen3': 'qwen3-flow-4stage',
    'qwen3-no-flow-loss': 'qwen3-no-flow-loss-4stage',
    'qwen3-no-latent-align': 'qwen3-no-latent-align'}[VARIANT]
FLOW_WEIGHT = 0.0 if VARIANT in ('qwen3-no-flow-loss', 'qwen3-no-latent-align') else 0.1
FLUX = FLOW / 'weights/FLUX.2-klein-base-4B'


def reference_recipe():
    """Change only backbone locations; the Flow builder removes Plain token lengths."""
    import yaml
    config = yaml.safe_load((ROOT / 'configs/plain_reference.yaml').read_text().replace('${REPO}', str(ROOT)))
    config['model'].update(encoder_model_name_or_path=str(QWEN), qa_model_name_or_path=str(QWEN))
    config['data']['processor_name_or_path'] = str(QWEN)
    return config


def flow_defaults():
    import yaml
    defaults = yaml.safe_load((EXPERIMENT / 'configs/experiment.yaml').read_text().replace('${REPO}', str(ROOT)))
    defaults.update(plain_recipe=str(LOCAL_REFERENCE), output=str(DEFAULT_OUTPUT),
                    flow_weight=FLOW_WEIGHT)
    if QWEN3:
        defaults.update(phases=['teacher_alignment', 'teacher_joint', 'warmup', 'joint'],
                        teacher_description_weight=1.0, joint_description_weight=0.0,
                        description_captions=str(ROOT / 'assets/qwen3_descriptions/captions.jsonl.gz'))
    if VARIANT == 'qwen3-no-latent-align':
        defaults.update(phases=['joint'], latent_weight=0.0, teacher_description_weight=0.0,
                        source_run=str(ROOT / 'runs/qwen3-flow-4stage'))
    return defaults


def verify_descriptions():
    """The historical caption sidecar ships with code; sample indexes stay unchanged."""
    manifest = ROOT / 'manifests/qwen3_descriptions.json'
    lock = json.loads(manifest.read_text())
    path = ROOT / lock['path']
    if (path.stat().st_size != lock['bytes']
            or hashlib.sha256(path.read_bytes()).hexdigest() != lock['sha256']
            or hashlib.sha256((ROOT / 'manifests/training_data.json').read_bytes()).hexdigest()
            != lock['training_data_manifest_sha256']):
        raise ValueError('Frozen Qwen3 descriptions or training-data binding differ')
    return lock


def environment(*, gpu=True):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, (EXPERIMENT, FLOW, PROJECT / 'src', PLAIN / 'script'))),
               LVR_FLOW_CONFIG=str(LOCAL_FLOW),
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

