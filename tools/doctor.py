"""CPU-only dependency, path, data and benchmark checks; no model weights are loaded."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT, PROJECT, EXPERIMENT, FLOW, QWEN, FLUX, SUITE, environment, write_json
from release import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-only', action='store_true')
    args = parser.parse_args()
    expected = json.loads((ROOT / 'manifests/environment_training.json').read_text())['packages']
    required = ('torch', 'torchvision', 'transformers', 'lightning', 'diffusers', 'deepspeed',
                'huggingface-hub', 'accelerate', 'jsonargparse', 'numpy', 'pillow',
                'opencv-python-headless', 'datasets', 'pyarrow', 'setuptools')
    actual = {name: importlib.metadata.version(name) for name in required}
    differences = {name: [expected[name], actual[name]] for name in required if expected[name] != actual[name]}
    if sys.version_info[:2] != (3, 12):
        differences['python'] = ['3.12', sys.version.split()[0]]
    if differences:
        raise ValueError(f'Training environment differs: {differences}')
    for name in ('lvr', 'reconstruction_experiment', 'flow_reconstruction'):
        module = __import__(name)
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise ValueError(f'Imported {name} from another checkout: {module.__file__}')
    scorer = environment(gpu=False)['LVR_SCORING_PYTHON']
    code = ('import importlib.metadata as m,json,sys;'
            'print(json.dumps({"python":list(sys.version_info[:2]),"packages":'
            '{n:m.version(n) for n in ["numpy","scipy","shapely","matplotlib","pillow"]}}))')
    score_env = json.loads(subprocess.check_output([scorer, '-c', code], text=True, env=environment(gpu=False)))
    score_expected = json.loads((ROOT / 'manifests/environment_scoring.json').read_text())['packages']
    if score_env['python'] != [3, 13] or any(v != score_expected[k] for k, v in score_env['packages'].items()):
        raise ValueError(f'Scoring environment differs: {score_env}')
    report = {'status': 'passed', 'gpu_used': False, 'versions': actual, 'scoring': score_env,
              'scope': 'environment' if args.environment_only else 'environment_and_inputs'}
    if not args.environment_only:
        import yaml
        from prepare_training import read_records, CLEAN, NOSAT
        from lvr.eval.paper_benchmark import load_suite, answer_parsing_metadata
        bound = json.loads((ROOT / 'manifests/training_data.json').read_text())
        for relative, digest in bound['files'].items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f'Training data changed: {relative}')
        sizes = {'legacy_train': len(read_records(CLEAN / 'sft/train')),
                 'reasoning_train': len(read_records(NOSAT)),
                 'legacy_val': len(read_records(CLEAN / 'sft/test')),
                 'reasoning_val': len(read_records(PROJECT / 'data/processed/reasoning_v1/val'))}
        if sizes != {'legacy_train': 122706, 'reasoning_train': 49518, 'legacy_val': 1241, 'reasoning_val': 6330}:
            raise ValueError(f'Incorrect data counts: {sizes}')
        from lvr.data.crop_qa import CropQADataset
        from lvr.data.length_manifest import load_plain_qa_lengths
        items = CropQADataset(CLEAN / 'sft/train', CLEAN / 'align/train').items
        items += CropQADataset(NOSAT).items
        lengths = load_plain_qa_lengths(ROOT / '.local/plain_qa_lengths.json.gz', items, QWEN, 2000000, 8192)
        report['plain_length_binding'] = {'records': len(lengths), 'max_tokens': max(lengths), 'status': 'passed'}
        del items, lengths
        records, _, manifest = load_suite(SUITE)
        config = yaml.safe_load((ROOT / '.local/flow.yaml').read_text())
        for key in ('plain_recipe', 'flow_model', 'suite'):
            if not Path(config[key]).exists():
                raise FileNotFoundError(config[key])
        from download_models import specs
        for name in ('qwen', 'flow'):
            _, _, _, files, target, lock_name = specs(name)
            for relative, spec in files.items():
                path = target / relative
                if path.stat().st_size != spec['bytes']:
                    raise ValueError(f'Model file size mismatch: {path}')
                if path.suffix != '.safetensors' and sha256(path) != spec['sha256']:
                    raise ValueError(f'Model config checksum mismatch: {path}')
            if sha256(target / lock_name) != sha256(ROOT / 'manifests' / f'{name}_model.json'):
                raise ValueError(f'Model source lock differs: {target}')
        report.update(splits=sizes, benchmark_records=manifest['records'], benchmark_generations=len(records),
                      answer_parsing=answer_parsing_metadata(), model_check='all sizes + configuration hashes; download verifies all weight hashes')
    write_json(ROOT / '.local/doctor.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

