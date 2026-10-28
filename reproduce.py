#!/usr/bin/env python3
"""One entry point for the frozen Qwen2.5 Plain/Flow experiment."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from common import ROOT, PROJECT, EXPERIMENT, QWEN, SUITE, environment, python, run

FLOW_ARGS = ['--gpus', '4', '--cpu-threads', '8',
             '--teacher-batch-size', '8', '--teacher-optimizer-device', 'cpu',
             '--warmup-batch-size', '8', '--warmup-optimizer-device', 'cuda',
             '--joint-batch-size', '4', '--joint-optimizer-device', 'cuda']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use COMMAND --help for command-specific options.')
    commands = ('configure', 'download-models', 'download-raw', 'restore-data', 'verify-data',
                'prepare-data', 'prepare-benchmarks', 'cpu-check', 'cache', 'preflight', 'train', 'plain',
                'evaluate', 'score', 'export', 'doctor', 'test')
    parser.add_argument('command', choices=commands)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    extra = argv[1:]
    command = args.command
    if command in ('restore-data', 'verify-data'):
        run([sys.executable, ROOT / 'tools/release.py', 'restore' if command == 'restore-data' else 'verify', *extra], gpu=False)
    elif command in ('configure', 'prepare-data', 'prepare-benchmarks', 'download-models', 'download-raw', 'plain', 'doctor', 'test', 'export'):
        script = {'configure': 'configure', 'prepare-data': 'prepare_training',
                  'download-models': 'download_models', 'download-raw': 'download_raw',
                  'plain': 'plain', 'doctor': 'doctor', 'test': 'test', 'export': 'export',
                  'prepare-benchmarks': 'prepare_benchmarks'}[command]
        env = environment(gpu=command == 'plain' and '--dry-run' not in extra)
        if command.startswith('download-'):
            env.pop('HF_HUB_OFFLINE', None)
            env.pop('TRANSFORMERS_OFFLINE', None)
        run([python(), ROOT / 'tools' / f'{script}.py', *extra], env=env)
    elif command in ('cpu-check', 'cache', 'preflight', 'train'):
        if not (ROOT / '.local/flow.yaml').exists():
            raise SystemExit('Run: python reproduce.py configure')
        action = {'cpu-check': ['--prepare-only'], 'cache': ['--prepare-latents-only'],
                  'preflight': ['--preflight-only'], 'train': []}[command]
        gpu = command != 'cpu-check' and '--dry-run' not in extra and not (
            '--cache-device' in extra and extra[extra.index('--cache-device') + 1] == 'cpu')
        run([python(), EXPERIMENT / 'script/run.py', *FLOW_ARGS, *action, *extra], gpu=gpu)
    elif command == 'evaluate':
        run([python(), PROJECT / 'scripts/paper_benchmarks/evaluate.py', '--suite', SUITE,
             '--model', QWEN, '--gpus', '4', *extra], gpu='--dry-run' not in extra)
    else:
        env = environment(gpu=False)
        run([env['LVR_SCORING_PYTHON'], PROJECT / 'scripts/paper_benchmarks/score.py',
             '--suite', SUITE, *extra], env=env)


if __name__ == '__main__':
    main()

