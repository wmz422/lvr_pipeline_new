"""Reproduce the no-SAT Plain baseline from the pinned base model (two epochs by default)."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import yaml

from common import ROOT, PLAIN, PROJECT, environment, run, write_json
from release import sha256


def recipe(output, epochs):
    config = yaml.safe_load((ROOT / 'configs/plain_reference.yaml').read_text().replace('${REPO}', str(ROOT)))
    config.pop('ckpt_path', None)
    config['data']['train_length_manifest_path'] = str(ROOT / '.local/plain_qa_lengths.json.gz')
    config.update(exp_name=str(output), version='plain_sft')
    trainer = config['trainer']
    trainer.update(max_epochs=epochs, max_steps=1346 * epochs, default_root_dir=str(output / 'plain_sft'),
                   num_sanity_val_steps=1, use_distributed_sampler=False)
    for logger in trainer['logger']:
        logger['init_args']['save_dir'] = str(output / 'plain_sft')
    directory = str(output / 'plain_sft/checkpoints')
    trainer['callbacks'] = [
        {'class_path': 'continuation_runtime.RestoreDataCursor', 'init_args': {'steps_per_epoch': 1346}},
        {'class_path': 'lvr.training.audit.RunAudit'},
        {'class_path': 'lvr.training.audit.HostMemoryAudit'},
        {'class_path': 'lightning.pytorch.callbacks.LearningRateMonitor', 'init_args': {'logging_interval': 'step'}},
        {'class_path': 'continuation_runtime.HalfEpochModelCheckpoint', 'init_args': {
            'steps': [696] + [1346 * epoch + 673 for epoch in range(1, epochs)],
            'filename': 'candidate-{step:06d}', 'save_top_k': -1, 'save_last': False,
            'auto_insert_metric_name': False, 'dirpath': directory}},
        {'class_path': 'continuation_runtime.EpochModelCheckpoint', 'init_args': {'dirpath': directory}},
        {'class_path': 'continuation_runtime.RecoveryModelCheckpoint', 'init_args': {'dirpath': directory}},
    ]
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/plain')
    p.add_argument('--epochs', type=int, choices=range(1, 5), default=2)
    p.add_argument('--resume', type=Path, help='A complete four-rank native ZeRO checkpoint from this output')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    output = args.output.expanduser().resolve()
    config = recipe(output, args.epochs)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not Path(config['data']['train_length_manifest_path']).is_file():
        raise ValueError('Run reproduce.py prepare-data to verify and relocate the frozen Plain lengths')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {'recipe': config, 'data_lock_sha256': sha256(ROOT / 'manifests/training_data.json'),
                   'code': {str(path.relative_to(ROOT)): sha256(path)
                            for path in [*sorted((PROJECT / 'src').rglob('*.py')),
                                         PLAIN / 'script/train.py', PLAIN / 'script/continuation_runtime.py',
                                         Path(__file__)]}}
        bound = output / 'workflow_plan.json'
        if bound.exists() and json.loads(bound.read_text()) != binding:
            raise ValueError('Existing Plain inputs or recipe changed')
        if (output / 'training_started.json').exists() and not args.resume:
            raise ValueError('Existing training state: pass --resume CHECKPOINT or choose a new --output')
        if args.resume:
            checkpoint = args.resume.resolve()
            if not checkpoint.is_relative_to(output) or not (checkpoint / 'latest').is_file():
                raise ValueError('Resume requires a complete native checkpoint inside this output')
            config['ckpt_path'] = str(checkpoint)
        write_json(bound, binding)
        path = output / 'recipe.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node=4', str(PLAIN / 'script/train.py'), 'fit', '--config', str(path)]
        write_json(output / 'training_started.json', {'command': command})
        run(command)
        write_json(output / 'training_complete.json', {'steps': 1346 * args.epochs, 'epochs': args.epochs})


if __name__ == '__main__':
    main()
