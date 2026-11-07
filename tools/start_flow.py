"""Single-command, manually configured Qwen VL Flow training and evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from common import (ROOT, PROJECT, EXPERIMENT, VARIANT, QWEN, QWEN_LOCK, LOCAL_FLOW,
                    DEFAULT_OUTPUT, FLOW_WEIGHT, reference_recipe, flow_defaults,
                    environment, python, run, write_json, verify_descriptions)
from release import load_lock, safe_target, sha256

sys.path.insert(0, str(ROOT))
from reproduce import FLOW_ARGS


def launcher():
    spec = importlib.util.spec_from_file_location('one_command_flow', EXPERIMENT / 'script/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_binding():
    if VARIANT != 'qwen25':
        verify_descriptions()
    return {'root': str(ROOT), 'variant': VARIANT,
            'locks': {name: sha256(ROOT / 'manifests' / name)
                      for name in ('data_release.json', 'training_data.json', QWEN_LOCK.name, 'flow_model.json',
                                   *(['qwen3_descriptions.json'] if VARIANT != 'qwen25' else []))}}


def inputs_present():
    _, rows = load_lock(ROOT / 'manifests/data_release.json')
    for row in rows:
        path = safe_target(ROOT, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes']:
            return False
    return True


def bootstrap(data_source=None):
    env = environment(gpu=False)
    def invoke(*command):
        run([python(), ROOT / 'reproduce.py', *command], env=env)
    invoke('configure')
    marker = ROOT / '.local' / f'inputs_ready_{VARIANT}.json'
    binding = bootstrap_binding()
    ready = marker.exists() and json.loads(marker.read_text()) == binding
    if not ready:
        if inputs_present():
            invoke('verify-data')
        else:
            invoke('restore-data', *(['--source', data_source] if data_source else []))
        from download_models import specs
        all_models_present = True
        for name in ('qwen', 'flow'):
            _, _, _, files, target, lock_name = specs(name)
            if not (target / lock_name).is_file() or not all((target / relative).is_file() for relative in files):
                all_models_present = False
        invoke('download-models', *(['--verify-only'] if all_models_present else []))
    invoke('prepare-data')
    if not ready:
        invoke('prepare-benchmarks', '--verify-only')
    invoke('doctor')
    if not ready:
        write_json(marker, binding)


def main(argv=None):
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--skip-bootstrap', action='store_true')
    parser.add_argument('--data-source')
    parser.add_argument('--preflight', action='store_true')
    wrapper, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if wrapper.preflight and '--skip-preflight' in forwarded:
        parser.error('--preflight cannot be combined with --skip-preflight')
    if not wrapper.preflight and '--preflight-only' not in forwarded and '--skip-preflight' not in forwarded:
        forwarded.append('--skip-preflight')
    if '--dry-run' in forwarded:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Resolve defaults without writing .local or requiring downloaded model/data files.
    local = LOCAL_FLOW
    os.environ['LVR_FLOW_CONFIG'] = str(local if local.exists() else EXPERIMENT / 'configs/experiment.yaml')
    launch = launcher()
    args = launch.parse_args([*FLOW_ARGS, *forwarded])
    if not local.exists():
        args.defaults = flow_defaults()
    if args.defaults['flow_weight'] != FLOW_WEIGHT:
        raise ValueError(f'Local configuration does not match {VARIANT} Flow loss weight')
    if not local.exists() and not any(a == '--output' or a.startswith('--output=') for a in forwarded):
        args.output = DEFAULT_OUTPUT
    plain = reference_recipe()
    counts = json.loads((ROOT / 'manifests/training_data.json').read_text())
    configs, steps = launch.build_recipes(args, plain, {k: counts[k] for k in ('train', 'val')},
                                         {'protocol_sha256': '${GT_PROTOCOL_SHA}'})
    summary = {'variant': VARIANT, 'base_model': str(QWEN), 'flow_loss_weight': FLOW_WEIGHT,
               'stages': list(configs),
               'loss_weights': {phase: {key: config['model'].get(key, 0.0) for key in (
                   'qa_loss_weight', 'description_weight', 'flow_weight', 'latent_weight')}
                   for phase, config in configs.items()},
               'hardware': launch.hardware_summary(args, configs), 'optimizer_steps': steps,
               'output': str(args.output), 'validation': 'Each stage validates; full 11-benchmark predicted evaluation follows Joint',
               'gpu_preflight': not args.skip_preflight, 'bootstrap': not wrapper.skip_bootstrap,
               'automatic_hardware_tuning': False,
               'evaluation': launch.evaluation_command(args, plain, steps)}
    if args.dry_run:
        print(json.dumps({'status': 'dry_run', 'gpu_started': False, 'downloads_started': False, **summary}, indent=2))
        return
    print(json.dumps(summary, indent=2), flush=True)
    if not wrapper.skip_bootstrap:
        bootstrap(wrapper.data_source)
    elif not local.exists():
        run([python(), ROOT / 'reproduce.py', 'configure'], gpu=False)
    run([python(), ROOT / 'reproduce.py', 'train', *forwarded])


if __name__ == '__main__':
    main()
