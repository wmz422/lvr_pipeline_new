"""Consolidate a native ZeRO checkpoint to the full Lightning format used by evaluation."""
import argparse
import importlib.util
from pathlib import Path

from common import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('export_runner', PROJECT / 'scripts/latent_prediction/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.consolidate_deepspeed_checkpoint(args.checkpoint.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()

