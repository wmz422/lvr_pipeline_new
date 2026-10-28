"""Run CPU storage, recipe, scoring, gradient and resume tests."""
import argparse
import os
from pathlib import Path
import sys

from common import ROOT, PROJECT, EXPERIMENT, PLAIN, FLOW, run, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packaging-only', action='store_true')
    args = parser.parse_args()
    env = environment(gpu=False)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    suites = [ROOT / 'tests']
    if not args.packaging_only:
        suites += [FLOW / 'tests', EXPERIMENT / 'tests', PLAIN / 'tests', PROJECT / 'tests']
    for suite in suites:
        run([sys.executable, '-m', 'unittest', 'discover', '-s', suite, '-v'], env=env)


if __name__ == '__main__':
    main()
