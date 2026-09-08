#!/usr/bin/env python3
"""Download pinned public benchmark data and prepare the evaluator's layout."""

import argparse
from pathlib import Path
from _downloads import BENCHMARKS, prepare_benchmark, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    parser.add_argument("--output", type=Path, default=Path("benchmark/bench"))
    parser.add_argument("--list", action="store_true", help="Show sources without downloading")
    args = parser.parse_args()
    for name in args.benchmarks:
        spec = sources()[name]
        print(f"{name}: {spec['repo_id']}@{spec['revision']}")
        if not args.list:
            print(prepare_benchmark(name, args.output))


if __name__ == "__main__":
    main()
