"""各 benchmark：base / vstar / mmvp / blink / boxed local diagnostics / HR / MME-RW."""

from lvr.eval.benchmarks.base import BaseBenchmark
from lvr.eval.benchmarks.blink import BLINKBenchmark
from lvr.eval.benchmarks.hr_bench import HRBench4KBenchmark, HRBench8KBenchmark
from lvr.eval.benchmarks.mme_realworld import MMERealWorldLiteBenchmark
from lvr.eval.benchmarks.mmvp import MMVPBenchmark
from lvr.eval.benchmarks.vstar import VSTARBenchmark
from lvr.eval.benchmarks.vstar_boxed import VSTARBoxedSingleBBoxBenchmark

# benchmark 名 -> 类
BENCHMARKS = {
    "vstar": VSTARBenchmark,
    "vstar_boxed_single": VSTARBoxedSingleBBoxBenchmark,
    "mmvp": MMVPBenchmark,
    "blink": BLINKBenchmark,
    "hr_bench_4k": HRBench4KBenchmark,
    "hr_bench_8k": HRBench8KBenchmark,
    "mme_realworld_lite": MMERealWorldLiteBenchmark,
}

__all__ = [
    "BaseBenchmark",
    "VSTARBenchmark",
    "VSTARBoxedSingleBBoxBenchmark",
    "MMVPBenchmark",
    "BLINKBenchmark",
    "HRBench4KBenchmark",
    "HRBench8KBenchmark",
    "MMERealWorldLiteBenchmark",
    "BENCHMARKS",
]
