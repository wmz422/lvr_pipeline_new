"""Benchmark adapters, inference and scoring."""

__all__ = ["sft_generate", "baseline_generate", "extract_answer"]


def __getattr__(name):
    # Pure benchmark scorers also run in the lightweight geometry environment.
    # Keep the public convenience imports without requiring torch at import time.
    if name in ("sft_generate", "baseline_generate"):
        from lvr.eval import inference
        return getattr(inference, name)
    if name == "extract_answer":
        from lvr.eval.scoring import extract_answer
        return extract_answer
    raise AttributeError(name)
