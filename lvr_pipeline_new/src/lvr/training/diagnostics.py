"""Opt-in, rank-local diagnostics; no tensors, CUDA calls or collectives here."""

from contextlib import contextmanager
from contextvars import ContextVar
import faulthandler
from functools import wraps
import json
import math
import os
from pathlib import Path
import socket
import sys
import time


_recorder = None
_scope = ContextVar("lvr_diagnostic_scope", default=())
DEBUG_DEFAULTS = {
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "TORCH_CPP_LOG_LEVEL": "INFO",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_NCCL_DESYNC_DEBUG": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "10000",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "NCCL_DEBUG": "INFO",
}


def diagnostic_environment(environment, run_directory, enabled):
    """Prepare a fresh per-phase child environment, before torch is imported."""
    result = dict(environment)
    result.pop("LVR_DIAGNOSTICS_DIR", None)
    if enabled:
        directory = Path(run_directory).resolve() / "diagnostics"
        result["LVR_DIAGNOSTICS_DIR"] = str(directory)
        result.setdefault("LVR_DIAGNOSTICS_INTERVAL_SECONDS", "120")
        for key, value in DEBUG_DEFAULTS.items():
            result.setdefault(key, value)
        result.setdefault("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", str(directory / "nccl_trace_rank_"))
        result.setdefault("NCCL_DEBUG_FILE", str(directory / "nccl.%h.%p.log"))
    return result


def enabled():
    # A forked DataLoader worker must not write to its parent's rank log.
    return _recorder is not None and _recorder.pid == os.getpid()


def trace_event(event, **fields):
    if enabled():
        _recorder.emit(event, **fields)


@contextmanager
def trace_span(name, **fields):
    if not enabled():
        yield
        return
    _recorder.span_sequence += 1
    token = _scope.set((*_scope.get(), f"{name}:{_recorder.span_sequence}"))
    try:
        trace_event(f"{name}.begin", **fields)
        try:
            yield
        except BaseException as error:
            trace_event(f"{name}.error", exception_type=type(error).__name__)
            raise
        else:
            trace_event(f"{name}.end")
    finally:
        _scope.reset(token)


def traced(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not enabled():
                return function(*args, **kwargs)
            with trace_span(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


class _Recorder:
    def __init__(self, directory, interval):
        self.pid = os.getpid()
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.started = time.monotonic()
        self.sequence = self.span_sequence = 0
        directory.mkdir(parents=True, exist_ok=True)
        self.events = (directory / f"events_rank{self.rank}.jsonl").open("a", buffering=1)
        try:
            self.stacks = (directory / f"stacks_rank{self.rank}.log").open("a", buffering=1)
            self.stacks.write(f"\n# rank={self.rank} pid={self.pid} unix={time.time()} interval={interval}s\n")
            # The C watchdog can dump Python stacks even when the main thread is
            # blocked. Periodic snapshots are evidence, NOT a declaration of a hang.
            faulthandler.dump_traceback_later(interval, repeat=True, file=self.stacks, exit=False)
        except BaseException:
            self.events.close()
            if hasattr(self, "stacks"):
                self.stacks.close()
            raise

    def emit(self, event, **fields):
        self.sequence += 1
        record = dict(event=event, rank=self.rank, pid=self.pid, sequence=self.sequence,
                      unix_time=time.time(), elapsed_seconds=time.monotonic() - self.started,
                      scope=list(_scope.get()), **fields)
        self.events.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.events.flush()

    def close(self):
        # Cancel before closing the fd, otherwise a future dump could write into
        # an unrelated file which reuses this descriptor.
        faulthandler.cancel_dump_traceback_later()
        self.stacks.close()
        self.events.close()


@contextmanager
def diagnostic_session():
    """Wrap CLI construction, model loading, distributed setup and the full fit."""
    global _recorder
    directory = os.environ.get("LVR_DIAGNOSTICS_DIR")
    if not directory:
        yield
        return
    if _recorder is not None:
        raise RuntimeError("A diagnostic session is already active")
    interval = float(os.environ.get("LVR_DIAGNOSTICS_INTERVAL_SECONDS", "120"))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("LVR_DIAGNOSTICS_INTERVAL_SECONDS must be finite and positive")
    _recorder = _Recorder(Path(directory), interval)
    try:
        keys = (*DEBUG_DEFAULTS, "TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "NCCL_DEBUG_FILE")
        trace_event("session.start", hostname=socket.gethostname(), python=sys.version,
                    local_rank=os.environ.get("LOCAL_RANK"), world_size=os.environ.get("WORLD_SIZE"),
                    stack_interval_seconds=interval,
                    debug_environment={key: os.environ.get(key) for key in keys})
        print(f"[diagnostics] rank={_recorder.rank} pid={_recorder.pid} directory={directory} "
              f"stacks every {interval:g}s (snapshots do not imply a hang)", flush=True)
        try:
            yield
        except BaseException as error:
            trace_event("session.error", exception_type=type(error).__name__)
            faulthandler.dump_traceback(file=_recorder.stacks, all_threads=True)
            raise
        else:
            trace_event("session.end")
    finally:
        _recorder.close()
        _recorder = None
