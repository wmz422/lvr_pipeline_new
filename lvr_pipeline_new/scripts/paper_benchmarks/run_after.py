"""Persistently wait for an exact training session and its dev-selected checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lvr.eval.paper_benchmark import ROOT, file_sha, load_suite, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": stat[19], "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def dependency_state(job):
    path = Path(job["session_file"])
    if not path.is_file():
        return "failed", "The bound training session marker disappeared"
    session = json.loads(path.read_text())
    if any(session.get(k) != v for k, v in job["session_identity"].items()):
        return "failed", "Training session identity changed"
    if session.get("status") == "complete":
        if session.get("exit_code") != 0:
            return "failed", "Training session did not exit successfully"
        return "ready", "Training and candidate validation finished successfully"
    if session.get("status") == "failed":
        return "failed", "Training or candidate validation failed"
    if session.get("status") not in ("starting", "running"):
        return "failed", f"Unexpected training status: {session.get('status')}"
    if process_identity(job["launcher_process"]["pid"]) != job["launcher_process"]:
        return "failed", "Bound launcher stopped without a successful completion marker"
    return "waiting_for_experiment", "Training, four candidate validations, and selection must finish first"


def selected_checkpoint(job):
    # Import the exact selector used by the training launcher only after it exits.
    sys.path.insert(0, str(ROOT / "scripts/reasoning_data"))
    from validate_plain_sft_candidates import checkpoint_identity, digest, select_candidate_summaries
    root = Path(job["validation_results"])
    policy = json.loads(Path(job["selection_policy"]).read_text())
    if file_sha(Path(job["selection_policy"])) != job["selection_policy_sha256"]:
        raise ValueError("Development checkpoint selection policy changed")
    plans, summaries = [], []
    for step in job["expected_steps"]:
        directory = root / f"step_{step:06d}"
        summary = json.loads((directory / "summary.json").read_text())
        plan = json.loads((directory / "plan.json").read_text())
        if summary["status"] != "complete" or summary["checkpoint_step"] != step or plan["checkpoint_step"] != step:
            raise ValueError("Candidate validation is incomplete or has the wrong checkpoint step")
        if summary["plan_sha256"] != digest(plan):
            raise ValueError("Validation summary is not bound to its plan")
        plans.append(plan)
        summaries.append(summary)
    for key in ("config_sha256", "samples_sha256", "samples", "selection_policy_sha256", "code_sha256"):
        if any(plan[key] != plans[0][key] for plan in plans):
            raise ValueError(f"Candidate validation conditions differ: {key}")
    expected = select_candidate_summaries(summaries, job["expected_steps"], policy)
    selected = json.loads((root / "candidate_selection.json").read_text())
    if any(selected.get(k) != v for k, v in expected.items()):
        raise ValueError("Saved candidate selection does not match the frozen development rule")
    if selected.get("test_benchmarks_used_for_selection") is not False:
        raise ValueError("Benchmark results must not select the checkpoint")
    step = selected["selected_checkpoint_step"]
    plan = plans[job["expected_steps"].index(step)]
    source = Path(selected["checkpoint"])
    export = Path(plan["checkpoint"])
    if not export.is_file() or checkpoint_identity(source) != plan["source_checkpoint_identity"]:
        raise ValueError("Selected native checkpoint changed or its export is missing")
    if export.stat().st_size != plan["checkpoint_bytes"] or export.stat().st_mtime_ns != plan["checkpoint_mtime_ns"]:
        raise ValueError("Selected validation export changed")
    binding = json.loads(export.with_suffix(export.suffix + ".source.json").read_text())
    if binding != {"source": checkpoint_identity(source), "export": checkpoint_identity(export)}:
        raise ValueError("Selected export is not bound to the selected native checkpoint")
    return export, step, {"selection": selected, "selected_validation_plan_sha256": file_sha(root / f"step_{step:06d}/plan.json"),
                          "selection_file_sha256": file_sha(root / "candidate_selection.json")}


def gpus_idle(count):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    pids = [line.strip() for line in output.splitlines() if line.strip()]
    if pids:
        return False, f"Waiting for GPU processes to exit: {', '.join(sorted(set(pids)))}"
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True).splitlines()
    if len(rows) < count:
        raise RuntimeError("The queued evaluation requires eight GPUs")
    return True, "All GPUs released"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Read-only dependency status; never run inference")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if args.check:
        state, reason = dependency_state(job)
        print(json.dumps({"status": state, "reason": reason}, ensure_ascii=False))
        return 1 if state == "failed" else 0
    status = Path(job["status_file"])
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {"job_sha256": file_sha(args.job), "pid": os.getpid(), "started_at": now()}
        if status.exists():
            previous = json.loads(status.read_text())
            if previous.get("job_sha256") != binding["job_sha256"]:
                raise ValueError("Queued job changed; use a new queue status path")
            if previous.get("status") == "complete":
                print("ALREADY COMPLETE", flush=True)
                return 0
        try:
            _, _, manifest = load_suite(Path(job["suite"]))
            if file_sha(Path(job["suite"]) / "manifest.json") != job["suite_manifest_sha256"]:
                raise ValueError("Queued benchmark protocol changed")
            previous_state = None
            while True:
                state, reason = dependency_state(job)
                write_json(status, {**binding, "status": state, "reason": reason, "updated_at": now()})
                if state != previous_state:
                    print(now(), state, reason, flush=True)
                    previous_state = state
                if state == "failed":
                    raise RuntimeError(reason)
                if state == "ready":
                    break
                time.sleep(30)
            export, step, selection = selected_checkpoint(job)
            for relative, sha in job["evaluation_code"].items():
                if file_sha(ROOT / relative) != sha:
                    raise ValueError(f"Queued evaluation code changed: {relative}")
            while True:
                idle, reason = gpus_idle(job["gpus"])
                if idle:
                    break
                write_json(status, {**binding, "status": "waiting_for_gpu_release", "reason": reason, "updated_at": now()})
                time.sleep(30)
            command = [sys.executable, str(Path(__file__).with_name("evaluate.py")), "--suite", job["suite"],
                       "--mode", "plain", "--model", job["model"], "--checkpoint", str(export),
                       "--expected-step", str(step), "--gpus", str(job["gpus"]), "--output", job["output"]]
            output = Path(job["output"])
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "checkpoint_selection.json", selection)
            write_json(status, {**binding, "status": "evaluating", "selected_step": step, "command": command, "updated_at": now()})
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1")
            with (output / "launcher.log").open("a") as log:
                subprocess.run(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / "summary.json").read_text())
            if summary["status"] != "complete" or summary["generations"] != manifest["generations"]:
                raise RuntimeError("Evaluation exited without complete coverage")
            write_json(status, {**binding, "status": "complete", "selected_step": step, "summary": str(output / "summary.json"), "finished_at": now()})
            print(now(), "COMPLETE", flush=True)
        except BaseException as error:
            write_json(status, {**binding, "status": "failed", "error": str(error), "updated_at": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
