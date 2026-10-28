#!/usr/bin/env python3
"""Loop two independent baseline ablations, without reconstruction or latent loops.

no_alignment: teacher_joint -> warmup -> joint, initialized from the base model.
no_alignment_no_description: same, with Description disabled in teacher_joint/joint.
Warmup retains the baseline configuration; its training objective is latent alignment
only (Description is a validation diagnostic). Each experiment trains its own teacher
and warmup, then evaluates Predicted on all old5/new6 benchmarks before the next run.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sys

import yaml
from lvr.data.defaults import split_records, training_data_root

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("ablation_reports", ROOT / "scripts/latent_prediction/run_r12.py")
reports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reports)
workflow = reports.workflow

PHASES = ("teacher_joint", "warmup", "joint")
EXPERIMENTS = ("no_alignment", "no_alignment_no_description")
MODES = ("predicted",)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True,
                        help="Saved baseline matching the selected training release; never copy trained weights")
    parser.add_argument("--training-data-root", type=Path, default=ROOT / "data/default_training",
                        help="Selected QA/bbox release; explicit data permits historical reproduction")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/baseline_stage_ablations_20261015")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--eval-gpus", type=int, help="Defaults to --gpus")
    parser.add_argument("--batch-size", type=int, default=16, help="Per GPU; global batch stays 128")
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmark/reasoning_v1")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/latent_prediction")
    parser.add_argument("--bench-root", type=Path, default=Path("/workspace/wmz/lvr_pipeline_new/benchmark/bench"))
    parser.add_argument("--vlmeval-root", type=Path, default=ROOT / "third_party/VLMEvalKit")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted stages from native checkpoints")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true", help="Print recipes/commands; no writes or GPU use")
    actions.add_argument("--prepare-only", action="store_true", help="Check data and export plans/recipes; CPU only")
    actions.add_argument("--train-only", action="store_true")
    actions.add_argument("--eval-only", action="store_true", help="Evaluate this output's completed stages")
    args = parser.parse_args(argv)
    args.eval_gpus = args.gpus if args.eval_gpus is None else args.eval_gpus
    if args.gpus < 1 or args.batch_size < 1 or 128 % (args.gpus * args.batch_size):
        parser.error("gpus × batch-size must divide global batch 128")
    if not 1 <= args.eval_gpus <= 8:
        parser.error("eval-gpus must be 1..8")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("experiments must not repeat")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if (args.output == args.baseline_run or args.output.is_relative_to(args.baseline_run)
            or args.baseline_run.is_relative_to(args.output)):
        parser.error("output and baseline-run must be separate directories")
    args.selection, args.max_new_tokens, args.num_loops, args.preflight = "clean", 512, 1, False
    return args


def load_baseline(args):
    configs = {phase: yaml.safe_load((args.baseline_run / "recipes" / f"{phase}.yaml").read_text())
               for phase in workflow.PHASES}
    reference = configs["teacher_joint"]
    data = reference["data"]
    args.model = Path(data["processor_name_or_path"]).resolve()
    args.image_root = Path(data["image_root"]).resolve()
    args.captions = Path(data["caption_path"]).resolve()
    args.release = Path(data["additional_train_path"]).resolve().parent
    args.deepspeed_config = Path(reference["trainer"]["strategy"]["init_args"]["config"]).resolve()
    data_root = training_data_root(args, ROOT)
    records = split_records(data_root / "sft/train") + split_records(args.release / "train")
    expected_paths = dict(train_path=data_root / "sft/train", val_path=data_root / "sft/test",
                          train_boxes_path=data_root / "align/train", val_boxes_path=data_root / "align/test",
                          additional_train_path=args.release / "train", additional_val_path=args.release / "val",
                          processor_name_or_path=args.model, image_root=args.image_root, caption_path=args.captions)
    for phase, config in configs.items():
        model, trainer = config["model"], config["trainer"]
        if (model.get("reconstruction_weight", 0) != 0 or model.get("num_loops", 1) != 1
                or config["data"].get("include_reconstruction", False)):
            raise ValueError("Expected a no-reconstruction, single-round baseline")
        if model["description_weight"] != 1 or model["qa_loss_weight"] != (0 if phase == "teacher_alignment" else 1):
            raise ValueError(f"Unexpected baseline text objectives: {phase}")
        if trainer["devices"] * trainer["accumulate_grad_batches"] * config["data"]["batch_size"] != 128:
            raise ValueError(f"Expected baseline global batch 128: {phase}")
        updates = math.ceil(math.ceil(math.ceil(records / trainer["devices"]) / config["data"]["batch_size"])
                            / trainer["accumulate_grad_batches"])
        if trainer["max_steps"] != (min(200, updates) if phase == "warmup" else updates):
            raise ValueError(f"Baseline updates differ from selected training release ({updates} per epoch): {phase}")
        for key, expected in expected_paths.items():
            if Path(config["data"][key]).resolve() != expected.resolve():
                raise ValueError(f"Baseline data path differs: {phase}/{key}")
        if phase.startswith("teacher"):
            if model["gaussian"] or not model["train_qa"]:
                raise ValueError("Expected deterministic teacher with trainable QA language model")
            for key in ("encoder_model_name_or_path", "qa_model_name_or_path"):
                if Path(model[key]).resolve() != args.model:
                    raise ValueError(f"Baseline model path differs: {phase}/{key}")
        elif Path(model["base_model_path"]).resolve() != args.model:
            raise ValueError(f"Baseline model path differs: {phase}")
    return configs


def build_recipes(args, baseline, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown ablation: {experiment}")
    configs, steps = {}, {}
    for phase in PHASES:
        config = copy.deepcopy(baseline[phase])
        # Remove both weight initialization and optimizer resume from the old run.
        config.pop("init_checkpoint", None)
        config.pop("ckpt_path", None)
        config.update(exp_name=str(args.output), version=phase)
        config["trainer"].update(devices=args.gpus, accumulate_grad_batches=128 // (args.gpus * args.batch_size))
        config["data"]["batch_size"] = args.batch_size
        if phase != "teacher_joint":
            config["model"]["source_checkpoint"] = str(args.output / "teacher_joint/checkpoints/last_full.ckpt")
        if phase == "warmup":
            config["trainer"]["limit_train_batches"] = 200 * config["trainer"]["accumulate_grad_batches"]
        if phase == "joint":
            config["init_checkpoint"] = str(args.output / "warmup/checkpoints/last_full.ckpt")
        if experiment == "no_alignment_no_description" and phase in ("teacher_joint", "joint"):
            config["model"]["description_weight"] = 0.0
            config["data"]["include_descriptions"] = False
        configs[phase] = config
        steps[phase] = config["trainer"]["max_steps"]
    return configs, steps


def validate_completed(args, configs, steps):
    for phase in PHASES:
        run = args.output / phase
        complete = json.loads((run / "complete.json").read_text())
        if (complete.get("status") != "complete" or complete.get("optimizer_steps") != steps[phase]
                or complete.get("recipe_sha256") != workflow.digest(configs[phase])):
            raise ValueError(f"Missing or mismatched completed stage: {run}")
        workflow.rank_states(run, args.gpus, steps[phase])
        if not (run / "checkpoints/last_full.ckpt").is_file():
            raise FileNotFoundError(run / "checkpoints/last_full.ckpt")


def execute(args, inputs, configs, steps, environment):
    if not args.eval_only:
        for phase in PHASES:
            workflow.run_stage(args, phase, configs[phase], steps[phase], environment)
    validate_completed(args, configs, steps)
    if not args.train_only:
        reports.run_logged(old_evaluation_command(args), args.output / "benchmark_old/evaluate.log", environment)
        workflow.run_evaluation(args, environment, modes=MODES)
        reports.collect_results(args, inputs, reports.run_rescoring(args, environment, modes=MODES),
                                experiment=args.experiment, stages={phase: [phase] for phase in PHASES}, modes=MODES)


def old_evaluation_command(args):
    return [*reports.old_evaluation_command(args), "--modes", "predicted"]


def run_experiment(args, inputs, configs, steps, plan, environment):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = args.output / "workflow_plan.json"
        if binding.exists() and json.loads(binding.read_text()) != plan:
            raise ValueError(f"Existing data/config/code changed; choose a new --output: {args.output}")
        for phase in PHASES:
            run = args.output / phase
            if (not args.prepare_only and not args.eval_only and not args.resume
                    and (run / "training_started.json").exists() and not (run / "complete.json").exists()):
                raise RuntimeError(f"Interrupted stage: {run}. Use --resume to recover its checkpoint.")
        workflow.write_json(binding, plan)
        workflow.write_json(args.output / "input_check.json", dict(status="passed", **inputs))
        for phase, config in configs.items():
            path = args.output / "recipes" / f"{phase}.yaml"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False))
        if args.prepare_only:
            print(f"PREPARED {args.output} (CPU only)", flush=True)
            return
        workflow.write_json(args.output / "workflow_status.json", dict(status="running", experiment=args.experiment))
        try:
            execute(args, inputs, configs, steps, environment)
        except Exception as error:
            workflow.write_json(args.output / "workflow_status.json", dict(status="failed", error=str(error)))
            raise
        workflow.write_json(args.output / "workflow_status.json", dict(
            status="training_complete" if args.train_only else "complete", experiment=args.experiment,
            plan_sha256=workflow.digest(plan)))


def main(argv=None):
    args = parse_args(argv)
    baseline = load_baseline(args)
    print("Checking baseline data and old5/new6 evaluation inputs...", flush=True)
    inputs = workflow.inspect_inputs(args, check_images=not args.dry_run)
    original = json.loads((args.baseline_run / "input_check.json").read_text())
    for key in ("training_bindings", "model_configuration", "train_records", "val_records"):
        if inputs[key] != original[key]:
            raise ValueError(f"Inputs differ from the baseline: {key}")
    inputs["old_benchmarks"] = reports.inspect_old_benchmarks(args)
    code_paths = sorted({Path(__file__), ROOT / "scripts/run_stage_ablations.sh",
                         ROOT / "scripts/reasoning_data/train_and_evaluate.py",
                         *ROOT.glob("src/lvr/**/*.py"), *ROOT.glob("scripts/latent_prediction/*.py"),
                         *ROOT.glob("scripts/reasoning_benchmarks/*.py"),
                         *ROOT.glob("configs/latent_prediction/*")})
    shared = dict(format="lvr-baseline-stage-ablations-v1", baseline_run=str(args.baseline_run),
                  baseline_recipes=baseline, inputs={k: v for k, v in inputs.items() if k != "checked"},
                  code={str(p.relative_to(ROOT)): workflow.file_sha(p) for p in code_paths if p.is_file()},
                  deepspeed_sha256=workflow.file_sha(args.deepspeed_config), eval_gpus=args.eval_gpus,
                  evaluation=dict(num_loops=1, modes=list(MODES), selection=args.selection,
                                  old_max_new_tokens=256, new_max_new_tokens=512,
                                  original_pixels="processor_default", answer_policy=reports.ANSWER_POLICY,
                                  mira_policy=reports.MIRA_POLICY))
    environment = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       LVR_PREFLIGHT_DIAGNOSTICS="0")
    environment.pop("LVR_DIAGNOSTICS_DIR", None)
    for key, value in {"OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                       "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1"}.items():
        environment.setdefault(key, value)
    if not (args.dry_run or args.prepare_only):
        import torch
        required = args.eval_gpus if args.eval_only else args.gpus if args.train_only else max(args.gpus, args.eval_gpus)
        if torch.cuda.device_count() < required:
            raise RuntimeError(f"Need {required} visible CUDA devices")
    for experiment in args.experiments:
        current = copy.copy(args)
        current.output, current.experiment = args.output / experiment, experiment
        configs, steps = build_recipes(current, baseline, experiment)
        plan = dict(shared, experiment=experiment, recipes=configs, optimizer_steps=steps)
        print(json.dumps(dict(experiment=experiment, output=str(current.output), optimizer_steps=steps,
                              description_weights={p: c["model"]["description_weight"] for p, c in configs.items()},
                              reconstruction_weight=0, num_loops=1), indent=2), flush=True)
        if args.dry_run:
            for phase, config in configs.items():
                print(yaml.safe_dump(config, sort_keys=False))
                print(shlex.join(workflow.training_command(current, phase, current.output / "recipes" / f"{phase}.yaml")))
            print(shlex.join(old_evaluation_command(current)))
            for mode in MODES:
                for shard in range(current.eval_gpus):
                    print(shlex.join(workflow.evaluation_command(current, mode, shard)))
                print(shlex.join(reports.rescore_command(current, mode, current.output / "benchmark_latest" / mode)))
            continue
        run_experiment(current, inputs, configs, steps, plan, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
