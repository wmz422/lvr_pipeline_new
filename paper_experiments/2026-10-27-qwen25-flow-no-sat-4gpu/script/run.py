#!/usr/bin/env python3
"""Three stages, clean data, no description, image reconstruction, unified evaluation."""
import argparse
import copy
import fcntl
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
PROJECT = WORKSPACE / "lvr_pipeline_new"
FLOW = WORKSPACE / "doc/flow_reconstruction_parallel_20261015"
sys.path[:0] = [str(EXPERIMENT), str(FLOW), str(PROJECT / "src")]
from flow_reconstruction.flow import verify_model_lock
from flow_reconstruction.gt import digest_file
from reconstruction_experiment.prepare import protocol, prepare, record_file, validate_ready, write_json

PHASES = ("teacher_joint", "warmup", "joint")


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    config_path = Path(os.environ.get("LVR_FLOW_CONFIG", EXPERIMENT / "configs/experiment.yaml"))
    defaults = yaml.safe_load(config_path.read_text().replace("${REPO}", str(WORKSPACE)))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "flow_data", "flow_latents"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(defaults[name]))
    for name in ("gpus", "teacher_batch_size", "student_batch_size"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=defaults[name])
    parser.add_argument("--eval-gpus", type=int, help="Evaluation workers; default: min(training GPUs, 8)")
    for name in ("warmup", "joint"):
        parser.add_argument(f"--{name}-batch-size", type=int,
                            help="Override this student stage's per-GPU batch")
    for name in ("teacher", "warmup", "joint"):
        parser.add_argument(f"--{name}-optimizer-device", choices=("cuda", "cpu"), default="cuda",
                            help="Place the unchanged AdamW optimizer states on this device")
        parser.add_argument(f"--{name}-zero-stage", type=int, choices=(2, 3),
                            help="Override --zero-stage for this training phase")
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), default=3,
                        help="DeepSpeed ZeRO stage, shared by all phases unless overridden (default: 3)")
    parser.add_argument("--parameter-offload", choices=("none", "cpu"), default="none",
                        help="Optionally offload model parameters to CPU; requires ZeRO-3 in every phase")
    parser.add_argument("--flow-generator-batch-size", type=int, default=defaults["flow_generator_batch_size"],
                        help="Frozen Flow forward sub-batch; independent of the global training batch")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the separate four-update GPU smoke check; no automatic hardware tuning is performed")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--teacher-sees-question", action=argparse.BooleanOptionalAction,
                        default=defaults["teacher_sees_question"], help="Independent teacher-input ablation; default false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-device", choices=("cpu", "cuda"), default=defaults["cache_device"])
    parser.add_argument("--cache-workers", type=int, help="Default: one CPU worker, or one per requested GPU")
    parser.add_argument("--cache-vae-batch-size", type=int, default=defaults["cache_vae_batch_size"])
    parser.add_argument("--cache-shard-mib", type=int, default=defaults["cache_shard_mib"])
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true", help="Print recipes/commands; no writes or GPU use")
    actions.add_argument("--prepare-only", action="store_true", help="Write bound recipes and run CPU checks")
    actions.add_argument("--prepare-data-only", action="store_true", help="Prepare all reconstruction images on CPU")
    actions.add_argument("--prepare-latents-only", action="store_true", help="Prepare RGB and reusable FP32 VAE latents, without training")
    actions.add_argument("--preflight-only", action="store_true", help="Prepare data and run four GPU updates per stage")
    actions.add_argument("--train-only", action="store_true", help="GPU preflight and full training, without evaluation")
    actions.add_argument("--eval-only", action="store_true", help="Evaluate this run's completed final student")
    args = parser.parse_args(argv)
    if args.skip_preflight and args.preflight_only:
        parser.error("--skip-preflight cannot be combined with --preflight-only")
    if args.eval_gpus is None:
        args.eval_gpus = min(args.gpus, 8)
    if args.gpus < 1 or not 1 <= args.eval_gpus <= 8:
        parser.error("Training GPUs must be positive; the unchanged benchmark launcher supports 1..8 evaluation GPUs")
    for name in ("teacher", "warmup", "joint"):
        if getattr(args, name + "_zero_stage") is None:
            setattr(args, name + "_zero_stage", args.zero_stage)
    if args.parameter_offload == "cpu" and any(getattr(args, name + "_zero_stage") != 3
                                               for name in ("teacher", "warmup", "joint")):
        parser.error("CPU parameter offload requires ZeRO-3 in every phase")
    for batch in (args.teacher_batch_size,
                  args.warmup_batch_size or args.student_batch_size,
                  args.joint_batch_size or args.student_batch_size):
        if batch < 1 or 128 % (args.gpus * batch):
            parser.error("Each per-GPU batch × GPUs must divide global batch 128")
    if (args.cpu_threads < 1 or args.flow_generator_batch_size < 1
            or args.warmup_batch_size == 0 or args.joint_batch_size == 0):
        parser.error("CPU threads and explicit stage batches must be positive")
    args.output = args.output.expanduser().resolve()
    args.flow_data = args.flow_data.expanduser().resolve()
    args.flow_latents = args.flow_latents.expanduser().resolve()
    if args.cache_workers is None:
        args.cache_workers = args.gpus if args.cache_device == "cuda" else 1
    if min(args.cache_workers, args.cache_vae_batch_size, args.cache_shard_mib) < 1:
        parser.error("Cache workers, batch size and shard size must be positive")
    args.defaults = defaults
    return args


def load_inputs(args):
    reference = Path(args.defaults["plain_recipe"])
    plain = yaml.safe_load(reference.read_text())
    data, trainer = plain["data"], plain["trainer"]
    if (data["original_pixels"] != 2000000
            or trainer["devices"] * data["batch_size"] * trainer["accumulate_grad_batches"] != 128):
        raise ValueError("Plain reference is not the requested 2MP/global-batch-128 experiment")
    if (args.defaults["original_pixels"] != data["original_pixels"]
            or args.defaults["crop_pixels"] != data["crop_pixels"]
            or args.defaults["global_batch_size"] != 128):
        raise ValueError("Experiment pixel/global-batch settings must match the Plain reference")
    model = Path(data["processor_name_or_path"])
    if json.loads((model / "config.json").read_text())["model_type"] != "qwen2_5_vl":
        raise ValueError("Expected Qwen2.5-VL")
    source = json.loads((model / "source.json").read_text())
    if (source["repository"] != "Qwen/Qwen2.5-VL-7B-Instruct" or source["status"] != "complete"
            or source["revision"] != "cc594898137f460bfe9f0759e9844b3ce807cfb5"):
        raise ValueError("Qwen2.5-VL-7B-Instruct source/revision differs from Plain")
    model_files = {}
    for item in source["verified_files"]:
        path = model / item["name"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Base model file size differs: {path}")
        if path.suffix != ".safetensors" and digest_file(path) != item["sha256"]:
            raise ValueError(f"Base model configuration/tokenizer differs: {path}")
        model_files[item["name"]] = {"bytes": item["bytes"], "source_sha256": item["sha256"],
                                    "mtime_ns": path.stat().st_mtime_ns}
    configs = {}
    for key in ("train_path", "val_path", "train_boxes_path", "val_boxes_path", "additional_train_path", "additional_val_path"):
        path = record_file(data[key])
        manifest = json.loads(path.read_text())
        if isinstance(manifest, dict):
            for shard in manifest["shards"]:
                if digest_file(path.parent / shard["file"]) != shard["sha256"]:
                    raise ValueError(f"Changed data shard: {path.parent / shard['file']}")
            records = manifest["records"]
        else:
            records = len(manifest)
            if key == "additional_train_path" and any(row["source"] == "sat_perspective" for row in manifest):
                raise ValueError("The current Plain reference excludes SAT training examples")
        configs[key] = {"path": str(path), "sha256": digest_file(path), "records": records}
    counts = {"train": configs["train_path"]["records"] + configs["additional_train_path"]["records"],
              "val": configs["val_path"]["records"] + configs["additional_val_path"]["records"]}
    if counts != {"train": 172224, "val": 7571}:
        raise ValueError(f"Clean Plain SFT data changed: {counts}")
    for split in ("train", "val"):
        if configs[split + "_path"]["records"] != configs[split + "_boxes_path"]["records"]:
            raise ValueError("Clean QA/bbox count mismatch")
    lock = verify_model_lock(args.defaults["flow_model"])
    settings = protocol(data, PROJECT)
    return plain, counts, settings, {"plain_recipe": str(reference), "plain_recipe_sha256": digest_file(reference),
        "splits": configs, "counts": counts, "flow_model_revision": lock["revision"],
        "model_source_sha256": digest_file(model / "source.json"), "model_files": model_files,
        "flow_model_lock_sha256": digest_file(Path(args.defaults["flow_model"]) / "weights_lock.json"),
        "model_config_sha256": digest_file(model / "config.json"),
        "deepspeed_config_sha256": digest_file(trainer["strategy"]["init_args"]["config"])}


def build_recipes(args, plain, counts, settings, *, preflight=False):
    output = args.output / "preflight" if preflight else args.output
    configs, steps = {}, {}
    for phase in PHASES:
        teacher = phase == "teacher_joint"
        batch = (args.teacher_batch_size if teacher else
                 getattr(args, phase + "_batch_size") or args.student_batch_size)
        accumulation = 128 // (args.gpus * batch)
        if counts["train"] % args.gpus:
            raise ValueError("Training samples must divide GPU count without sampler padding")
        tail = counts["train"] // args.gpus % (batch * accumulation)
        if tail and (tail < accumulation or tail % accumulation):
            raise ValueError(f"{phase}: tail {tail} cannot split into {accumulation} equal nonempty "
                             "microbatches; use per-GPU batch >= 2 with GPUs × batch dividing 128")
        per_epoch = math.ceil(math.ceil(math.ceil(counts["train"] / args.gpus) / batch) / accumulation)
        epochs = args.defaults["teacher_epochs" if teacher else "student_epochs"]
        updates = args.defaults["warmup_steps"] if phase == "warmup" else per_epoch * epochs
        if preflight:
            updates = args.defaults["preflight_steps"]
        steps[phase] = updates
        config = copy.deepcopy(plain)
        config.pop("init_checkpoint", None)
        config.pop("ckpt_path", None)
        config.update(exp_name=str(output), version=phase)
        data, trainer = config["data"], config["trainer"]
        optimizer_device = getattr(args, ("teacher" if teacher else phase) + "_optimizer_device")
        stage = getattr(args, ("teacher" if teacher else phase) + "_zero_stage")
        strategy = trainer["strategy"]["init_args"]
        deepspeed = (copy.deepcopy(strategy["config"]) if isinstance(strategy["config"], dict)
                     else json.loads(Path(strategy["config"]).read_text()))
        if stage != 3 or optimizer_device == "cpu" or args.parameter_offload == "cpu":
            zero = deepspeed["zero_optimization"]
            zero["stage"] = stage
            zero.pop("offload_optimizer", None)
            zero.pop("offload_param", None)
            if stage == 2:
                for key in list(zero):
                    if key.startswith("stage3_") or key in ("sub_group_size", "gather_16bit_weights_on_model_save"):
                        zero.pop(key)
            if optimizer_device == "cpu":
                zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
                deepspeed["zero_force_ds_cpu_optimizer"] = False
            if args.parameter_offload == "cpu":
                zero["offload_param"] = {"device": "cpu", "pin_memory": True}
            strategy["config"] = deepspeed
        data.pop("train_length_manifest_path", None)
        data.update(batch_size=batch, num_latents=8, include_encoder=True, include_descriptions=False,
                    caption_path=None, encoder_uses_question=args.teacher_sees_question,
                    max_length=args.defaults["max_length"], include_reconstruction=False)
        data.update({key: args.defaults[key] for key in (
            "num_workers", "val_num_workers", "prefetch_factor", "pin_memory", "persistent_workers")})
        trainer.update(devices=args.gpus, accumulate_grad_batches=accumulation, max_steps=updates,
                       max_epochs=1 if preflight or phase == "warmup" else epochs,
                       limit_train_batches=updates * accumulation if preflight or phase == "warmup" else 1.0,
                       val_check_interval=1.0, num_sanity_val_steps=0 if preflight else 1,
                       default_root_dir=str(output / phase), log_every_n_steps=1 if preflight else 10,
                       use_distributed_sampler=False)
        if preflight:
            trainer["limit_val_batches"] = 2
        trainer["callbacks"] = [
            {"class_path": "reconstruction_experiment.runtime.UpdateCheckpoint", "init_args": {
                "steps": sorted({max(1, updates // 2), updates}), "filename": "candidate-{step:06d}",
                "save_top_k": -1, "save_last": "link", "auto_insert_metric_name": False,
                "dirpath": str(output / phase / "checkpoints")}},
            {"class_path": "reconstruction_experiment.runtime.RecoveryCheckpoint", "init_args": {
                "filename": "recovery-{step:06d}", "every_n_train_steps": 400,
                "save_top_k": 1, "save_last": False, "save_on_train_epoch_end": False,
                "auto_insert_metric_name": False, "dirpath": str(output / phase / "checkpoints")}},
            {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor", "init_args": {"logging_interval": "step"}},
            {"class_path": "reconstruction_experiment.runtime.RestoreDataCursor", "init_args": {
                "steps_per_epoch": per_epoch}},
            {"class_path": "reconstruction_experiment.runtime.RunAudit"},
            {"class_path": "lvr.training.audit.HostMemoryAudit"},
            {"class_path": "reconstruction_experiment.audit.NoDescriptionAudit"},
        ]
        for logger in trainer["logger"]:
            logger["init_args"]["save_dir"] = str(output / phase)
        reference_batch = args.defaults["teacher_batch_size" if teacher else "student_batch_size"]
        reference_accumulation = 128 // (args.defaults["gpus"] * reference_batch)
        trainer["precision"] = None
        trainer["plugins"] = [{"class_path": "reconstruction_experiment.precision." + (
            "FlowDeepSpeedPrecision" if teacher else "MatchedDeepSpeedPrecision"),
            "init_args": {"precision": "bf16-mixed", "reference_accumulation": reference_accumulation}}]
        if teacher:
            config["model"].update(num_latents=8, description_weight=0.0, qa_loss_weight=1.0,
                encoder_type="qwen", encoder_uses_question=args.teacher_sees_question,
                lr_schedule_steps=updates, reconstruction_weight=0.0,
                flow_model_path=args.defaults["flow_model"], flow_weight=args.defaults["flow_weight"],
                flow_mapper_hidden_size=1024, flow_gradient_checkpointing=True,
                flow_protocol_sha256=settings["protocol_sha256"],
                flow_generator_batch_size=args.flow_generator_batch_size,
                flow_vae_batch_size=args.cache_vae_batch_size, flow_latent_cache_bytes=0)
            data.update(flow_data_root=str(args.flow_data), flow_protocol_sha256=settings["protocol_sha256"],
                        flow_latent_root=str(args.flow_latents), flow_vae_model_path=args.defaults["flow_model"],
                        flow_layout_buckets=True, flow_stress_first=preflight)
        else:
            config["model"] = {
                "source_checkpoint": str(output / "teacher_joint/checkpoints/student_source.ckpt"),
                "base_model_path": data["processor_name_or_path"], "prediction_phase": phase,
                "learning_rate": plain["model"]["learning_rate"],
                "head_learning_rate": plain["model"]["head_learning_rate"],
                "weight_decay": plain["model"]["weight_decay"], "lr_warmup_ratio": 0.03,
                "lr_schedule_steps": updates, "latent_weight": 1.0, "cosine_weight": 1.0,
                "qa_loss_weight": 1.0, "description_weight": 0.0,
                "energy_floor": 1e-6, "gradient_checkpointing": True, "seed": 42, "num_loops": 1,
                "allow_question_conditioned_teacher": args.teacher_sees_question,
            }
            if phase == "joint":
                config["init_checkpoint"] = str(output / "warmup/checkpoints/last_full.ckpt")
        configs[phase] = config
    return configs, steps


def training_command(args, phase, recipe):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={args.gpus}", "--module",
            "reconstruction_experiment.stage1" if phase == "teacher_joint" else "reconstruction_experiment.student",
            "fit", "--config", str(recipe)]


def evaluation_command(args, plain, steps, *, dry_run=False):
    command = [sys.executable, str(PROJECT / "scripts/paper_benchmarks/evaluate.py"),
               "--suite", args.defaults["suite"], "--model", plain["data"]["processor_name_or_path"],
               "--mode", "predicted", "--checkpoint", str(args.output / "joint/checkpoints/last_full.ckpt"),
               "--source-checkpoint", str(args.output / "teacher_joint/checkpoints/student_source.ckpt"),
               "--expected-step", str(steps["joint"]), "--gpus", str(args.eval_gpus),
               "--output", str(args.output / "evaluation/predicted")]
    return command + (["--dry-run"] if dry_run else [])


def cache_command(args):
    return [sys.executable, str(EXPERIMENT / "script/prepare_latents.py"),
            "--flow-data", str(args.flow_data), "--output", str(args.flow_latents),
            "--model", args.defaults["flow_model"], "--device", args.cache_device,
            "--workers", str(args.cache_workers), "--batch-size", str(args.cache_vae_batch_size),
            "--shard-mib", str(args.cache_shard_mib)]


def hardware_summary(args, configs):
    phases = {}
    for phase, config in configs.items():
        trainer, data = config["trainer"], config["data"]
        value = trainer["strategy"]["init_args"]["config"]
        deepspeed = value if isinstance(value, dict) else json.loads(Path(value).read_text())
        zero = deepspeed["zero_optimization"]
        phases[phase] = {"gpus": args.gpus, "per_gpu_batch": data["batch_size"],
                         "accumulation": trainer["accumulate_grad_batches"],
                         "global_batch": args.gpus * data["batch_size"] * trainer["accumulate_grad_batches"],
                         "zero_stage": zero["stage"],
                         "optimizer_device": zero.get("offload_optimizer", {}).get("device", "cuda"),
                         "parameter_device": zero.get("offload_param", {}).get("device", "cuda"),
                         "gradient_scale_reference": trainer["plugins"][0]["init_args"]["reference_accumulation"]}
    return {"global_batch": 128, "epoch_tail_global_batch": 64, "phases": phases,
            "eval_gpus": args.eval_gpus, "flow_generator_sub_batch": args.flow_generator_batch_size,
            "automatic_hardware_tuning": False}


def execution_settings(args):
    return {"cpu_threads": args.cpu_threads,
            "skip_preflight": args.skip_preflight,
            "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "1"),
            "kernel_cache": os.environ.get("PYTORCH_KERNEL_CACHE_PATH", str(EXPERIMENT / ".cache/torch/kernels"))}


def execution_environment(args):
    settings = execution_settings(args)
    return dict(os.environ, PYTHONPATH=os.pathsep.join((str(EXPERIMENT), str(FLOW), str(PROJECT / "src"))),
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                SETUPTOOLS_USE_DISTUTILS="local", PYTORCH_CUDA_ALLOC_CONF=settings["cuda_allocator"],
                NCCL_P2P_DISABLE=settings["nccl_p2p_disable"], PYTORCH_KERNEL_CACHE_PATH=settings["kernel_cache"])


def make_plan(args, configs, steps, inputs, settings):
    files = [*sorted((EXPERIMENT / "reconstruction_experiment").glob("*.py")),
             *sorted((EXPERIMENT / "script").glob("*.py")), *sorted((EXPERIMENT / "script").glob("*.sh")),
             EXPERIMENT / "configs/experiment.yaml", *sorted((FLOW / "flow_reconstruction").glob("*.py")),
             FLOW / "scripts/export_teacher.py", *sorted((PROJECT / "src/lvr").rglob("*.py")),
             PROJECT / "scripts/reasoning_data/train_and_evaluate.py", PROJECT / "scripts/latent_prediction/run.py",
             PROJECT / "scripts/paper_benchmarks/evaluate.py"]
    files += [WORKSPACE / "reproduce.py", WORKSPACE / "train_flow.sh", WORKSPACE / "tools/start_flow.py"]
    return {"schema": "qwen25-no-sat-four-gpu-offline-vae-v2", "phases": list(PHASES),
            "execution": execution_settings(args),
            "hardware": hardware_summary(args, configs),
            "inputs": inputs, "recipes": configs, "optimizer_steps": steps,
            "reconstruction_protocol": settings,
            "latent_cache": {"root": str(args.flow_latents), "dtype": "float32", "online_vae": False,
                             "preparation_command": cache_command(args)},
            "description_execution": "disabled in data, train, validation and all three phases",
            "evaluation_mode": "predicted", "evaluation_gpus": args.eval_gpus,
            "answer_template_sha256": digest_file(Path(args.defaults["suite"]).parent / "configs/answer_parsing.json"),
            "suite_manifest_sha256": digest_file(Path(args.defaults["suite"]) / "manifest.json"),
            "code_sha256": {str(path): digest_file(path) for path in files},
            "dependencies": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "lightning", "diffusers", "deepspeed", "jsonargparse")}}


def save_plan(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    binding = output / "workflow_plan.json"
    if binding.exists() and json.loads(binding.read_text()) != plan:
        raise ValueError(f"Existing inputs/code/recipe differ; use a new output: {output}")
    write_json(binding, plan)
    for phase, config in plan["recipes"].items():
        recipe = output / "recipes" / (phase + ".yaml")
        recipe.parent.mkdir(parents=True, exist_ok=True)
        if not (output / phase / "training_started.json").exists():
            recipe.write_text(yaml.safe_dump(config, sort_keys=False))


def run_logged(command, path, environment):
    path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + shlex.join(command), flush=True)
    with path.open("a") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Command exited {code}; see {path}")


def execute_stages(args, configs, steps, environment, workflow):
    export = import_script("qwen25_export_teacher", FLOW / "scripts/export_teacher.py").export_teacher
    workflow.training_command = training_command
    for phase in PHASES:
        workflow.run_stage(args, phase, configs[phase], steps[phase], environment)
        if phase == "teacher_joint":
            export(args.output / "teacher_joint/checkpoints/last_full.ckpt",
                   args.output / "teacher_joint/checkpoints/student_source.ckpt")
    write_json(args.output / "training_complete.json", {"status": "complete", "optimizer_steps": steps})


def main(argv=None):
    args = parse_args(argv)
    plain, counts, settings, inputs = load_inputs(args)
    configs, steps = build_recipes(args, plain, counts, settings)
    plan = make_plan(args, configs, steps, inputs, settings)
    environment = execution_environment(args)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "gpu_started": False, "counts": counts,
                          "hardware": hardware_summary(args, configs),
                          "optimizer_steps": steps, "recipes": configs,
                          "latent_preparation": cache_command(args),
                          "evaluation": evaluation_command(args, plain, steps)}, indent=2))
        return
    Path(environment["PYTORCH_KERNEL_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_plan(args.output, plan)
        print(f"OUTPUT {args.output}\nSTEPS {steps}\nTEACHER_SEES_QUESTION {args.teacher_sees_question}", flush=True)
        if args.prepare_data_only:
            prepare(args.flow_data, settings, counts, args.defaults["prepare_workers"])
            return
        if args.prepare_latents_only:
            prepare(args.flow_data, settings, counts, args.defaults["prepare_workers"])
            run_logged(cache_command(args), args.output / "prepare_latents.log", environment)
            return
        if not args.eval_only:
            run_logged([sys.executable, str(EXPERIMENT / "script/check_cpu.py"),
                        "--output", str(args.output / "cpu_checks"), "--launch-output", str(args.output),
                        "--flow-data", str(args.flow_data), "--gpus", str(args.gpus),
                        "--flow-latents", str(args.flow_latents),
                        "--eval-gpus", str(args.eval_gpus),
                        "--teacher-batch-size", str(args.teacher_batch_size),
                        "--student-batch-size", str(args.student_batch_size),
                        "--cpu-threads", str(args.cpu_threads),
                        "--teacher-optimizer-device", args.teacher_optimizer_device,
                        "--warmup-optimizer-device", args.warmup_optimizer_device,
                        "--joint-optimizer-device", args.joint_optimizer_device,
                        "--zero-stage", str(args.zero_stage),
                        "--teacher-zero-stage", str(args.teacher_zero_stage),
                        "--warmup-zero-stage", str(args.warmup_zero_stage),
                        "--joint-zero-stage", str(args.joint_zero_stage),
                        "--parameter-offload", args.parameter_offload,
                        "--flow-generator-batch-size", str(args.flow_generator_batch_size),
                        *(["--warmup-batch-size", str(args.warmup_batch_size)] if args.warmup_batch_size else []),
                        *(["--joint-batch-size", str(args.joint_batch_size)] if args.joint_batch_size else [])],
                       args.output / "cpu_checks.log", environment)
        if args.prepare_only:
            write_json(args.output / "preparation.json", {"status": "cpu_prepared", "gpu_started": False,
                "reconstruction_images_ready": validate_ready(args.flow_data, settings, counts),
                "optimizer_steps": steps, "gpu_preflight": "pending; runs before full training"})
            return
        import torch
        evaluate = not (args.train_only or args.preflight_only)
        needed = args.eval_gpus if args.eval_only else max(args.gpus, args.eval_gpus if evaluate else 1)
        if torch.cuda.device_count() < needed:
            raise RuntimeError(f"Need {needed} visible GPUs; found {torch.cuda.device_count()}")
        workflow = import_script("qwen25_stage_workflow", PROJECT / "scripts/reasoning_data/train_and_evaluate.py")
        if not args.eval_only:
            if not args.resume and any((base / phase / "training_started.json").exists()
                                       for base in (args.output, args.output / "preflight") for phase in PHASES):
                raise ValueError("This run has training state; use --resume or a new --output")
            prepare(args.flow_data, settings, counts, args.defaults["prepare_workers"])
            run_logged(cache_command(args), args.output / "prepare_latents.log", environment)
            pre_configs, pre_steps = build_recipes(args, plain, counts, settings, preflight=True)
            pre_args = copy.copy(args)
            pre_args.output = args.output / "preflight"
            save_plan(pre_args.output, make_plan(pre_args, pre_configs, pre_steps, inputs, settings))
            if not args.skip_preflight:
                execute_stages(pre_args, pre_configs, pre_steps, environment, workflow)
            if args.preflight_only:
                return
            execute_stages(args, configs, steps, environment, workflow)
        if not args.train_only:
            for phase in PHASES:
                done = json.loads((args.output / phase / "complete.json").read_text())
                if done["optimizer_steps"] != steps[phase] or done["recipe_sha256"] != workflow.digest(configs[phase]):
                    raise ValueError("Evaluation requires all completed stages from this recipe")
                workflow.rank_states(args.output / phase, args.gpus, steps[phase])
            run_logged(evaluation_command(args, plain, steps), args.output / "evaluation/launcher.log", environment)
            validation = json.loads((args.output / "evaluation/predicted/status.json").read_text())
            summary = json.loads((args.output / "evaluation/predicted/summary.json").read_text())
            if (validation["status"] != "complete" or validation["generations"] != 10107
                    or len(summary["metrics"]) != 11 or sum(m["generations"] for m in summary["metrics"]) != 10107):
                raise ValueError("Incomplete unified evaluation")
            write_json(args.output / "workflow_status.json", {"status": "complete", "optimizer_steps": steps,
                                                              "evaluation": str(args.output / "evaluation/predicted")})


if __name__ == "__main__":
    main()
