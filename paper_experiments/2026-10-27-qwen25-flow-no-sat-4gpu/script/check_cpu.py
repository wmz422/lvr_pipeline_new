#!/usr/bin/env python3
"""Parse every recipe and inspect real Qwen2.5 processor inputs without loading 7B weights."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningArgumentParser
from transformers import AutoProcessor

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1] / "lvr_pipeline_new"
FLOW = EXPERIMENT.parents[1] / "doc/flow_reconstruction_parallel_20261015"
sys.path[:0] = [str(EXPERIMENT), str(FLOW), str(PROJECT / "src")]
spec = importlib.util.spec_from_file_location("reconstruction_launch", EXPERIMENT / "script/run.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
from reconstruction_experiment.data import FlowDataModule, StudentDataModule
from flow_reconstruction.gt import GTConfig, process_sample, read_records
from flow_reconstruction.stage1 import FlowStageCLI
from lvr.data.crop_qa import CropQACollator, CropQADataModule, CropQADataset
from lvr.data.prompt import cap_qwen_pixels
from lvr.models.latent_prediction import LatentPrediction, prediction_prompt
from lvr.predict_latents import PredictionCLI
from reconstruction_experiment.prepare import samples, write_json, protocol, prepare
from reconstruction_experiment.teacher import NoDescriptionFlowTeacher


def check_recipes(args, plain, counts, settings):
    checked = []
    for sees_question in (False, True):
        variant = copy.copy(args)
        variant.teacher_sees_question = sees_question
        for preflight in (False, True):
            configs, steps = launch.build_recipes(variant, plain, counts, settings, preflight=preflight)
            assert list(configs) == ["teacher_joint", "warmup", "joint"]
            if not preflight:
                assert steps == {"teacher_joint": 1346, "warmup": 200, "joint": 1346}
            for phase, config in configs.items():
                teacher = phase == "teacher_joint"
                cls, data_cls, cli_cls = ((NoDescriptionFlowTeacher, FlowDataModule, FlowStageCLI) if teacher
                                         else (LatentPrediction, StudentDataModule, PredictionCLI))
                parser = LightningArgumentParser(exit_on_error=False)
                parser.add_lightning_class_args(Trainer, "trainer")
                parser.add_lightning_class_args(cls, "model")
                parser.add_lightning_class_args(data_cls, "data")
                cli_cls.add_arguments_to_parser(object.__new__(cli_cls), parser)
                parser.add_argument("--ckpt_path", type=str, default=None)
                parser.add_argument("--seed_everything", default=42)
                parsed = parser.parse_object(config)
                assert parsed.model.description_weight == 0 and not parsed.data.include_descriptions
                assert parsed.data.caption_path is None and not parsed.data.qa_crops
                assert parsed.data.encoder_uses_question == sees_question
                assert parsed.data.original_pixels == plain["data"]["original_pixels"]
                assert parsed.trainer.devices * parsed.data.batch_size * parsed.trainer.accumulate_grad_batches == 128
                reference_batch = args.defaults["teacher_batch_size" if teacher else "student_batch_size"]
                assert config["trainer"]["plugins"][0]["init_args"]["reference_accumulation"] == (
                    128 // (args.defaults["gpus"] * reference_batch))
                assert not parsed.trainer.use_distributed_sampler
                for callback in config["trainer"]["callbacks"][:2]:
                    expected_root = args.output / "preflight" if preflight else args.output
                    assert callback["init_args"]["dirpath"] == str(expected_root / phase / "checkpoints")
                checked.append({"phase": phase, "preflight": preflight, "teacher_sees_question": sees_question,
                                "batch": parsed.data.batch_size, "accumulation": parsed.trainer.accumulate_grad_batches})
    return checked


def check_real_loader(args, plain, output):
    """Real QA processor and offline loaders; only CPU-probe Flow RGB is reduced."""
    from types import SimpleNamespace
    from reconstruction_experiment.data import AccumulationBatchSampler, LatentFlowCollator
    from reconstruction_experiment.latent_cache import build_plan, bind_plan, run_worker, finalize
    root = output / "loader_probe"
    data = copy.deepcopy(plain["data"])
    for prefix, legacy_count in (("train", 6), ("val", 2)):
        for key, count in ((prefix + "_path", legacy_count), (prefix + "_boxes_path", legacy_count),
                           ("additional_" + prefix + "_path", 2)):
            path = root / (key + ".json")
            write_json(path, read_records(data[key])[:count])
            data[key] = str(path)
    # A dedicated debug GT protocol bounds CPU VAE cost. Formal recipes remain 2MP.
    settings = protocol(dict(data, original_pixels=3072, crop_pixels=1024), PROJECT)
    prepare(root / "gt", settings, {"train": 8, "val": 4}, workers=2)
    cache_plan = build_plan(root / "gt", args.defaults["flow_model"], device_type="cpu",
                            batch_size=2, shard_bytes=2**20)
    cache_root = root / "latents" / cache_plan["plan_sha256"]
    bind_plan(cache_root, cache_plan)
    run_worker(cache_root, 0, 1, "cpu")
    finalize(cache_root)
    data.pop("train_length_manifest_path", None)
    data.update(num_latents=8, include_encoder=True, max_length=args.defaults["max_length"],
                batch_size=2, num_workers=0, val_num_workers=0, pin_memory=False, persistent_workers=False)
    flow_data = dict(data, flow_data_root=str(root / "gt"), flow_protocol_sha256=settings["protocol_sha256"],
                     flow_latent_root=str(cache_root), flow_vae_model_path=args.defaults["flow_model"],
                     flow_layout_buckets=True, flow_stress_first=False)
    results = []
    for cls, kwargs in ((FlowDataModule, flow_data), (StudentDataModule, data)):
        module = cls(**kwargs)
        module.trainer = SimpleNamespace(world_size=4, global_rank=0, accumulate_grad_batches=1)
        module.setup("fit")
        train, val = module.train_dataloader(), module.val_dataloader()
        assert isinstance(train.batch_sampler, AccumulationBatchSampler)
        assert len(val.sampler) == 1 and val.sampler.num_replicas == 4
        batch = next(iter(train))
        expected = [module.train_dataset.items[i]["id"] for i in next(iter(train.batch_sampler))]
        assert batch["ids"] == expected and len(batch["ids"]) == 2
        assert "description" not in batch
        if cls is FlowDataModule:
            assert isinstance(train.collate_fn, LatentFlowCollator)
            assert all(x.dtype == torch.float32 and x.shape[:2] == (1, 128) for x in batch["flow"]["targets"])
            assert batch["flow"]["ids"] == batch["ids"]
            plain_batch = train.collate_fn.base_collator([module.train_dataset[i] for i in next(iter(train.batch_sampler))])
            for branch in ("qa", "encoder"):
                for key, value in plain_batch[branch].items():
                    assert torch.equal(value, batch[branch][key])
        results.append({"module": cls.__name__, "train_ids": batch["ids"],
                        "validation_ids": next(iter(val))["ids"], "passed": True,
                        "qa_original_pixels": data["original_pixels"], "cpu_probe_flow_original_pixels": 3072})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--launch-output", type=Path)
    forwarded = ("gpus", "eval-gpus", "teacher-batch-size", "student-batch-size", "flow-data", "flow-latents",
                 "warmup-batch-size", "joint-batch-size", "teacher-optimizer-device",
                 "warmup-optimizer-device", "joint-optimizer-device", "cpu-threads", "zero-stage",
                 "teacher-zero-stage", "warmup-zero-stage", "joint-zero-stage", "parameter-offload",
                 "flow-generator-batch-size")
    for key in forwarded:
        p.add_argument("--" + key)
    a = p.parse_args()
    torch.set_num_threads(2)
    launch_args = ["--dry-run"]
    if a.launch_output:
        launch_args += ["--output", str(a.launch_output)]
    for key in forwarded:
        value = getattr(a, key.replace("-", "_"))
        if value is not None:
            launch_args += ["--" + key, value]
    args = launch.parse_args(launch_args)
    plain, counts, settings, _ = launch.load_inputs(args)
    recipes = check_recipes(args, plain, counts, settings)
    print("CPU_RECIPE_PARSE_PASSED", len(recipes), flush=True)
    data = plain["data"]
    sets = {}
    selected = {}
    for split, prefix in (("train", "train"), ("val", "val")):
        legacy = CropQADataset(data[prefix + "_path"], data[prefix + "_boxes_path"], caption_path=None).items
        reasoning = CropQADataset(data["additional_" + prefix + "_path"]).items
        all_items = legacy + reasoning
        ids = {r["id"] for r in all_items}
        assert len(ids) == len(all_items) == counts[split]
        gt_ids = {r["id"] for r in samples(settings, split)}
        assert ids == gt_ids
        sets[split] = ids
        # Exercise each source and long/multi-image cases; this is not an exhaustive token audit.
        choices = [legacy[0], max(legacy, key=lambda r: len(r["boxes"])),
                   max(legacy, key=lambda r: len(r["question"]) + len(r["answer"])),
                   max(reasoning, key=lambda r: len(r["images"]) + len(r["auxiliary_images"])),
                   max(reasoning, key=lambda r: len(r["question"]) + len(r["answer"]))]
        by_source = {}
        for row in reasoning:
            by_source.setdefault(row["source"], row)
        choices.extend(by_source.values())
        selected.update({r["id"]: r for r in choices})
    assert not sets["train"].intersection(sets["val"])
    print("CLEAN_QA_RECONSTRUCTION_IDS_PASSED", counts, flush=True)
    processor = AutoProcessor.from_pretrained(data["processor_name_or_path"], local_files_only=True)
    processor.tokenizer.padding_side = "right"
    cap_qwen_pixels(processor, data["original_pixels"])
    def collator(question=False):
        return CropQACollator(processor, data["image_root"], num_latents=8,
            original_pixels=data["original_pixels"], crop_pixels=data["crop_pixels"],
            max_length=args.defaults["max_length"], include_encoder=True, qa_crops=False,
            encoder_uses_question=question, include_descriptions=False)
    records = []
    for item in selected.values():
        item = dict(item, description="")
        batch = collator()([item])
        assert "description" not in batch and "labels" not in batch["encoder"]
        assert batch["qa"]["slot_mask"].sum() == batch["encoder"]["slot_mask"].sum() == 8
        original_count = len(item.get("images") or [item["question_image"]])
        assert len(batch["qa"]["image_grid_thw"]) == original_count
        records.append({"id": item["id"], "qa_tokens": batch["qa"]["input_ids"].shape[1],
                        "encoder_tokens": batch["encoder"]["input_ids"].shape[1],
                        "original_images": original_count,
                        "encoder_images": len(batch["encoder"]["image_grid_thw"])})
        print("PROCESSOR_PASS", item["id"], records[-1]["qa_tokens"], records[-1]["encoder_tokens"], flush=True)
    probe = dict(next(iter(selected.values())), question="QUESTION_MARKER_B8", answer="ANSWER_MARKER_F6", description="")
    for sees_question in (False, True):
        batch = collator(sees_question)([probe])
        encoded = processor.tokenizer.decode(batch["encoder"]["input_ids"][0])
        assert ("QUESTION_MARKER_B8" in encoded) == sees_question
        assert "ANSWER_MARKER_F6" not in encoded
        prompt = prediction_prompt(batch["qa"], processor.tokenizer.pad_token_id)
        decoded = processor.tokenizer.decode(prompt["input_ids"][0])
        assert "QUESTION_MARKER_B8" in decoded and "ANSWER_MARKER_F6" not in decoded
    # Prepare representative real legacy/reasoning targets with the new 2MP protocol.
    target_examples = []
    for split in ("train", "val"):
        wanted = {r["id"] for r in selected.values()}
        chosen = {}
        for sample in samples(settings, split):
            if sample["id"] in wanted:
                kind = "helpers" if sample["helpers"] else "multiple" if len(sample["boxes"]) > 1 else "single"
                chosen.setdefault(kind, sample)
        for sample in chosen.values():
            row = process_sample(sample, a.output / "sample_reconstruction", GTConfig(**settings["gt_config"]))
            target_examples.append({"id": row["id"], "layout": row["layout"]["kind"],
                                    "target_size": [row["target"]["width"], row["target"]["height"]]})
    evaluation = launch.evaluation_command(args, plain, {"joint": 1346}, dry_run=True)
    import subprocess
    subprocess.run(evaluation, check=True, cwd=PROJECT)
    loaders = check_real_loader(args, plain, a.output)
    write_json(a.output / "validation.json", {"status": "passed", "gpu_started": False,
        "full_model_instantiated": False, "recipes_checked": recipes, "data_counts": counts,
        "all_clean_qa_ids_equal_reconstruction_ids": True, "description_collator_execution": "absent",
        "teacher_question_toggle_and_answer_exclusion": "passed", "processor_samples": records,
        "reconstruction_examples": target_examples,
        "real_datamodule_checks": loaders,
        "limits": "Representative real processor checks; full 7B GPU preflight remains required."})
    print("CPU_CHECKS_PASSED", a.output / "validation.json", flush=True)


if __name__ == "__main__":
    main()
