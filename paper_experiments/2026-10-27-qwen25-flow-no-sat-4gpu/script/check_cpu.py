#!/usr/bin/env python3
"""Parse every recipe and inspect the selected Qwen VL processor without full model weights."""
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
from reconstruction_experiment.description_teacher import DescriptionFlowTeacher


def check_recipes(args, plain, counts, settings):
    checked = []
    for sees_question in (False, True):
        variant = copy.copy(args)
        variant.teacher_sees_question = sees_question
        for preflight in (False, True):
            configs, steps = launch.build_recipes(variant, plain, counts, settings, preflight=preflight)
            assert list(configs) == list(launch.phases(args))
            if not preflight:
                assert steps == {phase: 200 if phase == "warmup" else 1346 for phase in launch.phases(args)}
            for phase, config in configs.items():
                teacher = launch.teacher_phase(phase)
                teacher_cls = DescriptionFlowTeacher if config["model"]["description_weight"] else NoDescriptionFlowTeacher
                cls, data_cls, cli_cls = ((teacher_cls, FlowDataModule, FlowStageCLI) if teacher
                                         else (LatentPrediction, StudentDataModule, PredictionCLI))
                parser = LightningArgumentParser(exit_on_error=False)
                parser.add_lightning_class_args(Trainer, "trainer")
                parser.add_lightning_class_args(cls, "model")
                parser.add_lightning_class_args(data_cls, "data")
                cli_cls.add_arguments_to_parser(object.__new__(cli_cls), parser)
                parser.add_argument("--ckpt_path", type=str, default=None)
                parser.add_argument("--seed_everything", default=42)
                parsed = parser.parse_object(config)
                described = bool(config["model"]["description_weight"])
                assert parsed.data.include_descriptions == described
                assert bool(parsed.data.caption_path) == described and not parsed.data.qa_crops
                assert parsed.data.description_image_mode == "none"
                assert parsed.data.include_encoder == (teacher or parsed.model.latent_weight > 0)
                assert parsed.data.encoder_uses_question == (sees_question and parsed.data.include_encoder)
                assert parsed.data.original_pixels == plain["data"]["original_pixels"]
                assert parsed.data.num_latents == args.defaults.get("num_latents", 8)
                if teacher:
                    assert parsed.model.num_latents == parsed.data.num_latents
                assert parsed.trainer.devices * parsed.data.batch_size * parsed.trainer.accumulate_grad_batches == 128
                reference_batch = args.defaults["teacher_batch_size" if teacher else "student_batch_size"]
                assert config["trainer"]["plugins"][0]["init_args"]["reference_accumulation"] == (
                    128 // (args.defaults["gpus"] * reference_batch))
                assert not parsed.trainer.use_distributed_sampler
                for callback in config["trainer"]["callbacks"][:2]:
                    expected_root = args.output / "preflight" if preflight else args.output
                    assert callback["init_args"]["dirpath"] == str(expected_root / phase / "checkpoints")
                checked.append({"phase": phase, "preflight": preflight, "teacher_sees_question": sees_question,
                                "num_latents": parsed.data.num_latents,
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
    needs_flow = any(map(launch.teacher_phase, launch.phases(args)))
    if needs_flow:
        settings = protocol(dict(data, original_pixels=3072, crop_pixels=1024), PROJECT)
        prepare(root / "gt", settings, {"train": 8, "val": 4}, workers=2)
        cache_plan = build_plan(root / "gt", args.defaults["flow_model"], device_type="cpu",
                                batch_size=2, shard_bytes=2**20)
        cache_root = root / "latents" / cache_plan["plan_sha256"]
        bind_plan(cache_root, cache_plan)
        run_worker(cache_root, 0, 1, "cpu")
        finalize(cache_root)
    data.pop("train_length_manifest_path", None)
    data.update(num_latents=args.defaults.get("num_latents", 8), include_encoder=args.defaults.get("latent_weight", 1.0) > 0,
                include_descriptions=False, caption_path=None, max_length=args.defaults["max_length"],
                batch_size=2, num_workers=0, val_num_workers=0, pin_memory=False, persistent_workers=False)
    if not data["include_encoder"]:
        data["predicted_latents_only"] = True
    modules = [(StudentDataModule, data)]
    if needs_flow:
        described = bool(args.defaults.get("teacher_description_weight", 0))
        flow_data = dict(data, include_encoder=True, include_descriptions=described,
                        caption_path=args.defaults.get("description_captions") if described else None,
                        flow_data_root=str(root / "gt"), flow_protocol_sha256=settings["protocol_sha256"],
                        flow_latent_root=str(cache_root), flow_vae_model_path=args.defaults["flow_model"],
                        flow_layout_buckets=True, flow_stress_first=False)
        modules.insert(0, (FlowDataModule, flow_data))
    results = []
    for cls, kwargs in modules:
        module = cls(**kwargs)
        module.trainer = SimpleNamespace(world_size=4, global_rank=0, accumulate_grad_batches=1)
        module.setup("fit")
        train, val = module.train_dataloader(), module.val_dataloader()
        assert isinstance(train.batch_sampler, AccumulationBatchSampler)
        assert len(val.sampler) == 1 and val.sampler.num_replicas == 4
        batch = next(iter(train))
        expected = [module.train_dataset.items[i]["id"] for i in next(iter(train.batch_sampler))]
        assert batch["ids"] == expected and len(batch["ids"]) == 2
        assert ("description" in batch) == kwargs["include_descriptions"]
        assert ("encoder" in batch) == kwargs["include_encoder"]
        if kwargs["include_descriptions"]:
            assert "pixel_values" not in batch["description"]
            assert not batch["description"]["mm_token_type_ids"].any()
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
                 "flow-generator-batch-size", "source-run")
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
        legacy = CropQADataset(data[prefix + "_path"], data[prefix + "_boxes_path"],
            caption_path=args.defaults.get("description_captions") if args.defaults.get("teacher_description_weight", 0) else None).items
        reasoning = CropQADataset(data["additional_" + prefix + "_path"]).items
        all_items = legacy + reasoning
        ids = {r["id"] for r in all_items}
        assert len(ids) == len(all_items) == counts[split]
        gt_ids = {r["id"] for r in samples(settings, split)}
        assert ids == gt_ids
        if args.defaults.get("teacher_description_weight", 0):
            assert all(item.get("description", "").strip() for item in all_items)
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
    needs_encoder = args.defaults.get("latent_weight", 1.0) > 0
    num_latents = args.defaults.get("num_latents", 8)
    def collator(question=False, descriptions=False):
        return CropQACollator(processor, data["image_root"], num_latents=num_latents,
            original_pixels=data["original_pixels"], crop_pixels=data["crop_pixels"],
            max_length=args.defaults["max_length"], include_encoder=needs_encoder, qa_crops=False,
            encoder_uses_question=question, include_descriptions=descriptions,
            description_image_mode="none")
    records = []
    for item in selected.values():
        batch = collator()([item])
        assert "description" not in batch
        assert batch["qa"]["slot_mask"].sum() == num_latents
        if needs_encoder:
            assert "labels" not in batch["encoder"] and batch["encoder"]["slot_mask"].sum() == num_latents
        else:
            assert "encoder" not in batch
        if args.defaults.get("teacher_description_weight", 0):
            described = collator(descriptions=True)([item])["description"]
            assert "pixel_values" not in described and "image_grid_thw" not in described
            assert not described["mm_token_type_ids"].any() and (described["labels"] != -100).any()
        original_count = len(item.get("images") or [item["question_image"]])
        assert len(batch["qa"]["image_grid_thw"]) == original_count
        records.append({"id": item["id"], "num_latents": num_latents,
                        "qa_tokens": batch["qa"]["input_ids"].shape[1],
                        "encoder_tokens": batch["encoder"]["input_ids"].shape[1] if needs_encoder else 0,
                        "original_images": original_count,
                        "encoder_images": len(batch["encoder"]["image_grid_thw"]) if needs_encoder else 0})
        print("PROCESSOR_PASS", item["id"], records[-1]["qa_tokens"], records[-1]["encoder_tokens"], flush=True)
    probe = dict(next(iter(selected.values())), question="QUESTION_MARKER_B8", answer="ANSWER_MARKER_F6", description="")
    for sees_question in (False, True):
        batch = collator(sees_question)([probe])
        if needs_encoder:
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
        "all_clean_qa_ids_equal_reconstruction_ids": True,
        "description_collator_execution": "image_free_teacher_only" if args.defaults.get("teacher_description_weight", 0) else "absent",
        "teacher_question_toggle_and_answer_exclusion": "passed", "processor_samples": records,
        "reconstruction_examples": target_examples,
        "real_datamodule_checks": loaders,
        "base_model": data["processor_name_or_path"], "num_latents": num_latents,
        "flow_loss_weight": args.defaults["flow_weight"],
        "limits": "Representative real processor checks; full-model GPU preflight remains required."})
    print("CPU_CHECKS_PASSED", a.output / "validation.json", flush=True)


if __name__ == "__main__":
    main()
