"""Persist training progress, initial weights and sampler order for matched experiments."""

import hashlib
import json
from pathlib import Path
import time

from lightning.pytorch import Callback
import torch

from lvr.training.diagnostics import trace_event, traced


class HostMemoryAudit(Callback):
    """Record RAM without collecting tensors or changing training computation."""

    def snapshot(self, trainer, event):
        record = {"at_unix": time.time(), "event": event, "rank": trainer.global_rank,
                  "step": trainer.global_step, "sanity_checking": trainer.sanity_checking}
        try:
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[0] in ("Rss:", "Pss:", "Private_Dirty:", "Locked:"):
                    record[parts[0].rstrip(":")] = int(parts[1]) * 1024
            root = Path("/sys/fs/cgroup")
            record["cgroup_current"] = int((root / "memory.current").read_text())
            record["cgroup_stat"] = {k: int(v) for k, v in
                                     (line.split() for line in (root / "memory.stat").read_text().splitlines())
                                     if k in ("anon", "file", "shmem", "kernel", "unevictable")}
            # Some PyTorch builds expose this API but return no host statistics.
            record["pinned_host_stats"] = dict(torch.cuda.memory.host_memory_stats())
        except (OSError, AttributeError, RuntimeError) as exc:
            record["sampling_error"] = str(exc)
        path = Path(trainer.default_root_dir) / f"host_memory_rank{trainer.global_rank}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def on_train_start(self, trainer, module):
        self.snapshot(trainer, "train_start")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step <= 2 or trainer.global_step % 50 == 0:
            self.snapshot(trainer, "train_batch_end")

    def on_validation_start(self, trainer, module):
        self.snapshot(trainer, "validation_start")

    def on_validation_end(self, trainer, module):
        self.snapshot(trainer, "validation_end")

    def on_save_checkpoint(self, trainer, module, checkpoint):
        self.snapshot(trainer, "save_checkpoint")

    def on_train_end(self, trainer, module):
        self.snapshot(trainer, "train_end")


class RunAudit(Callback):
    def setup(self, trainer, module, stage):
        trace_event("trainer.setup", stage=str(stage))

    def on_fit_start(self, trainer, module):
        trace_event("fit.start")

    def on_fit_end(self, trainer, module):
        trace_event("fit.end")

    def on_train_epoch_start(self, trainer, module):
        trace_event("train_epoch.start", epoch=trainer.current_epoch)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        trace_event("train_batch.begin", batch_idx=batch_idx, optimizer_step=trainer.global_step,
                    sample_ids=batch["ids"])

    def on_before_backward(self, trainer, module, loss):
        trace_event("backward.begin", optimizer_step=trainer.global_step)

    def on_after_backward(self, trainer, module):
        trace_event("backward.end", optimizer_step=trainer.global_step)

    def on_before_optimizer_step(self, trainer, module, optimizer):
        trace_event("optimizer_step.begin", optimizer_step=trainer.global_step)

    def on_validation_start(self, trainer, module):
        trace_event("validation.start")

    def on_validation_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx=0):
        trace_event("validation_batch.begin", batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        trace_event("validation_batch.end", batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_validation_end(self, trainer, module):
        trace_event("validation.end")

    def on_exception(self, trainer, module, exception):
        trace_event("trainer.exception", exception_type=type(exception).__name__)

    @traced("train_start_audit")
    def on_train_start(self, trainer, module):
        self.started = time.time()
        self.directory = Path(trainer.default_root_dir)
        self.path = self.directory / f"runtime_rank{trainer.global_rank}.json"
        loader = trainer.train_dataloader
        assert module.hparams.encoder_uses_question == trainer.datamodule.hparams.encoder_uses_question
        assert module.hparams.num_latents == trainer.datamodule.hparams.num_latents
        assert bool(module.hparams.description_weight) == trainer.datamodule.hparams.include_descriptions
        for key in ("encoder_type", "encoder_includes_original"):
            assert getattr(module.hparams, key) == getattr(trainer.datamodule.hparams, key)
        def full_numel(parameter):
            # ZeRO-3 leaves a zero-sized placeholder and records the original
            # size in ds_numel after parameters have been partitioned.
            return int(getattr(parameter, "ds_numel", parameter.numel()))

        zero3 = bool(getattr(trainer.strategy, "zero_stage_3", False))
        qa_hash = hashlib.sha256()
        if zero3:
            # Gathering every multi-billion-parameter tensor merely to hash 32
            # values would defeat the memory purpose of ZeRO-3. Hash stable
            # names/shapes here; checkpoint paths/hashes audit the actual state.
            for name, parameter in module.qa.named_parameters():
                qa_hash.update(name.encode())
                qa_hash.update(str(tuple(getattr(parameter, "ds_shape", parameter.shape))).encode())
            qa_hash_scope = "names_and_shapes_zero3"
            optimizer_groups = [{
                "parameters": sum(full_numel(p) for p in module.parameters() if p.requires_grad),
                "initial_lr": None,
                "partitioned": True,
            }]
        else:
            for name, parameter in module.qa.named_parameters():
                values = parameter.detach().flatten()
                qa_hash.update(name.encode())
                qa_hash.update(torch.cat((values[:16], values[-16:])).float().cpu().numpy().tobytes())
            qa_hash_scope = "parameter_samples"
            optimizer = trainer.optimizers[0]
            strategy_config = getattr(trainer.strategy, "config", {})
            zero2 = (isinstance(strategy_config, dict)
                     and strategy_config.get("zero_optimization", {}).get("stage") == 2)
            # ZeRO-2's optimizer param_groups contain flattened FP32 partitions.
            # Validate its original model groups, rather than comparing master
            # partition identities with the unpartitioned model parameters.
            logical_groups = optimizer.bit16_groups if zero2 else [g["params"] for g in optimizer.param_groups]
            parameters = [p for group in logical_groups for p in group]
            assert len(parameters) == len({id(p) for p in parameters})
            assert {id(p) for p in parameters} == {id(p) for p in module.parameters() if p.requires_grad}
            optimizer_groups = [{"parameters": sum(p.numel() for p in parameters),
                                 "initial_lr": group.get("initial_lr", group["lr"]),
                                 **({"partitioned": True} if zero2 else {})}
                                for parameters, group in zip(logical_groups, optimizer.param_groups)]
        self.record = {
            "started_at_unix": self.started, "rank": trainer.global_rank,
            "expected_optimizer_steps": int(trainer.estimated_stepping_batches),
            "train_samples": len(loader.dataset), "batch_size_per_gpu": loader.batch_size,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "world_size": trainer.world_size,
            "num_latents": module.hparams.num_latents,
            "description_weight": module.hparams.description_weight,
            "description_input": "latent_and_fixed_instruction_only" if module.hparams.description_weight else None,
            "encoder_uses_question": module.hparams.encoder_uses_question,
            "encoder_type": module.hparams.encoder_type,
            "encoder_includes_original": module.hparams.encoder_includes_original,
            "qformer_use_geometry": module.hparams.qformer_use_geometry,
            "qa_initial_parameter_sample_sha256": qa_hash.hexdigest(),
            "qa_initial_parameter_hash_scope": qa_hash_scope,
            "initial_checkpoint": getattr(module, "initial_checkpoint", None),
            "initial_global_step": trainer.global_step,
            "initial_epoch": trainer.current_epoch,
            "initial_optimizer_state_entries": len(trainer.optimizers[0].state),
            "sampler_indices_sha256": hashlib.sha256(json.dumps(list(iter(loader.sampler))).encode()).hexdigest(),
            "sample_ids_sha256": hashlib.sha256(json.dumps([x['id'] for x in loader.dataset.items]).encode()).hexdigest(),
            "optimizer_groups": optimizer_groups,
            "qa_trainable_parameters": sum(full_numel(p) for p in module.qa.parameters() if p.requires_grad),
            "encoder_trainable_parameters": sum(full_numel(p) for p in module.encoder.parameters() if p.requires_grad)
            if module.encoder is not None else 0,
            "status": "training", "optimizer_steps": 0,
        }
        self.path.write_text(json.dumps(self.record, indent=2) + '\n')

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        trace_event("train_batch.end", batch_idx=batch_idx, optimizer_step=trainer.global_step)
        step = trainer.global_step
        if step <= 2 or step % 10 == 0:
            self.record.update(optimizer_steps=step, updated_at_unix=time.time(),
                               elapsed_seconds=time.time()-self.started,
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                               latest_loss=float(outputs['loss'].detach()),
                               current_batch_samples=len(batch['ids']))
            self.path.write_text(json.dumps(self.record, indent=2) + '\n')
            if trainer.is_global_zero:
                print(f"[progress] step={step}/{self.record['expected_optimizer_steps']} "
                      f"elapsed={self.record['elapsed_seconds']:.1f}s "
                      f"loss={self.record['latest_loss']:.6f} "
                      f"allocated={self.record['peak_allocated_gib']:.2f}GiB", flush=True)

    def on_train_end(self, trainer, module):
        trace_event("train.end", optimizer_step=trainer.global_step)
        self.record.update(status='training_complete', optimizer_steps=trainer.global_step,
                           finished_at_unix=time.time(), elapsed_seconds=time.time()-self.started,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        self.path.write_text(json.dumps(self.record, indent=2) + '\n')
