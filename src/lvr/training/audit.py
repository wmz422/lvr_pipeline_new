"""Persist training progress, initial weights and sampler order for matched experiments."""

import hashlib
import json
from pathlib import Path
import time

from lightning.pytorch import Callback
import torch


class RunAudit(Callback):
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
        qa_hash = hashlib.sha256()
        for name, parameter in module.qa.named_parameters():
            values = parameter.detach().flatten()
            qa_hash.update(name.encode())
            qa_hash.update(torch.cat((values[:16], values[-16:])).float().cpu().numpy().tobytes())
        parameters = [p for group in trainer.optimizers[0].param_groups for p in group["params"]]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in parameters} == {id(p) for p in module.parameters() if p.requires_grad}
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
            "initial_checkpoint": getattr(module, "initial_checkpoint", None),
            "initial_global_step": trainer.global_step,
            "initial_epoch": trainer.current_epoch,
            "initial_optimizer_state_entries": len(trainer.optimizers[0].state),
            "sampler_indices_sha256": hashlib.sha256(json.dumps(list(iter(loader.sampler))).encode()).hexdigest(),
            "sample_ids_sha256": hashlib.sha256(json.dumps([x['id'] for x in loader.dataset.items]).encode()).hexdigest(),
            "optimizer_groups": [{"parameters": sum(p.numel() for p in g['params']),
                                  "initial_lr": g.get('initial_lr', g['lr'])}
                                 for g in trainer.optimizers[0].param_groups],
            "qa_trainable_parameters": sum(p.numel() for p in module.qa.parameters() if p.requires_grad),
            "encoder_trainable_parameters": sum(p.numel() for p in module.encoder.parameters() if p.requires_grad)
            if module.encoder is not None else 0,
            "status": "training", "optimizer_steps": 0,
        }
        self.path.write_text(json.dumps(self.record, indent=2) + '\n')

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
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
        self.record.update(status='training_complete', optimizer_steps=trainer.global_step,
                           finished_at_unix=time.time(), elapsed_seconds=time.time()-self.started)
        self.path.write_text(json.dumps(self.record, indent=2) + '\n')
