"""GPU acceptance timing only; formal training keeps synchronization-free execution."""
import json
from pathlib import Path
import time
import torch
from lightning.pytorch.callbacks import Callback

class ParallelTimingAudit(Callback):
    def on_train_start(self,trainer,module):
        audit=next(cb for cb in trainer.callbacks if type(cb).__name__=='RunAudit')
        self.path=Path(audit.path).parent/f'parallel_timing_rank{trainer.global_rank}.jsonl'
        self.total_gib=torch.cuda.get_device_properties(module.device).total_memory/2**30
        self.previous_end=time.perf_counter()
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        torch.cuda.synchronize(module.device);self.started=time.perf_counter();self.before_step=trainer.global_step
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        torch.cuda.synchronize(module.device)
        runtime=module.__dict__.get('_flow_runtime')
        stats=runtime.last_stats if runtime is not None else {}
        ended=time.perf_counter()
        record={'rank':trainer.global_rank,'batch_idx':batch_idx,'optimizer_step':trainer.global_step,'optimizer_step_before':self.before_step,
                'seconds':ended-self.previous_end,'gpu_compute_seconds':ended-self.started,'batch_size':len(batch['ids']),
                'stress_case':bool(getattr(trainer.datamodule.hparams,'flow_stress_first',False) and self.before_step<2),
                'allocated_gib':torch.cuda.max_memory_allocated(module.device)/2**30,
                'reserved_gib':torch.cuda.max_memory_reserved(module.device)/2**30,'total_gib':self.total_gib,
                'generator_batch_sizes':stats.get('generator_batch_sizes',[]),
                'vae_batch_sizes':stats.get('vae_batch_sizes',[]),'vae_cache_hits':stats.get('vae_cache_hits',0)}
        self.previous_end=ended
        with self.path.open('a') as stream:stream.write(json.dumps(record)+'\n')
