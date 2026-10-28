"""Short-run assertions for the reasoning data integration."""
import json
from pathlib import Path
import torch
from lightning.pytorch import Callback


class ReasoningAcceptance(Callback):
    def __init__(self,stop_after_updates=0):
        self.stop_after_updates=stop_after_updates;self.shapes=set();self.gradient_steps=0;self.batches=0;self.global_norms=[];self.last_norm_step=-1

    def on_train_start(self,trainer,module):
        self.tied_backbones=[]
        for name in ('encoder','qa'):
            backbone=getattr(module,name,None)
            if backbone is not None and hasattr(backbone,'get_input_embeddings') and getattr(backbone.config,'tie_word_embeddings',False):
                if backbone.get_input_embeddings().weight is not backbone.get_output_embeddings().weight:
                    raise AssertionError('Checkpoint split shared word embeddings: '+name)
                self.tied_backbones.append(name)
        self.stop_this_run=trainer.global_step<self.stop_after_updates
        self.initial_step=trainer.global_step
        self.hook=module.register_forward_hook(self.inspect_forward)

    def inspect_forward(self,module,args,output):
        if not isinstance(output,dict):return
        for name in ('mu','predicted','teacher'):
            value=output.get(name)
            if value is not None:
                if tuple(value.shape[1:])!=(8,512):raise AssertionError((name,value.shape))
                if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite '+name)
                self.shapes.add(tuple(value.shape))

    def on_before_backward(self,trainer,module,loss):
        if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite acceptance loss')

    def on_before_optimizer_step(self,trainer,module,optimizer):
        # Inspect available local gradient shards; DeepSpeed also checks its global norm.
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        if gradients and not torch.stack([torch.isfinite(g).all() for g in gradients]).all():raise FloatingPointError('Nonfinite acceptance gradient')
        self.gradient_steps+=1

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        self.batches+=1
        if trainer.global_step!=self.last_norm_step and trainer.global_step>self.initial_step:
            engine=trainer.strategy.model
            if hasattr(engine,'get_global_grad_norm'):
                norm=engine.get_global_grad_norm()
                if norm is None or not torch.isfinite(torch.as_tensor(norm)).all():raise FloatingPointError('Missing or nonfinite distributed gradient norm')
                self.global_norms.append(float(norm));self.last_norm_step=trainer.global_step
        if self.stop_this_run and trainer.global_step>=self.stop_after_updates:trainer.should_stop=True

    def on_train_end(self,trainer,module):
        self.hook.remove()
        if module.encoder is not None and not self.shapes:raise AssertionError('No latent forward was inspected')
        if not self.global_norms and not self.gradient_steps:raise AssertionError('No gradients were inspected')
        root=Path(trainer.default_root_dir)
        path=root/f'acceptance_rank{trainer.global_rank}_from{self.initial_step}.json'
        path.write_text(json.dumps({'initial_step':self.initial_step,'final_step':trainer.global_step,
            'batches':self.batches,'optimizer_callbacks':self.gradient_steps,'latent_shapes':sorted(self.shapes),
            'distributed_gradient_norms':self.global_norms,
            'shared_word_embeddings_preserved':self.tied_backbones,
            'finite_loss':True,'available_local_gradients_finite':True,'status':'passed'},indent=2))
