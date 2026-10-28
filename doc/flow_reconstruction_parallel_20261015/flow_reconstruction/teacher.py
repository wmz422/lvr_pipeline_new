"""Teacher extension: existing supervised losses plus standard flow matching."""
import inspect
import copy
import json
import math
from pathlib import Path
import torch
from lvr.models.visual_bottleneck import VisualBottleneck,_read_checkpoint,load_checkpoint_weights
from .flow import FrozenFlowRuntime,LatentMapper,MODEL_REVISION

class FlowVisualBottleneck(VisualBottleneck):
    def __init__(self,encoder_model_name_or_path: str,qa_model_name_or_path: str,
                 num_latents: int=8,latent_dim: int=512,gaussian: bool=False,
                 learning_rate: float=1e-5,head_learning_rate: float=1e-4,
                 kl_weight: float=0.,kl_warmup_ratio: float=.1,lr_warmup_ratio: float=.03,
                 weight_decay: float=.01,gradient_checkpointing: bool=True,seed: int=42,
                 train_qa: bool=False,encoder_uses_question: bool=False,encoder_type: str='qwen',
                 encoder_includes_original: bool=True,qformer_hidden_size: int=768,
                 qformer_num_layers: int=6,qformer_num_heads: int=12,
                 qformer_cross_attention_frequency: int=2,qformer_use_geometry: bool=False,
                 description_weight: float=0.,qa_loss_weight: float=1.,
                 description_image_mode: str='none',lr_schedule_steps: int|None=None,
                 reconstruction_weight: float=0.,reconstruction_hidden_size: int=512,
                 reconstruction_num_layers: int=2,reconstruction_num_heads: int=8,
                 reconstruction_normalize: bool=False,flow_model_path: str='',
                 flow_weight: float=0.,flow_mapper_hidden_size: int=1024,
                 flow_gradient_checkpointing: bool=True,flow_protocol_sha256: str='',
                  flow_generator_batch_size: int=16,flow_vae_batch_size: int=4,flow_latent_cache_bytes: int=134217728):
        arguments=dict(locals());arguments.pop('self');arguments.pop('__class__',None)
        flow_args={key:arguments.pop(key) for key in list(arguments) if key.startswith('flow_')}
        if gaussian or reconstruction_weight or num_latents!=8 or latent_dim!=512:
            raise ValueError('Flow experiment uses deterministic eight-token latent and disables old reconstruction')
        if not math.isfinite(flow_weight) or flow_weight<=0:raise ValueError('Specify a positive flow loss weight')
        if not flow_protocol_sha256:raise ValueError('Require a bound GT protocol')
        config=json.loads((Path(flow_model_path)/'transformer/config.json').read_text())
        if config['joint_attention_dim']!=7680:raise ValueError('Expected Klein Base 4B condition dimension')
        super().__init__(**arguments)
        # Replace introspection-derived locals with stable serializable constructor inputs.
        self.hparams.clear();self.hparams.update({**arguments,**flow_args})
        self._hparams_initial=copy.deepcopy(self.hparams)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.flow_mapper=LatentMapper(latent_dim,flow_mapper_hidden_size,config['joint_attention_dim'])
        # A plain external runtime keeps the frozen ~8 GB out of optimizer/teacher checkpoints.
        self.__dict__['_flow_runtime']=None

    def flow_runtime(self,device=None):
        if self.__dict__['_flow_runtime'] is None:
            self.__dict__['_flow_runtime']=FrozenFlowRuntime(self.hparams.flow_model_path,
                                    checkpointing=self.hparams.flow_gradient_checkpointing,device=self.device if device is None else device,
                                     generator_batch_size=self.hparams.flow_generator_batch_size,vae_batch_size=self.hparams.flow_vae_batch_size,
                                     latent_cache_bytes=self.hparams.flow_latent_cache_bytes)
        return self.__dict__['_flow_runtime']

    def forward(self,batch,sample=None):
        outputs=super().forward(batch,sample=sample)
        if 'flow' not in batch:raise ValueError('Missing flow GT/reference branch')
        condition=self.flow_mapper(outputs['mu'])
        outputs['flow_mse']=self.flow_runtime(condition.device).loss(batch['flow']['targets'],batch['flow']['references'],
               condition,batch['ids'],validation=not self.training,seed=self.hparams.seed,
               target_keys=batch['flow'].get('target_keys'),reference_keys=batch['flow'].get('reference_keys'))
        return outputs

    def training_step(self,batch,batch_idx):
        out=self(batch)
        loss=self.hparams.qa_loss_weight*out['ce']+self.hparams.description_weight*out['description_ce']
        loss=loss+self.hparams.flow_weight*out['flow_mse']
        sizes=self.__dict__['_flow_runtime'].last_stats['generator_batch_sizes']
        self.log_dict({'train/flow_generator_batch_mean':sum(sizes)/len(sizes),
                       'train/flow_generator_calls':len(sizes),'train/loss':loss,'train/ce':out['ce'],
                       'train/description_ce':out['description_ce'],'train/flow_mse':out['flow_mse'],
                       'train/flow_weighted':self.hparams.flow_weight*out['flow_mse'],
                       'train/mu_rms':out['mu'].square().mean().sqrt()},
                      on_step=True,on_epoch=True,sync_dist=True,batch_size=len(batch['ids']))
        return loss

    def validation_step(self,batch,batch_idx):
        # Base implementation keeps the original CE/description mismatch diagnostics.
        out=self(batch,sample=False)
        metrics={'val/ce':out['ce'],'val/ce_mu':out['ce'],'val/flow_mse':out['flow_mse']}
        if self.hparams.description_weight:
            metrics['val/description_ce']=out['description_ce']
            if len(batch['ids'])>1:
                wrong=self.answer_loss(batch['description'],out['mu'].roll(1,dims=0))
                metrics['val/description_mismatched_ce']=wrong
                metrics['val/description_mismatch_gap']=wrong-out['description_ce']
        self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True,batch_size=len(batch['ids']))

    def configure_optimizers(self):
        groups=[]
        for backbone in (self.encoder,self.qa):
            params=[p for p in backbone.parameters() if p.requires_grad] if backbone is not None else []
            if params:
                new_encoder=backbone is self.encoder and self.hparams.encoder_type=='qformer'
                groups.append({'params':params,'lr':self.hparams.head_learning_rate if new_encoder else self.hparams.learning_rate})
        heads=[self.queries,*self.mu_head.parameters(),*self.projector.parameters(),*self.flow_mapper.parameters()]
        groups.append({'params':heads,'lr':self.hparams.head_learning_rate})
        optimizer=torch.optim.AdamW(groups,weight_decay=self.hparams.weight_decay,foreach=True)
        total=self.hparams.lr_schedule_steps or self.trainer.estimated_stepping_batches
        warmup=max(1,int(total*self.hparams.lr_warmup_ratio))
        def schedule(step):
            if step<warmup:return (step+1)/warmup
            progress=min(1.,(step-warmup)/max(1,total-warmup))
            return .1+.9*(1+math.cos(math.pi*progress))/2
        return {'optimizer':optimizer,'lr_scheduler':{
            'scheduler':torch.optim.lr_scheduler.LambdaLR(optimizer,schedule),'interval':'step'}}

    def initialize_from_checkpoint(self,path):
        ckpt=_read_checkpoint(path);hp=ckpt['hyper_parameters']
        if hp.get('flow_protocol_sha256')!=self.hparams.flow_protocol_sha256:
            raise ValueError('Teacher phase initialization must use this experiment GT protocol')
        for key in ('num_latents','latent_dim','gaussian','encoder_type','encoder_uses_question',
                    'encoder_includes_original','flow_mapper_hidden_size'):
            if hp[key]!=self.hparams[key]:raise ValueError(f'Phase initialization differs: {key}')
        if ckpt.get('flow_metadata',{}).get('model_revision')!=MODEL_REVISION:
            raise ValueError('Frozen generator revision differs')
        load_checkpoint_weights(self,ckpt['state_dict'],assign=False)
        self.initial_checkpoint={'path':str(path),'source_global_step':ckpt.get('global_step'),
                                 'optimizer_restored':False,'scheduler_restored':False}
        return self.initial_checkpoint

    def on_save_checkpoint(self,checkpoint):
        checkpoint['flow_metadata']={'model_revision':MODEL_REVISION,
            'gt_protocol_sha256':self.hparams.flow_protocol_sha256,
            'objective':'unweighted_standard_flow_matching','condition':'mapper_latent_only',
            'frozen_weights_external':True,'reference_inputs':'question_images_only',
            'parallel_implementation':'native_homogeneous_layout_v1','reduction':'sample_mean_then_batch_mean'}

    def on_load_checkpoint(self,checkpoint):
        info=checkpoint.get('flow_metadata',{})
        if info.get('model_revision')!=MODEL_REVISION or info.get('gt_protocol_sha256')!=self.hparams.flow_protocol_sha256:
            raise ValueError('Resume model/GT protocol mismatch')
