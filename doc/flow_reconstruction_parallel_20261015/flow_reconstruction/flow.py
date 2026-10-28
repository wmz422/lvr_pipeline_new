"""Standard conditional flow matching through a frozen Klein Base generator."""
from __future__ import annotations
from collections import OrderedDict,defaultdict
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

MODEL_REVISION='a3b4f4849157f664bdbc776fd7453c2783562f4d'

class LatentMapper(nn.Module):
    def __init__(self,latent_dim=512,hidden_size=1024,condition_dim=7680):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(latent_dim),nn.Linear(latent_dim,hidden_size),
                               nn.GELU(),nn.Linear(hidden_size,condition_dim))
        nn.init.normal_(self.net[-1].weight,std=.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,latent):
        return self.net(latent.to(dtype=next(self.parameters()).dtype))#这里把latent映射到生成器条件的latent维度

def pipeline_class():
    from diffusers import Flux2KleinPipeline
    return Flux2KleinPipeline

def velocity_prediction(transformer,noisy,refs,condition,timestep,*,checkpointing=False):
    """One native forward for a homogeneous image-layout batch, without padding."""
    p=pipeline_class();batch=noisy.shape[0]
    if condition.shape[0]!=batch or any(x.shape[0]!=batch for x in refs):
        raise ValueError('Target/reference/condition batch cardinality differs')
    target_tokens=p._pack_latents(noisy)
    target_ids=p._prepare_latent_ids(noisy).to(noisy.device)
    ref_tokens=[p._pack_latents(x) for x in refs]
    if refs:
        tokens=torch.cat([target_tokens,*ref_tokens],dim=1)
        # The upstream helper accepts reference batches of one; layouts are shared.
        ref_ids=p._prepare_image_ids([x[:1] for x in refs]).to(noisy.device).expand(batch,-1,-1)
        ids=torch.cat([target_ids,ref_ids],dim=1)
    else:tokens,ids=target_tokens,target_ids
    txt_ids=p._prepare_text_ids(condition).to(condition.device)
    dtype=next(transformer.parameters()).dtype
    tokens=tokens.to(dtype=dtype);condition=condition.to(dtype=dtype)
    def predict(x,c):
        return transformer(hidden_states=x,timestep=timestep.to(dtype=dtype),guidance=None,
                           encoder_hidden_states=c,txt_ids=txt_ids,img_ids=ids,return_dict=False)[0]
    if checkpointing and not getattr(transformer,'gradient_checkpointing',False) and torch.is_grad_enabled() and condition.requires_grad:
        predicted=checkpoint(predict,tokens,condition,use_reentrant=False)
    else:predicted=predict(tokens,condition)
    return predicted[:,:target_tokens.shape[1]],target_tokens

def conditional_flow_loss(transformer,clean,refs,condition,sigma,timestep,noise,#clean和refs分别是过VAE之后的gt图和参考图，noise和clean形状相同为随机高斯噪声，sigma用于构造当前timestep下的噪声和真实的融合目标噪声
                          *,checkpointing=False,reduction='mean'):
    if clean.shape!=noise.shape:raise ValueError('Noise shape differs from target')
    batch=clean.shape[0]
    sigma=torch.as_tensor(sigma,device=clean.device,dtype=torch.float32)
    if sigma.numel() not in (1,batch) or not torch.isfinite(sigma).all() or (sigma<0).any() or (sigma>1).any():
        raise ValueError('Invalid per-example noise fractions')
    sigma=sigma.reshape(-1,1,1,1)
    noisy=(1-sigma)*clean+sigma*noise#从这个点出发
    pred,_=velocity_prediction(transformer,noisy,refs,condition,timestep,#noisy是目前的位置
                               checkpointing=checkpointing)
    target=pipeline_class()._pack_latents(noise-clean)#要学习的目标速度
    # Reference predictions are excluded. Each sample keeps its original MSE.
    values=(pred.float()-target.float()).square().flatten(1).mean(1)#速度差值作为loss
    if reduction=='none':return values
    if reduction!='mean':raise ValueError('Unsupported reduction')
    return values.mean()

def verify_model_lock(model_path):
    p=Path(model_path);lock=json.loads((p/'weights_lock.json').read_text())
    if lock.get('status')!='complete' or lock.get('revision')!=MODEL_REVISION:
        raise ValueError('Model download is incomplete or revision differs')
    if lock.get('text_encoder_downloaded') is not False:raise ValueError('Unexpected text encoder policy')
    for name,spec in lock['files'].items():
        file=p/name
        if not file.is_file() or file.stat().st_size!=spec['bytes']:
            raise ValueError(f'Missing/wrong-size model component: {name}')
    return lock

class FrozenFlowRuntime:
    """External fixed components are intentionally outside teacher state_dict/ZeRO.

    Only LatentMapper is a registered, optimized teacher submodule. Generator and
    VAE move explicitly to the latent's device and never acquire parameter grads.
    """
    def __init__(self,model_path,*,checkpointing=True,device=None,generator_batch_size=16,vae_batch_size=4,latent_cache_bytes=134217728):
        from diffusers import AutoencoderKLFlux2,Flux2Transformer2DModel,FlowMatchEulerDiscreteScheduler
        verify_model_lock(model_path)
        device=torch.device('cpu' if device is None else device)
        placement={'':device}
        self.transformer=Flux2Transformer2DModel.from_pretrained(
            model_path,subfolder='transformer',local_files_only=True,torch_dtype=torch.bfloat16,device_map=placement,low_cpu_mem_usage=True)
        self.vae=AutoencoderKLFlux2.from_pretrained(
            model_path,subfolder='vae',local_files_only=True,torch_dtype=torch.float32,device_map=placement,low_cpu_mem_usage=True)
        self.scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,subfolder='scheduler',local_files_only=True)
        self.transformer.requires_grad_(False).eval();self.vae.requires_grad_(False).eval()
        if checkpointing:self.transformer.enable_gradient_checkpointing()
        self.checkpointing=checkpointing
        self._device=device
        self.condition_dim=self.transformer.config.joint_attention_dim
        if min(generator_batch_size,vae_batch_size)<1 or latent_cache_bytes<0:raise ValueError('Invalid parallel batch/cache configuration')
        self.generator_batch_size=generator_batch_size;self.vae_batch_size=vae_batch_size
        self.latent_cache_bytes=latent_cache_bytes;self._encoded_cache=OrderedDict();self._cache_bytes=0
        self.last_stats={'generator_batch_sizes':[],'vae_batch_sizes':[],'vae_cache_hits':0}


    def move(self,device):
        device=torch.device(device)
        if self._device!=device:
            self.transformer.to(device);self.vae.to(device);self._device=device
            self._encoded_cache.clear();self._cache_bytes=0

    @torch.no_grad()
    def encode_many(self,images,keys=None):
        """Encode compatible RGB images in native FP32 VAE batches; deduplicate fixed pixels."""
        if keys is None:keys=[None]*len(images)
        if len(keys)!=len(images):raise ValueError('VAE image/key count differs')
        outputs=[None]*len(images);groups=defaultdict(list);pending={};aliases={}
        for i,(image,key) in enumerate(zip(images,keys)):
            if image.ndim!=3 or image.shape[0]!=3 or image.shape[-1]%16 or image.shape[-2]%16:
                raise ValueError('Expected aligned RGB CHW')
            tag=(str(key),tuple(image.shape)) if key is not None else None
            if tag is not None and tag in self._encoded_cache:
                outputs[i]=self._encoded_cache[tag];self._encoded_cache.move_to_end(tag)
                self.last_stats['vae_cache_hits']+=1
            elif tag is not None and tag in pending:
                aliases[i]=pending[tag];self.last_stats['vae_cache_hits']+=1
            else:
                if tag is not None:pending[tag]=i
                groups[tuple(image.shape)].append((i,tag))
        for members in groups.values():
            for offset in range(0,len(members),self.vae_batch_size):
                chunk=members[offset:offset+self.vae_batch_size]
                rgb=torch.stack([images[i] for i,_ in chunk]).to(device=self._device,dtype=torch.float32)
                if not torch.isfinite(rgb).all() or rgb.min()<-1.001 or rgb.max()>1.001:
                    raise ValueError('VAE pixels must be finite and in [-1,1]')
                with torch.autocast(device_type=self._device.type,enabled=False):
                    x=self.vae.encode(rgb).latent_dist.mode()
                x=pipeline_class()._patchify_latents(x).float()
                mean=self.vae.bn.running_mean.view(1,-1,1,1)
                std=torch.sqrt(self.vae.bn.running_var.view(1,-1,1,1)+self.vae.config.batch_norm_eps)
                x=((x-mean)/std).detach()
                self.last_stats['vae_batch_sizes'].append(len(chunk))
                for local,(index,tag) in enumerate(chunk):
                    value=x[local:local+1].clone();outputs[index]=value
                    size=value.numel()*value.element_size()
                    if tag is not None and size<=self.latent_cache_bytes:
                        while self._encoded_cache and self._cache_bytes+size>self.latent_cache_bytes:
                            _,previous=self._encoded_cache.popitem(last=False)
                            self._cache_bytes-=previous.numel()*previous.element_size()
                        self._encoded_cache[tag]=value;self._cache_bytes+=size
        for index,source in aliases.items():outputs[index]=outputs[source]
        return outputs

    @torch.no_grad()
    def encode(self,image):
        return self.encode_many([image])[0]

    def loss(self,targets,references,condition,ids,*,validation=False,seed=42,
             target_keys=None,reference_keys=None):
        if not targets or len(targets)!=len(references) or len(targets)!=len(condition) or len(ids)!=len(targets):
            raise ValueError('Flow batch cardinality mismatch')
        self.move(condition.device)
        self.last_stats={'generator_batch_sizes':[],'vae_batch_sizes':[],'vae_cache_hits':0}
        if target_keys is None:target_keys=[None]*len(targets)
        if reference_keys is None:reference_keys=[[None]*len(refs) for refs in references]
        if len(target_keys)!=len(targets) or len(reference_keys)!=len(targets):raise ValueError('Flow cache-key count differs')
        images=[];keys=[];positions=[]
        for target,refs,key,rkeys in zip(targets,references,target_keys,reference_keys):
            if len(refs)!=len(rkeys):raise ValueError('Reference cache-key count differs')
            start=len(images);images.extend([target,*refs]);keys.extend([key,*rkeys]);positions.append((start,len(refs)))
        encoded=self.encode_many(images,keys)
        examples=[];buckets=defaultdict(list)
        # Draw randomness in the original sample order, independently for every row.
        for index,((start,count),sample_id) in enumerate(zip(positions,ids)):
            clean=encoded[start];refs=encoded[start+1:start+1+count]
            if validation:
                example_seed=seed+int(hashlib.sha256(sample_id.encode()).hexdigest()[:8],16)
                rng=torch.Generator(device=condition.device).manual_seed(example_seed)
            else:rng=None
            n=self.scheduler.config.num_train_timesteps
            step=torch.randint(n,(1,),device=condition.device,generator=rng)
            sigma=self.scheduler.sigmas.to(condition.device)[step].float()
            timestep=self.scheduler.timesteps.to(condition.device)[step].float()/1000
            noise=torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=rng)
            examples.append((clean,refs,sigma,timestep,noise))
            signature=(tuple(clean.shape[1:]),tuple(tuple(r.shape[1:]) for r in refs))
            buckets[signature].append(index)
        losses=[None]*len(targets)
        for members in buckets.values():
            for offset in range(0,len(members),self.generator_batch_size):
                indices=members[offset:offset+self.generator_batch_size]
                rows=[examples[i] for i in indices]
                clean=torch.cat([r[0] for r in rows])
                refs=[torch.cat([r[1][j] for r in rows]) for j in range(len(rows[0][1]))]
                sigma=torch.cat([r[2] for r in rows]);timestep=torch.cat([r[3] for r in rows]);noise=torch.cat([r[4] for r in rows])
                c=condition[indices]
                values=conditional_flow_loss(self.transformer,clean,refs,c,sigma,timestep,noise,
                    checkpointing=self.checkpointing,reduction='none')
                for local,index in enumerate(indices):losses[index]=values[local]
                self.last_stats['generator_batch_sizes'].append(len(indices))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate(self,condition,references,height,width,*,steps=50,seed=42):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        if height%16 or width%16 or min(height,width,steps)<1:raise ValueError('Invalid sampling dimensions/steps')
        if condition.shape[0]!=1:raise ValueError('Generate one variable-size canvas at a time')
        self.move(condition.device);p=pipeline_class()
        rng=torch.Generator(device=condition.device).manual_seed(seed)
        shape=(1,self.transformer.config.in_channels,height//16,width//16)
        x=torch.randn(shape,device=condition.device,dtype=torch.float32,generator=rng)
        refs=[self.encode(r) for r in references]
        schedule=FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        import numpy as np
        sigmas=np.linspace(1.,1/steps,steps)
        schedule.set_timesteps(steps,device=condition.device,sigmas=sigmas,
                              mu=compute_empirical_mu(shape[2]*shape[3],steps))
        for timestep in schedule.timesteps:
            pred,_=velocity_prediction(self.transformer,x,refs,condition,
                                        timestep.reshape(1).float()/1000)
            packed=p._pack_latents(x)
            packed=schedule.step(pred.float(),timestep,packed,return_dict=False)[0]
            x=packed.transpose(1,2).reshape(shape)
        mean=self.vae.bn.running_mean.view(1,-1,1,1)
        std=torch.sqrt(self.vae.bn.running_var.view(1,-1,1,1)+self.vae.config.batch_norm_eps)
        with torch.autocast(device_type=self._device.type,enabled=False):
            decoded=self.vae.decode(p._unpatchify_latents(x*std+mean).float(),return_dict=False)[0]
        return ((decoded.float()+1)/2).clamp(0,1)
