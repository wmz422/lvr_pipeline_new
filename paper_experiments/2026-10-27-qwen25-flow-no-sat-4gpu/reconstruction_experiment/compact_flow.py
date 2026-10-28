"""Keep prepared Flow RGB in uint8 through DataLoader and host/device transfer.

The original encode_many grouping, FP32 VAE mode/BN normalization, LRU cache,
noise schedule, loss and generator backward are retained. Only RGB decoding
moves to the VAE device; mixed legacy float and compact inputs are supported.
"""
from collections import defaultdict
import torch
from flow_reconstruction.flow import FrozenFlowRuntime, pipeline_class


class CompactFrozenFlowRuntime(FrozenFlowRuntime):
    @torch.no_grad()
    def encode_many(self,images,keys=None):
        """Same frozen FP32 VAE computation; normalize uint8 only on cache misses."""
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
                groups[(tuple(image.shape), image.dtype)].append((i,tag))
        for members in groups.values():
            for offset in range(0,len(members),self.vae_batch_size):
                chunk=members[offset:offset+self.vae_batch_size]
                compact = images[chunk[0][0]].dtype == torch.uint8
                rgb=torch.stack([images[i] for i,_ in chunk]).to(device=self._device,dtype=torch.float32)
                if compact:
                    # This is an owned FP32 stack, never an input/cache tensor.
                    # Normalize in place to avoid two extra full-size GPU RGB allocations.
                    rgb.div_(127.5).sub_(1)
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

