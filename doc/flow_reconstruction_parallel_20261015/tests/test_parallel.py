"""Check true native batch execution and equality to independent-image FM gradients."""
from collections import Counter,OrderedDict
import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from diffusers import Flux2Transformer2DModel,FlowMatchEulerDiscreteScheduler
from flow_reconstruction.flow import LatentMapper,FrozenFlowRuntime,conditional_flow_loss
from flow_reconstruction.sampling import DistributedLayoutSampler

def tiny():
    return Flux2Transformer2DModel(in_channels=8,out_channels=8,num_layers=1,num_single_layers=1,
        attention_head_dim=16,num_attention_heads=2,joint_attention_dim=16,timestep_guidance_channels=16,
        mlp_ratio=1,axes_dims_rope=(4,4,4,4),guidance_embeds=False).requires_grad_(False).eval()

class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm2d(8);self.config=SimpleNamespace(batch_norm_eps=1e-5);self.calls=[]
    def encode(self,rgb):
        assert rgb.dtype==torch.float32
        self.calls.append(rgb.shape[0])
        value=torch.nn.functional.avg_pool2d(rgb[:,:2],8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))

def runtime(model,batch,vae_batch):
    result=FrozenFlowRuntime.__new__(FrozenFlowRuntime)
    result.transformer=model;result.vae=FakeVAE().requires_grad_(False).eval();result._device=torch.device('cpu')
    result.scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    result.checkpointing=False;result.generator_batch_size=batch;result.vae_batch_size=vae_batch
    result.latent_cache_bytes=1024*1024;result._encoded_cache=OrderedDict();result._cache_bytes=0
    result.last_stats={'generator_batch_sizes':[],'vae_batch_sizes':[],'vae_cache_hits':0}
    return result

class NativeParallelTests(unittest.TestCase):
    def test_batch16_is_one_native_forward_and_matches_serial_loss_and_gradients(self):
        torch.manual_seed(3);model=tiny();mapper=LatentMapper(4,16,16)
        mu=torch.randn(16,8,4,requires_grad=True);clean=torch.randn(16,8,2,3);noise=torch.randn_like(clean)
        refs=[torch.randn(16,8,2,2),torch.randn(16,8,1,2)]
        sigma=torch.linspace(.05,.95,16);timestep=sigma.clone();saved=copy.deepcopy(model.state_dict())
        outputs=[];gradients=[];weight_gradients=[];counts=[]
        for native in (False,True):
            mapper.zero_grad(set_to_none=True);mu.grad=None;calls=[]
            hook=model.register_forward_pre_hook(lambda module,args,kwargs:calls.append(kwargs['hidden_states'].shape[0]),with_kwargs=True)
            condition=mapper(mu)
            if native:
                model.enable_gradient_checkpointing()
                loss=conditional_flow_loss(model,clean,refs,condition,sigma,timestep,noise,checkpointing=True)
            else:
                loss=torch.stack([conditional_flow_loss(model,clean[i:i+1],[r[i:i+1] for r in refs],condition[i:i+1],sigma[i],timestep[i:i+1],noise[i:i+1]) for i in range(16)]).mean()
            loss.backward();hook.remove();outputs.append(loss.detach());gradients.append(mu.grad.clone())
            weight_gradients.append(mapper.net[-1].weight.grad.clone());counts.append(calls)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=2e-5,atol=1e-6)
        torch.testing.assert_close(gradients[0],gradients[1],rtol=5e-5,atol=2e-7)
        torch.testing.assert_close(weight_gradients[0],weight_gradients[1],rtol=5e-5,atol=2e-7)
        self.assertEqual(counts[0],[1]*16);self.assertEqual(counts[1],[16])
        self.assertGreater(gradients[1].abs().sum().item(),0);self.assertGreater(weight_gradients[1].abs().sum().item(),0)
        self.assertTrue(all(param.grad is None for param in model.parameters()))
        self.assertTrue(all(torch.equal(saved[k],v) for k,v in model.state_dict().items()))

    def test_mixed_layout_groups_keep_sample_weighting_and_independent_noise(self):
        torch.manual_seed(9);base=tiny();mapper=LatentMapper(4,16,16)
        mu=torch.randn(16,8,4,requires_grad=True)
        shared=torch.rand(3,16,32)*2-1;second=torch.rand(3,32,16)*2-1
        targets=[torch.rand(3,32,48 if i<12 else 32)*2-1 for i in range(16)]
        references=[[shared] if i<12 else [shared,second] for i in range(16)]
        ids=[str(i) for i in range(16)];keys=[f'target_{i}' for i in range(16)]
        refkeys=[['shared'] if i<12 else ['shared','second'] for i in range(16)]
        results=[];grads=[];runtimes=[]
        for batch,vae_batch in [(1,1),(16,4)]:
            r=runtime(copy.deepcopy(base),batch,vae_batch);mapper.zero_grad(set_to_none=True);mu.grad=None
            loss=r.loss(targets,references,mapper(mu),ids,validation=True,target_keys=keys,reference_keys=refkeys)
            loss.backward();results.append(loss.detach());grads.append(mu.grad.clone());runtimes.append(r)
        torch.testing.assert_close(results[0],results[1],rtol=2e-5,atol=1e-6)
        torch.testing.assert_close(grads[0],grads[1],rtol=5e-5,atol=2e-7)
        self.assertEqual(runtimes[1].last_stats['generator_batch_sizes'],[12,4])
        self.assertIn(4,runtimes[1].last_stats['vae_batch_sizes'])
        self.assertGreater(runtimes[1].last_stats['vae_cache_hits'],0)
        count=len(runtimes[1].vae.calls)
        with torch.no_grad():
            cached=runtimes[1].loss(targets,references,mapper(mu),ids,validation=True,target_keys=keys,reference_keys=refkeys)
        torch.testing.assert_close(cached,results[1]);self.assertEqual(len(runtimes[1].vae.calls),count)

    def test_distributed_sampler_coverage_padding_determinism_and_epoch_shuffle(self):
        for n,world,batch in [(61,8,16),(317,8,4),(347,8,16),(3,8,16)]:
            dataset=list(range(n));layouts=[(16*(1+i%3),32,((16,16),)) for i in dataset]
            ranks=[DistributedLayoutSampler(dataset,layouts,batch,num_replicas=world,rank=r) for r in range(world)]
            parts=[list(s) for s in ranks];combined=[i for part in parts for i in part]
            self.assertEqual(set(combined),set(dataset));self.assertEqual(len(combined),((n+world-1)//world)*world)
            self.assertTrue(all(len(part)==len(parts[0]) for part in parts))
            self.assertEqual(parts[0],list(ranks[0]))
            if n>100:
                ranks[0].set_epoch(1);self.assertNotEqual(parts[0],list(ranks[0]))

    def test_reference_predictions_remain_excluded_for_batch16(self):
        clean=torch.randn(16,8,2,3);noise=torch.randn_like(clean)
        expected=(noise-clean).flatten(2).transpose(1,2)
        class TargetOnly(nn.Module):
            def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(()),requires_grad=False)
            def forward(self,hidden_states,**kwargs):
                out=torch.full_like(hidden_states,999);out[:,:expected.shape[1]]=expected;return (out,)
        loss=conditional_flow_loss(TargetOnly(),clean,[torch.randn(16,8,1,2)],torch.randn(16,8,16),torch.linspace(0,1,16),torch.ones(16),noise)
        self.assertEqual(loss.item(),0.)

if __name__=='__main__':unittest.main()
