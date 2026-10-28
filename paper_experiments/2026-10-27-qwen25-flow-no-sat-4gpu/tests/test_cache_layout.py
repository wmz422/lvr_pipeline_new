import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_latent_cache import fixture,ProbeEncoder
from reconstruction_experiment.latent_cache import build_plan,bind_plan,encode_shard,finalize,LatentStore,shard_metadata
from script.prepare_latents import validate_equivalent_layout
from tuning.adopt_stream import linked_plan,link_shards,finalize_parallel

class CacheLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.gt=self.root/'gt';fixture(self.gt);self.original=self.root/'original';self.stream=self.root/'stream';self.stream.mkdir();self.out=self.root/'linked';self.out.mkdir()
        with patch('reconstruction_experiment.latent_cache.vae_identity',return_value={'unit_encoder':True}):
            self.plan=build_plan(self.gt,self.root/'model',batch_size=2,shard_bytes=3000,device_type='cpu')
        bind_plan(self.original,self.plan);self.source={};self.chunks={}
        for i,shard in enumerate(self.plan['shards']):
            encode_shard(self.original,self.plan,i,ProbeEncoder());meta=shard_metadata(self.original,self.plan,i)
            name=f'rank0_{len(self.plan["shards"])-i:06d}.f32';os.link(self.original/shard['file'],self.stream/name)
            self.chunks[name]={k:meta[k] for k in ('bytes','sha256')}
            for key,item in meta['tensors'].items():self.source[key]={**item,'file':name}
        finalize(self.original)
        self.linked=linked_plan(self.plan,self.source,self.chunks)
    def test_reordered_shards_are_reused_without_payload_copy(self):
        self.assertNotEqual(self.plan['plan_sha256'],self.linked['plan_sha256'])
        validate_equivalent_layout(self.plan,self.linked);bind_plan(self.out,self.linked)
        link_shards(self.stream,self.out,self.linked,self.source,self.chunks);finalize(self.out)
        a,b=LatentStore(self.original),LatentStore(self.out)
        for key in self.source:
            self.assertTrue(torch.equal(a.read(key),b.read(key)))
            self.assertTrue(os.path.samefile(self.out/b.tensors[key]['file'],self.stream/self.source[key]['file']))
        # Resume verifies and reuses committed shards, including their file identity.
        link_shards(self.stream,self.out,self.linked,self.source,self.chunks)
    def test_parallel_finalization_matches_canonical_and_rejects_corruption(self):
        bind_plan(self.out,self.linked)
        link_shards(self.stream,self.out,self.linked,self.source,self.chunks)
        finalize(self.out)
        expected={name:(self.out/name).read_bytes() for name in ('index.json','status.json')}
        for name in expected:(self.out/name).unlink()
        finalize_parallel(self.out,workers=4)
        for name,content in expected.items():self.assertEqual((self.out/name).read_bytes(),content)
        payload=self.out/self.linked['shards'][0]['file']
        raw=bytearray(payload.read_bytes());raw[0]^=1;payload.write_bytes(raw)
        (self.out/'status.json').unlink()
        with self.assertRaises(ValueError):finalize_parallel(self.out,workers=4)
        self.assertFalse((self.out/'status.json').exists())
    def test_changed_semantics_missing_tensors_and_bad_offsets_rejected(self):
        mutations=[]
        value=copy.deepcopy(self.linked);value['encoder']={'wrong_weights':True};mutations.append(value)
        value=copy.deepcopy(self.linked);value['manifest_sha256']['train']='changed';mutations.append(value)
        value=copy.deepcopy(self.linked);value['shards'].pop();mutations.append(value)
        value=copy.deepcopy(self.linked);value['shards'][0]['items'][0]['offset']=4;mutations.append(value)
        value=copy.deepcopy(self.linked);value['shards'][0]['items'][0]['png_sha256']='changed';mutations.append(value)
        value=copy.deepcopy(self.linked);value['encoding']['batch_size']=8;mutations.append(value)
        for value in mutations:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_equivalent_layout(self.plan,value)
if __name__=='__main__':unittest.main()
