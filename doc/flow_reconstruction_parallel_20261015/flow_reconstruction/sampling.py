"""Deterministic layout grouping without dropping samples or adding bucket-wise repeats."""
from collections import defaultdict
import torch
from torch.utils.data import DistributedSampler

def layout_key(row):
    return (row['target']['width'],row['target']['height'],tuple((r['width'],r['height']) for r in row['references']))

class DistributedLayoutSampler(DistributedSampler):
    def __init__(self,dataset,layouts,batch_size,*,num_replicas=1,rank=0,seed=42,stress_first=False,stress_accumulation=1):
        super().__init__(dataset,num_replicas=num_replicas,rank=rank,shuffle=True,seed=seed,drop_last=False)
        if len(layouts)!=len(dataset) or batch_size<1:raise ValueError('Invalid layout sampler inputs')
        self.layouts=layouts;self.batch_size=batch_size;self.stress_first=stress_first;self.stress_accumulation=stress_accumulation
        self.buckets=defaultdict(list)
        for index,key in enumerate(layouts):self.buckets[key].append(index)

    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        keys=sorted(self.buckets)
        keys=[keys[i] for i in torch.randperm(len(keys),generator=g).tolist()]
        chunks=[];tails=[];size=self.batch_size
        for key in keys:
            members=self.buckets[key]
            members=[members[i] for i in torch.randperm(len(members),generator=g).tolist()]
            full=len(members)//size*size
            chunks.extend([members[i:i+size] for i in range(0,full,size)])
            tails.extend(members[full:])
        full=len(tails)//size*size
        chunks.extend([tails[i:i+size] for i in range(0,full,size)])
        chunks=[chunks[i] for i in torch.randperm(len(chunks),generator=g).tolist()]
        order=[index for chunk in chunks for index in chunk]+tails[full:]
        padding=self.total_size-len(order)
        if padding:order+=(order*((padding+len(order)-1)//len(order)))[:padding]
        full_steps=self.num_samples//size
        local=[]
        for step in range(full_steps):
            start=step*self.num_replicas*size+self.rank*size;local.extend(order[start:start+size])
        remaining=self.num_samples%size
        start=full_steps*self.num_replicas*size+self.rank*remaining
        local.extend(order[start:start+remaining])
        if self.stress_first and local:
            # Acceptance only: two worst-case logical updates, then ordinary data.
            worst=max(self.buckets,key=lambda k:k[0]*k[1]+sum(w*h for w,h in k[2]))
            source=self.buckets[worst][0]
            items=getattr(self.dataset,'items',None)
            longest=max(range(len(items)),key=lambda i:items[i].get('processed_token_length') or
                (len(items[i].get('auxiliary_images',[]))+len(items[i].get('boxes',[])))) if items else source
            count=size*self.stress_accumulation
            for update,index in enumerate([source,longest]):
                begin=update*count;end=min(begin+count,len(local))
                if begin<end:local[begin:end]=[index]*(end-begin)
        if len(local)!=self.num_samples:raise RuntimeError('Distributed sampler cardinality mismatch')
        return iter(local)
