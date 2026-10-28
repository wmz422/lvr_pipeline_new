"""Add prepared references and RGB GT to the unchanged QA/teacher collator."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from lvr.data.crop_qa import CropQADataModule
from .gt import digest_file,load_protocol
from .sampling import DistributedLayoutSampler,layout_key
from torch.utils.data import DataLoader

def read_tensor(path):
    with Image.open(path) as im:
        if im.mode!='RGB' or im.width%16 or im.height%16:raise ValueError('Unprepared flow image')
        value=np.array(im,dtype=np.uint8,copy=True)
    return torch.from_numpy(value).permute(2,0,1).float().div(127.5).sub(1)

class FlowCollator:
    def __init__(self,base_collator,root,index,cache_keys=True):
        self.base_collator,self.root,self.index=base_collator,Path(root),index
        self.cache_keys=cache_keys

    def __call__(self,examples):
        batch=self.base_collator(examples)
        targets,references=[],[]
        for example in examples:
            row=self.index[example['id']]
            targets.append(read_tensor(self.root/row['target']['path']))
            references.append([read_tensor(self.root/r['path']) for r in row['references']])
        # No question, answer, helper captions, bbox coordinates, or task prompt
        # enter this generator branch. The target is only its FM noisy input/label.
        batch['flow']={'targets':targets,'references':references,'ids':list(batch['ids']),
            'target_keys':[self.index[e['id']]['target']['pixel_sha256'] for e in examples],
            'reference_keys':[[r['pixel_sha256'] for r in self.index[e['id']]['references']] for e in examples]}
        if not self.cache_keys:
            batch['flow'].pop('target_keys');batch['flow'].pop('reference_keys')
        return batch

class FlowDataModule(CropQADataModule):
    def __init__(self,processor_name_or_path,image_root,train_path='data/sft/train',
                 val_path='data/sft/test',train_boxes_path='data/align/train',val_boxes_path='data/align/test',
                 batch_size=16,num_workers=8,num_latents=8,original_pixels=1048576,crop_pixels=262144,
                 max_length=8192,train_limit=None,val_limit=None,seed=42,include_encoder=True,qa_crops=False,
                 encoder_uses_question=False,encoder_type='qwen',encoder_includes_original=True,
                 include_descriptions=False,encoder_image_mode='question_crops',description_image_mode='none',
                 auxiliary_manifest_path=None,caption_path=None,additional_train_path=None,additional_val_path=None,
                 val_num_workers=None,prefetch_factor=2,pin_memory=True,persistent_workers=True,
                 include_reconstruction=False,flow_data_root: str='',flow_protocol_sha256: str='',
                 flow_layout_buckets: bool=True,flow_stress_first: bool=False):
        arguments=dict(locals());arguments.pop('self');arguments.pop('__class__',None)
        arguments.pop('flow_data_root');arguments.pop('flow_protocol_sha256')
        arguments.pop('flow_layout_buckets');arguments.pop('flow_stress_first')
        if include_reconstruction or qa_crops or not include_encoder:raise ValueError('Incompatible flow data branch')
        super().__init__(**arguments)
        self.hparams.clear();self.hparams.update(arguments)
        self.hparams['flow_data_root']=flow_data_root
        self.hparams['flow_protocol_sha256']=flow_protocol_sha256
        self.hparams['flow_layout_buckets']=flow_layout_buckets;self.hparams['flow_stress_first']=flow_stress_first
        self._hparams_initial=copy.deepcopy(self.hparams)
        self._flow_indices={}

    def setup(self,stage=None):
        super().setup(stage)
        root=Path(self.hparams.flow_data_root)
        status=json.loads((root/'status.json').read_text())
        protocol=load_protocol(root/'protocol.json')
        if status['status']!='complete':raise ValueError('Full GT processing is not complete')
        if protocol['protocol_sha256']!=self.hparams.flow_protocol_sha256:
            raise ValueError('GT protocol differs from training recipe')
        for split,dataset in [('train',self.train_dataset),('val',self.val_dataset)]:
            if dataset is None or split in self._flow_indices:continue
            manifest=root/(split+'.jsonl')
            if digest_file(manifest)!=status['splits'][split]['manifest_sha256']:raise ValueError('Changed GT manifest')
            index={}
            with manifest.open() as f:
                for line in f:
                    row=json.loads(line)
                    if row['id'] in index:raise ValueError('Duplicate prepared sample ID')
                    index[row['id']]={'id':row['id'],'target':row['target'],'references':row['references']}
            expected=[item['id'] for item in dataset.items]
            if len(set(expected))!=len(expected) or any(i not in index for i in expected):
                raise ValueError('QA samples and flow GT do not align')
            if self.hparams.train_limit is None and self.hparams.val_limit is None and len(expected)!=len(index):
                raise ValueError('Unexpected GT cardinality')
            self._flow_indices[split]=index

    def _loader(self,dataset,training):
        loader=super()._loader(dataset,training)
        split='train' if training else 'val'
        collator=FlowCollator(loader.collate_fn,self.hparams.flow_data_root,self._flow_indices[split],cache_keys=not self.hparams.flow_stress_first)
        if training and self.hparams.flow_layout_buckets:
            world=self.trainer.world_size if self.trainer is not None else 1
            rank=self.trainer.global_rank if self.trainer is not None else 0
            layouts=[layout_key(self._flow_indices[split][item['id']]) for item in dataset.items]
            sampler=DistributedLayoutSampler(dataset,layouts,loader.batch_size,num_replicas=world,rank=rank,
                seed=self.hparams.seed,stress_first=self.hparams.flow_stress_first,
                stress_accumulation=128//(world*loader.batch_size))
            kwargs={}
            if loader.num_workers:
                kwargs.update(prefetch_factor=loader.prefetch_factor,persistent_workers=loader.persistent_workers)
            return DataLoader(dataset,batch_size=loader.batch_size,sampler=sampler,num_workers=loader.num_workers,
                pin_memory=loader.pin_memory,collate_fn=collator,drop_last=loader.drop_last,**kwargs)
        loader.collate_fn=collator
        return loader
