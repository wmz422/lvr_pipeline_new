#!/usr/bin/env python3
"""Export a baseline-compatible teacher source without generator/mapper/optimizer."""
import argparse
import inspect
from pathlib import Path
import sys
import torch
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path('/workspace/wmz/workspace/lvr_pipeline_new')
sys.path[:0]=[str(PACKAGE),str(ROOT/'src')]
from lvr.models.visual_bottleneck import VisualBottleneck,_read_checkpoint

def export_teacher(source,output):
    source,output=Path(source),Path(output)
    checkpoint=_read_checkpoint(source)
    if checkpoint.get('flow_metadata',{}).get('condition')!='mapper_latent_only':
        raise ValueError('Require a complete flow teacher checkpoint')
    parameters=inspect.signature(VisualBottleneck).parameters
    hp={k:v for k,v in checkpoint['hyper_parameters'].items() if k in parameters}
    state={k:v for k,v in checkpoint['state_dict'].items() if not k.startswith('flow_mapper.')}
    if len(state)==len(checkpoint['state_dict']):raise ValueError('No mapper found in source')
    provenance={'flow_source':str(source.resolve()),'flow_metadata':checkpoint['flow_metadata'],
                'source_global_step':checkpoint.get('global_step'),'optimizer_exported':False}
    if output.exists():
        old=_read_checkpoint(output)
        if old.get('source_provenance')!=provenance:raise ValueError('Refuse to overwrite another export')
        if set(old['state_dict'])!=set(state):raise ValueError('Existing export tensor keys differ')
        for k,v in state.items():
            if not torch.equal(old['state_dict'][k],v):raise ValueError('Existing export differs/corrupt')
        return output
    result={'state_dict':state,'hyper_parameters':hp,'global_step':checkpoint.get('global_step',0),
            'epoch':checkpoint.get('epoch',0),'pytorch-lightning_version':checkpoint.get('pytorch-lightning_version'),
            'source_provenance':provenance}
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.partial');torch.save(result,tmp);tmp.replace(output)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(export_teacher(a.source,a.output))

if __name__=='__main__':main()
