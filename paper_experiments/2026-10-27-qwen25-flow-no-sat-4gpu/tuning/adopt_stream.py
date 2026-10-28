"""Verify streamed FP32 cache and pack it into the standard resumable training format."""
import argparse, copy, fcntl, hashlib, json, mmap, os, sys, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; WS=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(WS/'doc/flow_reconstruction_parallel_20261015'),str(WS/'lvr_pipeline_new/src')]
from reconstruction_experiment.latent_cache import build_plan,bind_plan,read_plan,fingerprint,shard_metadata
from reconstruction_experiment.prepare import write_json
from flow_reconstruction.gt import digest_file
from script.prepare_latents import validate_equivalent_layout

def checked_chunk(path,expected):
    meta=json.loads(path.read_text());payload=path.parent/meta['file']
    if meta['binding_sha256']!=expected or payload.stat().st_size!=meta['bytes'] or digest_file(payload)!=meta['sha256']:raise ValueError('Corrupt stream payload')
    return path,meta

def inventory(stream,plan,verify_workers=4):
    binding={'encoder':plan['encoder'],'gt_protocol_sha256':plan['gt_protocol_sha256'],'batch_size':plan['encoding']['batch_size'],'device_type':'cuda','workers':4}
    expected=fingerprint(binding); source={}; chunks={}; totals={rank:0 for rank in range(4)}
    for rank in range(4):
        if json.loads((stream/f'worker{rank}_binding.json').read_text())!=binding: raise ValueError('Stream encoder/GT binding mismatch')
        done=json.loads((stream/f'worker{rank}_complete.json').read_text())
        if done['status']!='complete' or done['binding_sha256']!=expected or done['rows']!=plan['counts']: raise ValueError('Incomplete stream worker')
    with ThreadPoolExecutor(max_workers=verify_workers) as pool:
        checked=pool.map(lambda path:checked_chunk(path,expected),sorted(stream.glob('rank*_*.json')))
        for path,meta in checked:
            chunks[meta['file']]={k:meta[k] for k in ('sha256','bytes')}
            rank=int(path.stem.split('_')[0][4:]); totals[rank]+=len(meta['tensors'])
            for key,item in meta['tensors'].items():
                if key in source or item['offset']<0 or item['offset']+item['bytes']>meta['bytes']: raise ValueError('Invalid stream tensor inventory')
                source[key]={**item,'file':meta['file']}
    required={item['key']:item for shard in plan['shards'] for item in shard['items']}
    if set(source)!=set(required): raise ValueError(f'Stream inventory mismatch: missing={len(set(required)-set(source))}, extra={len(set(source)-set(required))}')
    for key,item in required.items():
        if any(source[key][k]!=item[k] for k in ('shape','bytes')): raise ValueError('Stream tensor shape/size mismatch')
    for rank in range(4):
        if totals[rank]!=json.loads((stream/f'worker{rank}_complete.json').read_text())['images']: raise ValueError('Worker completion count mismatch')
    return source,chunks

def pack_worker(rank,workers,stream,out,plan,source):
    handles=OrderedDict()
    try:
        for index in range(rank,len(plan['shards']),workers):
            shard=plan['shards'][index]; path=out/shard['file'];path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if shard_metadata(out,plan,index) is not None: continue
                started=time.monotonic(); tensors={}; temp=path.with_suffix('.partial');sha=hashlib.sha256()
                with temp.open('wb') as target:
                    for item in shard['items']:
                        original=source[item['key']];file=original['file']
                        if file not in handles:
                            if len(handles)>=16:
                                _,(fd,view)=handles.popitem(last=False);view.close();fd.close()
                            fd=(stream/file).open('rb');handles[file]=(fd,mmap.mmap(fd.fileno(),0,access=mmap.ACCESS_READ))
                        handles.move_to_end(file); view=handles[file][1]
                        raw=view[original['offset']:original['offset']+original['bytes']]
                        if hashlib.sha256(raw).hexdigest()!=original['sha256']: raise ValueError('Stream tensor checksum mismatch')
                        if target.tell()!=item['offset']: raise ValueError('Packed offset mismatch')
                        target.write(raw);sha.update(raw)
                        tensors[item['key']]={k:item[k] for k in ('shape','offset','bytes')}
                        tensors[item['key']]['sha256']=original['sha256']
                    target.flush();os.fsync(target.fileno())
                if temp.stat().st_size!=shard['bytes']: raise ValueError('Packed shard size mismatch')
                temp.replace(path)
                write_json(path.with_suffix('.json'),{'plan_sha256':plan['plan_sha256'],'bytes':shard['bytes'],'sha256':sha.hexdigest(),'tensors':tensors,'seconds':time.monotonic()-started})
                print('PACK_STREAM',rank,index+1,len(plan['shards']),flush=True)
    finally:
        for fd,view in handles.values(): view.close();fd.close()

def linked_plan(canonical,source,chunks):
    required={item['key']:item for shard in canonical['shards'] for item in shard['items']}
    groups={name:[] for name in chunks}
    for key,item in source.items():groups[item['file']].append(key)
    plan=copy.deepcopy(canonical);plan['shards']=[]
    for index,name in enumerate(sorted(chunks)):
        items=[{**required[key],'offset':source[key]['offset']} for key in sorted(groups[name],key=lambda key:source[key]['offset'])]
        plan['shards'].append({'file':f'shards/{index:06d}.f32','items':items,'bytes':chunks[name]['bytes']})
    plan.pop('plan_sha256');plan['plan_sha256']=fingerprint(plan)
    validate_equivalent_layout(canonical,plan)
    return plan

def link_shards(stream,out,plan,source,chunks):
    if stream.stat().st_dev!=out.stat().st_dev:raise ValueError('Hardlink cache requires the same filesystem; use --copy-shards')
    for index,name in enumerate(sorted(chunks)):
        shard=plan['shards'][index];path=out/shard['file'];path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if shard_metadata(out,plan,index) is not None:continue
            original=stream/name
            if path.exists():
                if not os.path.samefile(path,original):raise ValueError('Refuse to replace another cache payload')
            else:os.link(original,path)
            tensors={item['key']:{**{k:item[k] for k in ('shape','offset','bytes')},'sha256':source[item['key']]['sha256']} for item in shard['items']}
            write_json(path.with_suffix('.json'),{'plan_sha256':plan['plan_sha256'],'bytes':shard['bytes'],'sha256':chunks[name]['sha256'],'tensors':tensors,'seconds':0})
        if (index+1)%100==0:print('LINK_STREAM',index+1,len(plan['shards']),flush=True)

def finalize_parallel(root,workers=4):
    """Apply the canonical finalizer's full checks with concurrent payload hashes.

    Keep this in the adopter: the encoder-bound cache module remains unchanged.
    map preserves shard order, producing the same index/status as finalize().
    """
    root=Path(root);plan=read_plan(root);tensors={};shards={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked=pool.map(lambda index:shard_metadata(root,plan,index),range(len(plan['shards'])))
        for index,(spec,meta) in enumerate(zip(plan['shards'],checked)):
            if meta is None:raise ValueError(f'Missing VAE cache shard {index}')
            shards[spec['file']]={key:meta[key] for key in ('bytes','sha256')}
            for key,item in meta['tensors'].items():
                if key in tensors:raise ValueError('Duplicate cache tensor')
                tensors[key]={**item,'file':spec['file']}
    if len(tensors)!=plan['unique_images']:raise ValueError('Incomplete unique-image cache')
    write_json(root/'index.json',{'plan_sha256':plan['plan_sha256'],'tensors':tensors,'shards':shards})
    write_json(root/'status.json',{'status':'complete','plan_sha256':plan['plan_sha256'],'index_sha256':digest_file(root/'index.json'),'images':len(tensors),'bytes':plan['bytes']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--copy-shards',action='store_true');args=p.parse_args()
    stream=ROOT/'data/vae_stream_fp32';out=ROOT/'data/vae_latents_fp32';out.mkdir(parents=True,exist_ok=True)
    with (out/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=build_plan(ROOT/'data/reconstruction_2mp',WS/'doc/flow_reconstruction_parallel_20261015/weights/FLUX.2-klein-base-4B')
        print('PLAN',plan['unique_images'],plan['bytes']/2**30,len(plan['shards']),flush=True)
        source,chunks=inventory(stream,plan,args.workers)
        if not args.copy_shards:plan=linked_plan(plan,source,chunks)
        bind_plan(out,plan)
        write_json(out/'stream_provenance.json',{'source':str(stream),'chunks':chunks,'adopter_sha256':digest_file(__file__),'stream_encoder_sha256':digest_file(ROOT/'tuning/stream_encode.py'),'plan_sha256':plan['plan_sha256'],'storage':'copy' if args.copy_shards else 'hardlink'})
        if args.copy_shards:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(pack_worker,rank,args.workers,stream,out,plan,source) for rank in range(args.workers)]
                for future in futures: future.result()
        else:link_shards(stream,out,plan,source,chunks)
        finalize_parallel(out,args.workers)
        print('ADOPT_STREAM_COMPLETE',flush=True)
if __name__=='__main__':main()
