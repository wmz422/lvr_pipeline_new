"""Bounded resumable per-puzzle subprocesses; this produces unapproved candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from common import ROOT,BUILD,write_json


def prepare_sources():
    root=BUILD/'sliding_sources';root.mkdir(exist_ok=True)
    manifest=root/'manifest.json'
    if manifest.exists():return [ROOT/r['path'] for r in json.loads(manifest.read_text())['images']]
    # COCO train2017 archive already belongs to the original training source pool.
    # New val will exclude source hashes found in old train, during release splitting.
    archive=ROOT/'data/raw/coco2017/train2017.zip'; records=[]
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.jpg'))
        names.sort(key=lambda n:hashlib.sha256(('9302026:'+n).encode()).hexdigest())
        for name in names[:3000]:
            p=root/Path(name).name
            if not p.exists():p.write_bytes(z.read(name))
            records.append({'path':str(p.relative_to(ROOT)),'source_member':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    write_json(manifest,{'source':'COCO train2017','archive':str(archive.relative_to(ROOT)),'images':records})
    return [ROOT/r['path'] for r in records]


def main():
    p=argparse.ArgumentParser();p.add_argument('--per-task',type=int,default=100);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--tasks',nargs='*',default=['form-board','paper-fold','hinge-folding','sliding-puzzle','rushhour']);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--levels',nargs='+',type=int,choices=range(1,6),default=[1,2,3,4,5])
    a=p.parse_args();sources=prepare_sources() if 'sliding-puzzle' in a.tasks else []
    output=BUILD/'mentis';output.mkdir(exist_ok=True)
    def one(spec):
        task,idx=spec;level=a.levels[idx%len(a.levels)];seed=930202600+idx+100000*['form-board','paper-fold','hinge-folding','sliding-puzzle','rushhour'].index(task)
        target=output/task/f'seed_{seed}'
        if (target/'record.json').exists():
            old=json.loads((target/'record.json').read_text())
            if old['generation_seed']!=seed or old['requested_level']!=level:raise ValueError('Cached generation parameters differ: '+str(target))
            return {'task':task,'seed':seed,'status':'cached'}
        target.mkdir(parents=True,exist_ok=True)
        cmd=[str(ROOT/'.venv-mentis/bin/python'),str(Path(__file__).with_name('mentis_one.py')),
             '--task',task,'--level',str(level),'--seed',str(seed),'--output',str(target)]
        if task=='sliding-puzzle':cmd+=['--source-image',str(sources[idx%len(sources)])]
        timeout=90 if task=='rushhour' else 45;start=time.monotonic()
        env={**os.environ,'MPLBACKEND':'Agg','MPLCONFIGDIR':str(BUILD/'mplconfig'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        with (target/'generation.log').open('w') as f:
            try:
                r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
                status='generated' if r.returncode==0 and (target/'record.json').exists() else 'rejected'
            except subprocess.TimeoutExpired:status='timeout'
        result={'task':task,'seed':seed,'requested_level':level,'status':status,'seconds':round(time.monotonic()-start,3),'output':str(target.relative_to(ROOT))}
        write_json(target/'status.json',result);return result
    specs=[(task,i) for i in range(a.offset,a.offset+a.per_task) for task in a.tasks]
    with (output/f'events_{a.offset}_{a.per_task}.jsonl').open('a') as log,ThreadPoolExecutor(a.workers) as pool:
        for future in as_completed([pool.submit(one,s) for s in specs]):
            row=future.result();line=json.dumps(row);log.write(line+'\n');log.flush();print(line,flush=True)


if __name__=='__main__':main()
