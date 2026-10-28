"""Convert fixed public paired-image sources; candidates are not a release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from collections import Counter,deque
import json
from pathlib import Path
import re
import sqlite3
import tarfile
import threading
import pyarrow.parquet as pq
from common import *


def rows_parquet(directory):
    for p in sorted(directory.glob('*.parquet')):
        offset=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                yield p,offset,row
                offset+=1


def embedded(row,prefix):
    result=[]
    fields=sorted((k for k in row if k.startswith(prefix)),key=lambda k:int(k.rsplit('_',1)[-1]))
    for k in fields:
        if row[k] and row[k].get('bytes'):
            result.append(asset(row[k]['bytes'],step=len(result)+1 if prefix=='reasoning_image_' else None,
                                role='auxiliary_view' if prefix=='reasoning_image_' else 'problem',name=k))
    return result


def run_source(source,iterator,convert):
    CANDIDATES.mkdir(parents=True,exist_ok=True)
    out=CANDIDATES/(source+'.jsonl'); done=CANDIDATES/(source+'.complete.json')
    if done.exists():print('SKIP',source,flush=True);return
    counts=Counter();temp=out.with_suffix('.building.jsonl')
    with temp.open('w') as f,(CANDIDATES/(source+'.rejected.jsonl')).open('w') as rej:
        numbered=enumerate(iterator)
        def process(pair):
            i,item=pair
            try:return i,convert(i,item),None
            except ValueError as e:return i,None,str(e)
        with ThreadPoolExecutor(8) as pool:
            while batch:=list(islice(numbered,64)):
                for i,record,error in pool.map(process,batch):
                    counts['input']+=1
                    if error:
                        counts[error]+=1;rej.write(json.dumps({'source':source,'ordinal':i,'reason':error})+'\n')
                    else:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');counts['candidates']+=1
                f.flush()
                print(source,dict(counts),flush=True)
    temp.replace(out);write_json(done,dict(counts));print('COMPLETE',source,dict(counts),flush=True)


def zebra(which):
    source='zebra_jigsaw' if which=='Visual Jigsaw' else 'zebra_scientific_'+which.lower().replace(' ','_')
    lock=json.loads((ROOT/'data/reasoning_sources.json').read_text())['sources']['zebra_cot']
    config=('2D Visual Reasoning - ' if which=='Visual Jigsaw' else 'Scientific Reasoning - ')+which
    db=sqlite3.connect(ROOT/'data/reasoning_browser/catalog.sqlite')
    excluded={str(uid).rsplit('/',1)[-1] for uid,m in db.execute('select uid,metadata from samples where dataset=?',('zebra_'+which.lower().replace(' ','_'),)) if json.loads(m).get('overlap_with')}
    db.close()
    def convert(i,item):
        p,offset,row=item
        if str(i) in excluded:raise ValueError('benchmark_overlap_candidate')
        existing={k for k,v in row.items() if '_image_' in k and v and v.get('bytes')}
        refs=set(re.findall(r'\[((?:problem|reasoning)_image_\d+)\]',str(row['Question'])+'\n'+str(row['Text Reasoning Trace'])))
        if refs-existing:raise ValueError('missing_image_reference')
        problem=embedded(row,'problem_image_');helpers=embedded(row,'reasoning_image_')
        group=source+':'+digest([a['pixel_sha256'] for a in problem])
        return candidate(source,lock['revision'],i,which,'空间想象与变换' if which=='Visual Jigsaw' else '抽象与科学推理',
                row['Question'] or '',row['Final Answer'] or '',problem,helpers,group,
                answer_type='choice' if re.fullmatch('[A-E]',str(row['Final Answer']).strip()) else 'symbolic',
                provenance={'file':str(p.relative_to(ROOT)),'row':offset,'raw_trace_retained_in_source':True})
    run_source(source,rows_parquet(ROOT/'data/raw/Zebra-CoT'/config),convert)


def think(name):
    source='thinkmorph_jigsaw' if name=='Jigsaw_Assembly' else 'thinkmorph_navigation'
    lock=json.loads((RAW/name/'source_lock.json').read_text())
    def convert(i,item):
        p,offset,row=item; problem=embedded(row,'problem_image_');helpers=embedded(row,'reasoning_image_')
        image_name=(row.get('problem_image_0') or {}).get('path','')
        family=re.sub(r'_(border|answer.*|cot)(\.[^.]+)?$','',image_name)
        group='image-family:'+family if name=='Jigsaw_Assembly' and family else 'map:'+digest([a['pixel_sha256'] for a in problem])
        return candidate(source,lock['revision'],row['pid'],name,'空间想象与变换' if name=='Jigsaw_Assembly' else '路径与行动规划',
                row['question'],row['answer'],problem,helpers,group,answer_type='choice' if name=='Jigsaw_Assembly' else 'grid_actions',
                provenance={'file':str(p.relative_to(ROOT)),'row':offset,'source_image_name':image_name})
    run_source(source,rows_parquet(RAW/name/'data'),convert)


def sat():
    lock=json.loads((RAW/'SAT_perspective/source_lock.json').read_text())
    for entry in lock['files']:
        if not (RAW/'SAT_perspective'/entry['path']).is_file(): raise RuntimeError('Incomplete SAT download')
    def convert(i,item):
        p,offset,row=item
        problem=[asset(im['bytes'],role='problem') for im in row['images']]
        helpers=[asset(im['bytes'],step=j,role='auxiliary_view') for j,im in enumerate(row['image_reason'],1)]
        options={chr(65+j):text for j,text in enumerate(row['choices'])}
        question=row['question']+'\n'+'\n'.join(f'({k}) {v}' for k,v in options.items())
        return candidate('sat_perspective',lock['revision'],i,'Perspective','空间想象与变换',question,
                chr(65+row['answer_idx']),problem,helpers,'sat-house:'+str(row['house_ind']),answer_type='choice',options=options,
                provenance={'file':str(p.relative_to(ROOT)),'row':offset,'house_ind':row['house_ind'],'cam_position':row['cam_position'],'cam_rotation':row['cam_rotation']})
    run_source('sat_perspective',rows_parquet(RAW/'SAT_perspective/data'),convert)


def check_map(grid,actions):
    start=next((i,j) for i,row in enumerate(grid) for j,x in enumerate(row) if x==1)
    goal=next((i,j) for i,row in enumerate(grid) for j,x in enumerate(row) if x==2)
    moves={'L':(0,-1),'R':(0,1),'U':(-1,0),'D':(1,0)};state=start
    for a in actions:
        di,dj=moves[a];i,j=state; nxt=(i+di,j+dj)
        if not(0<=nxt[0]<len(grid) and 0<=nxt[1]<len(grid[0])):continue
        if grid[nxt[0]][nxt[1]]==-1:raise ValueError('path_enters_hole')
        state=nxt
    if state!=goal:raise ValueError('path_misses_goal')
    q=deque([(start,0)]);seen={start}
    while q:
        state,n=q.popleft()
        if state==goal:
            if n!=len(actions):raise ValueError('nonshortest_path')
            return
        for di,dj in moves.values():
            nxt=state[0]+di,state[1]+dj
            if 0<=nxt[0]<len(grid) and 0<=nxt[1]<len(grid[0]) and grid[nxt[0]][nxt[1]]!=-1 and nxt not in seen:
                seen.add(nxt);q.append((nxt,n+1))


def mirage():
    lock=json.loads((RAW/'Mirage/download_manifest.json').read_text())
    with tarfile.open(RAW/'Mirage/vsp_spatial_planning_test.tar.gz') as test:
        testrows=[json.loads(x) for m in test.getmembers() if m.name.endswith('test_direct.jsonl') for x in test.extractfile(m)]
        excluded={digest(r['map_desc']) for r in testrows}
    with tarfile.open(RAW/'Mirage/vsp_spatial_planning.tar.gz') as t:
        members={m.name.lstrip('./'):m for m in t.getmembers() if m.isfile()}
        rows=[json.loads(x) for m in t.getmembers() if m.name.endswith('train_direct.jsonl') for x in t.extractfile(m)]
        tar_lock=threading.Lock()
        def get_image(ref,**kwargs):
            candidates=[m for k,m in members.items() if ref.lstrip('./').endswith(k) or k.endswith(ref.lstrip('./'))]
            if len(candidates)!=1:raise ValueError('ambiguous_image_reference')
            with tar_lock: payload=t.extractfile(candidates[0]).read()
            return asset(payload,**kwargs)
        def convert(i,row):
            key=digest(row['map_desc'])
            if key in excluded:raise ValueError('official_test_map_overlap')
            match=re.fullmatch(r'\\boxed\{(.+)\}',row['text_output']);assert match
            actions=[{'LEFT':'L','RIGHT':'R','UP':'U','DOWN':'D'}[a.strip()] for a in match[1].split(',')]
            check_map(row['map_desc'],actions)
            return candidate('mirage_vsp',lock['revision'],f"{len(row['map_desc'])}-{row['map_id']}",'VSP','路径与行动规划',
                    row['text_input'],'\\boxed{'+','.join(actions)+'}',[get_image(row['image_input'],role='problem')],
                    [get_image(row['image_output'],step=1,role='path')],'map:'+key,answer_type='grid_actions',
                    difficulty=len(row['map_desc']),provenance={'map_id':row['map_id'],'map_desc':row['map_desc'],'actions':actions,'execution_verified':True})
        run_source('mirage_vsp',rows,convert)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('sources',nargs='*');args=parser.parse_args()
    jobs={'zebra_jigsaw':lambda:zebra('Visual Jigsaw'),'zebra_geometry':lambda:zebra('Geometry'),
          'zebra_graph':lambda:zebra('Graph Algorithms'),'zebra_physics':lambda:zebra('Physics'),
          'thinkmorph_jigsaw':lambda:think('Jigsaw_Assembly'),'thinkmorph_navigation':lambda:think('Spatial_Navigation'),
          'sat':sat,'mirage':mirage}
    for k in args.sources or jobs:jobs[k]()
