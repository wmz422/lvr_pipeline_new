"""Recover visible grid tiles from known renderings; replay plans and exclude test maps."""
from collections import Counter,defaultdict
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT,RAW,BUILD,CANDIDATES,digest,write_json
from convert import check_map


def tiles(image,n):
    w,h=image.size
    if w!=h or w%n:raise ValueError('nonuniform_grid_image')
    step=w//n
    return [hashlib.sha256(image.crop((j*step,i*step,(j+1)*step,(i+1)*step)).tobytes()).hexdigest() for i in range(n) for j in range(n)]


def main():
    vocabulary=defaultdict(set)
    with tarfile.open(RAW/'Mirage/vsp_spatial_planning.tar.gz') as tar:
        members={m.name.lstrip('./'):m for m in tar.getmembers() if m.isfile()}
        rows=[json.loads(l) for m in tar.getmembers() if m.name.endswith('train_direct.jsonl') for l in tar.extractfile(m)]
        for row in rows:
            names=[k for k in members if row['image_input'].lstrip('./').endswith(k) or k.endswith(row['image_input'].lstrip('./'))]
            assert len(names)==1
            im=Image.open(io.BytesIO(tar.extractfile(members[names[0]]).read())).convert('RGB')
            n=len(row['map_desc'])
            for h,v in zip(tiles(im,n),sum(row['map_desc'],[])):vocabulary[(im.width//n,h)].add(v)
    with tarfile.open(RAW/'Mirage/vsp_spatial_planning_test.tar.gz') as tar:
        excluded={digest(json.loads(l)['map_desc']) for m in tar.getmembers() if m.name.endswith('test_direct.jsonl') for l in tar.extractfile(m)}
    counts=Counter();validated=[];rejected=[]
    for line in (CANDIDATES/'thinkmorph_navigation.jsonl').read_text().splitlines():
        row=json.loads(line);counts['input']+=1
        im=Image.open(ROOT/row['problem_images'][0]['path']).convert('RGB')
        candidates=[]
        for n in range(3,9):
            if im.width!=im.height or im.width%n:continue
            labels=[vocabulary.get((im.width//n,h),set()) for h in tiles(im,n)]
            if all(len(s)==1 for s in labels):
                flat=[next(iter(s)) for s in labels]
                if flat.count(1)==flat.count(2)==1:candidates.append([flat[i*n:(i+1)*n] for i in range(n)])
        try:
            if len(candidates)!=1:raise ValueError('grid_not_unambiguously_recovered')
            grid=candidates[0];key=digest(grid)
            if key in excluded:raise ValueError('official_vsp_test_map_overlap')
            actions=[s.strip() for s in row['answer'].replace('\\boxed{','').replace('}','').split(',')]
            check_map(grid,actions)
            row['group_id']='map:'+key;row['source_image_id']=row['group_id']
            row['provenance'].update(map_desc=grid,actions=actions,execution_verified=True,map_reconstruction='exact_visible_tile_hash')
            row['quality']['map_execution']='passed';validated.append(row);counts['validated']+=1
        except (ValueError,KeyError) as e:
            rejected.append({'id':row['id'],'reason':str(e)});counts[str(e)]+=1
    # Preserve original candidates; release must use the quality-gated overlay.
    out=BUILD/'quality';out.mkdir(exist_ok=True)
    (out/'thinkmorph_navigation.validated.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in validated))
    (out/'thinkmorph_navigation.rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
    write_json(out/'map_validation.json',{'counts':dict(counts),'tile_vocabulary':len(vocabulary),'test_maps':len(excluded)})
    print(dict(counts),flush=True)

if __name__=='__main__':main()
