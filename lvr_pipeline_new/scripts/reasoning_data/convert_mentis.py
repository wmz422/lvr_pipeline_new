"""Normalize generated Mentis samples, retaining complete ordered state sequences."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse,fcntl
import json
from pathlib import Path
from common import *


def canonical_polygon(points):
    # Remove dependence on choice order, ring start, orientation and display color.
    from shapely.geometry import Polygon
    return Polygon(points).normalize().wkt


def convert(path):
    m=json.loads(path.read_text());task=path.parent.parent.name;seed=m['generation_seed'];base=path.parent
    def resolve(ref):
        a=base/ref
        if a.is_file():return a
        a=base/m.get('metadata_subdirectory','')/ref
        if a.is_file():return a
        raise ValueError('missing_image_reference')
    if task=='form-board':
        inputs=[m['combined_image']];question='Which labeled pieces on the right are necessary to form the target outline on the left? Give the set of option letters.';answer=', '.join(m['solution_pieces']);atype='set'
        geo=m['geometry'];structure=[canonical_polygon(geo['target']),sorted((x['solution'],canonical_polygon(x['vertices'])) for x in geo['options'])]
        group='form-board:'+digest(structure)
    elif task=='paper-fold':
        inputs=[m['combined_image']];question=m['question']+' Choose one of A, B, C, D, E.';answer=m['correct_choice'];atype='choice'
        structure=m['geometry'];group='paper-fold:'+digest(structure)
    elif task=='hinge-folding':
        inputs=[m['initial_image'],m['target_image']]
        question='The first image shows an unfolded chain with labeled hinges. The second image shows the target configuration. Rotate the part of the chain to the right of each hinge counterclockwise to form the target. Give the rotations in hinge order as letter and angle in degrees, for example: A 90, B 180.'
        answer=m['rotation_sequence'];atype='hinge_rotations';structure=[m['shape_chain'],m['rotation_angles']];group='hinge-folding:'+digest(structure)
    elif task=='sliding-puzzle':
        inputs=[m['initial_image'],m['target_image']]
        question='The first image is the current sliding puzzle and the second is the target. Move the blank square up, down, left, or right by swapping it with an adjacent tile. Give a shortest sequence of blank-square moves, separated by commas.'
        answer=', '.join(m['solution_moves']);atype='sliding_actions';structure=[m['source_image_sha256'],m['initial_state'],m['target_state']];group='source-image:'+m['source_image_sha256']
    else:
        inputs=[m['initial_state_image']]
        question='Move the red car R to the green exit. Each labeled car can move forward along its arrow or backward along its rail, without rotating or colliding. Each move goes as far as possible until blocked; R can stop at the exit. Give a shortest sequence of label and direction pairs, separated by commas, for example: A forward, R backward.'
        labels={o['id']:('R' if o['id']=='red_car' else o.get('label')) for o in m['objects']}
        if any(not labels[a['object_id']] for a in m['actions']):raise ValueError('unlabeled_action_object')
        answer=', '.join(labels[a['object_id']]+(' forward' if a['direction']==1 else ' backward') for a in m['actions']);atype='rushhour_actions'
        structure=[m['board'],[{k:v for k,v in o.items() if k not in ('color','label')} for o in m['objects']]];group='rushhour:'+digest(structure)
    problem=[asset(resolve(ref),role='problem') for ref in inputs]
    helpers=[asset(resolve(ref),step=i,role='final_state' if i==len(m['cot_images']) else 'process_state') for i,ref in enumerate(m['cot_images'],1)]
    # Target-only first-level folding/sliding samples are QA candidates, not LVR pairs.
    if task in ('hinge-folding','sliding-puzzle') and len(helpers)==1:raise ValueError('only_target_helper')
    record=candidate('mentis_'+task,REV,seed,task,'空间想象与变换',question,answer,problem,helpers,group,
        answer_type=atype,difficulty=len(helpers),provenance={'metadata':str(path.relative_to(ROOT)),'generation_seed':seed,
        'parameters':{'requested_level':m['requested_level']},'structure_sha256':digest(structure),
        'generation_seconds':m['elapsed_seconds'],'execution_verified':m.get('execution_verified',False)})
    if task=='sliding-puzzle':record['source_image_id']=m['source_image_sha256']
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='*');parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    CANDIDATES.mkdir(exist_ok=True,parents=True)
    for taskdir in sorted((BUILD/'mentis').iterdir()):
        if not taskdir.is_dir():continue
        if args.tasks and taskdir.name not in args.tasks:continue
        name='mentis_'+taskdir.name;counts=Counter();seen=set();out=CANDIDATES/(name+'.jsonl');tmp=out.with_suffix('.tmp')
        def one(path):
            try:return path,convert(path),None
            except ValueError as e:return path,None,str(e)
        with tmp.open('w') as f,(CANDIDATES/(name+'.rejected.jsonl')).open('w') as rej,ThreadPoolExecutor(args.workers) as pool:
            for path,r,error in pool.map(one,sorted(taskdir.glob('*/record.json'))):
                counts['generated']+=1
                try:
                    if error:raise ValueError(error)
                    key=r['provenance']['structure_sha256']
                    if key in seen:raise ValueError('duplicate_structure')
                    seen.add(key);f.write(json.dumps(r,ensure_ascii=False)+'\n');counts['candidates']+=1
                except ValueError as e:
                    counts[str(e)]+=1;rej.write(json.dumps({'metadata':str(path.relative_to(ROOT)),'reason':str(e)})+'\n')
        tmp.replace(out);write_json(CANDIDATES/(name+'.stats.json'),dict(counts));print(name,dict(counts),flush=True)

if __name__=='__main__':
    BUILD.mkdir(parents=True,exist_ok=True)
    with (BUILD/'mentis_conversion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
