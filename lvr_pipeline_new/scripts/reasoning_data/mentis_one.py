"""Isolated, timed upstream generation with additional structural evidence."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/raw/reasoning_expansion/mentis-oculi/datasets'


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--level',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-image',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(SOURCE/a.task));random.seed(a.seed)
    start=time.monotonic();phases={}
    if a.task=='form-board':
        from generate import Puzzle
        from main import TARGET_SHAPES
        name,edges=list(TARGET_SHAPES.items())[a.seed%len(TARGET_SHAPES)]
        puzzle=Puzzle.from_edges(edges,grid_size=5); captured={}
        orig=puzzle._render_combined_view
        def capture(options,colors,*args,**kwargs):
            captured['target']=list(puzzle.target.exterior.coords)
            captured['options']=[{'solution':s,'vertices':list(poly.exterior.coords)} for s,poly in options]
            return orig(options,colors,*args,**kwargs)
        puzzle._render_combined_view=capture
        mask=puzzle.export(str(a.output),seed=a.seed,num_solution_pieces=a.level)
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        polys=[Polygon(x['vertices']) for x in captured['options'] if x['solution']]
        if unary_union(polys).symmetric_difference(puzzle.target).area>1e-7:raise ValueError('assembly_mismatch')
        if abs(sum(x.area for x in polys)-puzzle.target.area)>1e-7:raise ValueError('overlapping_solution_pieces')
        m={'shape_name':name,'question':'Which pieces in (2) are necessary to form the figure of (1)? Give the set of option letters.',
           'solution_pieces':[chr(65+i) for i,x in enumerate(mask) if x=='T'],'combined_image':'combined.png',
           'cot_images':[f'cot_{i:02d}.png' for i in range(a.level)],'geometry':captured,'execution_verified':True}
    elif a.task=='paper-fold':
        from main import generate_sequence
        captured={}
        # Trace only the return event of the pinned function; do not alter its algorithm.
        def trace(frame,event,arg):
            if event=='call':return trace if frame.f_code is generate_sequence.__code__ else None
            if event=='return':
                loc=frame.f_locals
                captured['axes']=loc['axes']
                captured['shapes']=[list(x.exterior.coords) for x in loc['shapes']]
                for k in ('holes','correct_holes'):
                    if k in loc:captured[k]=[[x.x,x.y] for x in loc[k]]
            return trace
        sys.settrace(trace)
        try:m=generate_sequence(n=2,min_steps=a.level,max_steps=a.level,seed=a.seed,save_dir=a.output,puzzle_id=a.seed)
        finally:sys.settrace(None)
        m['geometry']=captured
        metrics=m['distinguishability_metrics']
        if metrics['min_pairwise_distance']+1e-9<metrics['min_required_distance']:raise ValueError('indistinguishable_options')
    elif a.task=='hinge-folding':
        from generator import generate_hinge_folding_dataset
        generate_hinge_folding_dataset(str(a.output),num_samples=1,min_pieces=a.level+1,max_pieces=a.level+1,seed=a.seed)
        path=a.output/'puzzle_0001/metadata.json';m=json.loads(path.read_text());m['metadata_subdirectory']='puzzle_0001'
    elif a.task=='sliding-puzzle':
        from generate_puzzles import generate_puzzle,bfs_solve,apply_move
        m=generate_puzzle(a.source_image,a.output,puzzle_id=1,dimensions=2,num_scramble_moves=a.level,seed=a.seed)
        path=bfs_solve(m['initial_state'],m['target_state'],max_depth=a.level)
        if path is None or len(path)!=a.level or len(m['solution_moves'])!=a.level:raise ValueError('wrong_shortest_distance')
        state=copy.deepcopy(m['initial_state'])
        for action in m['solution_moves']:state=apply_move(state,action)
        if state!=m['target_state']:raise ValueError('invalid_solution')
        m['source_image_sha256']=hashlib.sha256(a.source_image.read_bytes()).hexdigest()
        m['source_image_path']=str(a.source_image.relative_to(ROOT));m['metadata_subdirectory']='puzzle_0001';m['execution_verified']=True
    else:
        from generator import generate_initial_state,compute_greedy_solution
        from main import save_puzzle,validate_puzzle_obviousness,execute_abstract_sequence
        stage=1 if a.level==1 else 2+(a.seed%2) if a.level==2 else 4+(a.seed%2)
        instance=generate_initial_state(seed=a.seed,difficulty_stage=stage)
        phases['generation_seconds']=time.monotonic()-start
        actions=compute_greedy_solution(instance,max_actions=5)
        phases['solve_seconds']=time.monotonic()-start-phases['generation_seconds']
        if not 1<=len(actions)<=5:raise ValueError('no_solution_within_5')
        valid,reason=validate_puzzle_obviousness(instance,actions)
        if not valid:raise ValueError('obviousness:'+reason)
        if not execute_abstract_sequence(instance,actions):raise ValueError('invalid_rushhour_solution')
        save_puzzle(instance,actions,len(actions),1,a.seed,a.output)
        sub=f'level_{len(actions):02d}/puzzle_0001';m=json.loads((a.output/sub/'metadata.json').read_text());m['metadata_subdirectory']=sub;m['execution_verified']=True
    m.update(generation_seed=a.seed,requested_level=a.level,elapsed_seconds=time.monotonic()-start,phase_timing=phases)
    tmp=a.output/'record.tmp';tmp.write_text(json.dumps(m,indent=2));tmp.replace(a.output/'record.json')
    print('SUCCESS',a.task,a.level,a.seed,flush=True)


if __name__=='__main__':main()
