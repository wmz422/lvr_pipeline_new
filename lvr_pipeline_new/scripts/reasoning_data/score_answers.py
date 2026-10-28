"""Task-aware answers, with executable alternative plans (run in .venv-mentis)."""
import argparse,copy,importlib.util,json,math,re,sys
from collections import deque
from common import ROOT,RAW


def final_text(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    if '<think>' in text:return ''
    match=re.search(r'<answer>(.*?)</answer>',text,re.S)
    if match:text=match[1]
    return text.strip().strip('`$ .')


def moves(text):
    text=re.sub(r'\\boxed\{([^{}]*)\}',r'\1',text)
    parts=re.split(r'[,;\s]+',text.strip('[]{}() .'))
    mapping={'left':'L','right':'R','up':'U','down':'D','l':'L','r':'R','u':'U','d':'D'}
    if not parts or any(p.lower() not in mapping for p in parts):return []
    return [mapping[p.lower()] for p in parts]


def score(row,prediction):
    text=final_text(prediction);kind=row['answer_type'];answer=final_text(row['answer'])
    if not text:return False,'empty_or_incomplete_answer'
    if kind=='choice':
        match=re.fullmatch(r'(?:[Aa]nswer\s*:?\s*)?\(?([A-Z])\)?[.]?',text)
        return bool(match and match[1]==answer),'choice_letter'
    if kind=='yes_no':return text.lower().strip('.')==answer.lower(),'boolean'
    if kind=='set':
        cleaned=re.sub(r'[{},;\s\[\]()]','',text)
        return bool(re.fullmatch('[A-Z]+',cleaned) and set(cleaned)==set(re.findall('[A-Z]',answer))),'unordered_option_set'
    if kind in ('hinge_rotations','sliding_actions','rushhour_actions'):
        metadata=json.loads((ROOT/row['provenance']['metadata']).read_text())
        if kind=='hinge_rotations':
            from mentis_geometry import canonical_hinge
            pairs=re.findall(r'([A-Z])\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(?:°|degrees)?',text)
            remaining=re.sub(r'([A-Z])\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(?:°|degrees)?','',text)
            angles={k:float(v)%360 for k,v in pairs};names=[chr(65+i) for i in range(len(metadata['rotation_angles']))]
            if len(pairs)!=len(names) or set(angles)!=set(names) or remaining.strip(' ,;[]()'):return False,'invalid_hinge_format'
            return canonical_hinge(metadata,[angles[k] for k in names])==canonical_hinge(metadata),'executed_final_piece_geometry'
        if kind=='sliding_actions':
            actions=moves(text);state=copy.deepcopy(metadata['initial_state']);goal=metadata['target_state'];n=len(state)
            if not actions or len(actions)>len(metadata['solution_moves']):return False,'invalid_or_nonshortest_sliding_plan'
            # The pinned generator denotes its blank by -1.
            for action in actions:
                positions=[(i,j) for i,line in enumerate(state) for j,value in enumerate(line) if value==-1]
                if len(positions)!=1:raise ValueError('Unknown sliding blank representation')
                i,j=positions[0];di,dj={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[action];x,y=i+di,j+dj
                if not(0<=x<n and 0<=y<n):return False,'illegal_sliding_move'
                state[i][j],state[x][y]=state[x][y],state[i][j]
            return state==goal,'executed_sliding_goal_and_shortest_length'
        directory=RAW/'mentis-oculi/datasets/rushhour';sys.path.insert(0,str(directory))
        # The scorer runs in a dedicated process to avoid upstream module-name collisions.
        from main import execute_abstract_sequence
        labels={('R' if o['id']=='red_car' else o.get('label')):o['id'] for o in metadata['objects'] if o.get('movable',True)}
        pairs=re.findall(r'\b([A-Z])\s+(forward|backward)\b',text,re.I)
        remaining=re.sub(r'\b([A-Z])\s+(forward|backward)\b','',text,flags=re.I)
        if not pairs or len(pairs)>len(metadata['actions']) or remaining.strip(' ,;[]().') or any(k.upper() not in labels for k,d in pairs):return False,'invalid_rushhour_plan'
        actions=[{'object_id':labels[k.upper()],'direction':1 if d.lower()=='forward' else -1} for k,d in pairs]
        return execute_abstract_sequence(metadata,actions),'executed_collision_bounded_plan_and_shortest_length'
    if row['source'] in ('mirage_vsp','thinkmorph_navigation'):
        actions=moves(text);grid=row['provenance']['map_desc'];h=len(grid);w=len(grid[0])
        start=next((i,j) for i,line in enumerate(grid) for j,v in enumerate(line) if v==1)
        target=next((i,j) for i,line in enumerate(grid) for j,v in enumerate(line) if v==2)
        state=start;delta={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
        for action in actions:
            di,dj=delta[action];i,j=state[0]+di,state[1]+dj
            if not(0<=i<h and 0<=j<w) or grid[i][j]==-1:return False,'illegal_grid_move'
            state=i,j
        if state!=target:return False,'grid_goal_missed'
        queue=deque([(start,0)]);seen={start}
        while queue:
            pos,n=queue.popleft()
            if pos==target:return len(actions)==n,'executed_shortest_grid_path'
            for di,dj in delta.values():
                nxt=pos[0]+di,pos[1]+dj
                if 0<=nxt[0]<h and 0<=nxt[1]<w and grid[nxt[0]][nxt[1]]!=-1 and nxt not in seen:seen.add(nxt);queue.append((nxt,n+1))
    # Scientific free-form expressions: conservative normalized exact metric,
    # explicitly separate from execution correctness and symbolic equivalence.
    norm=lambda value:re.sub(r'\s+','',value).replace('\\left','').replace('\\right','').replace('$','').lower()
    return norm(text)==norm(answer),'normalized_exact_not_symbolic_equivalence'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('output');args=parser.parse_args()
    with open(args.input) as source,open(args.output,'w') as dest:
        for line in source:
            r=json.loads(line);correct,method=score(r['record'],r['prediction']);dest.write(json.dumps({'id':r['record']['id'],'correct':bool(correct),'method':method})+'\n')

if __name__=='__main__':main()
