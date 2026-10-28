"""Select a deterministic pilot spanning difficulty, helper counts, and answer lengths."""
from collections import defaultdict
import argparse,hashlib,json
from common import BUILD,CANDIDATES,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pool-size',type=int,default=100);parser.add_argument('--source-pool-size',action='append',default=[]);args=parser.parse_args()
    overrides={key:int(value) for key,value in (s.split('=') for s in args.source_pool_size)}
    sources=defaultdict(list);invalid=set()
    path=BUILD/'quality/thinkmorph_navigation.rejected.jsonl'
    if path.exists():invalid={json.loads(l)['id'] for l in path.read_text().splitlines()}
    path=BUILD/'quality/overlap_decisions.jsonl'
    if path.exists():invalid.update(r['id'] for r in map(json.loads,path.read_text().splitlines()) if r['split']=='quarantine')
    path=BUILD/'quality/answers.rejected.jsonl'
    if path.exists():invalid.update(r['id'] for r in map(json.loads,path.read_text().splitlines()))
    for p in sorted(CANDIDATES.glob('*.jsonl')):
        if '.rejected.' in p.name or '.building.' in p.name:continue
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['id'] in invalid:continue
            family='zebra_scientific' if r['source'].startswith('zebra_scientific_') else r['source']
            sources[family].append(r)
    assert len(sources)==11,sorted(sources)
    selected=[];counts={}
    for source,rows in sorted(sources.items()):
        limit=overrides.get(source,args.pool_size)
        strata=defaultdict(list)
        for row in rows:
            key=(row['task'],str(row['difficulty']),len(row['auxiliary_images']),min(3,len(row['answer'])//500))
            strata[key].append(row)
        for group in strata.values():group.sort(key=lambda r:hashlib.sha256(('9302026:'+r['id']).encode()).hexdigest())
        keep={}
        # Include the actual largest input/sequence cases instead of testing only first rows.
        for key in [lambda r:len(r['auxiliary_images']),lambda r:len(r['problem_images']),lambda r:len(r['question'])+len(r['answer']),lambda r:max(i['width']*i['height'] for i in r['problem_images'])]:
            r=max(rows,key=key);keep[r['id']]=r
        while len(keep)<min(limit,len(rows)):
            for key in sorted(strata):
                if strata[key]:
                    r=strata[key].pop();keep[r['id']]=r
                    if len(keep)>=limit:break
        selected.extend(keep);counts[source]=len(keep)
    write_json(BUILD/'pilot_ids.json',{'seed':9302026,'ids':selected,'counts':counts,'selection':'stratified task/difficulty/helper count/answer length, plus actual maximum inputs and text','status':'candidate_selection_not_release'})
    print(counts,flush=True)

if __name__=='__main__':main()
