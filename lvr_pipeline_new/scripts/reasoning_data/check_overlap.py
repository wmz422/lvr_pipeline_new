"""Traceable overlap quarantine and grouped train/val assignment for candidates."""
from collections import defaultdict,Counter
import csv,json,sqlite3
from pathlib import Path
from common import ROOT,RAW,BUILD,CANDIDATES,digest,normalized,write_json,asset
from lvr.data.records import read_records


class Groups:
    def __init__(self):self.parent={}
    def find(self,x):
        if x not in self.parent:self.parent[x]=x
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,x,y):
        a,b=self.find(x),self.find(y)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def main():
    geometry=json.loads((BUILD/'quality/mentis_geometry.json').read_text())['records']
    official_geometry=defaultdict(list)
    for key,value in geometry.items():
        if key.startswith('reference:'):official_geometry[value['signature']].append(key)
    def distinct_geometry(sid,scope,ref):
        candidate=geometry.get('candidate:'+sid)
        reference=geometry.get('reference:'+scope+':'+ref.split('/metadata.json')[0]+'/metadata.json')
        return candidate and reference and candidate['signature']!=reference['signature']
    db=sqlite3.connect(BUILD/'exclusions.sqlite');assert (BUILD/'exclusions_index_report.json').exists()
    fingerprints=db.execute('select scope,identity,sha,pixel_sha,phash,width,height from images').fetchall()
    by_sha=defaultdict(list);by_pixel=defaultdict(list);held_phash=defaultdict(list);old_phash=defaultdict(list)
    for scope,identity,sha,pixel,ph,w,h in fingerprints:
        ref=(scope,identity)
        by_sha[sha].append(ref);by_pixel[pixel].append(ref)
        if True:
            target=old_phash if scope=='old_train' else held_phash
            n=int(ph,16)
            for i,(start,bits) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):target[(i,(n>>start)&((1<<bits)-1))].append((n,w/h,ref))
    held_questions=defaultdict(list)
    for scope,sid,q in db.execute("select scope,id,normalized from questions where scope!='old_train'"):
        if len(q)>=60:held_questions[q].append((scope,sid))
    structures=defaultdict(list)
    for scope,sid,sha in db.execute('select scope,id,fingerprint from structures'):structures[sha].append((scope,sid))
    db.close()
    overlay=BUILD/'quality/thinkmorph_navigation.validated.jsonl';navigation={}
    if overlay.exists():navigation={r['id']:r for r in map(json.loads,overlay.read_text().splitlines())}
    rejected_navigation={json.loads(l)['id']:json.loads(l)['reason'] for l in (BUILD/'quality/thinkmorph_navigation.rejected.jsonl').read_text().splitlines()}
    records=[]
    for p in sorted(CANDIDATES.glob('*.jsonl')):
        if '.rejected.' in p.name or '.building.' in p.name:continue
        for line in p.read_text().splitlines():
            r=json.loads(line);records.append(navigation.get(r['id'],r))
    counts=Counter();decisions={};group=Groups();image_groups={};seen_questions={};seen_problems={};near_cache={}
    out=BUILD/'quality';out.mkdir(exist_ok=True)
    with (out/'overlap_candidates.csv').open('w') as csvfile:
        writer=csv.writer(csvfile);writer.writerow(['candidate_id','reason','asset_sha256','reference_scope','reference_id'])
        for r in records:
            sid=r['id'];gid=r['group_id'];group.find(gid);reasons=set();force_train=False
            geo=geometry.get('candidate:'+sid)
            if r['source'] in ('mentis_hinge-folding','mentis_paper-fold','mentis_rushhour'):
                if geo is None:reasons.add('geometry_validation_failed')
                else:
                    gid=r['source']+':visible-geometry:'+geo['signature'];group.find(gid)
                    for ref in official_geometry.get(geo['signature'],[]):
                        reasons.add('official_visible_geometry_match');writer.writerow([sid,'official_visible_geometry_match','',ref,''])
            if sid in rejected_navigation:reasons.add(rejected_navigation[sid])
            q=normalized(r['question'])
            for scope,ref in held_questions.get(q,[]):
                reasons.add('heldout_question_match');writer.writerow([sid,'heldout_question_match','',scope,ref])
            for scope,ref in structures.get(r['provenance'].get('structure_sha256'),[]):
                reasons.add('official_structure_match');writer.writerow([sid,'official_structure_match','',scope,ref])
            source_assets=[]
            if r['source']=='mentis_sliding-puzzle':
                metadata=json.loads((ROOT/r['provenance']['metadata']).read_text())
                source_assets=[asset(ROOT/metadata['source_image_path'])]
                for im in source_assets:
                    n=int(im['phash'],16);aspect=im['width']/im['height']
                    for i,(start,bits) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):
                        if any(abs(aspect/ratio-1)<.02 and (n^value).bit_count()<=2 for value,ratio,ref in old_phash.get((i,(n>>start)&((1<<bits)-1)),[])):force_train=True
            for im in r['problem_images']+r['auxiliary_images']+source_assets:
                exact=set(by_sha.get(im['sha256'],[])+by_pixel.get(im['pixel_sha256'],[]))
                for scope,ref in exact:
                    if scope=='old_train':force_train=True
                    elif distinct_geometry(sid,scope,ref):writer.writerow([sid,'resolved_distinct_full_geometry',im['sha256'],scope,ref])
                    else:reasons.add('heldout_image_match');writer.writerow([sid,'heldout_image_match',im['sha256'],scope,ref])
            # Near matches are quarantined for review, never asserted to be semantic duplicates.
            near_images=source_assets+r['problem_images']+(r['auxiliary_images'] if 'jigsaw' in r['source'] else [])
            for im in near_images:
                sha=im['sha256']
                if sha not in near_cache:
                    n=int(im['phash'],16);aspect=im['width']/im['height'];matches=set()
                    for i,(start,bits) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):
                        for value,ratio,ref in held_phash.get((i,(n>>start)&((1<<bits)-1)),[]):
                            if abs(aspect/ratio-1)<0.12 and (n^value).bit_count()<=4:matches.add(ref)
                    near_cache[sha]=matches
                for scope,ref in near_cache[sha]:
                    if distinct_geometry(sid,scope,ref):
                        writer.writerow([sid,'resolved_distinct_full_geometry',sha,scope,ref]);continue
                    reasons.add('heldout_perceptual_candidate');writer.writerow([sid,'heldout_perceptual_candidate',sha,scope,ref])
            # Full problem identity and shared assembled-view identity bind families.
            keys=['pixels:'+im['pixel_sha256'] for im in r['problem_images']]
            if geo is not None:keys=['visible-geometry:'+geo['signature']]
            if 'jigsaw' in r['source']:keys += ['pixels:'+im['pixel_sha256'] for im in r['auxiliary_images']]
            if r['source'].startswith('zebra_scientific') and len(q)>=60:keys+=['science-question:'+q]
            for key in keys:
                if key in image_groups:group.union(gid,image_groups[key])
                else:image_groups[key]=gid
            problem_key=digest([q,[im['pixel_sha256'] for im in r['problem_images']]])
            if geo is not None:problem_key=gid
            if problem_key in seen_problems:reasons.add('duplicate_qa')
            else:seen_problems[problem_key]=sid
            decisions[sid]={'id':sid,'group_id':gid,'forced_train_by_old_training_image':force_train,'reasons':sorted(reasons),'source':r['source']}
    force_groups={group.find(d['group_id']) for d in decisions.values() if d['forced_train_by_old_training_image']}
    # Expansion must preserve the pilot's frozen memberships. If new evidence
    # joins previously separated splits, stop for investigation instead of
    # silently moving a held-out record or publishing a cross-split group.
    frozen_groups=defaultdict(set);frozen_count=0
    pilot=ROOT/'data/processed/reasoning_v0'
    if (pilot/'release_manifest.json').exists():
        for split in ('train','val'):
            for old in read_records(pilot/split):
                d=decisions.get(old['id'])
                if not d or d['reasons']:raise ValueError('New overlap evidence invalidates frozen pilot: '+old['id'])
                frozen_groups[group.find(d['group_id'])].add(split);frozen_count+=1
        for gid,splits in frozen_groups.items():
            if len(splits)!=1 or ('val' in splits and gid in force_groups):
                raise ValueError('Expanded overlap group conflicts with frozen pilot split: '+gid)
    for d in decisions.values():
        d['group_id']=group.find(d['group_id'])
        d['split']='quarantine' if d['reasons'] else 'train' if d['group_id'] in force_groups or int(digest(['9302026',d['group_id']])[:8],16)%10 else 'val'
        if not d['reasons'] and d['group_id'] in frozen_groups:d['split']=next(iter(frozen_groups[d['group_id']]))
        counts[(d['source'],d['split'])]+=1
    (out/'overlap_decisions.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in decisions.values()))
    write_json(out/'overlap_report.json',{'counts':[{'source':s,'split':p,'count':n} for (s,p),n in sorted(counts.items())],
        'records':len(records),'groups':len({d['group_id'] for d in decisions.values()}),'forced_train_groups':len(force_groups),'frozen_pilot_records_preserved':frozen_count,
        'near_check':'64-bit DCT perceptual hash distance <=4 with aspect-ratio tolerance 12%; candidate matches quarantined, not claimed confirmed semantic duplication',
        'cross_new_groups':'source identities, exact decoded problem images, Jigsaw helper image identity, and Scientific normalized question families',
        'excluded_references':list(dict.fromkeys(row[0] for row in fingerprints))})
    print('COMPLETE OVERLAP',len(records),dict(counts),flush=True)

if __name__=='__main__':main()
