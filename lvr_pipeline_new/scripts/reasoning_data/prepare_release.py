"""Join approved evidence into staging records; never publishes unfinished data."""
import argparse,hashlib,json,re
from collections import Counter,defaultdict
from functools import lru_cache
from common import ROOT,BUILD,CANDIDATES,write_json
from caption import CONFIG,CONFIG_ID,RETRY_CONFIG,RETRY_CONFIG_ID,clean
from review_captions import REVIEW_ID,REVIEW_CONFIG
from lvr.data.reasoning import assemble_description,validate_record
from form_caption_facts import check as check_form_caption
from hinge_caption_facts import check as check_hinge_caption
from graph_caption_facts import check as check_graph_caption


@lru_cache(maxsize=None)
def rush_movable_count(path):
    m=json.loads((ROOT/path).read_text())
    return sum(o.get('movable',True) for o in m['objects'])


@lru_cache(maxsize=1)
def confirmed_literal_rejections():
    audit=json.loads((BUILD/'visual_audit'/f'{CONFIG_ID}.json').read_text())
    rules=audit.get('confirmed_literal_rejections',[])
    for rule in rules:
        if not rule.get('source') or not (rule.get('image_sha256') or rule.get('source_image_id')):
            raise ValueError('A visual literal rejection must identify its source and image')
    return rules


@lru_cache(maxsize=1)
def confirmed_graph_isolation():
    audit=json.loads((BUILD/'visual_audit'/f'{CONFIG_ID}.json').read_text())
    result={}
    for entry in audit.get('confirmed_graph_isolated_counts',[]):
        if entry['source']!='zebra_scientific_graph_algorithms' or entry['image_sha256'] in result:
            raise ValueError('Invalid or duplicate visual graph isolation evidence')
        if not all(isinstance(n,int) and n>=0 for n in entry['isolated_counts'].values()):
            raise ValueError('Invalid visually checked graph isolation count')
        result[entry['image_sha256']]=entry['isolated_counts']
    return result


def evidence_rows(path,live_snapshot=False):
    with path.open() as handle:
        for line in handle:
            if not line.endswith('\n') and live_snapshot:
                # A worker may still be writing the last journal record. It is
                # absent evidence until a later snapshot reads its full line.
                continue
            yield json.loads(line)


def evidence(live_snapshot=False):
    captions={};reviews={}
    for config_id in (CONFIG_ID,RETRY_CONFIG_ID):
        for p in sorted((BUILD/'captions'/config_id).glob('worker_*.jsonl')):
            for r in evidence_rows(p,live_snapshot):captions.setdefault(r['image_sha256'],{})[config_id]=r
        for p in sorted((BUILD/'caption_reviews'/config_id).glob(REVIEW_ID+'*.jsonl')):
            for r in evidence_rows(p,live_snapshot):reviews[r['image_sha256'],r['caption_sha256']]=r
    return captions,reviews


def approve(row,im,c,reviews,geometries,bad_hashes):
    _,error=clean(c['text'],c['status']=='generated')
    if error:return 'invalid_caption:'+error,None
    for rule in confirmed_literal_rejections():
        if row['source']!=rule['source']:continue
        if rule.get('image_sha256') and im['sha256']!=rule['image_sha256']:continue
        if rule.get('source_image_id') and row.get('source_image_id')!=rule['source_image_id']:continue
        if rule['substring'].lower() in c['text'].lower():return 'confirmed_visual_caption_conflict:'+rule['reason'],None
    sha=hashlib.sha256(c['text'].encode()).hexdigest();review=reviews.get((im['sha256'],sha))
    if sha in bad_hashes:return 'visual_caption_rejected',None
    if not review:return 'missing_caption_review',None
    if review['status']!='passed':return 'caption_review_rejected',None
    text=c['text'].lower()
    negative=re.search(r'\b(?:no|without|not)\s+(?:any\s+)?(?:overlap\w*|touching\s+or\s+overlap\w*)|\bnon[- ]overlap',text)
    if row['source'] in ('mirage_vsp','thinkmorph_navigation'):
        holes=sum(x==-1 for line in row['provenance']['map_desc'] for x in line)
        numbers={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
        for match in re.finditer(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:(?:blue|circular|ice|water)\s+)*(?:patches|pools|circles|shapes|holes)\b',text):
            if numbers[match[1]]!=holes:return 'caption_map_hole_count_conflict',None
    if row['source']=='zebra_scientific_graph_algorithms':
        error=check_graph_caption(c['text'],im,confirmed_graph_isolation().get(im['sha256']))
        if error:return 'caption_graph_pixels:'+error,None
        # These renderers draw disjoint circular nodes joined by thin edges.
        # Avoid treating graph adjacency as physical contact between circles.
        if re.search(r'\btouch\w*\b',text):return 'unverified_graph_node_contact',None
        if 'overlap' in text and not negative:return 'unverified_graph_node_overlap',None
        if re.search(r'no enclosed|\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:circular\s+)?(?:nodes|circles)\b',text):return 'unverified_graph_count_or_topology',None
    if row['source']=='zebra_scientific_geometry' and 'overlap' in text and not negative:return 'unverified_geometry_overlap_claim',None
    if row['source']=='mentis_form-board':
        if re.search(r'letter\s+["\x27]?[a-z]\b|\b[a-z]-shaped',text):return 'unverified_letter_shape_analogy',None
        if 'overlap' in text and not negative:return 'caption_conflicts_with_nonoverlapping_assembly',None
        error=check_form_caption(c['text'],im)
        if error:return 'caption_pixel_geometry:'+error,None
    if row['source']=='mentis_hinge-folding':
        facts=geometries['candidate:'+row['id']]['state_facts'][im['step']-1];error=check_hinge_caption(c['text'],facts)
        if error:return 'caption_hinge_geometry:'+error,None
    if row['source']=='mentis_rushhour':
        # Movable vehicles and obstacles cannot overlap. Intersections with
        # dashed rails and the final red-car/exit overlay are visibly permitted.
        exit_overlap=im['step']==len(row['auxiliary_images']) and re.search(r'overlap\w*\s+(?:a\s+|the\s+)?green',text)
        if 'overlap' in text and not negative and not exit_overlap and not re.search(r'overlap\w*\s+(?:the\s+)?dashed',text):return 'unverified_vehicle_overlap_claim',None
        if 'atop' in text:return 'unverified_exit_relative_position',None
        if re.search(r'\bgreen border\b|\ba square\b',text):return 'unverified_vehicle_border_or_square_claim',None
        if re.search(r'vertical (?:green )?bar at (?:the )?top',text):return 'unverified_exit_orientation',None
        numbers={'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8}
        for match in re.finditer(r'\b(two|three|four|five|six|seven|eight)\s+(?:(?:colored|angular|rectangular)\s+)*(?:regions|polygons|rectangles|shapes)\b',text):
            if numbers[match[1]]<rush_movable_count(row['provenance']['metadata']):return 'visible_vehicle_undercount',None
    if row['source']=='mentis_paper-fold':
        if re.search(r'\b(door|knob|building|window)\b',text):return 'unsupported_paper_object_identity',None
        if 'overlap' in text and not negative:return 'unverified_paper_region_overlap',None
        if re.search(r'circle\w*\s+touch\w*',text):return 'unverified_hole_frame_contact',None
        if re.search(r'\bcentered\b',text):return 'unverified_hole_center_position',None
    return None,{'step':im['step'],'image_sha256':im['sha256'],'text':c['text'],'quality_status':'passed','caption_sha256':sha,
        'generation_config_id':c['config_id'],'generation_attempt':c.get('attempt',1)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v0');p.add_argument('--per-family',type=int,default=100)
    p.add_argument('--snapshot',action='store_true',help='Read completed live evidence into a snapshot_* version without changing the caption retry queue.')
    a=p.parse_args()
    if a.snapshot and (not re.fullmatch(r'snapshot_[a-z0-9_]+',a.version)):
        p.error('--snapshot requires a separate snapshot_* version')
    captions,reviews=evidence(a.snapshot);rejected=[];accepted=defaultdict(list);missing=set();overlays={};answer_rejections={};retry_images={}
    geometries=json.loads((BUILD/'quality/mentis_geometry.json').read_text())['records']
    audit_path=BUILD/'visual_audit'/f'{CONFIG_ID}.json'
    manual=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    manual_rejected_ids={r['id'] for r in manual.get('samples',[]) if r['status']=='rejected'}
    manual_rejected_hashes=set(manual.get('rejected_caption_sha256',[]))
    for name in ('thinkmorph_navigation.validated','answers.validated'):
        for r in map(json.loads,(BUILD/'quality'/f'{name}.jsonl').read_text().splitlines()):overlays[r['id']]=r
    for r in map(json.loads,(BUILD/'quality/answers.rejected.jsonl').read_text().splitlines()):answer_rejections[r['id']]=r['reason']
    overlap={r['id']:r for r in map(json.loads,(BUILD/'quality/overlap_decisions.jsonl').read_text().splitlines())}
    for path in sorted(CANDIDATES.glob('*.jsonl')):
        if '.rejected.' in path.name or '.building.' in path.name:continue
        for original in map(json.loads,path.read_text().splitlines()):
            r=overlays.get(original['id'],original);d=overlap.get(r['id']);reasons=[];parts=[]
            if not d:reasons.append('missing_overlap_review')
            elif d['reasons']:reasons.extend(d['reasons'])
            if r['id'] in answer_rejections:reasons.append(answer_rejections[r['id']])
            if r['id'] in manual_rejected_ids:reasons.append('visual_audit_rejected')
            eligible_for_retry=not reasons
            for im in r['auxiliary_images']:
                options=captions.get(im['sha256'],{})
                if not options:reasons.append('missing_caption');missing.add(r['id']);continue
                errors=[];part=None
                for config_id in (CONFIG_ID,RETRY_CONFIG_ID):
                    if config_id not in options:continue
                    error,part=approve(r,im,options[config_id],reviews,geometries,manual_rejected_hashes)
                    if not error:break
                    errors.append(error)
                if part is None:
                    reasons.extend(errors)
                    if eligible_for_retry and RETRY_CONFIG_ID not in options and errors and errors!=['missing_caption_review']:
                        retry_images[im['sha256']]={**im,'reasons':errors}
                    continue
                parts.append(part)
            if reasons:
                rejected.append({'id':r['id'],'source':r['source'],'reasons':sorted(set(reasons))});continue
            r.update(split=d['split'],group_id=d['group_id'],description_parts=parts,description=assemble_description(parts),description_valid=True)
            r['description_provenance']={**CONFIG,'config_id':CONFIG_ID,'image_sha256':[im['sha256'] for im in r['auxiliary_images']],
                'quality_status':'passed','independent_review':REVIEW_CONFIG,'review_config_id':REVIEW_ID}
            used={p['generation_config_id'] for p in parts}
            r['description_provenance']['generation_configs']={key:value for key,value in ((CONFIG_ID,CONFIG),(RETRY_CONFIG_ID,RETRY_CONFIG)) if key in used}
            r['description_provenance']['configuration_scope']='Primary defaults above; exact per-image settings are in generation_configs keyed by each description part.'
            r['description_provenance']['prompt_version']=' + '.join(r['description_provenance']['generation_configs'][key]['prompt_version'] for key in sorted(used))
            r['quality'].update(status='automated_checks_passed_pending_release_audit',overlap=d)
            r['quality']['additional_caption_checks']=['manual_image_literal_rejections_v1']
            if r['source']=='mentis_form-board':r['quality']['additional_caption_checks'].append('form_pixel_geometry_v3')
            if r['source']=='mentis_hinge-folding':r['quality']['additional_caption_checks'].append('hinge_state_geometry_v1')
            if r['source']=='zebra_scientific_graph_algorithms':r['quality']['additional_caption_checks'].extend(['graph_yellow_node_pixels_v1','graph_isolated_node_visual_facts_v1'])
            if r['source']=='mentis_form-board':r['question']='Which labeled pieces on the right are necessary to form the target outline on the left? Give the set of option letters.'
            validate_record(r)
            family='zebra_scientific' if r['source'].startswith('zebra_scientific_') else r['source'];accepted[family].append(r)
    selected=[]
    for family,rows in sorted(accepted.items()):
        strata=defaultdict(list)
        for r in rows:strata[r['source'],str(r['difficulty']),len(r['auxiliary_images'])].append(r)
        for group in strata.values():group.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
        keep={}
        for key in (lambda r:len(r['auxiliary_images']),lambda r:len(r['question'])+len(r['answer']),lambda r:len(r['problem_images'])):
            r=max(rows,key=key);keep[r['id']]=r
        target=min(a.per_family,len(rows)) if a.per_family else len(rows)
        while len(keep)<target:
            for key in sorted(strata):
                if strata[key]:
                    r=strata[key].pop();keep[r['id']]=r
                    if len(keep)>=target:break
        # Keep the grouped split's natural ratio, with two val examples per
        # subdivision for the small acceptance evaluation whenever available.
        protected=set(keep)
        for source in sorted({r['source'] for r in rows}):
            current=sum(r['source']==source and r['split']=='val' for r in keep.values())
            for r in sorted((r for r in rows if r['source']==source and r['split']=='val' and r['id'] not in keep),key=lambda r:r['id']):
                if current>=2:break
                if len(keep)>=target:
                    removable=[sid for sid,x in keep.items() if x['split']=='train' and x['source']==source]
                    if not removable:break
                    keep.pop(removable[-1])
                keep[r['id']]=r;current+=1
        selected.extend(keep.values())
    output=BUILD/f'staging_{a.version}.jsonl'
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    (BUILD/f'rejected_{a.version}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
    if not a.snapshot:
        write_json(BUILD/f'caption_retry_queue_{CONFIG_ID}.json',{'retry_config_id':RETRY_CONFIG_ID,'images':list(retry_images.values()),'maximum_generation_attempts':2,'input_policy':'single image plus fixed retry instruction only; reasons are never passed to the model'})
    write_json(BUILD/f'staging_{a.version}_report.json',{'status':'not_published','accepted_pool':{k:len(v) for k,v in accepted.items()},
        'selected':len(selected),'selected_by_source':dict(Counter(r['source'] for r in selected)),
        'reasons':dict(Counter(x for r in rejected for x in r['reasons']))})
    print(json.dumps(json.loads((BUILD/f'staging_{a.version}_report.json').read_text()),indent=2),flush=True)

if __name__=='__main__':main()
