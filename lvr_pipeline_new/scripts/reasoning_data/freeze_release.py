"""Publish immutable checksummed shards only after explicit data quality gates."""
import argparse,hashlib,json,shutil
from collections import Counter,defaultdict
from common import ROOT,RAW,BUILD,digest,write_json
from caption import CONFIG,CONFIG_ID,RETRY_CONFIG,RETRY_CONFIG_ID
from review_captions import REVIEW_CONFIG
from lvr.data.reasoning import validate_record
from lvr.data.records import read_records
from build_statistics import sliding_split_statistics
from pilot_amendments import validate_pilot_subset


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['v0','v1'],required=True);a=p.parse_args()
    target=ROOT/'data/processed'/f'reasoning_{a.version}'
    if target.exists():raise FileExistsError(f'Immutable release exists: {target}')
    rows=[json.loads(l) for l in (BUILD/f'preprocessed_{a.version}.jsonl').read_text().splitlines()]
    audit=json.loads((BUILD/'visual_audit'/f'{CONFIG_ID}.json').read_text())
    if audit['status']!='passed' or audit['caption_config_id']!=CONFIG_ID:raise ValueError('Visual audit is incomplete or stale')
    reviewed=Counter(r['family'] for r in audit['samples'])
    families={'zebra_scientific' if r['source'].startswith('zebra_scientific_') else r['source'] for r in rows}
    if len(families)!=11 or any(reviewed[f]<20 for f in families):raise ValueError('At least 20 visual samples per family required')
    preprocessed_counts=Counter(r['source'] for r in rows)
    rejected={r['id'] for r in audit['samples'] if r['status']=='rejected'}
    rejected_caption_hashes=set(audit.get('rejected_caption_sha256',[]))
    audit_excluded=[r['id'] for r in rows if r['id'] in rejected or any(part['caption_sha256'] in rejected_caption_hashes for part in r['description_parts'])]
    rows=[r for r in rows if r['id'] not in rejected and not any(part['caption_sha256'] in rejected_caption_hashes for part in r['description_parts'])]
    qualified_counts=Counter(r['source'] for r in rows)
    previous={}
    if a.version=='v1':
        acceptance=json.loads((BUILD/'training_acceptance_summary.json').read_text())
        if acceptance['status']!='passed':raise ValueError('Pilot training acceptance is incomplete')
        expanded=json.loads((BUILD/'batch_acceptance_v1.json').read_text())
        if expanded['status']!='passed' or expanded['input_sha256']!=sha(BUILD/'preprocessed_v1.jsonl'):
            raise ValueError('Expanded per-source model checks are incomplete or stale')
        v0=ROOT/'data/processed/reasoning_v0'
        previous={r['id']:r for split in ('train','val') for r in read_records(v0/split)}
        grouped=defaultdict(list)
        for row in rows:grouped[row['source']].append(row)
        selected=[]
        for source,group in grouped.items():
            if source.startswith('mentis_'):
                if len(group)<2000:raise ValueError(f'{source}: {len(group)} qualified records; 2000 required')
                group.sort(key=lambda r:(r['id'] not in previous,digest(r['id'])))
                group=group[:2000]
            selected.extend(group)
        rows=selected
        lookup={r['id']:r for r in rows}
        errata_path=BUILD/'pilot_caption_errata.json'
        errata=json.loads(errata_path.read_text()) if errata_path.exists() else {}
        withdrawals_path=BUILD/'pilot_caption_withdrawals.json'
        withdrawals=json.loads(withdrawals_path.read_text()) if withdrawals_path.exists() else {}
        corrections=validate_pilot_subset(previous,lookup,errata,rejected_caption_hashes,sha(v0/'release_manifest.json'),withdrawals)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate ids')
    group_splits=defaultdict(set)
    for r in rows:
        validate_record(r,check_files=True)
        if r['description_provenance']['config_id']!=CONFIG_ID or r['preprocessing']['status']!='passed':raise ValueError('Stale quality evidence')
        if r['split'] not in ('train','val'):raise ValueError('Unexpected split')
        group_splits[r['group_id']].add(r['split']);r['quality']['status']='passed'
    if any(len(s)!=1 for s in group_splits.values()):raise ValueError('Group leakage')
    stage=target.with_name(target.name+'.building');stage.mkdir(exist_ok=False,parents=True)
    manifest={'format':'lvr-reasoning-release-v1','record_schema':'lvr-reasoning-qa-v1','version':a.version,'status':'data_validated_training_pending' if a.version=='v0' else 'accepted',
        'splits':{},'description_config':CONFIG,'caption_config_id':CONFIG_ID,'caption_review_config':REVIEW_CONFIG,
        'assets_root':str(BUILD/'assets'),'visual_audit_sha256':digest(audit),'group_seed':9302026}
    manifest['description_generation_configs']={CONFIG_ID:CONFIG,RETRY_CONFIG_ID:RETRY_CONFIG}
    manifest['description_attempt_limit']=2
    manifest['description_configuration_scope']='Each description part records its actual generation_config_id and attempt.'
    manifest['description_config_counts']=dict(Counter(p['generation_config_id'] for r in rows for p in r['description_parts']))
    for split in ('train','val'):
        subset=sorted((r for r in rows if r['split']==split),key=lambda r:r['id']);directory=stage/split;directory.mkdir();shards=[]
        for n in range(0,len(subset),1000):
            path=directory/f'{n//1000:05d}.json';write_json(path,subset[n:n+1000]);shards.append({'file':path.name,'records':len(subset[n:n+1000]),'sha256':sha(path)})
        index={'format':'lvr-json-shards-v1','record_schema':'lvr-reasoning-qa-v1','records':len(subset),'shards':shards};write_json(directory/'index.json',index)
        manifest['splits'][split]={'records':len(subset),'index_sha256':sha(directory/'index.json')}
    stats=[]
    for source,split in sorted({(r['source'],r['split']) for r in rows}):
        subset=[r for r in rows if (r['source'],r['split'])==(source,split)]
        stats.append({'source':source,'split':split,'records':len(subset),'helpers':sum(len(r['auxiliary_images']) for r in subset),
            'description_qa_coverage':1.0,'description_helper_coverage':1.0,'difficulties':dict(Counter(str(r['difficulty']) for r in subset)),
            'max_processed_tokens':max(r['processed_token_length'] for r in subset)})
    write_json(stage/'stats.json',{'by_source_and_split':stats,'total':len(rows),'train_fraction':sum(r['split']=='train' for r in rows)/len(rows),
        'limits':'Description coverage is completeness after prescribed checks, not proof of factual perfection. Overlap checks cover only indexed assets.'})
    published_counts=Counter(r['source'] for r in rows)
    write_json(stage/'selection_report.json',{'preprocessed_by_source':dict(preprocessed_counts),
        'audit_excluded_ids':audit_excluded,'qualified_by_source':dict(qualified_counts),
        'published_by_source':dict(published_counts),
        'qualified_not_selected_by_source':{s:n-published_counts[s] for s,n in qualified_counts.items()},
        'selection':'v1 keeps 2000 per Mentis task, prioritizing qualified v0 QA identities then a fixed ID digest; other sources retain all qualified records. Retained pilot QA content and splits are preserved; declared caption corrections and two-attempt withdrawals are verified separately.',
        'meaning':'Qualified records outside the Mentis quota are not quality rejections.'})
    write_json(stage/'sliding_split_report.json',sliding_split_statistics(rows))
    locks={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in sorted(RAW.glob('*/download_manifest.json'))}
    locks['existing_sources']=json.loads((ROOT/'data/reasoning_sources.json').read_text())
    locks['caption_model']=json.loads((BUILD/'caption_model_lock.json').read_text())
    locks['caption_review_model']=json.loads((BUILD/'caption_review_model_lock.json').read_text())
    locks['mentis_environment']=json.loads((BUILD/'mentis_environment_lock.json').read_text())
    locks['sliding_source_images']=json.loads((BUILD/'sliding_sources/manifest.json').read_text())
    locks['pipeline_code_sha256']={str(p.relative_to(ROOT)):sha(p) for directory in ('scripts/reasoning_data','src/lvr/data','src/lvr/training','src/lvr/models') for p in sorted((ROOT/directory).rglob('*.py'))}
    for relative in ('src/lvr/stage1.py','src/lvr/predict_latents.py','configs/latent_prediction/deepspeed_zero3_80gb.json'):
        locks['pipeline_code_sha256'][relative]=sha(ROOT/relative)
    write_json(stage/'source_lock.json',locks);write_json(stage/'visual_audit.json',audit)
    if a.version=='v1':
        write_json(stage/'training_acceptance.json',acceptance)
        write_json(stage/'batch_acceptance.json',expanded)
        write_json(stage/'pilot_caption_errata.json',errata)
        write_json(stage/'pilot_caption_corrections.json',corrections)
        write_json(stage/'pilot_caption_withdrawals.json',withdrawals)
        resolution_path=BUILD/'pilot_caption_errata_resolution.json'
        if resolution_path.exists():write_json(stage/'pilot_caption_retry_evidence.json',json.loads(resolution_path.read_text()))
    write_json(stage/'validation_report.json',{'processor':json.loads((BUILD/f'preprocessing_{a.version}_report.json').read_text()),
        'published_records':len(rows),'schema_hashes_checked':True,'group_split_independence':True,
        'overlap':json.loads((BUILD/'quality/overlap_report.json').read_text())})
    shutil.copyfile(BUILD/f'rejected_{a.version}.jsonl',stage/'rejected.jsonl')
    shutil.copyfile(BUILD/'quality/overlap_candidates.csv',stage/'overlap_candidates.csv')
    manifest['files_sha256']={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    write_json(stage/'release_manifest.json',manifest);stage.rename(target);print('FROZEN',target,len(rows),flush=True)

if __name__=='__main__':main()
