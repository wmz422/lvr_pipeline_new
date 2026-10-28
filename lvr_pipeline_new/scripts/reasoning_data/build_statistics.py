"""Reconcile unique generation attempts, candidates, captions and frozen releases."""
import json,math,statistics
from datetime import datetime,timezone
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from common import ROOT,BUILD,CANDIDATES,write_json,digest
from caption import CONFIG_ID,RETRY_CONFIG_ID
from review_captions import REVIEW_ID
from lvr.data.records import read_records


def distribution(values):
    values=sorted(values)
    if not values:return {'count':0}
    return {'count':len(values),'min':values[0],'p50':values[math.ceil(.5*len(values))-1],
        'p90':values[math.ceil(.9*len(values))-1],'p99':values[math.ceil(.99*len(values))-1],'max':values[-1],
        'quantile_method':'nearest rank'}


def read_json(path):return json.loads(path.read_text())


def recorded_runtime_statistics(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row.get('runtime_device','not_recorded')].append(row)
    result={}
    for device,group in sorted(grouped.items()):
        durations=[row['seconds_per_image'] for row in group if 'seconds_per_image' in row]
        result[device]={'entries':len(group),'timed_entries':len(durations),
            'entries_without_timing':len(group)-len(durations),'seconds_per_image':distribution(durations),
            'sum_recorded_worker_seconds':sum(durations) if durations else None}
    return result


def sliding_split_statistics(rows):
    rows=[r for r in rows if r['source']=='mentis_sliding-puzzle']
    source_images=defaultdict(set);pairs=defaultdict(set);states={};counts=Counter()
    with ThreadPoolExecutor(16) as pool:
        metadata=pool.map(read_json,(ROOT/r['provenance']['metadata'] for r in rows))
        for row,m in zip(rows,metadata):
            assert row['source_image_id']==m['source_image_sha256']
            split=row['split'];source_images[split].add(row['source_image_id']);counts[split]+=1
            key=digest([m['grid_size'],m['initial_state'],m['target_state']]);pairs[split].add(key)
            state=states.setdefault(key,{'grid_size':m['grid_size'],'initial_state':m['initial_state'],
                'target_state':m['target_state'],'split_counts':{'train':0,'val':0}})
            state['split_counts'][split]+=1
    shared_images=source_images['train'] & source_images['val']
    assert not shared_images,'Sliding source photographs cross train and val'
    return {'records':len(rows),'by_split':{split:{'records':counts[split],
            'source_photographs':len(source_images[split]),'abstract_state_pairs':len(pairs[split])} for split in ('train','val')},
        'source_photographs_shared_train_val':len(shared_images),
        'abstract_state_pairs_shared_train_val':len(pairs['train'] & pairs['val']),
        'unique_abstract_state_pairs':len(states),'grid_sizes':sorted({s['grid_size'] for s in states.values()}),
        'states':states,'scope':'Photo identities are held apart. Abstract 2x2 arrangements may repeat across splits; this is not evidence of unseen-arrangement or larger-grid generalization.'}


def lines(path):
    with path.open() as f:
        for line in f:
            try:yield json.loads(line)
            except json.JSONDecodeError:continue


def main():
    generation={}
    for task in ('form-board','paper-fold','hinge-folding','sliding-puzzle','rushhour'):
        directory=BUILD/'mentis'/task
        with ThreadPoolExecutor(16) as pool:statuses=list(pool.map(read_json,directory.glob('*/status.json')))
        durations=[r['seconds'] for r in statuses];levels=defaultdict(Counter)
        for r in statuses:levels[str(r['requested_level'])][r['status']]+=1
        events=[r for p in (BUILD/'mentis').glob('events_*.jsonl') for r in lines(p) if r['task']==task]
        generation[task]={'qualified_target':2000,'unique_seed_statuses':dict(Counter(r['status'] for r in statuses)),
            'unique_seeds':len(statuses),'event_statuses_including_restarts':dict(Counter(r['status'] for r in events)),
            'raw_records':len(list(directory.glob('*/record.json'))),'requested_difficulty_statuses':dict(levels),
            'seconds_per_last_seed_attempt_median':statistics.median(durations),
            'seconds_per_last_seed_attempt_p90':sorted(durations)[int(.9*(len(durations)-1))],
            'conversion':json.loads((CANDIDATES/('mentis_'+task+'.stats.json')).read_text())}
        if task=='rushhour':
            def phase_timing(path):return read_json(path).get('phase_timing',{})
            with ThreadPoolExecutor(16) as pool:timings=list(pool.map(phase_timing,directory.glob('*/record.json')))
            generation[task]['successful_record_phase_seconds']={name:distribution(t[name] for t in timings if name in t)
                for name in ('generation_seconds','solve_seconds')}
            generation[task]['phase_timing_scope']='Successful puzzles with saved metadata only; rejected/timed-out attempts retain total subprocess time in status.json.'
    captioned=set();caption_stats={}
    for config in (CONFIG_ID,RETRY_CONFIG_ID):
        cache={r['image_sha256']:r for p in (BUILD/'captions'/config).glob('worker_*.jsonl') for r in lines(p)}
        reviews={(r['image_sha256'],r['caption_sha256']):r for p in (BUILD/'caption_reviews'/config).glob(REVIEW_ID+'*.jsonl') for r in lines(p)}
        captioned.update(sha for sha,r in cache.items() if r['status']=='generated')
        lengths=[len(r['text'].split()) for r in cache.values()]
        caption_stats[config]={'unique_images_attempted':len(cache),'generation_status':dict(Counter(r['status'] for r in cache.values())),
            'generation_failure_reasons':dict(Counter(r['error'] for r in cache.values() if r.get('error'))),
            'review_status':dict(Counter(r['status'] for r in reviews.values())),
            'review_state_scope':'Independent model verdicts before manual and local source quality gates.',
            'caption_word_length_median':statistics.median(lengths) if lengths else None,
            'caption_word_lengths':distribution(lengths),
            'generation_output_token_lengths':distribution(r['output_tokens'] for r in cache.values() if 'output_tokens' in r),
            'generation_seconds_per_image':distribution(r['seconds_per_image'] for r in cache.values() if 'seconds_per_image' in r),
            'sum_recorded_seconds_per_image':sum(r.get('seconds_per_image',0) for r in cache.values()),
            'generation_runtime_by_recorded_device':recorded_runtime_statistics(cache.values()),
            'review_runtime_by_recorded_device':recorded_runtime_statistics(reviews.values()),
            'timing_scope':'Recorded worker processing seconds for unique active-config entries, including CPU exceptions; excludes model initialization. Sums are not elapsed wall time or CPU core-seconds. Legacy journals without a device field are reported as not_recorded.'}
    candidates={}
    for p in sorted(CANDIDATES.glob('*.jsonl')):
        if '.rejected.' in p.name or '.building.' in p.name:continue
        stats=Counter();difficulties=Counter();helper_counts=Counter();sizes=Counter()
        for r in lines(p):
            stats['qa']+=1;helpers=r['auxiliary_images'];stats['helper_references']+=len(helpers)
            stats['qa_with_generated_descriptions']+=all(im['sha256'] in captioned for im in helpers)
            stats['helper_references_with_generated_descriptions']+=sum(im['sha256'] in captioned for im in helpers)
            difficulties[str(r['difficulty'])]+=1;helper_counts[str(len(helpers))]+=1
            for im in r['problem_images']+helpers:sizes[f"{im['width']}x{im['height']}"]+=1
        candidates[p.stem]={**stats,'difficulty_counts':dict(difficulties),'helper_count_distribution':dict(helper_counts),'image_dimensions':dict(sizes)}
    releases={}
    for version in ('v0','v1'):
        root=ROOT/'data/processed'/('reasoning_'+version)
        if (root/'release_manifest.json').exists():
            by_source=defaultdict(list)
            for split in ('train','val'):
                for row in read_records(root/split):by_source[row['source']].append(row)
            measured={}
            for source,rows in by_source.items():
                measured[source]={'records':len(rows),
                    'processed_token_lengths':{key:distribution(r['preprocessing']['lengths'][key] for r in rows) for key in ('qa','encoder','description')},
                    'helpers_per_qa':distribution(len(r['auxiliary_images']) for r in rows),
                    'originals_per_qa':distribution(len(r['problem_images']) for r in rows),
                    'description_words_per_qa':distribution(len(r['description'].split()) for r in rows),
                    'image_dimensions':dict(Counter(f"{im['width']}x{im['height']}" for r in rows for im in r['problem_images']+r['auxiliary_images']))}
            releases[version]={'manifest':json.loads((root/'release_manifest.json').read_text()),'stats':json.loads((root/'stats.json').read_text()),
                'measured_distributions_by_source':measured,
                'sliding_split_statistics':sliding_split_statistics([r for group in by_source.values() for r in group])}
    write_json(BUILD/'build_statistics.json',{'captured_utc':datetime.now(timezone.utc).isoformat(),
        'workflow_stage':read_json(BUILD/'full_workflow.json')['stage'] if (BUILD/'full_workflow.json').exists() else None,
        'generation':generation,'candidates':candidates,'captions':caption_stats,'releases':releases,
        'meaning':'Candidate generation coverage is not approval. Frozen releases separately require all description and quality gates.'})
    print('BUILD_STATISTICS',len(candidates),sum(s['qa'] for s in candidates.values()),flush=True)


if __name__=='__main__':main()
