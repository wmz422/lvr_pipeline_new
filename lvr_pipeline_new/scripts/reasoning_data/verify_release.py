"""Read an immutable release and verify its data and recorded acceptance evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from common import BUILD, ROOT, digest, write_json
from caption import clean
from graph_caption_facts import check as check_graph_caption
from lvr.data.reasoning import SCHEMA, validate_record
from lvr.data.records import read_records


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_processor_fingerprint(row, config):
    staged = {key: value for key, value in row.items()
              if key not in ('preprocessing', 'processed_token_length')}
    # Freezing changes only this quality marker after preprocessing; restore
    # its input value without changing any field of the published record.
    staged['quality'] = {**row['quality'], 'status': 'automated_checks_passed_pending_release_audit'}
    require(digest([staged, config]) == row['preprocessing']['cache_key'],
            'Processor evidence belongs to different record content or configuration: ' + row['id'])


def verify(release):
    manifest = json.loads((release / 'release_manifest.json').read_text())
    version = manifest['version']
    require(manifest['format'] == 'lvr-reasoning-release-v1', 'Unknown release format')
    require(manifest['record_schema'] == SCHEMA, 'Unknown record schema')
    require(version in ('v0', 'v1'), 'Unknown release version')
    require(manifest['status'] == ('accepted' if version == 'v1' else 'data_validated_training_pending'), 'Unexpected release status')
    files = {str(path.relative_to(release)) for path in release.rglob('*') if path.is_file()}
    require(files == set(manifest['files_sha256']) | {'release_manifest.json'}, 'Manifest file inventory differs')
    for name, expected in manifest['files_sha256'].items():
        path = (release / name).resolve()
        require(path.is_relative_to(release), 'Release file path escapes package')
        require(sha(path) == expected, 'Release file checksum differs: ' + name)

    rows = []
    for split in ('train', 'val'):
        directory = release / split
        expected = manifest['splits'][split]
        require(sha(directory / 'index.json') == expected['index_sha256'], 'Index checksum differs')
        records = read_records(directory)
        require(len(records) == expected['records'], 'Split count differs')
        require(all(row['split'] == split for row in records), 'Record has wrong split')
        rows.extend(records)
    require(len({row['id'] for row in rows}) == len(rows), 'Duplicate record IDs')

    audit = json.loads((release / 'visual_audit.json').read_text())
    require(audit['status'] == 'passed', 'Visual audit is unfinished')
    require(audit['caption_config_id'] == manifest['caption_config_id'], 'Stale visual audit')
    rejected_ids = {row['id'] for row in audit['samples'] if row['status'] == 'rejected'}
    rejected_captions = set(audit.get('rejected_caption_sha256', []))
    inspected = defaultdict(set)
    for row in audit['samples']:
        inspected[row['family']].add(row['id'])

    groups = defaultdict(set)
    source_identities = defaultdict(set)
    images = {}
    counts = Counter()
    families = Counter()
    sources = defaultdict(list)
    configs = manifest['description_generation_configs']
    graph_isolation = {entry['image_sha256']: entry['isolated_counts']
                       for entry in audit.get('confirmed_graph_isolated_counts', [])}
    graph_parts_checked = 0
    processor_config = json.loads((release / 'validation_report.json').read_text())['processor']['config']
    require(manifest['description_attempt_limit'] == 2, 'Unexpected description retry policy')
    for row in rows:
        validate_record(row)
        require(row['id'] not in rejected_ids, 'Visually rejected QA was released')
        require(row['quality']['status'] == 'passed', 'Unapproved QA was released')
        overlap = row['quality']['overlap']
        require(not overlap['reasons'] and overlap['split'] == row['split'] and overlap['group_id'] == row['group_id'], 'Invalid overlap decision')
        require(not (overlap['forced_train_by_old_training_image'] and row['split'] == 'val'), 'Old training image entered independent val')
        groups[row['group_id']].add(row['split'])
        require(isinstance(row.get('source_image_id'), str) and bool(row['source_image_id']), 'Missing source image/scene identity')
        source_identities[row['source'], row['source_image_id']].add(row['split'])
        counts[row['source'], row['split']] += 1
        sources[row['source']].append(row)
        family = 'zebra_scientific' if row['source'].startswith('zebra_scientific_') else row['source']
        families[family] += 1
        require(len(inspected[family]) >= 20, 'Insufficient distinct visual audit samples: ' + family)
        require(row['description_provenance']['config_id'] == manifest['caption_config_id'], 'Unexpected primary caption configuration')
        for part in row['description_parts']:
            _, error = clean(part['text'], True)
            require(error is None, 'Caption text fails the publication checks: ' + row['id'] + ':' + str(error))
            config = part['generation_config_id']
            require(config in configs, 'Unrecorded caption generation configuration')
            require(row['description_provenance']['generation_configs'][config] == configs[config], 'Per-image generation settings differ')
            require(part['generation_attempt'] == (1 if config == manifest['caption_config_id'] else 2), 'Unexpected caption attempt')
            caption_sha = hashlib.sha256(part['text'].encode()).hexdigest()
            require(caption_sha == part['caption_sha256'], 'Caption checksum differs')
            require(caption_sha not in rejected_captions, 'Visually rejected caption was released')
            if version == 'v1' and row['source'] == 'zebra_scientific_graph_algorithms':
                helper = row['auxiliary_images'][part['step'] - 1]
                error = check_graph_caption(part['text'], helper, graph_isolation.get(helper['sha256']))
                require(error is None, 'Graph caption fails its explicit count checks: ' + row['id'] + ':' + str(error))
                graph_parts_checked += 1
        prep = row['preprocessing']
        verify_processor_fingerprint(row, processor_config)
        require(prep['status'] == 'passed' and prep['id'] == row['id'], 'Unprocessed QA was released')
        require(all(0 < prep['lengths'][key] <= 8192 for key in ('qa', 'encoder', 'description')), 'Sequence budget violated')
        require(row['processed_token_length'] == max(prep['lengths'].values()), 'Processed token length differs')
        require(len(prep['grids']['qa']) == len(row['problem_images']), 'QA image grid count differs')
        require(len(prep['grids']['encoder']) == len(row['problem_images']) + len(row['auxiliary_images']), 'Teacher image grid count differs')
        for image in row['problem_images'] + row['auxiliary_images']:
            identity = {key: image[key] for key in ('path', 'sha256', 'pixel_sha256', 'width', 'height')}
            if image['sha256'] in images:
                require(images[image['sha256']] == identity, 'Inconsistent shared image metadata')
            images[image['sha256']] = identity
    require(len(sources) == 13 and len(families) == 11, 'Missing source subdivisions')
    require(all(len(splits) == 1 for splits in groups.values()), 'Group crosses train and val')
    require(all(len(splits) == 1 for splits in source_identities.values()), 'Source image/scene identity crosses train and val')
    if version == 'v0':
        require(all(count == 100 for count in families.values()), 'Pilot family quota differs')
    else:
        require(all(len(group) == 2000 for source, group in sources.items() if source.startswith('mentis_')), 'Mentis qualified quota differs')
        selection = json.loads((release / 'selection_report.json').read_text())
        published = {source: len(group) for source, group in sources.items()}
        require(selection['published_by_source'] == published, 'Selection report published counts differ')
        require(all(count == published[source] + selection['qualified_not_selected_by_source'][source]
                    for source, count in selection['qualified_by_source'].items()), 'Qualified pool selection counts do not reconcile')
        sliding = json.loads((release / 'sliding_split_report.json').read_text())
        require(sliding['records'] == published['mentis_sliding-puzzle'], 'Sliding split report count differs')
        require(sliding['source_photographs_shared_train_val'] == 0, 'Sliding photos cross splits')
        require(sliding['unique_abstract_state_pairs'] == len(sliding['states']), 'Sliding abstract state count differs')
        for split in ('train', 'val'):
            group = [row for row in sources['mentis_sliding-puzzle'] if row['split'] == split]
            require(sliding['by_split'][split]['records'] == len(group), 'Sliding split count differs')
            require(sliding['by_split'][split]['source_photographs'] == len({row['source_image_id'] for row in group}), 'Sliding source-photo count differs')
            require(sum(state['split_counts'][split] for state in sliding['states'].values()) == len(group), 'Sliding abstract state counts do not reconcile')
        from pilot_amendments import validate_pilot_subset
        pilot = release.with_name('reasoning_v0')
        previous = {row['id']: row for split in ('train', 'val') for row in read_records(pilot / split)}
        corrections = validate_pilot_subset(previous, {row['id']: row for row in rows},
            json.loads((release / 'pilot_caption_errata.json').read_text()), rejected_captions, sha(pilot / 'release_manifest.json'),
            json.loads((release / 'pilot_caption_withdrawals.json').read_text()))
        require(corrections == json.loads((release / 'pilot_caption_corrections.json').read_text()), 'Pilot caption corrections differ from recorded evidence')

    for index, (expected, image) in enumerate(images.items(), 1):
        path = ROOT / image['path']
        require(sha(path) == expected, 'Image checksum differs: ' + str(path))
        with Image.open(path) as opened:
            opened.load()
            rgb = opened.convert('RGB')
            require(rgb.size == (image['width'], image['height']), 'Decoded image dimensions differ')
            pixel_sha = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
            require(pixel_sha == image['pixel_sha256'], 'Decoded image checksum differs')
        if index % 1000 == 0:
            print('VERIFY_IMAGES', index, len(images), flush=True)

    stats = json.loads((release / 'stats.json').read_text())
    require(stats['total'] == len(rows), 'Published total differs from stats')
    require({(row['source'], row['split']): row['records'] for row in stats['by_source_and_split']} == dict(counts), 'Per-source split counts differ from stats')
    for entry in stats['by_source_and_split']:
        group = [row for row in sources[entry['source']] if row['split'] == entry['split']]
        require(entry['helpers'] == sum(len(row['auxiliary_images']) for row in group), 'Helper count differs')
        require(entry['difficulties'] == dict(Counter(str(row['difficulty']) for row in group)), 'Difficulty counts differ')
        require(entry['description_qa_coverage'] == entry['description_helper_coverage'] == 1.0, 'Description coverage differs')

    acceptance_path = release / 'training_acceptance.json' if version == 'v1' else BUILD / 'training_acceptance_summary.json'
    training = json.loads(acceptance_path.read_text())
    require(training['status'] == 'passed', 'Training acceptance incomplete')
    for name, expected in training['evidence_sha256'].items():
        require(sha(Path(name)) == expected, 'Training evidence changed: ' + name)
    batch_path = release / 'batch_acceptance.json' if version == 'v1' else BUILD / 'batch_acceptance_v0.json'
    batch = json.loads(batch_path.read_text())
    require(batch['status'] == 'passed' and batch['shared_word_embeddings_preserved'], 'GPU batch acceptance incomplete')
    batch_input = BUILD / 'preprocessed_v1.jsonl' if version == 'v1' else release / 'release_manifest.json'
    require(batch['input_sha256'] == sha(batch_input), 'GPU batch input evidence changed')
    require(any(case['helper_counts'][1:] == [1, 3, 5] and len(case['ids']) == 4 for case in batch['cases']), 'Missing mixed helper-count GPU case')
    for case in batch['cases']:
        require(case['status'] == 'passed' and case['helper_and_question_isolation'] == 'passed', 'GPU isolation check failed')
        require(case['latent_shape'] == [len(case['ids']), 8, 512], 'GPU latent shape differs')
        require(math.isfinite(case['qa_ce']) and math.isfinite(case['description_ce']), 'Nonfinite GPU loss')
    for source, group in sources.items():
        cases = [case for case in batch['cases'] if case['case'].startswith(source + ':')]
        require(bool(cases), 'Source lacks GPU acceptance: ' + source)
        require(max(max(case['helper_counts']) for case in cases) >= max(len(row['auxiliary_images']) for row in group), 'Largest helper count lacks GPU coverage')
        require(max(case['qa_images'] for case in cases) >= max(len(row['problem_images']) for row in group), 'Largest original-image count lacks GPU coverage')
        for key in ('qa', 'encoder', 'description'):
            require(max(max(case['lengths'][key]) for case in cases) >= max(row['preprocessing']['lengths'][key] for row in group), 'Longest branch lacks GPU coverage: ' + source + ':' + key)

    current_errata = []
    errata_path = BUILD / 'pilot_caption_errata.json'
    if version == 'v0' and errata_path.exists():
        declared = json.loads(errata_path.read_text())
        require(declared['manifest_sha256'] == sha(release / 'release_manifest.json'), 'Pilot errata manifest differs')
        current_errata = declared['records']
    withdrawal_path = (release / 'pilot_caption_withdrawals.json') if version == 'v1' else BUILD / 'pilot_caption_withdrawals.json'
    withdrawn_ids = []
    if withdrawal_path.exists():
        declaration = json.loads(withdrawal_path.read_text())
        original_manifest = release.with_name('reasoning_v0') / 'release_manifest.json'
        require(not declaration or declaration['manifest_sha256'] == sha(original_manifest), 'Pilot withdrawal manifest differs')
        withdrawn_ids = sorted(row['id'] for row in declaration.get('records', []))
    return {'status': 'passed', 'release': str(release), 'manifest_sha256': sha(release / 'release_manifest.json'),
            'known_caption_errata': current_errata,
            'withdrawn_pilot_ids': withdrawn_ids,
            'pilot_subset_transition': corrections if version == 'v1' else None,
            'records': len(rows), 'splits': dict(Counter(row['split'] for row in rows)),
            'unique_images_decoded_and_hashed': len(images), 'source_subdivisions': len(sources),
            'source_families': len(families), 'group_split_independence': True,
            'source_identity_split_independence': True,
            'processor_fingerprints_verified': len(rows),
            'graph_caption_parts_checked': graph_parts_checked,
            'description_qa_coverage': 1.0, 'description_helper_coverage': 1.0,
            'published_counts_by_source': {source: len(group) for source, group in sources.items()},
            'training_evidence_sha256_verified': len(training['evidence_sha256']),
            'per_source_gpu_extremes_covered': True,
            'scope': 'Integrity, complete image decoding, schema, split and recorded GPU evidence. Does not prove every caption factual or absence of all semantic overlap.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.release.resolve())
    require(not args.output.resolve().is_relative_to(args.release.resolve()), 'Do not modify an immutable release')
    write_json(args.output, report)
    print('RELEASE_VERIFIED', report['records'], report['unique_images_decoded_and_hashed'], flush=True)


if __name__ == '__main__':
    main()
