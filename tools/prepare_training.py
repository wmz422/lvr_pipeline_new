"""Rebuild the exact VStar-decontaminated, no-SAT training indexes on CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from common import ROOT, PROJECT, write_json
from release import sha256

CLEAN = ROOT / 'paper_experiments/datasets/vstar_decontaminated_v1'
NOSAT = ROOT / 'paper_experiments/2026-10-26-qwen25-no-sat-ablation/data/reasoning_train_without_sat.json'


def read_records(path):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text())
    index = json.loads((path / 'index.json').read_text())
    result = []
    for shard in index['shards']:
        source = path / shard['file']
        if sha256(source) != shard['sha256']:
            raise ValueError(f'Corrupt annotation shard: {source}')
        rows = json.loads(source.read_text())
        if len(rows) != shard['records']:
            raise ValueError('Incorrect shard count')
        result.extend(rows)
    if len(result) != index['records']:
        raise ValueError('Incorrect record count')
    return result


def immutable_write(path, payload, expected=None):
    if expected and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'Rebuilt bytes differ from the frozen experiment: {path}')
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Refuse to replace different data: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        temp.write_bytes(payload)
        temp.replace(path)


def prepare():
    expected = json.loads((ROOT / 'manifests/training_data.json').read_text())
    matches_path = ROOT / 'paper_experiments/2026-10-23-vstar-contamination-audit/results/matches.json'
    if sha256(matches_path) != expected['vstar_matches_sha256']:
        raise ValueError('VStar exclusion evidence changed')
    matches = json.loads(matches_path.read_text())
    blocked = {row['training_image'] for row in matches}
    if len(blocked) != 191:
        raise ValueError('Expected 191 frozen VStar identities')
    counts = {}
    for kind in ('sft', 'align'):
        for split in ('train', 'test'):
            source = PROJECT / 'data' / kind / split
            index = json.loads((source / 'index.json').read_text())
            emitted = {**index, 'shards': [], 'records': 0}
            for spec in index['shards']:
                path = source / spec['file']
                if sha256(path) != spec['sha256']:
                    raise ValueError('Source shard checksum mismatch')
                rows = json.loads(path.read_text())
                kept = [row for row in rows if row['question_image'] not in blocked
                        and '/benchmark/vstar/' not in row['question_image']]
                if not kept:
                    continue
                payload = ('[\n' + ',\n'.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))
                                              for row in kept) + '\n]\n').encode()
                target = CLEAN / kind / split / spec['file']
                immutable_write(target, payload, expected['files'][str(target.relative_to(ROOT))])
                emitted['shards'].append({'file': spec['file'], 'records': len(kept),
                                          'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
                emitted['records'] += len(kept)
            target = CLEAN / kind / split / 'index.json'
            immutable_write(target, (json.dumps(emitted, ensure_ascii=False, indent=2) + '\n').encode(),
                            expected['files'][str(target.relative_to(ROOT))])
            counts[f'{kind}/{split}'] = emitted['records']
    before = read_records(PROJECT / 'data/processed/reasoning_v1/train')
    kept = [row for row in before if row['source'] != 'sat_perspective']
    if len(before) != 55443 or len(kept) != 49518 or len({r['id'] for r in kept}) != 49518:
        raise ValueError('Reasoning split differs from the frozen release')
    immutable_write(NOSAT, (json.dumps(kept, ensure_ascii=False) + '\n').encode(),
                    expected['files'][str(NOSAT.relative_to(ROOT))])
    val = read_records(PROJECT / 'data/processed/reasoning_v1/val')
    if counts != {'sft/train': 122706, 'sft/test': 1241, 'align/train': 122706, 'align/test': 1241} or len(val) != 6330:
        raise ValueError(f'Unexpected split sizes: {counts}, reasoning val={len(val)}')
    report = {'status': 'passed', 'train': 172224, 'val': 7571, 'removed_sat_train': 5925,
              'sat_val': sum(r['source'] == 'sat_perspective' for r in val),
              'reasoning_train_sources': dict(Counter(r['source'] for r in kept)),
              'resolution_filter': None, 'retained_records_and_order_unchanged': True}
    del before, kept, val
    from rebind_lengths import rebind
    report['plain_length_relocation'] = rebind()
    write_json(ROOT / '.local/data_preparation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
