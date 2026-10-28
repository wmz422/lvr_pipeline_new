"""Offline input adapter for the recorded latent-prediction benchmark protocol.

Only local JSONL and saved Arrow datasets are accepted. Native VLMEvalKit
prompts/scorers are retained; no download or hub fallback is permitted.
"""

import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH_ROOT = Path('/workspace/wmz/lvr_pipeline_new/benchmark/bench')
SUPPORTED = ('VStarBench', 'HRBench4K', 'HRBench8K', 'MME-RealWorld-Lite', 'BLINK')
SUFFIX = '\nAnswer with only the option letter.'


def locations():
    source = Path(os.environ.get('LVR_BENCH_ROOT', DEFAULT_BENCH_ROOT)).expanduser().resolve()
    protocol = Path(os.environ.get('LVR_BENCH_PROTOCOL', ROOT / 'benchmark/latent_prediction')).resolve()
    cache = Path(os.environ.get('LMUData', ROOT / 'benchmark/LMUData-local')).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['LMUData'] = str(cache)  # VLMEvalKit ignores this variable unless it exists.
    return source, protocol, cache


def source_files(source, name):
    if name == 'VStarBench':
        files = [source / 'vstar_bench/test_questions.jsonl']
    else:
        subdirs = {'HRBench4K': ['hr_bench_4k'], 'HRBench8K': ['hr_bench_8k'],
                   'MME-RealWorld-Lite': ['mme_realworld_lite'],
                   'BLINK': ['BLINK/Counting/val', 'BLINK/Spatial_Relation/val']}[name]
        files = []
        for subdir in subdirs:
            directory = source / subdir
            state = directory / 'state.json'
            if not state.is_file():
                raise FileNotFoundError(f'Offline benchmark missing: {state}')
            files += [state, directory / 'dataset_info.json']
            files += [directory / x['filename'] for x in json.loads(state.read_text())['_data_files']]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'Offline benchmark missing: {path}; no network fallback')
    return files


def _local_rows(source, name, wanted):
    if name == 'VStarBench':
        for line in (source / 'vstar_bench/test_questions.jsonl').read_text().splitlines():
            item = json.loads(line)
            if str(item['question_id']) not in wanted:
                continue
            text = item['text']
            matches = list(re.finditer(r'(?m)^\(([A-Z])\) (.*)$', text))
            if not matches:
                raise ValueError(f'Invalid local VStar options: {item["question_id"]}')
            row = dict(index=str(item['question_id']), question=text[:matches[0].start()].rstrip(),
                       answer=item['label'], category=item['category'])
            row.update({m[1]: m[2] for m in matches})
            yield row, [source / 'vstar_bench' / item['image']]
        return
    from datasets import Image, load_from_disk
    folders = {'HRBench4K': ['hr_bench_4k'], 'HRBench8K': ['hr_bench_8k'],
               'MME-RealWorld-Lite': ['mme_realworld_lite'],
               'BLINK': ['BLINK/Counting/val', 'BLINK/Spatial_Relation/val']}[name]
    for folder in folders:
        dataset = load_from_disk(str(source / folder))
        id_column = 'idx' if name == 'BLINK' else 'index'
        positions = [i for i, value in enumerate(dataset[id_column]) if str(value) in wanted]
        if name == 'BLINK':
            for key in ('image_1', 'image_2', 'image_3', 'image_4'):
                dataset = dataset.cast_column(key, Image(decode=False))
        for position in positions:
            item = dataset[position]
            index = str(item[id_column])
            if name == 'BLINK':
                row = dict(index=index, question=item['question'] + '\nSelect from the following choices.',
                           answer=item['answer'].strip('()'), category=folder.split('/')[1])
                row.update({chr(65 + i): choice for i, choice in enumerate(item['choices'])})
                images = [item[key] for key in ('image_1', 'image_2', 'image_3', 'image_4') if item[key] is not None]
            elif name == 'MME-RealWorld-Lite':
                options = item['multi-choice options']
                row = {key: item[key] for key in ('question', 'answer', 'category', 'l2-category')}
                row.update(index=index, **{'multi-choice options': 'The choices are listed below:\n' + '\n'.join(options)})
                row.update({chr(65 + i): option[4:] for i, option in enumerate(options)})
                cached = source / folder / 'images' / f'{index}.jpg'
                images = [cached if cached.is_file() else item['bytes']]
            else:
                row = {key: value for key, value in item.items() if key != 'image'}
                row['index'] = index
                cached = source / folder / 'images' / f'{index}.jpg'
                images = [cached if cached.is_file() else item['image']]
            yield row, images


def _materialize(value, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Path):
        if not value.is_file():
            raise FileNotFoundError(f'Offline image missing: {value}')
        if target.is_symlink() and target.resolve() == value.resolve():
            return
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Refusing to replace existing image: {target}; use a fresh LMUData directory')
        target.symlink_to(value.resolve())
        return
    if isinstance(value, dict):
        if value.get('bytes') is None:
            return _materialize(Path(value['path']), target)
        raw = value['bytes']
    else:
        raw = base64.b64decode(value, validate=True) if isinstance(value, str) else value
    if target.is_file():
        if target.read_bytes() == raw:
            return
        raise FileExistsError(f'Image content mismatch: {target}; use a fresh LMUData directory')
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_bytes(raw)
    temporary.replace(target)


def build_local_dataset(name):
    """Load a native scorer backed exclusively by locally prepared metadata/images."""
    from pandas import DataFrame
    from vlmeval.dataset.image_mcq import HRBenchDataset, ImageMCQDataset, MMERealWorld

    source, protocol, cache = locations()
    spec = json.loads((protocol / 'data_config.json').read_text())[name]
    annotations = {row['index']: row for row in json.loads((protocol / 'crop_annotations.json').read_text())['samples'].values()
                   if row['benchmark'] == name}
    wanted = {str(index) for index in spec['indices']}
    if set(annotations) != wanted:
        raise ValueError(f'Protocol IDs and annotations disagree: {name}')
    files = source_files(source, name)
    fingerprint = dict(version=1, source=str(source), protocol=str(protocol),
                       protocol_sha256=hashlib.sha256((protocol / 'crop_annotations.json').read_bytes()
                                                     + (protocol / 'data_config.json').read_bytes()).hexdigest(),
                       files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files])
    fingerprint = json.loads(json.dumps(fingerprint))
    metadata = cache / 'local_protocol' / f'{name}.json'
    metadata.parent.mkdir(parents=True, exist_ok=True)
    base = HRBenchDataset if name.startswith('HRBench') else MMERealWorld if name == 'MME-RealWorld-Lite' else ImageMCQDataset

    class LocalDataset(base):
        def load_data(self, dataset):
            return DataFrame(rows)

    with metadata.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = json.loads(metadata.read_text()) if metadata.is_file() else {}
        if saved.get('fingerprint') == fingerprint:
            rows = saved['rows']
        else:
            rows = []
            print(f'[local-benchmark] preparing {name}: {len(wanted)} questions from {source}', flush=True)
            for row, images in _local_rows(source, name, wanted):
                annotation = annotations[str(row['index'])]
                if len(images) != len(annotation['images']):
                    raise ValueError(f'Image count mismatch: {name}/{row["index"]}')
                paths = []
                for image, recorded in zip(images, annotation['images']):
                    relative = Path(recorded['path'])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError(f'Unsafe protocol image path: {relative}')
                    target = cache / relative
                    _materialize(image, target)
                    paths.append(str(target))
                row['image_path'] = paths
                rows.append(row)
        if len(rows) != len(wanted) or {str(row['index']) for row in rows} != wanted:
            raise ValueError(f'Missing/duplicate local questions: {name}')
        dataset = LocalDataset(dataset=name)
        for _, row in dataset.data.iterrows():
            annotation = annotations[str(row['index'])]
            prompt = dataset.build_prompt(row)
            question = '\n'.join(part['value'] for part in prompt if part['type'] == 'text') + SUFFIX
            if question != annotation['qa_prompt']:
                raise ValueError(f'Local prompt differs from recorded protocol: {name}/{row["index"]}')
            for path in row['image_path'] if isinstance(row['image_path'], list) else [row['image_path']]:
                if not Path(path).is_file():
                    raise FileNotFoundError(f'Offline image missing: {path}')
        if saved.get('fingerprint') != fingerprint:
            temporary = metadata.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(fingerprint=fingerprint, rows=rows), ensure_ascii=False) + '\n')
            temporary.replace(metadata)
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-root', type=Path, default=os.environ.get('LVR_BENCH_ROOT', DEFAULT_BENCH_ROOT))
    parser.add_argument('--protocol', type=Path, default=os.environ.get('LVR_BENCH_PROTOCOL', ROOT / 'benchmark/latent_prediction'))
    parser.add_argument('--cache', type=Path, default=os.environ.get('LMUData', ROOT / 'benchmark/LMUData-local'))
    parser.add_argument('--datasets', nargs='+', choices=SUPPORTED, default=list(SUPPORTED))
    args = parser.parse_args()
    os.environ.update(LVR_BENCH_ROOT=str(args.bench_root.resolve()), LVR_BENCH_PROTOCOL=str(args.protocol.resolve()),
                      LMUData=str(args.cache.resolve()), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    counts = {name: len(build_local_dataset(name)) for name in args.datasets}
    print(json.dumps(dict(status='local_data_ready', counts=counts, total=sum(counts.values())), indent=2), flush=True)


if __name__ == '__main__':
    main()
