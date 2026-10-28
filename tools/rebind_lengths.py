"""Relocate the Plain length fingerprint without recomputing or changing any length."""
import gzip
import json
from pathlib import Path

from common import ROOT, PROJECT
from release import sha256


def rebind():
    from lvr.data.crop_qa import CropQADataset
    from lvr.data.length_manifest import plain_qa_items_fingerprint
    from prepare_training import CLEAN, NOSAT, immutable_write
    bindings = json.loads((ROOT / 'manifests/training_data.json').read_text())
    relative = 'paper_experiments/2026-10-26-qwen25-no-sat-ablation/data/plain_qa_lengths.json.gz'
    source = ROOT / relative
    if sha256(source) != bindings['files'][relative]:
        raise ValueError('Frozen length sidecar changed')
    original = json.loads(gzip.decompress(source.read_bytes()))
    items = CropQADataset(CLEAN / 'sft/train', CLEAN / 'align/train', caption_path=None).items
    items += CropQADataset(NOSAT).items
    def original_paths():
        for item in items:
            if item.get('images'):
                images = [{**im, 'path': bindings['length_fingerprint_project_root'] + '/' +
                           str(Path(im['path']).relative_to(PROJECT))} for im in item['images']]
                yield {**item, 'images': images}
            else:
                yield item
    if (len(items) != original['records'] or
            plain_qa_items_fingerprint(original_paths()) != original['items_sha256']):
        raise ValueError('Rows or ordering differ from the measured length sidecar; cannot rebind')
    relocated = {**original, 'items_sha256': plain_qa_items_fingerprint(items)}
    output = ROOT / '.local/plain_qa_lengths.json.gz'
    immutable_write(output, gzip.compress(json.dumps(relocated, sort_keys=True).encode(), mtime=0))
    return {'status': 'passed', 'records': len(items), 'lengths_unchanged': True,
            'original_items_sha256': original['items_sha256'], 'items_sha256': relocated['items_sha256'],
            'path': str(output)}


if __name__ == '__main__':
    print(json.dumps(rebind(), indent=2))
