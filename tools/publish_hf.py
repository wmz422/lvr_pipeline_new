"""Publish the immutable data-v1 archives and verify the public Hugging Face snapshot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from common import ROOT, write_json
from release import load_lock, sha256


def retry_network(function, *args, **kwargs):
    import httpx
    for attempt in range(5):
        try:
            return function(*args, **kwargs)
        except httpx.TransportError:
            if attempt == 4:
                raise
            print('Retrying a transient network connection error', flush=True)
            time.sleep(2 ** attempt)


def prepare(directory, repo_id):
    lock_path = ROOT / 'manifests/data_release.json'
    lock, rows = load_lock(lock_path)
    names = [s['file'] for s in lock['archives']] + [lock_path.name, lock['files_manifest']]
    for spec in lock['archives']:
        path = directory / spec['file']
        if path.stat().st_size != spec['bytes']:
            raise ValueError(f'Archive size changed: {path}')
    for name in (lock_path.name, lock['files_manifest']):
        source = ROOT / 'manifests' / name
        target = directory / name
        if target.exists() and sha256(source) != sha256(target):
            raise ValueError(f'Release metadata differs: {name}')
        if not target.exists():
            shutil.copyfile(source, target)
    for name in ('sources.json', 'reasoning_sources.json'):
        shutil.copyfile(ROOT / 'lvr_pipeline_new/data' / name, directory / name)
        names.append(name)
    shutil.copyfile(ROOT / 'manifests/benchmark_downloads.json', directory / 'benchmark_downloads.json')
    names.append('benchmark_downloads.json')
    card = (ROOT / 'docs/dataset_card.md').read_text().replace('${DATASET_REPO}', repo_id)
    (directory / 'README.md').write_text(card)
    names.append('README.md')
    checksums = {s['file']: s['sha256'] for s in lock['archives']}
    checksums.update({name: sha256(directory / name) for name in names if name not in checksums})
    (directory / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(checksums.items())))
    checksums['SHA256SUMS'] = sha256(directory / 'SHA256SUMS')
    names.append('SHA256SUMS')
    return lock, names, checksums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='Wing22/qwen25-flow-repro-data-v1')
    parser.add_argument('--directory', type=Path, default=ROOT / 'artifacts/data-v1')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    lock, names, checksums = prepare(args.directory, args.repo_id)
    if args.prepare_only:
        print(json.dumps({'repo_id': args.repo_id, 'public': True, 'upload_files': names,
                          'archive_bytes': sum(s['bytes'] for s in lock['archives'])}, indent=2))
        return
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    api = HfApi()
    identity = retry_network(api.whoami)['name']
    if identity.lower() != args.repo_id.split('/')[0].lower():
        raise ValueError('Use the account owning the requested dataset namespace')
    state_path = ROOT / '.local/publication/hf_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    binding = {'repo_id': args.repo_id, 'release_sha256': sha256(ROOT / 'manifests/data_release.json')}
    if state and any(state.get(k) != v for k, v in binding.items()):
        raise ValueError('Publication state belongs to a different release')
    try:
        info = retry_network(api.repo_info, args.repo_id, repo_type='dataset')
    except RepositoryNotFoundError:
        api.create_repo(args.repo_id, repo_type='dataset', private=False, exist_ok=False)
        state = {**binding, 'created_by_this_publication': True}
        write_json(state_path, state)
        info = retry_network(api.repo_info, args.repo_id, repo_type='dataset')
    if info.private:
        raise ValueError('Existing target is private; refusing an implicit visibility change')
    existing = {f.rfilename for f in info.siblings}
    if existing - set(names) - {'.gitattributes'}:
        raise ValueError('Existing dataset contains files outside this release')
    if 'data_release.json' in existing:
        remote_lock = Path(retry_network(hf_hub_download, args.repo_id, 'data_release.json', repo_type='dataset', revision=info.sha))
        if sha256(remote_lock) != binding['release_sha256']:
            raise ValueError('Refuse to overwrite another data release')
    elif not state.get('created_by_this_publication'):
        raise ValueError('Existing repository is not a recognized resumable publication')
    # Hash the immutable archives before starting uploads; use only the allowlisted files.
    for spec in lock['archives']:
        if sha256(args.directory / spec['file']) != spec['sha256']:
            raise ValueError(f'Archive checksum mismatch: {spec["file"]}')
        print('VERIFIED LOCAL', spec['file'], flush=True)
    api.upload_large_folder(args.repo_id, folder_path=args.directory, repo_type='dataset',
                            private=False, allow_patterns=names, num_workers=args.workers,
                            print_report=True, print_report_every=30)
    public_api = HfApi(token=False)
    info = retry_network(public_api.repo_info, args.repo_id, repo_type='dataset', files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    for name in names:
        item = remote.get(name)
        if item is None or item.size != (args.directory / name).stat().st_size:
            raise ValueError(f'Remote file missing or wrong size: {name}')
        if item.lfs:
            if item.lfs.sha256 != checksums[name]:
                raise ValueError(f'Remote object checksum mismatch: {name}')
        else:
            path = Path(retry_network(hf_hub_download, args.repo_id, name, repo_type='dataset', revision=info.sha,
                                        token=False, force_download=True))
            if sha256(path) != checksums[name]:
                raise ValueError(f'Remote file checksum mismatch: {name}')
    receipt = {**binding, 'status': 'complete', 'private': info.private, 'revision': info.sha,
               'url': f'https://huggingface.co/datasets/{args.repo_id}',
               'download_base_url': f'https://huggingface.co/datasets/{args.repo_id}/resolve/{info.sha}',
               'files_verified': len(names), 'archives': len(lock['archives']),
               'archive_bytes': sum(s['bytes'] for s in lock['archives']),
               'verification': 'Unauthenticated pinned snapshot: all sizes and SHA256; LFS objects use server-reported SHA256'}
    write_json(ROOT / '.local/publication/huggingface.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
