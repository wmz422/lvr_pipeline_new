"""Fetch immutable expansion inputs, checking upstream hashes; safe to resume."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
RAW = ROOT / 'data/raw/reasoning_expansion'
sys.path.insert(0, str(ROOT / 'scripts'))
from download_reasoning_data import download, write_json


def fetch(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'lvr-reasoning-build/1'}), timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 8))


def github_blob(repo, rev, entry, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.read_bytes() if target.exists() else fetch(f'https://raw.githubusercontent.com/{repo}/{rev}/'+entry['path'])
    assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == entry['sha'], entry['path']
    if not target.exists():
        tmp=target.with_suffix(target.suffix+'.partial'); tmp.write_bytes(raw); tmp.replace(target)
    return {'path':entry['path'], 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(), 'git_blob_sha1':entry['sha']}


def mentis():
    repo='Jana-Z/mentis-oculi'; rev='c0d486ca77481073d26a8c3447a1e348bb58d39a'
    path=RAW/'mentis-oculi'; path.mkdir(exist_ok=True)
    tree_path=path/'tree.json'
    if not tree_path.exists(): tree_path.write_bytes(fetch(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1'))
    tree=json.loads(tree_path.read_text()); assert not tree.get('truncated') and tree['sha']==rev
    tasks={'form-board','paper-fold','hinge-folding','rushhour','sliding-puzzle'}
    selected=[]
    for e in tree['tree']:
        p=Path(e['path']); parts=p.parts
        if e['type']!='blob' or len(parts)<3 or parts[0]!='datasets' or parts[1] not in tasks: continue
        if p.suffix in ('.py','.toml','.md') or (p.name=='metadata.json' and '/output/level_' in e['path']): selected.append(e)
    # Download sources first so probes can start while benchmark metadata downloads.
    selected.sort(key=lambda e:e['path'].endswith('metadata.json'))
    with cf.ThreadPoolExecutor(12) as pool:
        records=list(pool.map(lambda e:github_blob(repo,rev,e,path/e['path']),selected))
    write_json(path/'download_manifest.json',{'repo_id':repo,'revision':rev,'files':records})
    print('COMPLETE mentis sources and benchmark metadata',len(records),flush=True)


def hf_source(spec):
    repo,rev=spec['repo'],spec['revision']; name=repo.split('/')[-1]
    dest=RAW/name; dest.mkdir(exist_ok=True)
    manifest=dest/'source_lock.json'
    if manifest.exists(): source=json.loads(manifest.read_text())
    else:
        data=json.loads(fetch(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true'))
        assert data['sha']==rev
        files=[]
        for e in data['siblings']:
            p=e['rfilename']
            if p=='README.md' or (p.endswith('.parquet') and (name!='SAT_perspective' or '/val-' in p)):
                f={'path':p,'bytes':e['size']}
                if e.get('lfs'): f['sha256']=e['lfs']['sha256']
                else: f['git_blob_sha1']=e['blobId']
                files.append(f)
        source={'repo_id':repo,'revision':rev,'files':files,'card':data.get('cardData',{})};write_json(manifest,source)
    with cf.ThreadPoolExecutor(4) as pool:
        records=list(pool.map(lambda e:download(source,e,dest,False),source['files']))
    write_json(dest/'download_manifest.json',{**source,'files':records})
    print('COMPLETE',repo,flush=True)


def mirage():
    audit=json.loads((WORK/'doc/2026-09-30-image-only-reasoning-datasets/mirage_audit.json').read_text())
    dest=RAW/'Mirage';dest.mkdir(exist_ok=True)
    for e in audit['archives']:
        target=dest/e['file']
        data=target.read_bytes() if target.exists() else fetch(f"https://raw.githubusercontent.com/{audit['repo']}/{audit['revision']}/data/vsp_spatial_planning/{e['file']}")
        assert hashlib.sha256(data).hexdigest()==e['sha256']
        target.write_bytes(data)
    write_json(dest/'download_manifest.json',audit)
    print('COMPLETE Mirage',flush=True)


if __name__=='__main__':
    meta=json.loads((WORK/'doc/2026-09-30-spatial-planning-data-benchmarks/metadata_audit.json').read_text())
    specs=[r for r in meta['records'] if r['repo'] in ('array/SAT_perspective','ThinkMorph/Jigsaw_Assembly','ThinkMorph/Spatial_Navigation')]
    with cf.ThreadPoolExecutor(5) as pool:
        jobs=[pool.submit(mentis),pool.submit(mirage)]+[pool.submit(hf_source,s) for s in specs]
        for f in cf.as_completed(jobs): f.result()
