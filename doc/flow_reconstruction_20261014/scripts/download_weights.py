#!/usr/bin/env python3
"""Download only immutable Klein Base image components; never load CUDA/models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = 'black-forest-labs/FLUX.2-klein-base-4B'
REVISION = 'a3b4f4849157f664bdbc776fd7453c2783562f4d'
REQUIRED = ['model_index.json', 'LICENSE.md', 'README.md',
            'scheduler/scheduler_config.json', 'transformer/config.json',
            'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
            'transformer/diffusion_pytorch_model.safetensors']

def curl(url, destination, *, resume=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
               '--retry', '3', '--retry-delay', '3', '--connect-timeout', '20',
               '--speed-time', '120', '--speed-limit', '1024']
    if resume and destination.exists():
        command += ['--continue-at', '-']
    subprocess.run(command + [url, '--output', str(destination)], check=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def check(path, spec):
    if not path.is_file() or path.stat().st_size != spec['size']: return None
    sha = sha256_file(path)
    if spec.get('lfs'):
        if sha != spec['lfs']['sha256']: return None
    else:
        content = path.read_bytes()
        git_sha = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if git_sha != spec['blobId']: return None
    return sha

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metadata', type=Path)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--endpoint', default='https://huggingface.co', help='Mirror URL; immutable revision and checksums stay enforced')
    a = p.parse_args(); a.endpoint=a.endpoint.rstrip('/'); a.output.mkdir(parents=True, exist_ok=True)
    if a.metadata:
        metadata = json.loads(a.metadata.read_text())
    else:
        tmp = a.output/'model_metadata.json.partial'
        curl(f'{a.endpoint}/api/models/{REPO}/revision/{REVISION}?blobs=true', tmp)
        metadata = json.loads(tmp.read_text())
        tmp.replace(a.output/'model_metadata.json')
    if metadata['sha'] != REVISION: raise ValueError('Wrong immutable model revision')
    specs = {s['rfilename']:s for s in metadata['siblings']}
    remaining = sum(specs[n]['size'] for n in REQUIRED if not (a.output/n).is_file())
    if not a.verify_only and shutil.disk_usage(a.output).free < remaining + 2*1024**3:
        raise RuntimeError('Need enough disk for missing weights plus 2 GiB reserve')
    lock = {'repo_id':REPO, 'revision':REVISION, 'text_encoder_downloaded':False,
            'components':['transformer','vae','scheduler'], 'files':{}, 'status':'downloading'}
    lockpath = a.output/'weights_lock.json'
    def save():
        temp=lockpath.with_suffix('.tmp');temp.write_text(json.dumps(lock,indent=2));temp.replace(lockpath)
    save()
    for name in REQUIRED:
        path = a.output/name; spec = specs[name]
        digest = check(path,spec)
        if digest is None:
            if a.verify_only: raise ValueError(f'Missing/corrupt component: {name}')
            if path.exists(): raise ValueError(f'Refusing to overwrite corrupt existing file: {path}')
            partial = path.with_name(path.name+'.partial')
            print(json.dumps({'event':'download_start','file':name,'bytes':spec['size']}),flush=True)
            curl(f'{a.endpoint}/{REPO}/resolve/{REVISION}/{name}',partial,resume=True)
            digest = check(partial,spec)
            if digest is None: raise ValueError(f'Download checksum/size mismatch: {name}')
            partial.replace(path)
        lock['files'][name]={'bytes':spec['size'],'sha256':digest};save()
        print(json.dumps({'event':'verified','file':name,'bytes':spec['size']}),flush=True)
    lock['status']='complete';lock['verified_unix']=time.time();save()
    print(json.dumps({'status':'complete','path':str(a.output.resolve()),'revision':REVISION}),flush=True)

if __name__=='__main__': main()
