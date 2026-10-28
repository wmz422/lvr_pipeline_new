"""Deterministic RGB GT construction. This module does not import torch."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
from PIL import Image, ImageDraw

@dataclass(frozen=True)
class GTConfig:
    crop_pixels: int = 262144
    masked_original_pixels: int = 1048576
    helper_pixels: int = 262144
    reference_pixels: int = 1048576
    target_max_side: int = 2048
    reference_max_side: int = 2048
    alignment: int = 16
    mask_color: str = 'black'
    padding_color: str = 'white'

    def __post_init__(self):
        if min(self.crop_pixels,self.masked_original_pixels,self.helper_pixels,
               self.reference_pixels,self.target_max_side,self.reference_max_side,self.alignment)<1:
            raise ValueError('Pixel/side/alignment budgets must be positive')
        if self.mask_color not in ('black','white') or self.padding_color not in ('black','white'):
            raise ValueError('Use a fixed black or white background')

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def read_records(path):
    path=Path(path)
    if path.is_dir():path=path/'index.json'
    value=json.loads(path.read_text())
    if isinstance(value,list):return value
    if value.get('format')!='lvr-json-shards-v1':raise ValueError(f'Unknown record index: {path}')
    result=[]
    for spec in value['shards']:
        raw=(path.parent/spec['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Corrupt shard')
        rows=json.loads(raw)
        if len(rows)!=spec['records']:raise ValueError('Incorrect shard count')
        result.extend(rows)
    if len(result)!=value['records']:raise ValueError('Incorrect split count')
    return result

def load_protocol(path):
    value=json.loads(Path(path).read_text())
    expected=value.pop('protocol_sha256')
    actual=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if expected!=actual:raise ValueError('GT protocol content/hash differs')
    value['protocol_sha256']=expected
    return value

def crop_bounds(size,box):
    w,h=size
    if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):raise ValueError('Invalid bbox')
    x1,y1,x2,y2=map(float,box)
    a,b=max(0,math.floor(min(x1,x2))),max(0,math.floor(min(y1,y2)))
    c,d=min(w,math.ceil(max(x1,x2))),min(h,math.ceil(max(y1,y2)))
    if x1==x2 and 0<=x1<w:c=a+1
    if y1==y2 and 0<=y1<h:d=b+1
    if c<=a or d<=b:raise ValueError(f'Empty bbox {box} for {size}')
    return a,b,c,d

def resized_size(size,pixels,max_side,*,enlarge=False):
    w,h=size
    scale=min(math.sqrt(pixels/(w*h)),max_side/max(w,h))
    if not enlarge:scale=min(scale,1.)
    return max(1,round(w*scale)),max(1,round(h*scale))

def resize(im,pixels,max_side,*,enlarge=False):
    size=resized_size(im.size,pixels,max_side,enlarge=enlarge)
    return im.resize(size,Image.Resampling.LANCZOS) if size!=im.size else im.copy()

def pad_aligned(im,alignment,color):
    w,h=im.size
    out=Image.new('RGB',(((w+alignment-1)//alignment)*alignment,
                         ((h+alignment-1)//alignment)*alignment),color)
    out.paste(im,(0,0))
    return out,[0,0,w,h]

def single_crop(original,box,cfg):
    bb=crop_bounds(original.size,box)
    image=resize(original.crop(bb),cfg.crop_pixels,cfg.target_max_side,enlarge=True)
    image,valid=pad_aligned(image,cfg.alignment,cfg.padding_color)
    return image,{'kind':'single_crop','boxes_pixels':[list(bb)],'valid_rect':valid,'tiles':1}

def multiple_crop(original,boxes,cfg):
    bounds=[crop_bounds(original.size,b) for b in boxes]
    im=resize(original,cfg.masked_original_pixels,cfg.target_max_side)
    mask=Image.new('L',original.size,0);draw=ImageDraw.Draw(mask)
    for a,b,c,d in bounds:draw.rectangle((a,b,c-1,d-1),fill=255)
    mask=mask.resize(im.size,Image.Resampling.NEAREST)
    canvas=Image.new('RGB',im.size,cfg.mask_color);canvas.paste(im,(0,0),mask)
    canvas,valid=pad_aligned(canvas,cfg.alignment,cfg.mask_color)
    return canvas,{'kind':'masked_original','boxes_pixels':[list(b) for b in bounds],
                   'valid_rect':valid,'tiles':1,'mask_color':cfg.mask_color,
                   'roi_pixels_before_padding':sum(mask.histogram()[128:])}

def horizontal_helpers(images,cfg):
    if not images:raise ValueError('Empty auxiliary image list')
    unequal=len({im.size for im in images})>1
    box=(max(im.width for im in images),max(im.height for im in images))
    size=resized_size(box,cfg.helper_pixels,cfg.target_max_side)
    tiles=[];rects=[]
    for i,im in enumerate(images):
        # Equal images are untouched except shared budget resize. Exceptions use a
        # common enclosing canvas; no stretching, reordering, or duplicate removal.
        tile=Image.new('RGB',box,cfg.padding_color)
        tile.paste(im,((box[0]-im.width)//2,(box[1]-im.height)//2))
        if tile.size!=size:tile=tile.resize(size,Image.Resampling.LANCZOS)
        tiles.append(tile);rects.append([i*size[0],0,(i+1)*size[0],size[1]])
    out=Image.new('RGB',(size[0]*len(tiles),size[1]),cfg.padding_color)
    for i,tile in enumerate(tiles):out.paste(tile,(i*size[0],0))
    out,valid=pad_aligned(out,cfg.alignment,cfg.padding_color)
    return out,{'kind':'horizontal_auxiliary','tiles':len(tiles),'tile_size':list(size),
                'tile_rects':rects,'valid_rect':valid,'unequal_source_sizes':unequal}

def read_rgb(path,expected_sha=None,padding_color='white'):
    path=Path(path);stat=path.stat()
    return _cached_rgb(str(path.resolve()),stat.st_mtime_ns,stat.st_size,expected_sha,padding_color)

@lru_cache(maxsize=64)
def _cached_rgb(path,mtime,size,expected_sha,padding_color):
    raw=Path(path).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if expected_sha and sha!=expected_sha:raise ValueError(f'Source image hash mismatch: {path}')
    with Image.open(io.BytesIO(raw)) as handle:
        # Match the established teacher bbox coordinate space: do not auto-rotate
        # EXIF. Transparent files are explicitly composited, never dropped to black.
        if handle.mode=='RGBA' or 'transparency' in handle.info:
            rgba=handle.convert('RGBA');im=Image.new('RGBA',rgba.size,padding_color)
            im.alpha_composite(rgba);im=im.convert('RGB')
        else:im=handle.convert('RGB')
        im.load()
    return im,sha

def persist_image(image,root,group):
    pixel_sha=hashlib.sha256(str(image.size).encode()+b'\0'+image.tobytes()).hexdigest()
    relative=Path(group)/pixel_sha[:2]/(pixel_sha+'.png');path=Path(root)/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
        image.save(tmp,format='PNG',compress_level=0);tmp.replace(path)
    return {'path':str(relative),'width':image.width,'height':image.height,
            'pixel_sha256':pixel_sha,'sha256':digest_file(path)}

def samples(root,image_root,split):
    root,image_root=Path(root),Path(image_root);oldsplit='train' if split=='train' else 'test'
    qa=read_records(root/'data/sft'/oldsplit);align=read_records(root/'data/align'/oldsplit)
    if len(qa)!=len(align):raise ValueError('QA/annotation count mismatch')
    for q,a in zip(qa,align):
        if q['question_image']!=a['question_image'] or q['auxiliary_image']!=a['auxiliary_image']:
            raise ValueError('QA/bbox ordering mismatch')
        if not a['boxes']:raise ValueError('Empty legacy bbox list')
        yield {'id':f"{q['question_image']}:{q['raw']['index']}",'source':q['raw']['dataset'],
               'refs':[{'path':str(image_root/q['question_image'])}], 'boxes':a['boxes'],'helpers':[]}
    for row in read_records(root/'data/processed/reasoning_v1'/split):
        hs=row['auxiliary_images']
        if row.get('teacher_view_mode')!='auxiliary_images' or not hs:raise ValueError('Invalid helper mode')
        if [h['step'] for h in hs]!=list(range(1,len(hs)+1)):raise ValueError('Wrong helper order')
        if not row['problem_images']:raise ValueError('Empty reference list')
        yield {'id':row['id'],'source':row['source'],
               'refs':[{**r,'path':str(root/r['path'])} for r in row['problem_images']],
               'helpers':[{**h,'path':str(root/h['path'])} for h in hs],'boxes':[]}

@lru_cache(maxsize=2048)
def _prepared_reference(path,mtime,size,expected_sha,output,cfg):
    im,_=read_rgb(path,expected_sha,cfg.padding_color)
    ref=resize(im,cfg.reference_pixels,cfg.reference_max_side)
    ref,_=pad_aligned(ref,cfg.alignment,cfg.padding_color)
    return persist_image(ref,output,'references')

def process_sample(sample,output,cfg):
    refs=[];source_hashes=[];originals=[]
    for r in sample['refs']:
        im,sha=read_rgb(r['path'],r.get('sha256'),cfg.padding_color)
        originals.append(im);source_hashes.append({'path':r['path'],'sha256':sha,'role':'reference'})
        stat=Path(r['path']).stat()
        refs.append(_prepared_reference(r['path'],stat.st_mtime_ns,stat.st_size,r.get('sha256'),str(output),cfg))
    if sample['helpers']:
        images=[]
        for h in sample['helpers']:
            im,sha=read_rgb(h['path'],h.get('sha256'),cfg.padding_color)
            images.append(im);source_hashes.append({'path':h['path'],'sha256':sha,'role':'helper'})
        target,layout=horizontal_helpers(images,cfg)
    elif len(originals)!=1:
        raise ValueError('Legacy crop GT requires exactly one original image')
    elif len(sample['boxes'])==1:target,layout=single_crop(originals[0],sample['boxes'][0],cfg)
    else:target,layout=multiple_crop(originals[0],sample['boxes'],cfg)
    return {'schema':'lvr-flow-gt-v1','id':sample['id'],'source':sample['source'],
            'references':refs,'target':persist_image(target,output,'targets'),
            'layout':layout,'source_hashes':source_hashes}
