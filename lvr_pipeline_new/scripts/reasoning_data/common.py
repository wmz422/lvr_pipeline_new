"""Shared, deterministic assets and candidate serialization."""
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import uuid
from PIL import Image
import numpy as np
from scipy.fft import dctn

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'data/processed/reasoning_build'
RAW=ROOT/'data/raw/reasoning_expansion'
CANDIDATES=BUILD/'candidates'
REV='c0d486ca77481073d26a8c3447a1e348bb58d39a'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def normalized(text):
    text=re.sub(r'<image_start>.*?<image_end>|<image\d*>','',text)
    return re.sub(r'[^\w]','',unicodedata.normalize('NFKC',text).lower())


def clean_question(text):
    return re.sub(r'<image_start>\[.*?\]<image_end>|<image>|\[(?:problem|reasoning)_image_\d+\]','',text).strip()


def asset(payload,*,step=None,role=None,name=None):
    if isinstance(payload,(str,Path)):payload=Path(payload).read_bytes()
    sha=hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as im:
        im.load(); rgb=im.convert('RGB'); width,height=rgb.size
        if width<16 or height<16:raise ValueError('image_too_small')
        ext={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(im.format,'.image')
        pixel_sha=hashlib.sha256(f'{width}x{height}:'.encode()+rgb.tobytes()).hexdigest()
        a=dctn(np.asarray(rgb.resize((32,32)).convert('L'),dtype=float),norm='ortho')[:8,:8]
        bits=a>np.median(a.reshape(-1)[1:]); phash=f'{int("".join("1" if b else "0" for b in bits.flatten()),2):016x}'
    p=BUILD/'assets'/sha[:2]/(sha+ext);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        temp=p.with_suffix(p.suffix+'.'+uuid.uuid4().hex+'.partial');temp.write_bytes(payload);temp.replace(p)
    result={'path':str(p.relative_to(ROOT)),'sha256':sha,'pixel_sha256':pixel_sha,'phash':phash,'width':width,'height':height}
    if step is not None:result['step']=step
    if role:result['role']=role
    if name:result['name']=name
    return result


def candidate(source,revision,sid,task,category,question,answer,problem,helpers,group,**extra):
    if not question.strip():raise ValueError('empty_question')
    if not str(answer).strip():raise ValueError('empty_answer')
    if not problem:raise ValueError('no_problem_image')
    if not helpers:raise ValueError('no_auxiliary_image')
    if all(h['pixel_sha256'] in {p['pixel_sha256'] for p in problem} for h in helpers):raise ValueError('helpers_only_repeat_inputs')
    return {'schema':'lvr-reasoning-qa-v1','id':f'{source}/{sid}','source':source,'source_revision':revision,
            'category':category,'task':task,'difficulty':extra.pop('difficulty',None),'question':clean_question(question),
            'answer':str(answer).strip(),'answer_type':extra.pop('answer_type','exact'),
            'problem_images':problem,'auxiliary_images':helpers,'teacher_view_mode':'auxiliary_images',
            'group_id':group,'source_image_id':group,'description':'','description_parts':[],
            'description_valid':False,'description_provenance':{},'split':'candidate','quality':{'status':'pending_caption_and_review'},**extra}


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
