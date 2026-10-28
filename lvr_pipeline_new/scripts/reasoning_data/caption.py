"""Deterministic Qwen3-VL per-image captions, with durable hash/config keyed cache."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from common import ROOT,BUILD,CANDIDATES,digest

MODEL_REV='ebb281ec70b05090aa6165b016eac8ec08e71b17'
MODEL=Path('/workspace/wmz/.cache/Qwen3-VL-4B-Instruct')
PROMPT='Write exactly ONE short factual sentence, at most 35 words, describing the visible image. For a geometric diagram, describe the dominant colors, outlines, and whether colored regions are separated, touching, or overlapping. Call angular colored regions polygons; avoid naming polygon subtypes or letter-shaped forms. For other images, describe the main visible objects, scene, or grid and prominent colors. Omit exact counts, angles, motion, arrow directions, uncertain small details, explanations, and answer conclusions. Describe only what is visibly present.'
CONFIG={'model':'Qwen/Qwen3-VL-4B-Instruct','model_revision':MODEL_REV,'prompt_version':'visible-state-v4',
        'prompt':PROMPT,'do_sample':False,'max_new_tokens':112,'max_pixels':1048576,'min_pixels':262144}
CONFIG_ID=digest(CONFIG)
PRIMARY_CONFIG_ID=CONFIG_ID
RETRY_PROMPT='Write one short English sentence identifying the clearly visible colors and kinds of visual elements in this image. Mention textures or clearly readable text only when useful. Focus on what is present, without describing how shapes touch, overlap, connect, align, or move. Avoid counts, angles, directions, position claims, interpretations, and task solutions. Use at most 25 words.'
RETRY_CONFIG={**CONFIG,'prompt_version':'visible-elements-retry-v1','prompt':RETRY_PROMPT,'max_new_tokens':96}
RETRY_CONFIG_ID=digest(RETRY_CONFIG)


def clean(text,ended):
    if '<think>' in text and '</think>' not in text:return '', 'unclosed_think'
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S)
    text=' '.join(text.split())
    if not text:return text,'empty'
    if not ended:return text,'truncated'
    letters=[c for c in text if c.isalpha()]
    # English sentences can quote non-Latin diagram labels. Reject outputs
    # whose main text switched language, while allowing short visible labels.
    if not letters or sum(c.isascii() for c in letters)/len(letters)<.75:return text,'non_latin_dominant_caption'
    if len(text)<20 or re.search(r'\b(cannot (see|view|describe)|unable to|as an ai|correct answer|the answer is|therefore|thus,|in conclusion)\b',text,re.I):return text,'invalid_or_conclusion'
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',text)
    if len(sentences)>3:return text,'too_many_sentences'
    return text,None


def collect(limit):
    images={}
    pilot=BUILD/'pilot_ids.json'
    selected=set(json.loads(pilot.read_text())['ids']) if limit and pilot.exists() else None
    invalid=set()
    for name in ('answers.rejected.jsonl','thinkmorph_navigation.rejected.jsonl','overlap_decisions.jsonl'):
        path=BUILD/'quality'/name
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line)
                if name!='overlap_decisions.jsonl' or r['split']=='quarantine':invalid.add(r['id'])
    for p in sorted(CANDIDATES.glob('*.jsonl')):
        if '.rejected.' in p.name or '.building.' in p.name:continue
        count=0
        with p.open() as f:
            for line in f:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue # Last line of a live candidate writer.
                if 'auxiliary_images' not in r or r['id'] in invalid or (selected is not None and r['id'] not in selected):continue
                for im in r['auxiliary_images']:images[im['sha256']]=im
                count+=1
                if selected is None and limit and count>=limit:break
    return images


def main():
    global CONFIG,CONFIG_ID,PROMPT
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--gpu',type=int);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--per-source',type=int,default=100)
    p.add_argument('--max-images',type=int,default=0);p.add_argument('--retry',action='store_true');a=p.parse_args()
    if a.retry:CONFIG=RETRY_CONFIG;CONFIG_ID=RETRY_CONFIG_ID;PROMPT=RETRY_PROMPT
    out=BUILD/'captions'/CONFIG_ID;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2))
    cached={}
    for path in out.glob('worker_*.jsonl'):
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            cached[r['image_sha256']]=r
    source={im['sha256']:im for im in json.loads((BUILD/f'caption_retry_queue_{PRIMARY_CONFIG_ID}.json').read_text())['images']} if a.retry else collect(a.per_source)
    images=[im for sha,im in source.items() if int(sha,16)%a.workers==a.worker and sha not in cached]
    if a.max_images:images=images[:a.max_images]
    print('CAPTION_QUEUE',a.worker,len(images),CONFIG_ID,flush=True)
    if not images:return
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(2);device=f'cuda:{a.worker if a.gpu is None else a.gpu}'
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    processor.tokenizer.padding_side='left';processor.image_processor.max_pixels=CONFIG['max_pixels']
    processor.image_processor.min_pixels=CONFIG['min_pixels']
    if hasattr(processor.image_processor,'size'):
        processor.image_processor.size.longest_edge=CONFIG['max_pixels']
        processor.image_processor.size.shortest_edge=CONFIG['min_pixels']
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}],tokenize=False,add_generation_prompt=True)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    with (out/f'worker_{a.worker:02d}.jsonl').open('a') as f:
        for offset in range(0,len(images),a.batch_size):
            batch=images[offset:offset+a.batch_size];start=time.monotonic();pil=[]
            for im in batch:
                with Image.open(ROOT/im['path']) as x:pil.append(x.convert('RGB'))
            inputs=processor(text=[prompt]*len(batch),images=pil,padding=True,return_tensors='pt').to(device)
            with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=CONFIG['max_new_tokens'])
            ids=generated[:,inputs.input_ids.shape[1]:].cpu().tolist()
            texts=processor.batch_decode(ids,skip_special_tokens=True)
            for im,tokens,raw in zip(batch,ids,texts):
                text,error=clean(raw,any(t in eos for t in tokens))
                r={'image_sha256':im['sha256'],'path':im['path'],'config_id':CONFIG_ID,'text':text,'raw_text':raw,
                   'status':'generated' if error is None else 'retry','error':error,'output_tokens':len([t for t in tokens if t!=processor.tokenizer.pad_token_id]),
                   'attempt':2 if a.retry else 1,'quality_status':'pending_visual_review',
                   'seconds_per_image':round((time.monotonic()-start)/len(batch),3)}
                f.write(json.dumps(r,ensure_ascii=False)+'\n');cached[im['sha256']]=r
            f.flush();print('CAPTION_PROGRESS',a.worker,offset+len(batch),len(images),'seconds',round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':main()
