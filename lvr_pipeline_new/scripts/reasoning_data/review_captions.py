"""Image-only independent caption factuality gate; no QA/answer/hidden state input."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from common import ROOT,BUILD,digest
from caption import CONFIG_ID,RETRY_CONFIG_ID

REVIEW_MODEL=Path('/workspace/wmz/model_weights/Qwen3-VL-32B-Instruct')
PROMPT='Check this caption against the attached image, examining every claim. Reject wrong or unsupported shapes, colors, positions, counts, object names, and touching versus overlapping. In particular, do not confuse a C-shaped contour with an L or assume a triangle has a right angle. A plausible claim is not enough; all claims must be visibly supported. A short partial description is acceptable. Also flag natural-language solution paragraphs in the image; short labels, numbers, and equations are allowed. Return only JSON with keys accurate (boolean), issues (at most EIGHT words), prose_solution (boolean). Keep the JSON brief. Caption: '
REVIEW_CONFIG={'model':'Qwen/Qwen3-VL-32B-Instruct','local_path':str(REVIEW_MODEL),'prompt_version':'caption-evidence-check-v2','prompt':PROMPT,'do_sample':False,'max_new_tokens':96,'min_pixels':262144,'max_pixels':1048576}
REVIEW_ID=digest(REVIEW_CONFIG)


def main():
    global CONFIG_ID
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=7);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--retry',action='store_true');a=p.parse_args()
    if a.retry:CONFIG_ID=RETRY_CONFIG_ID
    out=BUILD/'caption_reviews'/CONFIG_ID;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(REVIEW_CONFIG,indent=2))
    path=out/(REVIEW_ID+f'_worker{a.worker}.jsonl');reviewed=set()
    for previous in out.glob(REVIEW_ID+'*.jsonl'):
        for l in previous.read_text().splitlines():
            try:r=json.loads(l);reviewed.add((r['image_sha256'],r['caption_sha256']))
            except json.JSONDecodeError:pass
    captions={}
    for p in (BUILD/'captions'/CONFIG_ID).glob('worker_*.jsonl'):
        for l in p.read_text().splitlines():
            try:r=json.loads(l)
            except json.JSONDecodeError:continue
            if r['status']=='generated':captions[r['image_sha256']]=r
    rows=[]
    for r in captions.values():
        r['caption_sha256']=hashlib.sha256(r['text'].encode()).hexdigest()
        if (r['image_sha256'],r['caption_sha256']) not in reviewed and int(r['image_sha256'],16)%a.workers==a.worker:rows.append(r)
    rows.sort(key=lambda r:r['image_sha256'])
    if a.limit:rows=rows[:a.limit]
    print('REVIEW_QUEUE',len(rows),flush=True)
    if not rows:return
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(3);device=f'cuda:{a.gpu}'
    processor=AutoProcessor.from_pretrained(REVIEW_MODEL,local_files_only=True);processor.tokenizer.padding_side='left'
    processor.image_processor.size.longest_edge=REVIEW_CONFIG['max_pixels'];processor.image_processor.size.shortest_edge=REVIEW_CONFIG['min_pixels']
    model=Qwen3VLForConditionalGeneration.from_pretrained(REVIEW_MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa').eval()
    with path.open('a') as f:
        offset=0;batch_size=a.batch_size
        while offset<len(rows):
            batch=rows[offset:offset+batch_size];pil=[];prompts=[];start=time.monotonic()
            for r in batch:
                with Image.open(ROOT/r['path']) as im:pil.append(im.convert('RGB'))
                prompts.append(processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT+r['text']}]}],tokenize=False,add_generation_prompt=True))
            try:
                inputs=processor(text=prompts,images=pil,padding=True,return_tensors='pt').to(device)
                with torch.inference_mode():tokens=model.generate(**inputs,do_sample=False,max_new_tokens=REVIEW_CONFIG['max_new_tokens'])
            except torch.cuda.OutOfMemoryError:
                if batch_size==1:raise
                if 'inputs' in locals():del inputs
                torch.cuda.empty_cache();batch_size=max(1,batch_size//2)
                print('REVIEW_REDUCE_BATCH',batch_size,flush=True);continue
            texts=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
            for r,raw in zip(batch,texts):
                match=re.search(r'\{.*\}',raw,re.S)
                try:
                    verdict=json.loads(match[0]);assert isinstance(verdict['accurate'],bool) and isinstance(verdict['prose_solution'],bool)
                except (TypeError,ValueError,AssertionError,KeyError):verdict={'accurate':False,'prose_solution':False,'issues':'invalid_verifier_output'}
                record={'image_sha256':r['image_sha256'],'caption_sha256':r['caption_sha256'],'review_config_id':REVIEW_ID,'verdict':verdict,'raw':raw,'status':'passed' if verdict['accurate'] and not verdict['prose_solution'] else 'rejected',
                    'runtime_batch_size':len(batch),'seconds_per_image':round((time.monotonic()-start)/len(batch),3)}
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            offset+=len(batch);f.flush();print('REVIEW_PROGRESS',offset,len(rows),'seconds',round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':main()
