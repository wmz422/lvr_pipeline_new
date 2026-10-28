"""Decode/hash all assets and run the real training processor on every staged QA."""
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor
from common import ROOT,BUILD,digest,write_json
from caption import MODEL

CONFIG={'processor':str(MODEL),'original_pixels':1048576,'crop_pixels':262144,'max_length':8192,'num_latents':8}
COLLATOR=None


def initialize():
    global COLLATOR
    import torch
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQACollator
    from lvr.data.prompt import cap_qwen_pixels
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);processor.tokenizer.padding_side='right';cap_qwen_pixels(processor,CONFIG['original_pixels'])
    COLLATOR=CropQACollator(processor,ROOT,include_descriptions=True,**{k:v for k,v in CONFIG.items() if k!='processor'})


def process(row):
    from lvr.data.reasoning import validate_record,as_crop_qa_item
    key=digest([row,CONFIG]);cache=BUILD/'preprocessing'/key[:2]/(key+'.json')
    if cache.exists():return json.loads(cache.read_text())
    try:
        validate_record(row,check_files=True);batch=COLLATOR([as_crop_qa_item(row)])
        lengths={k:int(batch[k]['attention_mask'].sum()) for k in ('qa','encoder','description')}
        grids={k:batch[k]['image_grid_thw'].tolist() for k in ('qa','encoder')}
        assert len(grids['qa'])==len(row['problem_images'])
        assert len(grids['encoder'])==len(row['problem_images'])+len(row['auxiliary_images'])
        result={'id':row['id'],'status':'passed','cache_key':key,'lengths':lengths,'processed_token_length':max(lengths.values()),'grids':grids}
    except (ValueError,AssertionError,OSError) as e:result={'id':row['id'],'status':'rejected','cache_key':key,'reason':str(e)}
    write_json(cache,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v0');p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    rows=[json.loads(l) for l in (BUILD/f'staging_{a.version}.jsonl').read_text().splitlines()];results=[]
    os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    with ProcessPoolExecutor(a.workers,initializer=initialize) as pool:
        for i,r in enumerate(pool.map(process,rows,chunksize=4),1):
            results.append(r)
            if i%100==0:print('PREPROCESS',i,len(rows),flush=True)
    passed={r['id']:r for r in results if r['status']=='passed'}
    output=[]
    for row in rows:
        if row['id'] not in passed:continue
        row['processed_token_length']=passed[row['id']]['processed_token_length'];row['preprocessing']=passed[row['id']];output.append(row)
    (BUILD/f'preprocessed_{a.version}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    report={'config':CONFIG,'records':len(rows),'passed':len(output),'rejected':[r for r in results if r['status']!='passed'],
        'max_lengths':{k:max(r['lengths'][k] for r in passed.values()) for k in ('qa','encoder','description')},
        'max_problem_images':max(len(r['problem_images']) for r in output),'max_helper_images':max(len(r['auxiliary_images']) for r in output),
        'longest_sample':max(passed.values(),key=lambda r:r['processed_token_length'])}
    write_json(BUILD/f'preprocessing_{a.version}_report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
