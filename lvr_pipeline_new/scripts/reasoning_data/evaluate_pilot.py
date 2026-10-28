"""Matched per-source Plain/Teacher/Predicted/Zero/Shuffled generation checks."""
import argparse,gc,json,subprocess,sys
from collections import defaultdict
from common import ROOT,BUILD,write_json
from caption import MODEL
from lvr.data.records import read_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--training',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--per-source',type=int,default=2);a=p.parse_args()
    from pathlib import Path
    import torch
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQACollator
    from lvr.data.reasoning import as_crop_qa_item
    from lvr.data.prompt import cap_qwen_pixels
    from lvr.models.visual_bottleneck import VisualBottleneck
    from lvr.models.latent_prediction import LatentPrediction
    torch.set_num_threads(3);device=f'cuda:{a.gpu}';training=Path(a.training);out=training/'pilot_evaluation';out.mkdir(exist_ok=True)
    sources=defaultdict(list)
    for r in read_records(Path(a.release)/'val'):sources[r['source']].append(r)
    selected=[]
    for source,rows in sorted(sources.items()):selected+=sorted(rows,key=lambda r:r['id'])[:a.per_source]
    assert len(sources)==13 and all(sources.values()),'Pilot val needs all 13 source subdivisions'
    # Paired examples use different IDs; shuffled donors are shared across all sources.
    items=[as_crop_qa_item(r) for r in selected]
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);processor.tokenizer.padding_side='right';cap_qwen_pixels(processor,1048576)
    plain_collator=CropQACollator(processor,ROOT,num_latents=0,include_encoder=False,include_answers=False)
    student_collator=CropQACollator(processor,ROOT,include_encoder=False,include_answers=False)
    teacher_collator=CropQACollator(processor,ROOT,include_answers=False)
    def gpu(branch):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in branch.items()}
    journal=out/'generation_progress.jsonl'
    predictions=[json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
    done={(r['id'],r['mode']) for r in predictions}
    assert all(r['id'] in {x['id'] for x in selected} for r in predictions),'Evaluation selection changed'
    def record(row,mode,tokens,donor=None):
        result={'id':row['id'],'source':row['source'],'mode':mode,
            'prediction':processor.tokenizer.decode(tokens[0],skip_special_tokens=True),'donor_id':donor,
            'generated_tokens':int(tokens.shape[-1]),'reached_token_limit':int(tokens.shape[-1])>=256}
        predictions.append(result);done.add((row['id'],mode))
        with journal.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    with torch.inference_mode():
        model=VisualBottleneck.from_checkpoint(training/'plain_sft/checkpoints/last_full.ckpt',device,base_model_path=MODEL)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for row,item in zip(selected,items):
                if (row['id'],'plain') in done:continue
                batch=plain_collator([item]);tokens=model.generate_answers(gpu(batch['qa']),max_new_tokens=256)
                record(row,'plain',tokens)
        del model;gc.collect();torch.cuda.empty_cache()
        model=VisualBottleneck.from_checkpoint(training/'teacher_joint/checkpoints/last_full.ckpt',device,base_model_path=MODEL)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for row,item in zip(selected,items):
                if (row['id'],'teacher') in done:continue
                batch=teacher_collator([item]);latent,_=model.encode(gpu(batch['encoder']))
                assert tuple(latent.shape)==(1,8,512) and torch.isfinite(latent).all()
                tokens=model.generate_answers(gpu(batch['qa']),latent,max_new_tokens=256)
                record(row,'teacher',tokens)
        del model;gc.collect();torch.cuda.empty_cache()
        model=LatentPrediction.from_checkpoint(training/'joint/checkpoints/last_full.ckpt',device,source_checkpoint=training/'teacher_joint/checkpoints/last_full.ckpt',base_model_path=MODEL)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predicted=[]
            for item in items:
                latent=model.predict_latents(gpu(student_collator([item])['qa']));assert tuple(latent.shape)==(1,8,512)
                predicted.append(latent)
            for i,(row,item) in enumerate(zip(selected,items)):
                qa=gpu(student_collator([item])['qa'])
                latents={'predicted':predicted[i],'zero':torch.zeros_like(predicted[i]),'shuffled':predicted[(i+1)%len(items)]}
                for mode,latent in latents.items():
                    if (row['id'],mode) in done:continue
                    if not torch.isfinite(latent).all():raise FloatingPointError('Nonfinite inference latent')
                    tokens=model.network.generate_answers(qa,latent,max_new_tokens=256)
                    record(row,mode,tokens,selected[(i+1)%len(items)]['id'] if mode=='shuffled' else None)
                print('EVALUATED',i+1,len(items),row['source'],flush=True)
    lookup={r['id']:r for r in selected};results=[]
    for mode in ('plain','teacher','predicted','zero','shuffled'):
        entries=[r for r in predictions if r['mode']==mode];input_path=out/f'{mode}_scorer_input.jsonl';output_path=out/f'{mode}_scores.jsonl'
        input_path.write_text(''.join(json.dumps({'record':lookup[r['id']],'prediction':r['prediction']},ensure_ascii=False)+'\n' for r in entries))
        subprocess.run([str(ROOT/'.venv-mentis/bin/python'),str(ROOT/'scripts/reasoning_data/score_answers.py'),str(input_path),str(output_path)],check=True)
        scored={r['id']:r for r in map(json.loads,output_path.read_text().splitlines())}
        results.extend({**r,**scored[r['id']]} for r in entries)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    metrics=[]
    for mode in ('plain','teacher','predicted','zero','shuffled'):
        for source in sorted(sources):
            group=[r for r in results if r['mode']==mode and r['source']==source]
            metrics.append({'source':source,'mode':mode,'n':len(group),'correct':sum(r['correct'] for r in group)})
    write_json(out/'summary.json',{'status':'passed','samples':len(selected),'modes':5,'metrics':metrics,
        'requested_per_source':a.per_source,'available_val_by_source':{k:len(v) for k,v in sources.items()},
        'checkpoints':{'plain':'plain_sft','teacher':'teacher_joint','predicted':'joint','zero':'joint','shuffled':'joint'},
        'meaning':'Pipeline acceptance only; small-sample accuracy does not establish training efficacy. Scientific metric is normalized exact, not symbolic equivalence.'})

if __name__=='__main__':main()
