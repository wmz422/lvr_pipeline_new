"""Exercise released legacy/state mixtures and per-source extremes on the real model."""
import argparse,copy,hashlib,json
from collections import defaultdict
from pathlib import Path
from common import ROOT,write_json
from caption import MODEL
from lvr.data.records import read_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--training-data-root',type=Path,default=ROOT/'data/default_training')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    import torch
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQADataset,CropQACollator
    from lvr.data.reasoning import as_crop_qa_item
    from lvr.data.prompt import cap_qwen_pixels
    from lvr.models.visual_bottleneck import VisualBottleneck
    torch.set_num_threads(3);device=f'cuda:{a.gpu}'
    if a.release.is_file():
        rows=[json.loads(line) for line in a.release.read_text().splitlines()]
        input_path=a.release
    else:
        rows=[r for split in ('train','val') for r in read_records(a.release/split)]
        input_path=a.release/'release_manifest.json'
    input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
    legacy=CropQADataset(a.training_data_root/'sft/train',a.training_data_root/'align/train',limit=1,
        caption_path=ROOT/'data/latent_prediction/captions.jsonl.gz')[0]
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    processor.tokenizer.padding_side='right';cap_qwen_pixels(processor,1048576)
    image_root=Path('/workspace/wmz/lvr_pipeline_new/data')
    collator=CropQACollator(processor,image_root,include_descriptions=True)
    student=CropQACollator(processor,image_root,include_encoder=False)
    plain=CropQACollator(processor,image_root,num_latents=0,include_encoder=False)
    mixed=[legacy]+[as_crop_qa_item(next(r for r in rows if len(r['auxiliary_images'])==n)) for n in (1,3,5)]
    groups=defaultdict(list)
    for r in rows:groups[r['source']].append(r)
    cases=[('legacy_and_1_3_5_helpers',mixed)]
    for source,group in sorted(groups.items()):
        extreme={}
        selectors=[lambda r:len(r['auxiliary_images']),lambda r:len(r['problem_images']),lambda r:r['processed_token_length']]
        selectors.extend(lambda r,k=k:r['preprocessing']['lengths'][k] for k in ('qa','encoder','description'))
        for key in selectors:
            r=max(group,key=key);extreme[r['id']]=r
        cases.extend((source+':'+r['id'],[as_crop_qa_item(r)]) for r in extreme.values())
    model=VisualBottleneck.from_checkpoint(a.checkpoint,device,base_model_path=MODEL)
    for backbone in (model.encoder,model.qa):
        if getattr(backbone.config,'tie_word_embeddings',False):
            assert backbone.get_input_embeddings().weight is backbone.get_output_embeddings().weight
    def move(value):
        if torch.is_tensor(value):return value.to(device)
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        return value
    results=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for name,items in cases:
            batch=collator(items)
            originals=sum(len(r.get('images',[r.get('question_image')])) for r in items)
            helper_counts=[len(r['auxiliary_images']) if r.get('teacher_view_mode')=='auxiliary_images' else len(r['boxes']) for r in items]
            assert len(batch['qa']['image_grid_thw'])==originals
            assert len(batch['encoder']['image_grid_thw'])==originals+sum(helper_counts)
            assert 'pixel_values' not in batch['description'] and 'image_grid_thw' not in batch['description']
            # Unreadable helper paths must not affect either student QA or Plain.
            isolated=copy.deepcopy(items)
            for r in isolated:
                for im in r.get('auxiliary_images',[]):im['path']='/nonexistent/helper_isolation_check.png'
            student_batch=student(isolated);plain_batch=plain(isolated)
            for key in ('input_ids','pixel_values','image_grid_thw'):
                assert torch.equal(student_batch['qa'][key],batch['qa'][key]),key
            assert len(plain_batch['qa']['image_grid_thw'])==originals
            changed=copy.deepcopy(items)
            for r in changed:r['question']='QUESTION_ISOLATION_SENTINEL'
            changed_batch=collator(changed)
            for key in ('input_ids','pixel_values','image_grid_thw'):
                assert torch.equal(changed_batch['encoder'][key],batch['encoder'][key]),key
            output=model(move(batch),sample=False)
            assert tuple(output['mu'].shape)==(len(items),8,512)
            for key in ('mu','ce','description_ce'):
                assert torch.isfinite(output[key]).all(),key
            results.append({'case':name,'ids':batch['ids'],'helper_counts':helper_counts,
                'qa_images':originals,'encoder_images':originals+sum(helper_counts),
                'latent_shape':list(output['mu'].shape),'qa_ce':float(output['ce']),
                'description_ce':float(output['description_ce']),
                'lengths':{k:batch[k]['attention_mask'].sum(-1).tolist() for k in ('qa','encoder','description')},
                'helper_and_question_isolation':'passed','status':'passed'})
            print('BATCH_ACCEPTANCE',name,results[-1]['latent_shape'],flush=True)
            del batch,student_batch,plain_batch,changed_batch,output
    write_json(a.output,{'status':'passed','release':str(a.release.resolve()),'input_sha256':input_sha,'checkpoint':str(a.checkpoint.resolve()),
        'scope':'Real GPU forwards, mixed old/new 1/3/5 helpers, and per-source maxima; optimizer checks are separate.',
        'cases':results,'shared_word_embeddings_preserved':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(a.gpu)})


if __name__=='__main__':main()
