"""Manifest-counted new-only/mixed recipes and resumable GPU acceptance runs."""
import argparse,copy,hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path
import yaml
from common import ROOT,write_json
from caption import MODEL
from lvr.data.records import read_records
from lvr.data.defaults import split_records

spec=importlib.util.spec_from_file_location('legacy_runner',ROOT/'scripts/latent_prediction/run.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
PHASES=('teacher_alignment','teacher_joint','warmup','joint','plain_sft')


def config_for(phase,release,output,*,mixed,gpus,batch,global_batch,short,training_data_root=None):
    manifest=json.loads((release/'release_manifest.json').read_text())
    train=read_records(release/'train');val=read_records(release/'val')
    if len(train)!=manifest['splits']['train']['records'] or len(val)!=manifest['splits']['val']['records']:raise ValueError('Release count mismatch')
    data_root=Path(training_data_root or ROOT/'data/default_training')
    old_records=split_records(data_root/'sft/train') if mixed else 0
    n=len(train)+old_records
    accumulation=global_batch//(gpus*batch)
    if accumulation<1 or accumulation*gpus*batch!=global_batch:raise ValueError('Global batch must divide GPU batch')
    steps=math.ceil(math.ceil(n/gpus)/batch/accumulation)
    target=({'teacher_alignment':22,'teacher_joint':20,'warmup':20,'joint':60,'plain_sft':20}[phase] if short else min(200,steps) if phase=='warmup' else steps)
    c=legacy.recipe(phase);c.update(exp_name=str(output),version=phase)
    c['trainer'].update(devices=gpus,accumulate_grad_batches=accumulation,max_steps=target,max_epochs=-1 if short else 1,
        limit_train_batches=1.0,limit_val_batches=2 if short else 1.0,val_check_interval=1.0,num_sanity_val_steps=0,log_every_n_steps=1 if short else 10)
    c['trainer']['strategy']={'class_path':'lightning.pytorch.strategies.DeepSpeedStrategy','init_args':{
        'config':str(ROOT/'configs/latent_prediction/deepspeed_zero3_80gb.json'),'exclude_frozen_parameters':False}}
    c['trainer']['callbacks']=[{'class_path':'lightning.pytorch.callbacks.ModelCheckpoint','init_args':{
        'filename':'epoch-{epoch:02d}-{step:06d}','save_top_k':1,'save_last':True,'save_on_train_epoch_end':True,'auto_insert_metric_name':False}},
        {'class_path':'lightning.pytorch.callbacks.LearningRateMonitor','init_args':{'logging_interval':'step'}},
        {'class_path':'lvr.training.audit.RunAudit'},
        {'class_path':'lvr.training.reasoning_audit.ReasoningAcceptance','init_args':{'stop_after_updates':20 if short and phase=='teacher_alignment' else 0}}]
    d=c['data'];d.update(processor_name_or_path=str(MODEL),image_root='/workspace/wmz/lvr_pipeline_new/data',batch_size=batch,num_workers=2,
        train_path=str(release/'train'),val_path=str(release/'val'),train_boxes_path=None,val_boxes_path=None,caption_path=None)
    if mixed:
        d.update(train_path=str(data_root/'sft/train'),val_path=str(data_root/'sft/test'),train_boxes_path=str(data_root/'align/train'),
            val_boxes_path=str(data_root/'align/test'),additional_train_path=str(release/'train'),additional_val_path=str(release/'val'),
            caption_path=str(ROOT/'data/latent_prediction/captions.jsonl.gz') if phase!='plain_sft' else None)
    if phase.startswith('teacher') or phase=='plain_sft':c['model'].update(encoder_model_name_or_path=str(MODEL),qa_model_name_or_path=str(MODEL))
    else:c['model'].update(source_checkpoint=str(output/'teacher_joint/checkpoints/last_full.ckpt'),base_model_path=str(MODEL))
    if phase=='teacher_joint':c['init_checkpoint']=str(output/'teacher_alignment/checkpoints/last_full.ckpt')
    if phase=='joint':c['init_checkpoint']=str(output/'warmup/checkpoints/last_full.ckpt')
    return c,{'records':n,'new_records':len(train),'old_records':old_records,'global_batch':global_batch,'steps_per_epoch':steps,'optimizer_steps':target}


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-data-root',type=Path,default=ROOT/'data/default_training')
    p.add_argument('--mixed',action='store_true');p.add_argument('--short',action='store_true');p.add_argument('--write-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--gpus',type=int,default=4);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--global-batch',type=int,default=128)
    p.add_argument('--phases',nargs='+',choices=PHASES,default=list(PHASES));a=p.parse_args();a.release=a.release.resolve();a.output=a.output.resolve()
    env={**os.environ,'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2',
        'NCCL_SOCKET_IFNAME':'lo','GLOO_SOCKET_IFNAME':'lo','NCCL_IB_DISABLE':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    for phase in a.phases:
        c,counts=config_for(phase,a.release,a.output,mixed=a.mixed,gpus=a.gpus,batch=a.batch_size,global_batch=a.global_batch,short=a.short,
                            training_data_root=a.training_data_root.resolve())
        run=a.output/phase;run.mkdir(parents=True,exist_ok=True);recipes=a.output/'recipes';recipes.mkdir(exist_ok=True)
        path=recipes/(phase+'.yaml');fingerprint=hashlib.sha256(yaml.safe_dump(c).encode()).hexdigest();marker=run/'complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['recipe_sha256']==fingerprint;continue
        path.write_text(yaml.safe_dump(c,sort_keys=False));write_json(recipes/(phase+'_counts.json'),counts)
        if a.write_only:continue
        native=run/'checkpoints/last.ckpt';runtime=run/'runtime_rank0.json'
        if runtime.exists() and not a.resume:raise RuntimeError('Existing incomplete run; use --resume')
        if runtime.exists() and not legacy.is_deepspeed_checkpoint(native):raise RuntimeError('No recoverable checkpoint; choose a fresh output')
        while True:
            current=copy.deepcopy(c)
            if legacy.is_deepspeed_checkpoint(native):current.pop('init_checkpoint',None);current['ckpt_path']=str(native)
            path.write_text(yaml.safe_dump(current,sort_keys=False))
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.gpus}','--module',
                'lvr.stage1' if phase.startswith('teacher') or phase=='plain_sft' else 'lvr.predict_latents','fit','--config',str(path)]
            with (run/'train.log').open('a') as log:code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'{phase} failed ({code}); see {run}/train.log')
            states=[json.loads((run/f'runtime_rank{i}.json').read_text()) for i in range(a.gpus)]
            steps={s['optimizer_steps'] for s in states};assert all(s['status']=='training_complete' for s in states)
            if steps=={counts['optimizer_steps']}:break
            if not(a.short and phase=='teacher_alignment' and steps=={20}):raise RuntimeError(f'Unexpected steps: {steps}')
            write_json(run/'before_resume.json',states)
        legacy.consolidate_deepspeed_checkpoint(native,run/'checkpoints/last_full.ckpt',force=True)
        write_json(marker,{'recipe_sha256':fingerprint,**counts,'status':'complete'});print('TRAINING_COMPLETE',phase,counts,flush=True)

if __name__=='__main__':main()
