"""Summarize completed GPU evidence; fail closed on missing phases or controls."""
import argparse,hashlib,json,math
from pathlib import Path
from common import BUILD,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--batch-report',type=Path,required=True);p.add_argument('--output',type=Path,default=BUILD/'training_acceptance_summary.json');a=p.parse_args()
    evidence={};phases=[]
    def read(path):
        data=path.read_bytes();evidence[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
    for phase,target in (('teacher_alignment',22),('teacher_joint',20),('warmup',20),('joint',60),('plain_sft',20)):
        root=a.training/phase;marker=read(root/'complete.json')
        assert marker['status']=='complete' and marker['optimizer_steps']==target,(phase,marker)
        reports=[]
        for rank in range(4):
            runtime=read(root/f'runtime_rank{rank}.json')
            assert runtime['status']=='training_complete' and runtime['optimizer_steps']==target
            # Runtime metrics are sampled every five updates. The two-update
            # resume segment has no sampled latest_loss; its acceptance hook
            # below checks the loss and gradients on every batch instead.
            if 'latest_loss' in runtime:assert math.isfinite(runtime['latest_loss'])
            chunks=[(0,20),(20,22)] if phase=='teacher_alignment' else [(0,target)]
            for start,end in chunks:
                report=read(root/f'acceptance_rank{rank}_from{start}.json')
                assert report['status']=='passed' and report['initial_step']==start and report['final_step']==end
                assert report['finite_loss'] and report['available_local_gradients_finite']
                assert len(report['distributed_gradient_norms'])==end-start
                assert all(math.isfinite(x) for x in report['distributed_gradient_norms'])
                if phase!='plain_sft':assert report['latent_shapes']==[[1,8,512]],report
                if phase=='joint':assert report['shared_word_embeddings_preserved']==['encoder','qa']
                reports.append(report)
            if phase=='teacher_alignment':
                assert runtime['initial_global_step']==20 and runtime['initial_optimizer_state_entries']>0
        if phase=='teacher_alignment':
            previous=read(root/'before_resume.json')
            assert len(previous)==4 and all(r['optimizer_steps']==20 and r['status']=='training_complete' for r in previous)
        assert (root/'checkpoints/last_full.ckpt').is_file()
        phases.append({'phase':phase,'optimizer_steps':target,'ranks':4,'finite_loss_and_gradients':True,
            'latent_shape':None if phase=='plain_sft' else [1,8,512],
            'checkpoint':str((root/'checkpoints/last_full.ckpt').resolve())})
    batches=read(a.batch_report);assert batches['status']=='passed'
    assert batches['shared_word_embeddings_preserved']
    tie_audit=read(BUILD/'checkpoint_tie_audit.json')
    assert all(r['equal'] for r in tie_audit['warmup'].values())
    resume_audit=read(BUILD/'scheduler_resume_audit.json')
    assert resume_audit['status']=='passed' and {r['rank'] for r in resume_audit['records']}==set(range(4))
    for record in resume_audit['records']:
        assert record['initial_global_step']==20 and record['initial_optimizer_state_entries']>0
        assert record['final_global_step']==22 and record['skipped_steps']==0
        assert record['lr_schedulers'] and all(s['last_epoch']==22 and s['_step_count']==23 for s in record['lr_schedulers'])
        evidence[record['checkpoint']]=record['sha256']
    assert any(c['helper_counts'][1:]==[1,3,5] and len(c['ids'])==4 for c in batches['cases'])
    assert all(c['status']=='passed' and c['helper_and_question_isolation']=='passed' for c in batches['cases'])
    evaluation=read(a.training/'pilot_evaluation/summary.json');assert evaluation['status']=='passed' and evaluation['modes']==5
    modes={r['mode'] for r in evaluation['metrics']};sources={r['source'] for r in evaluation['metrics']}
    assert modes=={'plain','teacher','predicted','zero','shuffled'} and len(sources)==13
    assert len(evaluation['metrics'])==65 and all(r['n']>=1 for r in evaluation['metrics'])
    predictions=a.training/'pilot_evaluation/predictions.jsonl'
    raw=predictions.read_bytes();evidence[str(predictions.resolve())]=hashlib.sha256(raw).hexdigest()
    rows=[json.loads(l) for l in raw.splitlines()];assert len(rows)==5*evaluation['samples']
    matched={mode:{r['id'] for r in rows if r['mode']==mode} for mode in modes}
    assert len({frozenset(ids) for ids in matched.values()})==1
    assert all(r['donor_id']!=r['id'] for r in rows if r['mode']=='shuffled')
    write_json(a.output,{'status':'passed','training':str(a.training.resolve()),'phases':phases,
        'save_resume':'Native distributed model, optimizer and scheduler states resumed from step 20 to step 22.',
        'scheduler_resume_state':resume_audit,
        'resume_scope':'Mid-epoch loader cursor is not stateful; sample-exact continuation is not claimed.',
        'batch_cases':len(batches['cases']),'evaluation':evaluation,'evidence_sha256':evidence,
        'meaning':'Data/model pipeline acceptance. This short run and small held-out sample do not establish benchmark gains.'})
    print('TRAINING_ACCEPTANCE_PASSED',a.output,flush=True)


if __name__=='__main__':main()
