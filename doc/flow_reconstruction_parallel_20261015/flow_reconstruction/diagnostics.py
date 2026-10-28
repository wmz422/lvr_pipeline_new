"""Compare objective gradients at the same latent, without accumulating parameter grads."""
import math
import torch

def latent_gradient_stats(qa_grad,description_grad,flow_grad,*,qa_weight,description_weight,flow_weight):
    if qa_grad.shape!=description_grad.shape or qa_grad.shape!=flow_grad.shape:
        raise ValueError('Gradient shapes differ at shared latent')
    for value in (qa_weight,description_weight,flow_weight):
        if not math.isfinite(value) or value<0:raise ValueError('Invalid loss coefficient')
    if not qa_weight+description_weight or flow_weight<=0:raise ValueError('Require text and FM objectives')
    for name,gradient in [('qa',qa_grad),('description',description_grad),('flow',flow_grad)]:
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite '+name+' latent gradient')
    qa,desc,fm=qa_grad.detach().float(),description_grad.detach().float(),flow_grad.detach().float()
    text=qa_weight*qa+description_weight*desc
    text_norm=float(text.norm());fm_norm=float(fm.norm());weighted_norm=flow_weight*fm_norm
    valid=text_norm>1e-12 and fm_norm>1e-12
    cosine=float((text*fm).sum()/(text_norm*fm_norm)) if valid else None
    if cosine is not None:cosine=max(-1.,min(1.,cosine))
    return {'qa_grad_l2':float(qa.norm()),'description_grad_l2':float(desc.norm()),
        'text_grad_l2':text_norm,'fm_grad_l2':fm_norm,'weighted_fm_grad_l2':weighted_norm,
        'weighted_fm_over_text':weighted_norm/text_norm if text_norm>1e-12 else None,
        'text_fm_cosine':cosine,'text_grad_rms':text_norm/math.sqrt(text.numel()),
        'fm_grad_rms':fm_norm/math.sqrt(fm.numel()),
        'combined_grad_l2':float((text+flow_weight*fm).norm()),'latent_elements':text.numel()}

def select_probe_records(manifest,sample_ids=None):
    """Six validation cases: crops, auxiliary sequences, two refs and largest canvas."""
    import json
    from pathlib import Path
    with Path(manifest).open() as stream:
        rows=[json.loads(line) for line in stream]
    if sample_ids:
        lookup={r['id']:r for r in rows}
        if len(set(sample_ids))!=len(sample_ids):raise ValueError('Repeated probe ID')
        return [(sample_id,lookup[sample_id]) for sample_id in sample_ids]
    rules=[
        ('single_crop',lambda r:r['layout']['kind']=='single_crop'),
        ('masked_original',lambda r:r['layout']['kind']=='masked_original'),
        ('auxiliary_single',lambda r:r['layout']['kind']=='horizontal_auxiliary' and r['layout']['tiles']==1),
        ('auxiliary_multiple',lambda r:r['layout']['kind']=='horizontal_auxiliary' and r['layout']['tiles']>1),
        ('multiple_references',lambda r:len(r['references'])>1)]
    chosen=[];seen=set()
    for name,predicate in rules:
        row=next((r for r in rows if predicate(r) and r['id'] not in seen),None)
        if row is None:raise ValueError('No probe example for '+name)
        chosen.append((name,row));seen.add(row['id'])
    candidates=[r for r in rows if r['layout']['kind']=='horizontal_auxiliary' and r['id'] not in seen]
    largest=max(candidates,key=lambda r:r['target']['height']*r['target']['width'])
    chosen.append(('largest_auxiliary',largest))
    return chosen
