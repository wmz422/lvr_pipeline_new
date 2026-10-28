"""Normalize unambiguous answers; quarantine source answers containing derivations."""
import hashlib,json,random,re
from collections import Counter
from common import BUILD,CANDIDATES,write_json


def main():
    accepted=[];rejected=[];counts=Counter()
    # The pinned SAT source always places the correct choice first. Permute
    # choices reproducibly so the training target cannot be solved by constant A.
    for r in map(json.loads,(CANDIDATES/'sat_perspective.jsonl').read_text().splitlines()):
        original=dict(r['options']);keys=list(original);correct=original[r['answer']]
        random.Random(int(hashlib.sha256(('sat-choice-order-v1:'+r['id']).encode()).hexdigest(),16)).shuffle(keys)
        suffix='\n'+'\n'.join(f'({k}) {v}' for k,v in original.items())
        assert r['question'].endswith(suffix)
        r['provenance'].update(original_options=original,original_answer=r['answer'],choice_order=keys,answer_normalization='sat_choice_permutation_v1')
        r['options']={chr(65+i):original[k] for i,k in enumerate(keys)}
        r['question']=r['question'][:-len(suffix)]+'\n'+'\n'.join(f'({k}) {v}' for k,v in r['options'].items())
        r['answer']=next(k for k,v in r['options'].items() if v==correct)
        accepted.append(r);counts['sat_choice_permutation_v1']+=1
    for path in sorted(CANDIDATES.glob('zebra_scientific_*.jsonl')):
        if '.rejected.' in path.name:continue
        for r in map(json.loads,path.read_text().splitlines()):
            answer=r['answer'];question=r['question'];method='source_final_answer'
            if r['source']=='zebra_scientific_graph_algorithms' and re.match(r'^Is\b',question,re.I) and re.match(r'^(Yes|No)\b',answer,re.I):
                r['answer']=re.match(r'^(Yes|No)\b',answer,re.I)[1].capitalize();r['answer_type']='yes_no';method='explicit_boolean_answer_for_boolean_question'
            elif re.search(r'\b(proof|substitut\w*|therefore|because|since|this means|this implies|by applying|we (get|obtain|have)|starting from|follows from|deriv\w*)\b',answer,re.I):
                rejected.append({'id':r['id'],'reason':'final_answer_contains_derivation_requires_resolution'});counts['quarantined']+=1;continue
            elif re.search(r'\b(show that|prove that|derive|explain why)\b',question,re.I):
                rejected.append({'id':r['id'],'reason':'question_requires_derivation_not_final_answer_only'});counts['quarantined']+=1;continue
            r['provenance']['answer_normalization']=method;accepted.append(r);counts[method]+=1
    out=BUILD/'quality'
    (out/'answers.validated.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    (out/'answers.rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
    write_json(out/'answers_report.json',dict(counts));print(dict(counts))

if __name__=='__main__':main()
