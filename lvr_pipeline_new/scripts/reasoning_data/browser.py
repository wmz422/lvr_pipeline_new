"""Local visualization of real candidate/released QA, helpers, and caption status."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import sqlite3
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import mimetypes
from common import ROOT,BUILD,CANDIDATES,write_json
from caption import CONFIG_ID,CONFIG

DB=BUILD/'browser.sqlite'


def build():
    captions={}
    for p in (BUILD/'captions'/CONFIG_ID).glob('worker_*.jsonl'):
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            captions[r['image_sha256']]=r
    temp=DB.with_suffix('.building.sqlite')
    if temp.exists():temp.unlink()
    db=sqlite3.connect(temp)
    db.execute('create table samples(id text primary key,source text,task text,split text,difficulty int,helpers int,status text,question text,record text)')
    paths=sorted(CANDIDATES.glob('*.jsonl'));counts=Counter()
    for p in paths:
        if '.rejected.' in p.name or '.building.' in p.name:continue
        with p.open() as f:
            for line in f:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if 'auxiliary_images' not in r:continue
                parts=[]
                for i,im in enumerate(r['auxiliary_images'],1):
                    c=captions.get(im['sha256'],{})
                    parts.append({'step':i,'image_sha256':im['sha256'],'text':c.get('text',''),'status':c.get('status','pending'),'quality_status':c.get('quality_status','pending')})
                r['description_parts']=parts;r['description']=' '.join(f"State {p['step']}: {p['text']}" for p in parts if p['text'])
                r['caption_status']='待审核' if all(p['status']=='generated' for p in parts) else '待标注'
                db.execute('insert or replace into samples values (?,?,?,?,?,?,?,?,?)',(r['id'],r['source'],r['task'],r['split'],r['difficulty'],len(parts),r['caption_status'],r['question'],json.dumps(r,ensure_ascii=False)))
    full=ROOT/'data/processed/reasoning_v1'
    if (full/'release_manifest.json').exists():
        # Use frozen final decisions only. A pilot or pre-retry rejection file
        # can contain pending work and is not a final verdict for the full pool.
        rejects={r['id']:r['reasons'] for r in map(json.loads,(full/'rejected.jsonl').read_text().splitlines())}
        validation=json.loads((full/'validation_report.json').read_text())
        for r in validation['processor']['rejected']:rejects[r['id']]=['preprocessing:'+r['reason']]
        for sid,reasons in rejects.items():
            old=db.execute('select record from samples where id=?',(sid,)).fetchone()
            if not old:continue
            r=json.loads(old[0]);r['caption_status']='已拒收';r['quality']['final_rejection_reasons']=reasons
            db.execute('update samples set status=?,record=? where id=?',('已拒收',json.dumps(r,ensure_ascii=False),sid))
        prepared=BUILD/'preprocessed_v1.jsonl'
        batch=json.loads((full/'batch_acceptance.json').read_text())
        if prepared.exists() and hashlib.sha256(prepared.read_bytes()).hexdigest()==batch['input_sha256']:
            excluded=set(json.loads((full/'selection_report.json').read_text())['audit_excluded_ids'])
            for line in prepared.read_text().splitlines():
                r=json.loads(line)
                if r['id'] in rejects or r['id'] in excluded:continue
                r['caption_status']='合格候选·未发布'
                db.execute('insert or replace into samples values (?,?,?,?,?,?,?,?,?)',(r['id'],r['source'],r['task'],r['split'],r['difficulty'],len(r['auxiliary_images']),r['caption_status'],r['question'],json.dumps(r,ensure_ascii=False)))
    for version in ('reasoning_v0','reasoning_v1'):
        root=ROOT/'data/processed'/version
        if not (root/'release_manifest.json').exists():continue
        manifest=json.loads((root/'release_manifest.json').read_text())
        errata={};withdrawals={}
        errata_path=BUILD/'pilot_caption_errata.json'
        if version=='reasoning_v0' and errata_path.exists():
            declaration=json.loads(errata_path.read_text())
            if declaration['manifest_sha256']!=hashlib.sha256((root/'release_manifest.json').read_bytes()).hexdigest():
                raise ValueError('Pilot caption errata manifest differs')
            errata={r['id']:r for r in declaration['records']}
            withdrawal_path=(full/'pilot_caption_withdrawals.json') if (full/'release_manifest.json').exists() else (BUILD/'pilot_caption_withdrawals.json')
            if withdrawal_path.exists():
                withdrawn=json.loads(withdrawal_path.read_text())
                if withdrawn and withdrawn['manifest_sha256']!=declaration['manifest_sha256']:
                    raise ValueError('Pilot caption withdrawal manifest differs')
                withdrawals={r['id']:r for r in withdrawn.get('records',[])}
        acceptance=BUILD/'training_acceptance_summary.json'
        trained=manifest['status']=='accepted' or (acceptance.exists() and json.loads(acceptance.read_text()).get('status')=='passed')
        status='已验收' if trained else '数据已校验·训练待验收'
        for split in ('train','val'):
            index=json.loads((root/split/'index.json').read_text())
            for shard in index['shards']:
                for r in json.loads((root/split/shard['file']).read_text()):
                    record_status='已拒收' if r['id'] in withdrawals else ('描述待修订' if r['id'] in errata else status)
                    r['caption_status']=record_status;r['release_version']=version
                    if r['id'] in errata:r['caption_erratum']=errata[r['id']]
                    if r['id'] in withdrawals:r['caption_withdrawal']=withdrawals[r['id']]
                    db.execute('insert or replace into samples values (?,?,?,?,?,?,?,?,?)',(r['id'],r['source'],r['task'],r['split'],r['difficulty'],len(r['auxiliary_images']),record_status,r['question'],json.dumps(r,ensure_ascii=False)))
    db.execute('create index filters on samples(source,split,status)');db.commit()
    count=db.execute('select count(*) from samples').fetchone()[0];db.close();temp.replace(DB);print('BROWSER_INDEX',count,flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,body,ctype='application/json',status=200):
        if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',ctype+'; charset=utf-8' if ctype.startswith(('application/json','text/')) else ctype)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store' if ctype=='application/json' else 'max-age=3600')
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        url=urlparse(self.path);q=parse_qs(url.query)
        try:
            if url.path=='/':return self.respond(Path(__file__).with_name('static').joinpath('index.html').read_bytes(),'text/html')
            if url.path=='/api/stats':
                with sqlite3.connect(DB) as db:
                    counts=[dict(zip(('source','split','status','count'),row)) for row in db.execute('select source,split,status,count(*) from samples group by source,split,status')]
                    return self.respond({'counts':counts,'total':sum(r['count'] for r in counts),'caption_model':CONFIG['model'],'prompt_version':CONFIG['prompt_version']})
            if url.path=='/api/samples':
                filters=[];args=[]
                for key in ('source','split','status'):
                    if q.get(key,[''])[0]:filters.append(key+'=?');args.append(q[key][0])
                if q.get('q',[''])[0]:filters.append('(question like ? or id like ?)');args.extend(['%'+q['q'][0]+'%']*2)
                where=' where '+' and '.join(filters) if filters else ''
                offset=max(0,int(q.get('offset',['0'])[0]));limit=min(50,max(1,int(q.get('limit',['20'])[0])))
                with sqlite3.connect(DB) as db:
                    total=db.execute('select count(*) from samples'+where,args).fetchone()[0]
                    rows=db.execute('select id,source,task,split,difficulty,helpers,status,question from samples'+where+' order by source,id limit ? offset ?',args+[limit,offset]).fetchall()
                return self.respond({'total':total,'rows':[dict(zip(('id','source','task','split','difficulty','helpers','status','question'),r)) for r in rows]})
            if url.path=='/api/sample':
                with sqlite3.connect(DB) as db:r=db.execute('select record from samples where id=?',(q.get('id',[''])[0],)).fetchone()
                return self.respond(json.loads(r[0])) if r else self.respond({'error':'not found'},status=404)
            if url.path=='/image':
                path=(ROOT/q.get('path',[''])[0]).resolve();assets=(BUILD/'assets').resolve()
                if not path.is_relative_to(assets):return self.respond({'error':'forbidden'},status=403)
                return self.respond(path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            return self.respond({'error':'not found'},status=404)
        except (ValueError,KeyError,FileNotFoundError) as e:self.respond({'error':str(e)},status=400)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=8767);a=p.parse_args()
    if a.build:build()
    if a.serve:
        print(f'http://localhost:{a.port}',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
