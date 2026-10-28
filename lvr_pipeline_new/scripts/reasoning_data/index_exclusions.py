"""Persistent exact/pixel/perceptual evidence for old splits and held-out assets."""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
import hashlib,io,json,sqlite3
import numpy as np
from PIL import Image
from scipy.fft import dctn
import pyarrow as pa
import pyarrow.parquet as pq
from common import ROOT,RAW,BUILD,normalized,digest,write_json

OLD_IMAGES=Path('/workspace/wmz/lvr_pipeline_new/data')
DB=BUILD/'exclusions.sqlite'


def fingerprint(payload):
    if isinstance(payload,Path):payload=payload.read_bytes()
    sha=hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as im:
        im.load();im=im.convert('RGB');w,h=im.size
        pixels=hashlib.sha256(f'{w}x{h}:'.encode()+im.tobytes()).hexdigest()
        a=dctn(np.asarray(im.resize((32,32)).convert('L'),dtype=float),norm='ortho')[:8,:8]
        bits=a>np.median(a.reshape(-1)[1:]);phash=f'{int("".join("1" if b else "0" for b in bits.flatten()),2):016x}'
    return sha,pixels,phash,w,h


def main():
    db=sqlite3.connect(DB)
    db.execute('create table if not exists images(scope text,identity text,sha text,pixel_sha text,phash text,width int,height int,primary key(scope,identity))')
    db.execute('create table if not exists questions(scope text,id text,normalized text,primary key(scope,id))')
    db.execute('create table if not exists structures(scope text,id text,fingerprint text,primary key(scope,id))')
    db.execute('create table if not exists completed(scope text primary key)');db.commit()
    def add_images(scope,pairs):
        done={r[0] for r in db.execute('select identity from images where scope=?',(scope,))}
        pairs=(p for p in pairs if p[0] not in done)
        def work(pair):return (scope,pair[0],*fingerprint(pair[1]))
        count=0
        with ThreadPoolExecutor(12) as pool:
            while batch:=list(islice(pairs,48)):
                db.executemany('insert or replace into images values(?,?,?,?,?,?,?)',pool.map(work,batch));db.commit();count+=len(batch)
                if count%480==0:print('FINGERPRINT',scope,count,flush=True)
    for split in ('test','train'):
        scope='old_'+split
        if db.execute('select 1 from completed where scope=?',(scope,)).fetchone():continue
        index=json.loads((ROOT/'data/sft'/split/'index.json').read_text());paths=set()
        for shard in index['shards']:
            rows=json.loads((ROOT/'data/sft'/split/shard['file']).read_text())
            for r in rows:
                sid=r['question_image']+':'+str(r['raw']['index']);paths.add(r['question_image'])
                db.execute('insert or replace into questions values(?,?,?)',(scope,sid,normalized(r['question'])))
        add_images(scope,((p,OLD_IMAGES/p) for p in sorted(paths)))
        db.execute('insert or replace into completed values(?)',(scope,));db.commit();print('COMPLETE',scope,len(paths),flush=True)
    # Existing authoritative reasoning catalog references all MathVision/MathVista/VisuLogic inputs.
    scope='reasoning_benchmarks'
    if not db.execute('select 1 from completed where scope=?',(scope,)).fetchone():
        src=sqlite3.connect(ROOT/'data/reasoning_browser/catalog.sqlite');pairs={}
        for uid,q,images in src.execute("select uid,question,images from samples where family='benchmark'"):
            db.execute('insert or replace into questions values(?,?,?)',(scope,uid,normalized(q)))
            for im in json.loads(images):pairs[im['file']]=ROOT/im['file']
        src.close();add_images(scope,iter(pairs.items()));db.execute('insert into completed values(?)',(scope,));db.commit()
    # BLINK all locally available tasks and splits, including Jigsaw.
    for file in sorted((ROOT/'benchmark/bench').rglob('data-*.arrow')):
        if any(part in ('mathvision','mathvista','visulogic') for part in file.parts):continue
        scope='benchmark:'+str(file.relative_to(ROOT/'benchmark/bench'))
        if db.execute('select 1 from completed where scope=?',(scope,)).fetchone():continue
        def pairs():
            ordinal=0
            with pa.memory_map(str(file),'r') as handle:
                for batch in pa.ipc.open_stream(handle):
                    for r in batch.to_pylist():
                        sid=str(r.get('idx',r.get('id',ordinal)));ordinal+=1
                        db.execute('insert or replace into questions values(?,?,?)',(scope,sid,normalized(r.get('question',''))))
                        for k,v in r.items():
                            if isinstance(v,dict) and v.get('bytes'):yield sid+'/'+k,v['bytes']
        add_images(scope,pairs());db.execute('insert into completed values(?)',(scope,));db.commit()
    for file in sorted((RAW/'eval_exclusions').rglob('*.parquet')):
        scope='benchmark:'+str(file.relative_to(RAW/'eval_exclusions'))
        if db.execute('select 1 from completed where scope=?',(scope,)).fetchone():continue
        def pairs():
            ordinal=0
            for batch in pq.ParquetFile(file).iter_batches(batch_size=16):
                for r in batch.to_pylist():
                    sid=str(r.get('idx',r.get('id',ordinal)));ordinal+=1
                    db.execute('insert or replace into questions values(?,?,?)',(scope,sid,normalized(r.get('question',''))))
                    for k,v in r.items():
                        items=v if isinstance(v,list) else [v]
                        for i,im in enumerate(items):
                            if isinstance(im,dict) and im.get('bytes'):yield sid+'/'+k+'/'+str(i),im['bytes']
        add_images(scope,pairs());db.execute('insert into completed values(?)',(scope,));db.commit()
    for folder in ('hr_bench_4k','hr_bench_8k','mme_realworld_lite','vstar_bench','MMVP'):
        scope='benchmark_image_files:'+folder
        if db.execute('select 1 from completed where scope=?',(scope,)).fetchone():continue
        directory=ROOT/'benchmark/bench'/folder
        paths=sorted(p for p in directory.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
        if not paths:raise ValueError('No benchmark image assets: '+folder)
        add_images(scope,((str(p.relative_to(directory)),p) for p in paths))
        if folder=='vstar_bench':
            for line in (directory/'test_questions.jsonl').read_text().splitlines():
                r=json.loads(line);db.execute('insert or replace into questions values(?,?,?)',(scope,str(r['question_id']),normalized(r['text'])))
        db.execute('insert into completed values(?)',(scope,));db.commit()
    source=RAW/'mentis-oculi/datasets'
    for task in ('form-board','paper-fold','hinge-folding','rushhour','sliding-puzzle'):
        scope='official_mentis:'+task
        if db.execute('select 1 from completed where scope=?',(scope,)).fetchone():continue
        paths=sorted((source/task/'output').glob('level_*/puzzle_*/metadata.json'));assert len(paths)==250
        pairs=[]
        for path in paths:
            r=json.loads(path.read_text());sid=str(path.relative_to(source/task))
            if task in ('form-board','paper-fold'):refs=[r['combined_image']]
            elif task=='rushhour':refs=[r['initial_state_image']]
            else:refs=[r['initial_image'],r['target_image']]
            for ref in refs:
                options=[path.parent/ref,path.parent.parent/ref,path.parent.parent.parent/ref]
                found=next((p for p in options if p.is_file()),None)
                if found is None:raise FileNotFoundError((path,ref))
                pairs.append((sid+'/'+Path(ref).name,found))
            if task=='hinge-folding':structure=[r['shape_chain'],r['rotation_angles']]
            elif task=='rushhour':structure=[r['board'],[{k:v for k,v in o.items() if k not in ('color','label')} for o in r['objects']]]
            else:structure=None
            if structure is not None:db.execute('insert or replace into structures values(?,?,?)',(scope,sid,digest(structure)))
        add_images(scope,iter(pairs));db.execute('insert into completed values(?)',(scope,));db.commit()
    db.execute('create index if not exists exact_image on images(sha)');db.execute('create index if not exists decoded_image on images(pixel_sha)');db.execute('create index if not exists exact_question on questions(normalized)');db.commit()
    report={'images':dict(db.execute('select scope,count(*) from images group by scope')),'questions':dict(db.execute('select scope,count(*) from questions group by scope')),'completed':[r[0] for r in db.execute('select scope from completed')],'scope':'Old train/validation originals; all existing prepared benchmark Arrow records, MathVision/MathVista/VisuLogic catalog images; pinned SAT-v2 val/test and BLINK Multi-view; official Mentis 1250 puzzles. Mirage map checks are separate.'}
    write_json(BUILD/'exclusions_index_report.json',report);print('COMPLETE EXCLUSION INDEX',report['images'],flush=True)

if __name__=='__main__':main()
