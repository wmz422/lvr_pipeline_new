"""Resume independent caption reviews on configurable GPU lanes."""
import argparse,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import BUILD,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--batch-size',type=int,default=6);p.add_argument('--retry',action='store_true')
    p.add_argument('--workers',type=int,help='Total hash partitions; defaults to GPU count')
    p.add_argument('--worker-ids',help='Comma-separated partitions to run on these GPU lanes');a=p.parse_args()
    gpus=[int(x) for x in a.gpus.split(',')]
    workers=a.workers or len(gpus)
    selected=[int(x) for x in a.worker_ids.split(',')] if a.worker_ids else list(range(workers))
    if len(set(selected))!=len(selected) or any(w<0 or w>=workers for w in selected):p.error('Invalid or repeated worker IDs')
    def lane(lane_id):
      results=[]
      for worker in selected[lane_id::len(gpus)]:
        cmd=[sys.executable,str(Path(__file__).with_name('review_captions.py')),'--gpu',str(gpus[lane_id]),'--worker',str(worker),'--workers',str(workers),'--batch-size',str(a.batch_size)]
        if a.retry:cmd.append('--retry')
        with (BUILD/'logs'/f'caption_review_{worker}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        results.append({'worker':worker,'returncode':r.returncode})
      return results
    with ThreadPoolExecutor(len(gpus)) as pool:results=sum(pool.map(lane,range(len(gpus))),[])
    name=('retry' if a.retry else 'primary')+'_'+str(workers)+'_'+'-'.join(map(str,selected))
    write_json(BUILD/f'review_run_{name}.json',results)
    if any(r['returncode'] for r in results):raise SystemExit(1)

if __name__=='__main__':main()
