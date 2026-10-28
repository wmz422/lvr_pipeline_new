"""Run caption shards in GPU lanes; caches keep worker identity stable."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,subprocess,sys
from common import ROOT,BUILD,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--per-source',type=int,default=100);p.add_argument('--retry',action='store_true');a=p.parse_args()
 gpus=[int(s) for s in a.gpus.split(',')];logs=BUILD/'logs';logs.mkdir(exist_ok=True)
 def lane(l):
  results=[]
  for worker in range(l,8,len(gpus)):
   cmd=[sys.executable,str(Path(__file__).with_name('caption.py')),'--worker',str(worker),'--workers','8','--gpu',str(gpus[l]),'--per-source',str(a.per_source)]
   if a.retry:cmd.append('--retry')
   with (logs/f'caption_pilot_{worker}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   results.append({'worker':worker,'returncode':r.returncode,'command':cmd})
  return results
 with ThreadPoolExecutor(len(gpus)) as pool:results=sum(pool.map(lane,range(len(gpus))),[])
 write_json(BUILD/'caption_pilot_run.json',results)
 if any(r['returncode'] for r in results):raise SystemExit(1)
if __name__=='__main__':main()
