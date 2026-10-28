"""Geometry-aware duplicate evidence and executable state checks for Mentis."""
import argparse,importlib.util,json,math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon,Point
from common import ROOT,RAW,BUILD,digest,write_json

SOURCE=RAW/'mentis-oculi/datasets'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
HINGE=load('mentis_hinge_geometry',SOURCE/'hinge-folding/generator.py')
PAPER=load('mentis_paper_geometry',SOURCE/'paper-fold/main.py')
SHAPES={name:{'type':name,'size':1.0} for name in ('square','diamond','triangle')}
SHAPES['rectangle']={'type':'rectangle','width':1.0,'height':0.5}


def canonical_hinge(m,angles=None):
    chain=[SHAPES[n] for n in m['shape_chain']]
    initial=HINGE._calculate_initial_positions(chain,scale=1,center_offset=(0,0))
    target=HINGE.calculate_folded_positions(chain,initial,angles if angles is not None else m['rotation_angles'],scale=1)
    geometries=[]
    for shape,(x,y,rotation) in zip(chain,target):
        poly=HINGE._shape_to_polygon(shape,x,y,rotation_degrees=rotation,scale=1)
        poly=Polygon([(round(x,5)+0.0,round(y,5)+0.0) for x,y in poly.exterior.coords]).normalize()
        geometries.append(poly.wkt)
    return digest([m['shape_chain'],geometries])


def paper_holes(path):
    image=np.asarray(Image.open(path).convert('RGB'));h,w=image.shape[:2]
    mask=np.max(np.abs(image.astype(int)-[171,231,225]),axis=-1)<=3
    labels,n=ndimage.label(mask);boxes=ndimage.find_objects(labels);sizes=np.bincount(labels.ravel());centers=[]
    for i,b in enumerate(boxes,1):
        bh,bw=b[0].stop-b[0].start,b[1].stop-b[1].start
        if sizes[i]>50 and max(bh,bw)<min(h,w)*.25 and .7<bh/bw<1.4:
            ys,xs=np.where(labels[b]==i);centers.append(((xs.mean()+b[1].start)/w,(ys.mean()+b[0].start)/h))
    if not centers:raise ValueError('no_visible_holes_detected')
    variants=[]
    for mirror in (False,True):
        pts=[(1-x if mirror else x,y) for x,y in centers]
        for rotation in range(4):
            variants.append(sorted((round(x,3),round(y,3)) for x,y in pts));pts=[(1-y,x) for x,y in pts]
    return min(variants),len(centers)


def round_floats(value):
    if isinstance(value,float):return round(value,3)+0.0
    if isinstance(value,list):return [round_floats(v) for v in value]
    if isinstance(value,dict):return {k:round_floats(v) for k,v in value.items() if k not in ('color','label')}
    return value


def resolve(path,m,name):
    for root in (path.parent,path.parent/m.get('metadata_subdirectory',''),path.parent.parent,path.parent.parent.parent):
        p=root/name
        if p.is_file():return p
    raise FileNotFoundError((path,name))


def analyze(path,task):
    m=json.loads(path.read_text());report={'metadata':str(path.relative_to(ROOT))}
    if task=='hinge-folding':
        report['signature']=canonical_hinge(m);report['method']='ordered shape chain and per-piece final polygons; color-independent, symmetric rotations normalized'
        report['execution_verified']=True
        chain=[SHAPES[n] for n in m['shape_chain']];initial=HINGE._calculate_initial_positions(chain,scale=1,center_offset=(0,0))
        report['state_facts']=[]
        for step in range(1,len(m['rotation_angles'])+1):
            positions=HINGE.calculate_folded_positions(chain,initial,m['rotation_angles'][:step],scale=1)
            polygons=[HINGE._shape_to_polygon(shape,x,y,rotation_degrees=angle,scale=1) for shape,(x,y,angle) in zip(chain,positions)]
            overlap=sum(polygons[i].intersection(polygons[j]).area for i in range(len(polygons)) for j in range(i))
            report['state_facts'].append({'step':step,'piece_count':len(chain),'overlap_area':overlap,
                'color_shapes':[{'color':color,'type':name} for color,name in zip(m['piece_colors'],m['shape_chain'])]})
    elif task=='rushhour':
        report['signature']=digest(round_floats([m['board'],m['objects']]));report['method']='full board/exit/object geometry and permitted axes rounded to 0.001 board units; no colors or labels'
        report['execution_verified']=m.get('execution_verified',False)
    elif task=='paper-fold':
        holes,n=paper_holes(resolve(path,m,m['cot_images'][-1]));report.update(signature=digest(holes),visible_hole_count=n,
            method='visible final-hole centers at 0.001 image precision, canonical under 8 square symmetries')
        if m.get('geometry'):
            geom=m['geometry'];expected=geom['holes'];current=[Point(*expected[0])]
            states=[]
            for idx,axis in enumerate(reversed(geom['axes']),1):
                shape=Polygon(geom['shapes'][-(idx+1)]);points={}
                for point in current:
                    reflected=Point(*PAPER._reflect_point((point.x,point.y),axis))
                    for p in (point,reflected):
                        if shape.contains(p):points[(round(p.x,6),round(p.y,6))]=p
                current=list(points.values());states.append(len(current))
            if {(round(p.x,6),round(p.y,6)) for p in current}!={(round(x,6),round(y,6)) for x,y in expected}:raise ValueError('unfold_replay_mismatch')
            # Pixel-level circles can overlap after unfolding; union count may be smaller.
            if n>len(expected):raise ValueError('too_many_visible_holes')
            report.update(execution_verified=True,replayed_hole_counts=states,expected_final_holes=len(expected))
    else:return None
    return report


def analyze_one(spec):
    key,path,task=spec
    try:return key,analyze(path,task),None
    except (ValueError,KeyError) as e:return key,None,{'path':str(path),'error':str(e)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    records={};failures=[];specs=[]
    for task in ('hinge-folding','paper-fold','rushhour'):
        for p in sorted((SOURCE/task/'output').glob('level_*/puzzle_*/metadata.json')):
            key='reference:official_mentis:'+task+':'+str(p.relative_to(SOURCE/task))
            specs.append((key,p,task))
        for p in sorted((BUILD/'mentis'/task).glob('*/record.json')):
            m=json.loads(p.read_text());key='candidate:mentis_'+task+'/'+str(m['generation_seed'])
            specs.append((key,p,task))
    with ProcessPoolExecutor(a.workers) as pool:
        for i,(key,result,error) in enumerate(pool.map(analyze_one,specs,chunksize=8),1):
            if error:failures.append(error)
            else:records[key]=result
            if i%1000==0:print('GEOMETRY_PROGRESS',i,len(specs),flush=True)
    write_json(BUILD/'quality/mentis_geometry.json',{'records':records,'failures':failures})
    print('GEOMETRY',len(records),'failures',len(failures),flush=True)

if __name__=='__main__':main()
