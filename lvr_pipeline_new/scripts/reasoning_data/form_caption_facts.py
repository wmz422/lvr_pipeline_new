"""Conservative visible-polygon checks for Form Board captions; no label synthesis."""
import re
import cv2
import numpy as np
from PIL import Image
from common import ROOT,BUILD,write_json

COLORS={'blue':[51,166,205],'pink':[236,125,144],'teal':[88,207,196],'yellow':[254,221,140]}
TYPES={'triang':3,'rectang':4,'square':4,'diamond':4,'parallelogram':4,'quadrilateral':4,'trapezoid':4,'pentagon':5,'hexagon':6}


def classify(mask):
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);shapes=[]
    for contour in contours:
        if cv2.contourArea(contour)<100:continue
        points=cv2.approxPolyDP(contour,.009*cv2.arcLength(contour,True),True).reshape(-1,2).astype(float)
        count=len(points);labels=['polygon']
        for stem,n in TYPES.items():
            if count!=n:continue
            vectors=np.roll(points,-1,axis=0)-points;lengths=np.linalg.norm(vectors,axis=1)
            cosines=np.abs(np.sum(vectors*np.roll(vectors,1,axis=0),axis=1)/(lengths*np.roll(lengths,1)))
            if stem in ('rectang','square') and not all(cosines<.18):continue
            if stem=='square' and max(lengths)/min(lengths)>1.15:continue
            if stem=='diamond' and max(lengths)/min(lengths)>1.15:continue
            if stem=='parallelogram':
                unit=vectors/lengths[:,None]
                if min(abs(np.dot(unit[0],unit[2])),abs(np.dot(unit[1],unit[3])))<.98:continue
            if stem=='trapezoid':
                unit=vectors/lengths[:,None]
                if max(abs(np.dot(unit[0],unit[2])),abs(np.dot(unit[1],unit[3])))<.98:continue
            labels.append(stem)
        shapes.append(labels)
    return shapes


def features(image):
    path=BUILD/'quality/form_pixel_facts_v3'/(image['sha256']+'.json')
    if path.exists():
        import json
        return json.loads(path.read_text())
    cv2.setNumThreads(1)
    with Image.open(ROOT/image['path']) as handle:
        small=handle.convert('RGB');small.thumbnail((1024,1024));rgb=np.asarray(small).astype('int16')
    result={};allmask=np.zeros(rgb.shape[:2],dtype='uint8');regions=[]
    for color,value in COLORS.items():
        mask=(np.max(np.abs(rgb-value),axis=-1)<7).astype('uint8');allmask|=mask
        result[color]=classify(mask)
        n,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
        regions.extend(cv2.dilate((components==i).astype('uint8'),np.ones((9,9),dtype='uint8')) for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>100)
        result[color+'_union']=classify(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((11,11),dtype='uint8')))
    result['all_union']=classify(cv2.morphologyEx(allmask,cv2.MORPH_CLOSE,np.ones((11,11),dtype='uint8')))
    result['any_shared_edge']=any(np.count_nonzero(x&y)>150 for i,x in enumerate(regions) for y in regions[i+1:])
    write_json(path,result);return result


def check(text,image):
    text=text.lower()
    if re.search(r'touch\w*\s+(?:at\s+)?(?:a|the)\s+corner',text):return 'unsupported_corner_only_contact'
    if re.search(r'touch\w*.*edges? of (?:the )?(?:white )?(?:background|image)',text):return 'false_canvas_boundary_contact'
    if re.search(r'\broof\b|house-like',text):return 'unsupported_object_analogy'
    facts=features(image)
    if re.search(r'touch\w*\s+(?:along\s+shared\s+|at\s+|along\s+)?edges|shar\w*\s+edges',text) and not facts['any_shared_edge']:return 'unsupported_shared_edge_contact'
    allpieces=sum((facts[c] for c in COLORS),[])
    for match in re.finditer(r'\b(triang\w*|rectang\w*|square\w*|diamond\w*|parallelogram\w*|quadrilateral\w*|trapezoid\w*|pentagon\w*|hexagon\w*)\b',text):
        stem=next(s for s in TYPES if match[0].startswith(s));prefix=text[max(0,match.start()-50):match.start()]
        colors=re.findall(r'\b(blue|pink|teal|yellow)\b',prefix)
        candidates=(facts[colors[-1]]+facts[colors[-1]+'_union']) if colors else allpieces+facts['all_union']
        if not any(stem in shape for shape in candidates):return 'unsupported_visible_polygon_type:'+match[0]
        # A coordinated color list qualifies every listed region, not just the
        # last color (e.g. "pink and yellow triangular sections").
        color_list=re.search(r'((?:(?:blue|pink|teal|yellow)(?:,\s*|\s+and\s+))+(?:blue|pink|teal|yellow))\s*$',prefix)
        if color_list:
            for color in re.findall(r'blue|pink|teal|yellow',color_list[1]):
                if not any(stem in shape for shape in facts[color]+facts[color+'_union']):return 'unsupported_coordinated_color_type:'+color+':'+stem
        if match[0].endswith('s'):
            mentioned=set(re.findall(r'\b(blue|pink|teal|yellow)\b',prefix+text[match.end():match.end()+60]))
            pieces=sum((facts[c] for c in mentioned),[]) if mentioned else allpieces
            if any(stem not in shape for shape in pieces):return 'unsupported_plural_polygon_types:'+stem
        counts=re.search(r'\b(two|three|four|five|six|[2-6])\s+(?:(?:connected|touching)\s+)?$',prefix)
        if counts:
            number={'two':2,'three':3,'four':4,'five':5,'six':6}.get(counts[1],int(counts[1]) if counts[1].isdigit() else 0)
            if sum(stem in shape for shape in allpieces)<number:return 'unsupported_polygon_count'
    return None


if __name__=='__main__':
    import argparse,json
    from concurrent.futures import ProcessPoolExecutor
    from common import CANDIDATES
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    images={im['sha256']:im for r in map(json.loads,(CANDIDATES/'mentis_form-board.jsonl').read_text().splitlines()) for im in r['auxiliary_images']}
    pending=[im for im in images.values() if not (BUILD/'quality/form_pixel_facts_v3'/(im['sha256']+'.json')).exists()]
    print('FORM_FACTS_PENDING',len(pending),flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for i,_ in enumerate(pool.map(features,pending,chunksize=8),1):
            if i%1000==0:print('FORM_FACTS',i,len(pending),flush=True)
