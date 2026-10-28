"""Readable visual audit sheets with every input/helper and its generated caption."""
import argparse,json,textwrap
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,BUILD,write_json
from caption import CONFIG_ID

FONT=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',16)
TITLE=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',18)


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v0');p.add_argument('--input');p.add_argument('--tag');p.add_argument('--families');p.add_argument('--exclude-inspected',action='store_true');a=p.parse_args()
    rows=[json.loads(l) for l in (Path(a.input) if a.input else BUILD/f'preprocessed_{a.version}.jsonl').read_text().splitlines()];families=defaultdict(list)
    if a.exclude_inspected:
        inspected={r['id'] for r in json.loads((BUILD/'visual_audit'/f'{CONFIG_ID}.json').read_text())['samples']}
        rows=[r for r in rows if r['id'] not in inspected]
    for r in rows:families['zebra_scientific' if r['source'].startswith('zebra_scientific_') else r['source']].append(r)
    out=BUILD/'visual_audit'/CONFIG_ID/(a.tag or a.version);out.mkdir(parents=True,exist_ok=True);selection={};selected_records=[]
    for family,rows in sorted(families.items()):
        if a.families and family not in a.families.split(','):continue
        groups=defaultdict(list)
        for r in rows:groups[(r['source'],str(r['difficulty']))].append(r)
        chosen=[]
        for group in groups.values():group.sort(key=lambda r:(-len(r['auxiliary_images']),r['id']))
        target=min(len(rows),max(20,5*len(groups)))
        while len(chosen)<target:
            for key in sorted(groups):
                if groups[key]:chosen.append(groups[key].pop(0))
                if len(chosen)>=target:break
        selection[family]=[r['id'] for r in chosen]
        selected_records.extend(chosen)
        for index in range(0,len(chosen),4):
            panels=[]
            for row in chosen[index:index+4]:
                assets=[(f'Input {i}',im,'') for i,im in enumerate(row['problem_images'],1)]
                assets.extend((f'State {im["step"]}',im,part['text']) for im,part in zip(row['auxiliary_images'],row['description_parts']))
                cols=min(6,len(assets));width=2400;cw=width//cols
                panel=Image.new('RGB',(width,120+((len(assets)+cols-1)//cols)*550),'white');draw=ImageDraw.Draw(panel)
                draw.text((12,4),row['id']+' | difficulty '+str(row['difficulty'])+' | answer: '+row['answer'][:120],fill='black',font=TITLE)
                question='\n'.join(textwrap.wrap(row['question'],220)[:4]);draw.multiline_text((12,30),question,fill='black',font=FONT)
                for n,(label,im,caption) in enumerate(assets):
                    x=(n%cols)*cw;y=120+(n//cols)*550
                    with Image.open(ROOT/im['path']) as handle:thumb=handle.convert('RGB');thumb.thumbnail((cw-16,330))
                    panel.paste(thumb,(x+(cw-thumb.width)//2,y+25));draw.text((x+8,y),label,fill='navy',font=TITLE)
                    draw.multiline_text((x+8,y+365),'\n'.join(textwrap.wrap(caption,width=max(24,(cw-16)//8))),fill='black',font=FONT,spacing=3)
                panels.append(panel)
            sheet=Image.new('RGB',(2400,sum(p.height for p in panels)),(225,229,235));y=0
            for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
            sheet.save(out/f'{family}_{index//4:02d}.jpg',quality=92)
    write_json(out/'selection.json',{'version':a.version,'caption_config_id':CONFIG_ID,'ids':selection,'status':'awaiting_visual_inspection'})
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected_records))
    print({s:len(ids) for s,ids in selection.items()})

if __name__=='__main__':main()
