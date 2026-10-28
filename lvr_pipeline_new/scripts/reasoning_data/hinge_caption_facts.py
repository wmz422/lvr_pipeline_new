"""Reject caption claims contradicted by rendered-chain geometry; never create captions."""
import re

PALETTE={'blue':'#0090C1','pink':'#E85D75','teal':'#2EC4B6','yellow':'#FFD670'}
NUMBERS={'two':2,'three':3,'four':4,'five':5,'six':6}


def check(text,facts):
    text=text.lower();overlap=facts['overlap_area']>1e-5
    negative=re.search(r'\b(no|not|without)\s+overlap\w*',text)
    if negative and overlap:return 'false_no_overlap_claim'
    if not negative and 'overlap' in text and not overlap:return 'false_overlap_claim'
    for m in re.finditer(r'\b(two|three|four|five|six)\s+(?:(?:colored|angular|overlapping)\s+)*(?:regions|polygons|shapes)\b',text):
        if NUMBERS[m[1]]!=facts['piece_count'] and not re.search(r'with (?:a |an )?(?:fifth|sixth)',text):return 'unverified_visible_piece_count'
    for m in re.finditer(r'\b(blue|pink|teal|yellow)\s+(triangle|rectangle|square|diamond)',text):
        allowed={'triangle':{'triangle'},'rectangle':{'rectangle','square','diamond'},'square':{'square','diamond'},'diamond':{'square','diamond'}}[m[2]]
        if not any(p['color']==PALETTE[m[1]] and p['type'] in allowed for p in facts['color_shapes']):return 'color_shape_pair_not_present'
    return None
