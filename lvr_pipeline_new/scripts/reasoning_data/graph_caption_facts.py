"""Check explicit yellow-node counts against the rendered graph, without writing labels."""
from functools import lru_cache
import re
import cv2
import numpy as np
from PIL import Image
from common import ROOT

NUMBERS = dict(zip(('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'), range(1, 11)))
YELLOW_COUNT = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|[1-9]|10)\s+yellow\s+(?:circular\s+)?(?:nodes?|circles?)\b', re.I)
COUNT_WORDS = r'one|two|three|four|five|six|seven|eight|nine|ten|single|\d+'
QUALIFIERS = r'(?:(?:yellow|gold|golden|orange|gray|grey|red|blue|green|purple|black|white|circular|small|large)\s+)*'
ISOLATED_COUNT = re.compile(r'\b(' + COUNT_WORDS + r')\s+' + QUALIFIERS
                            + r'(?:isolated|unconnected|disconnected|standalone)\s+'
                            + QUALIFIERS + r'(?:nodes?|circles?)\b', re.I)
ISOLATED_COUNT_AFTER = re.compile(r'\b(' + COUNT_WORDS + r')\s+' + QUALIFIERS
                                  + r'(?:nodes?|circles?)\s+(?:is|are|remains?|remain)\s+'
                                  + r'(?:isolated|unconnected|disconnected|standalone)\b', re.I)


@lru_cache(maxsize=65536)
def yellow_nodes(path):
    cv2.setNumThreads(1)
    with Image.open(ROOT / path) as opened:
        rgb = np.asarray(opened.convert('RGB'), dtype=float)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # Includes the renderer's yellow and gold, but excludes its orange, gray,
    # purple, black labels and green edges. External contours ignore islands
    # inside digit glyphs, which connected-component counting overcounts.
    mask = ((red > 180) & (green > 140) & (green > .78 * red) & (blue < .65 * green)).astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    for contour in contours:
        _, _, width, height = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 20 and .6 <= width / height <= 1.6:
            total += 1
    return total


def check(text, image, verified_isolation=None):
    isolated_spans = []
    for pattern in (ISOLATED_COUNT, ISOLATED_COUNT_AFTER):
        for match in pattern.finditer(text):
            token = match[1].lower()
            claimed = int(token) if token.isdigit() else (1 if token == 'single' else NUMBERS[token])
            colors = re.findall(r'\b(yellow|gold|golden|orange|gray|grey|red|blue|green|purple|black|white)\b', match[0].lower())
            colors = {'yellow' if color in ('gold', 'golden') else ('gray' if color == 'grey' else color) for color in colors}
            color = next(iter(colors)) if len(colors) == 1 else 'all'
            if verified_isolation is None or color not in verified_isolation:
                return 'unverified_isolated_node_count'
            if claimed != verified_isolation[color]:
                return f'isolated_node_count_conflict:claimed_{claimed}:verified_{verified_isolation[color]}'
            isolated_spans.append(match.span())
    matches = [match for match in YELLOW_COUNT.finditer(text)
               if not any(match.start() < end and match.end() > start for start, end in isolated_spans)]
    if not matches:
        return None
    visible = yellow_nodes(image['path'])
    for match in matches:
        token = match[1].lower()
        claimed = int(token) if token.isdigit() else NUMBERS[token]
        if claimed != visible:
            return f'yellow_node_count_conflict:claimed_{claimed}:visible_{visible}'
    return None
