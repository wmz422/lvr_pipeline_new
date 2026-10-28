"""Executable paths accept alternate shortest answers and reject invalid plans."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/reasoning_data'))
from score_answers import score


class ReasoningScoringTests(unittest.TestCase):
    def test_alternative_grid_path(self):
        row={'source':'mirage_vsp','answer_type':'grid_actions','answer':'R, D','provenance':{'map_desc':[[1,0],[0,2]]}}
        for answer in ('R, D','D, R','down, right'):self.assertTrue(score(row,answer)[0])
        for answer in ('L, R, R, D','R','R,D,U,D',''):self.assertFalse(score(row,answer)[0])

    def test_unordered_set(self):
        row={'source':'mentis_form-board','answer_type':'set','answer':'A, C, D'}
        self.assertTrue(score(row,'{D, A, C}')[0]);self.assertFalse(score(row,'A, B, C, D')[0])

    def test_all_official_answer_roundtrip(self):
        root=Path(__file__).resolve().parents[1]/'data/processed/reasoning_build/candidates'
        if not root.exists():self.skipTest('Local generated fixtures unavailable')
        for source in ('mentis_hinge-folding','mentis_sliding-puzzle','mentis_rushhour'):
            for line in (root/(source+'.jsonl')).read_text().splitlines()[:12]:
                row=json.loads(line);self.assertTrue(score(row,row['answer'])[0],row['id'])
        row=json.loads((root/'mentis_hinge-folding.jsonl').read_text().splitlines()[0])
        import re
        alternate=re.sub(r'\d+',lambda m:str(int(m[0])+360),row['answer'])
        self.assertTrue(score(row,alternate)[0])

if __name__=='__main__':unittest.main()
