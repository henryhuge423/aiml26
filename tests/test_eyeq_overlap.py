import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class EyeqOverlapTests(unittest.TestCase):
    def module(self):
        path=ROOT/'tools/audit_eyeq_overlap.py'
        self.assertTrue(path.exists(),'EyeQ overlap audit is missing')
        spec=importlib.util.spec_from_file_location('eyeq_overlap',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_radius_search_finds_six_bit_neighbor_and_rejects_seven(self):
        m=self.module();index=m.HashIndex([0,(1<<63)-1])
        self.assertEqual(index.query(sum(1<<b for b in (0,9,18,27,36,63)),6),[(0,6)])
        self.assertEqual(index.query(sum(1<<b for b in (0,9,18,27,36,45,63)),6),[])

    def test_index_matches_brute_force_across_random_bit_positions(self):
        m=self.module();rng=random.Random(7)
        hashes=[rng.getrandbits(64) for _ in range(40)];index=m.HashIndex(hashes)
        for value in hashes:
            query=value
            for bit in rng.sample(range(64),6):query^=1<<bit
            self.assertEqual(index.query(query,6),[(i,(query^h).bit_count()) for i,h in enumerate(hashes) if (query^h).bit_count()<=6])

    def test_mirror_and_rotation_are_compared_in_the_correct_orientation(self):
        m=self.module();rng=np.random.default_rng(12)
        a=rng.integers(0,256,(384,384),dtype=np.uint8)
        b=m.orient(a,5)
        result=m.compare(a,b,5)
        self.assertEqual(result['phash_distance'],0)
        self.assertAlmostEqual(result['ssim'],1.,places=6)
        self.assertTrue(result['confirmed'])
        self.assertFalse(m.compare(a,rng.integers(0,256,(384,384),dtype=np.uint8),0)['confirmed'])

    def test_patient_closure_propagates_protected_roles_through_existing_group(self):
        m=self.module()
        rows=[dict(path='eyeq_a',eyeq_split='train',patient_id='9'),
              dict(path='eyeq_b',eyeq_split='train',patient_id='9'),
              dict(path='old_a'),dict(path='old_b')]
        active=[dict(path='old_a',related_group='g1',split='test'),
                dict(path='old_b',related_group='g1',split='test')]
        report=m.component_rows(rows,active,[('eyeq_a','old_b')],[])
        self.assertEqual([r['active_roles'] for r in report],['test','test'])
        self.assertTrue(all(r['blocked_from_new_training'] for r in report))


if __name__=='__main__':unittest.main()
