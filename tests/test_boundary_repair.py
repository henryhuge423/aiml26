import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class BoundaryRepairTests(unittest.TestCase):
    def module(self):
        path=ROOT/'tools/repair_boundary_development.py'
        self.assertTrue(path.exists(),'boundary family repair is missing')
        spec=importlib.util.spec_from_file_location('boundary_repair',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_removes_entire_boundary_group_and_preserves_other_original_rows(self):
        m=self.module()
        rows=[dict(sample_id='b1',task='glaucoma',split='boundary',related_group='bad',label='1'),
              dict(sample_id='b2',task='glaucoma',split='boundary',related_group='bad',label='0'),
              dict(sample_id='b3',task='glaucoma',split='boundary',related_group='good',label='1'),
              dict(sample_id='t1',task='glaucoma',split='train',related_group='train',label='0')]
        original=[dict(r) for r in rows]
        clean,removed=m.repair_rows(rows,[rows[0],rows[3]])
        self.assertEqual([r['sample_id'] for r in clean],['b3'])
        self.assertEqual([r['sample_id'] for r in removed],['b1','b2'])
        self.assertEqual(rows,original)


if __name__=='__main__':unittest.main()
