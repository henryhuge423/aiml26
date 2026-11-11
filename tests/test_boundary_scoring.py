import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class BoundaryScoringTests(unittest.TestCase):
    def test_inference_pixels_match_original_training_evaluation_for_both_tasks(self):
        path=ROOT/'tools/score_repaired_boundary.py'
        self.assertTrue(path.exists(),'repaired-boundary inference is missing')
        spec=importlib.util.spec_from_file_location('boundary_score',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.data import ImageRows
        with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as scratch:
            directory=Path(scratch);image=directory/'sample.png'
            Image.new('RGB',(25,37),(91,130,220)).save(image)
            row=dict(sample_id='one',path='sample.png',sha256=hashlib.sha256(image.read_bytes()).hexdigest(),label='1',teacher_q='.4')
            for task in ('glaucoma','quality'):
                original=ImageRows([row],directory,32,task,train=False)
                inference=module.InferenceRows([row],directory,32,task)
                x,sample_id=inference[0]
                self.assertEqual(sample_id,'one')
                self.assertTrue(torch.equal(x,original[0][0]))


if __name__=='__main__':unittest.main()
