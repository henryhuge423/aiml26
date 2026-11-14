import json
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image,ImageDraw
from training.data import ImageRows,epoch_indices,sha256
from training.engine import loader

ROOT=Path(__file__).resolve().parents[1]


class DegradationTrainingTests(unittest.TestCase):
    def test_quality_guard_and_evaluation_originals(self):
        with self.assertRaisesRegex(ValueError,'quality targets'):
            ImageRows([],ROOT,32,'quality',True,degradation={'probability':.3})
        scratch=ROOT/'.test_tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path=Path(folder)/'image.png'
            image=Image.new('RGB',(40,24),'black');ImageDraw.Draw(image).ellipse((6,2,34,22),fill=(170,70,30));image.save(path)
            rows=[dict(sample_id='x',path='image.png',sha256=sha256(path),label='1')]
            original=ImageRows(rows,folder,32,'glaucoma',False)[0]
            configured=ImageRows(rows,folder,32,'glaucoma',False,degradation={'probability':1})[0]
            self.assertTrue(torch.equal(original[0],configured[0]))
            self.assertEqual(original[1],configured[1])

    def test_worker_count_keeps_variants_labels_and_metadata_exact(self):
        scratch=ROOT/'.test_tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root=Path(folder);(root/'data').mkdir();path=root/'data/image.png'
            image=Image.new('RGB',(48,32));draw=ImageDraw.Draw(image)
            draw.ellipse((8,2,40,30),fill=(150,75,30));draw.line((12,10,35,20),fill=(210,160,90));image.save(path)
            rows=[dict(sample_id=str(i),path='image.png',sha256=sha256(path),label=str(i%2)) for i in range(8)]
            config=dict(image_size=32,task='glaucoma',seed=23,batch_size=4,device='cpu',num_workers=0,degradation={'probability':1})
            serial=list(loader(rows,config,root,True,epoch=2))
            workers=list(loader(rows,dict(config,num_workers=2),root,True,epoch=2))
            for a,b in zip(serial,workers):
                self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]));self.assertEqual(a[2:],b[2:])
                self.assertTrue(all(json.loads(m)['applied'] for m in a[3]))
            self.assertEqual(sha256(path),rows[0]['sha256'])

    def test_milder_sampler_reduces_small_stratum_repetition(self):
        rows=[dict(source='large',label=str(i%2)) for i in range(10000)]+[dict(source='small',label=str(i%2)) for i in range(100)]
        equal=epoch_indices(rows,23,0,True)
        mild=epoch_indices(rows,23,0,True,'inverse_sqrt')
        self.assertLess(sum(i>=10000 for i in mild),sum(i>=10000 for i in equal)/2)
        self.assertEqual(mild,epoch_indices(rows,23,0,True,'inverse_sqrt'))
        with self.assertRaisesRegex(ValueError,'sampling'):epoch_indices(rows,23,0,True,'unknown')


if __name__=='__main__':unittest.main()
