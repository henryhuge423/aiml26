import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'.test_tmp';scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch)
        assert Path(self.tmp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def manifest(self, rows):
        p=self.root/'manifest.csv'
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return p

    def row(self, **kwargs):
        r=dict(sample_id='one',path='one.jpg',sha256='hash1',source='source',label='1',
               target_definition='test',related_group='patient-1',patient_id_status='known',
               split='train',task='glaucoma',eligible='true',review_status='publisher_label')
        return dict(r,**kwargs)

    def data_module(self):
        self.assertTrue((ROOT/'training/data.py').exists(),'training data guards not implemented')
        from training import data
        return data

    def test_rejects_group_crossing_split(self):
        p=self.manifest([self.row(),self.row(sample_id='two',path='two.jpg',sha256='hash2',split='test')])
        with self.assertRaisesRegex(ValueError,'group.*split'):
            self.data_module().load_rows(p,'glaucoma','train')

    def test_rejects_unapproved_candidate(self):
        p=self.manifest([self.row(eligible='false')])
        with self.assertRaisesRegex(ValueError,'eligible'):
            self.data_module().load_rows(p,'glaucoma','train')

    def test_rejects_quality_training_overlap_with_disease_test(self):
        p=self.manifest([self.row(split='test'),self.row(sample_id='quality',task='quality')])
        with self.assertRaisesRegex(ValueError,'group.*split'):
            self.data_module().load_rows(p,'quality','train')

    def test_quality_cache_requires_matching_content_hash(self):
        r=self.row(task='quality')
        with self.assertRaisesRegex(ValueError,'teacher.*hash'):
            self.data_module().attach_teacher_targets([r],{'one':{'sha256':'different','teacher_q':'0.5'}})

    def test_quality_cache_missing_or_nonfinite_target_rejected(self):
        r=self.row(task='quality')
        for cache in ({},{'one':{'sha256':'hash1','teacher_q':'nan'}}):
            with self.assertRaises(ValueError):self.data_module().attach_teacher_targets([r],cache)

    def test_sampling_order_reproducible_by_epoch(self):
        rows=[self.row(sample_id=str(i),source='s'+str(i%2),label=str(i%2)) for i in range(30)]
        data=self.data_module()
        self.assertEqual(data.epoch_indices(rows,9,2,True),data.epoch_indices(rows,9,2,True))
        self.assertNotEqual(data.epoch_indices(rows,9,2,True),data.epoch_indices(rows,9,3,True))

    def test_release_manifest_hash_mismatch_rejected(self):
        p=self.manifest([self.row()]); release=self.root/'release.json'
        release.write_text(json.dumps({'manifest_sha256':'wrong'}))
        with self.assertRaisesRegex(ValueError,'manifest.*hash'):
            self.data_module().load_rows(p,'glaucoma','train',release)

    def test_release_not_formally_frozen_rejected(self):
        p=self.manifest([self.row(),self.row(sample_id='two',path='two.jpg',sha256='h2',label='0',related_group='patient-2')])
        release=self.root/'release.json'
        release.write_text(json.dumps({'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                                       'research_training_released':False}))
        with self.assertRaisesRegex(ValueError,'release.*frozen'):
            self.data_module().load_rows(p,'glaucoma','train',release)

    def test_quality_training_singleton_rejected(self):
        p=self.manifest([self.row(task='quality',label='0.5')])
        with self.assertRaisesRegex(ValueError,'at least two'):
            self.data_module().load_rows(p,'quality','train')

    def test_superseded_release_cannot_start_training(self):
        p=self.manifest([self.row(),self.row(sample_id='two',path='two.jpg',sha256='h2',label='0',related_group='patient-2')])
        anchor=self.root/'anchor.txt';anchor.write_text('source')
        release=self.root/'release.json'
        release.write_text(json.dumps({'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'version':'formal-research-v1','research_training_released':True,'project_root_from_release':'.',
            'input_fingerprints':{'anchor.txt':hashlib.sha256(anchor.read_bytes()).hexdigest()}}))
        (self.root/'SUPERSEDED_BY.json').write_text('{"replacement":"new-version"}')
        with self.assertRaisesRegex(ValueError,'superseded'):
            self.data_module().load_rows(p,'glaucoma','train',release)

    def test_per_image_augmentation_preserves_cuda_model_rng(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA environment required')
        from PIL import Image
        p=self.root/'one.jpg';Image.new('RGB',(32,24),(80,110,90)).save(p)
        row=self.row(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        dataset=self.data_module().ImageRows([row],self.root,32,'glaucoma',True)
        torch.cuda.manual_seed_all(123);before=torch.cuda.get_rng_state().clone()
        dataset[0]
        self.assertTrue(torch.equal(before,torch.cuda.get_rng_state()),'image transform changed model CUDA RNG')


if __name__=='__main__':unittest.main()
