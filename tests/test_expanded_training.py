import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ExpandedTrainingTests(unittest.TestCase):
    def test_pipeline_rejects_gpu_evidence_without_epoch_crossing(self):
        import importlib.util
        import json
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('pipeline',ROOT/'tools/run_training_pipeline.py')
        pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
        scratch=ROOT/'.test_tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            directory=Path(folder);name='glaucoma_a'
            (directory/name/'resumed').mkdir(parents=True)
            config=dict(task='glaucoma',epochs=20)
            (directory/f'{name}.json').write_text(json.dumps(config))
            (directory/name/'resumed.json').write_text(json.dumps(config))
            for filename in ('last.pt','best.pt','validation_predictions.csv'):
                (directory/name/'resumed'/filename).touch()
            result=dict(model=name,passed=True,weight_max_abs_difference=0.,resumed_next_loss_abs_difference=0.,
                completed=dict(parameters_changed=True,last_loss=.4,fingerprint='fake'))
            (directory/'verification.json').write_text(json.dumps(dict(passed=True,models=[result])))
            with patch('training.engine.fingerprint_inputs',return_value='fake'):
                with self.assertRaisesRegex(ValueError,'epoch'):
                    pipeline.verify_preflight(directory,[name],directory)

    def test_partial_reuse_preserves_old_identity_guards(self):
        from training.cache import validated_reuse_rows
        old = dict(sample_id='old', path='a.png', sha256='hash', split='train', teacher_q='.4', crop_std='.1')
        new = dict(sample_id='new', path='b.png', sha256='other', split='val')
        self.assertEqual(validated_reuse_rows([old, new], {'old': old}, allow_missing=True), [old])
        with self.assertRaisesRegex(ValueError, 'identity'):
            validated_reuse_rows([old, new], {'old': old})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validated_reuse_rows([dict(old, split='val'), new], {'old': old}, allow_missing=True)
        with self.assertRaisesRegex(ValueError, 'allowed'):
            validated_reuse_rows([dict(new, split='test')], {}, allow_missing=True)

    def test_workers_match_single_process_and_reject_changed_source(self):
        import torch
        from PIL import Image
        from training.data import sha256
        from training.engine import loader
        scratch = ROOT / '.test_tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            image = Image.new('RGB', (32, 24), 'red')
            for x in range(16):
                for y in range(24):
                    image.putpixel((x, y), (0, 100, 255))
            path = root / 'data/a.png'
            image.save(path)
            rows = [dict(sample_id=str(i), path='a.png', sha256=sha256(path), label=str(i % 2)) for i in range(8)]
            config = dict(image_size=24, task='glaucoma', seed=77, batch_size=4, device='cpu', num_workers=2)
            parallel = loader(rows, config, root, True, epoch=3)
            self.assertEqual(parallel.num_workers, 2)
            actual = list(parallel)
            expected = list(loader(rows, dict(config, num_workers=0), root, True, epoch=3))
            again = list(loader(rows, config, root, True, epoch=3))
            changed_epoch = list(loader(rows, config, root, True, epoch=4))
            for a, b, c in zip(actual, expected, again):
                self.assertTrue(torch.equal(a[0], b[0]))
                self.assertTrue(torch.equal(a[0], c[0]))
                self.assertEqual(a[2], b[2])
            self.assertTrue(any(not torch.equal(a[0], b[0]) for a, b in zip(actual, changed_epoch)))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                list(loader(rows, config, root, True, epoch=3))


if __name__ == '__main__':
    unittest.main()
