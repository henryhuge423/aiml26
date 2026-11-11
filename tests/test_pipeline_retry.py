import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'run_training_pipeline', ROOT / 'tools/run_training_pipeline.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class PipelineRetryTests(unittest.TestCase):
    def ensure(self, *args):
        if not hasattr(pipeline, 'ensure_quality_preflight'):
            self.fail('pipeline has no retry-safe quality preflight helper')
        return pipeline.ensure_quality_preflight(*args)

    def test_valid_existing_preflight_is_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            proof = Path(temp) / 'formal_v3_quality'
            proof.mkdir()
            (proof / 'evidence.txt').write_text('valid')
            executed = []
            verified = []

            def execute(*args):
                executed.append(args)

            def verify(directory, names, config_dir):
                verified.append((directory, names, config_dir))
                self.assertEqual((directory / 'evidence.txt').read_text(), 'valid')

            self.ensure(proof, ['quality_student'], Path(temp), execute, verify)

            self.assertEqual(executed, [])
            self.assertEqual(len(verified), 1)
            self.assertEqual(list(Path(temp).glob('formal_v3_quality.backup_*')), [])

    def test_invalid_existing_preflight_is_backed_up_and_rerun(self):
        for invalid_proof in ('{"passed": false}', '{damaged json', '{"fingerprint": "stale"}'):
            with self.subTest(invalid_proof=invalid_proof), tempfile.TemporaryDirectory(dir=ROOT) as temp:
                proof = Path(temp) / 'formal_v3_quality'
                proof.mkdir()
                (proof / 'verification.json').write_text(invalid_proof)
                executions = []

                def verify(directory, names, config_dir):
                    if (directory / 'verification.json').read_text() != '{"passed": true}':
                        raise ValueError('GPU preflight is incomplete or mismatched')

                def execute(stage, args):
                    executions.append((stage, args))
                    self.assertFalse(proof.exists())
                    proof.mkdir()
                    (proof / 'verification.json').write_text('{"passed": true}')

                self.ensure(proof, ['quality_student'], Path(temp), execute, verify)

                backups = list(Path(temp).glob('formal_v3_quality.backup_*'))
                self.assertEqual(len(executions), 1)
                self.assertEqual(executions[0][0], 'quality_preflight')
                self.assertEqual(executions[0][1][0], Path(pipeline.ROOT / 'tools/verify_training_preflight.py'))
                self.assertEqual(executions[0][1][2], proof)
                self.assertEqual(len(backups), 1)
                self.assertEqual((backups[0] / 'verification.json').read_text(), invalid_proof)
                self.assertEqual((proof / 'verification.json').read_text(), '{"passed": true}')

    def test_new_preflight_must_pass_verification(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            proof = Path(temp) / 'formal_v3_quality'

            def execute(stage, args):
                proof.mkdir()
                (proof / 'verification.json').write_text('{"passed": false}')

            def verify(directory, names, config_dir):
                raise ValueError('GPU preflight is incomplete')

            with self.assertRaisesRegex(ValueError, 'incomplete'):
                self.ensure(proof, ['quality_student'], Path(temp), execute, verify)

            self.assertTrue(proof.exists())
            self.assertEqual((proof / 'verification.json').read_text(), '{"passed": false}')
            self.assertEqual(list(Path(temp).glob('formal_v3_quality.backup_*')), [])


if __name__ == '__main__':
    unittest.main()
