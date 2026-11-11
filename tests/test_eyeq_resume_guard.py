import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'tools/resume_eyeq_confirmation.py'


class EyeqResumeGuardTests(unittest.TestCase):
    def load_launcher(self):
        if not LAUNCHER.is_file():
            self.fail('guarded resume launcher does not exist')
        spec = importlib.util.spec_from_file_location('resume_eyeq_confirmation', LAUNCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_matching_real_implementation_evidence_allows_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            screen = Path(folder) / 'screen'
            cache = screen / 'confirmation_cache'
            cache.mkdir(parents=True)
            implementation = Path(folder) / 'continue_eyeq_confirmation.py'
            implementation.write_text('def compare_cached(a, b): return a == b\n')
            implementation_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
            (screen / 'cached_confirmation.json').write_text(json.dumps(
                {'implementation_sha256': implementation_hash}))
            calls = []

            result = self.load_launcher().guarded_resume(
                screen, implementation, lambda: calls.append('resumed') or 'ok')

            self.assertEqual(result, 'ok')
            self.assertEqual(calls, ['resumed'])

    def test_mismatched_real_implementation_evidence_rejects_without_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            screen = Path(folder) / 'screen'
            cache = screen / 'confirmation_cache'
            cache.mkdir(parents=True)
            implementation = Path(folder) / 'continue_eyeq_confirmation.py'
            implementation.write_text('def compare_cached(a, b): return a == b\n')
            (screen / 'cached_confirmation.json').write_text(json.dumps(
                {'implementation_sha256': 'stale implementation hash'}))
            calls = []

            with self.assertRaisesRegex(ValueError, 'implementation'):
                self.load_launcher().guarded_resume(
                    screen, implementation, lambda: calls.append('resumed'))

            self.assertEqual(calls, [])

    def test_binding_implementation_if_present_must_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            screen = Path(folder) / 'screen'
            cache = screen / 'confirmation_cache'
            cache.mkdir(parents=True)
            implementation = Path(folder) / 'continue_eyeq_confirmation.py'
            implementation.write_text('def compare_cached(a, b): return a == b\n')
            implementation_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
            (screen / 'cached_confirmation.json').write_text(json.dumps(
                {'implementation_sha256': implementation_hash}))
            (cache / 'binding.json').write_text(json.dumps(
                {'implementation_sha256': 'stale cache binding'}))

            with self.assertRaisesRegex(ValueError, 'binding'):
                self.load_launcher().guarded_resume(screen, implementation, lambda: None)

    def test_binding_implementation_mismatch_is_rejected_without_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            screen = Path(folder) / 'screen'
            cache = screen / 'confirmation_cache'
            cache.mkdir(parents=True)
            implementation = Path(folder) / 'continue_eyeq_confirmation.py'
            implementation.write_text('def compare_cached(a, b): return a == b\n')
            (cache / 'binding.json').write_text(json.dumps(
                {'implementation_sha256': 'stale cache binding'}))

            with self.assertRaisesRegex(ValueError, 'binding'):
                self.load_launcher().guarded_resume(screen, implementation, lambda: None)

    def test_legacy_binding_without_evidence_uses_original_rebuild_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            screen = Path(folder) / 'screen'
            cache = screen / 'confirmation_cache'
            cache.mkdir(parents=True)
            implementation = Path(folder) / 'continue_eyeq_confirmation.py'
            implementation.write_text('def compare_cached(a, b): return a == b\n')
            (cache / 'binding.json').write_text(json.dumps({'candidates_sha256': 'bound'}))
            calls = []

            result = self.load_launcher().guarded_resume(
                screen, implementation, lambda: calls.append('rebuild') or 'ok')

            self.assertEqual(result, 'ok')
            self.assertEqual(calls, ['rebuild'])


if __name__ == '__main__':
    unittest.main()
