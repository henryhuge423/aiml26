import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class KnownLinkTests(unittest.TestCase):
    def module(self):
        path=ROOT/'tools/finalize_eyeq_overlap.py'
        self.assertTrue(path.exists(),'full historical linkage reconciliation is missing')
        spec=importlib.util.spec_from_file_location('finalize_eyeq',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_unknown_patients_are_not_grouped_and_known_patients_are_namespaced(self):
        m=self.module()
        rows=[dict(path='a',source='odir',patient_group='7'),dict(path='b',source='odir',patient_group='7'),
              dict(path='c',source='hygd',patient_group='7'),dict(path='d',source='odir',patient_group='UNKNOWN'),
              dict(path='e',source='odir',patient_group='UNKNOWN')]
        self.assertEqual(m.known_edges(rows,[]),[('b','a')])

    def test_both_fiqs_resolutions_connect_even_when_excluded_from_active_manifest(self):
        m=self.module()
        self.assertEqual(m.known_edges([],[dict(path='external/fiqs/images/00000.PNG')]),
                         [('external/fiqs/images/00000.PNG','external/fiqs/full_size_images/00000.PNG')])

    def test_finalizer_rejects_inputs_changed_since_pixel_screen(self):
        m=self.module()
        self.assertTrue(hasattr(m,'verify_screen_inputs'),'screen binding validation is missing')
        with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as scratch:
            directory=Path(scratch);screen=directory/'screen';screen.mkdir()
            source=directory/'source.csv';source.write_text('original metadata')
            (screen/'status.json').write_text(json.dumps(dict(stage='completed')))
            (screen/'inputs.json').write_text(json.dumps({'source.csv':hashlib.sha256(source.read_bytes()).hexdigest()}))
            m.verify_screen_inputs(screen,directory)
            source.write_text('changed metadata')
            with self.assertRaisesRegex(ValueError,'changed'):
                m.verify_screen_inputs(screen,directory)


if __name__=='__main__':unittest.main()
