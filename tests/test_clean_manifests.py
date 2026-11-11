import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from openpyxl import Workbook

TOOL = Path(__file__).resolve().parents[1] / 'tools/prepare_clean_manifests.py'


class CleaningManifestTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test_tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        assert Path(self.temp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.audit, self.out = [self.root / n for n in ('data', 'audit', 'out')]
        self.audit.mkdir()

    def write_csv(self, name, rows):
        with (self.audit / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def output_csv(self, name):
        with (self.out/name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    def build(self):
        self.assertTrue(TOOL.exists(), 'cleaning manifest builder has not been implemented')
        spec = importlib.util.spec_from_file_location('clean_manifests', TOOL)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.build_manifests(self.data, self.audit, self.out)

    def setup_data(self, conflict=False):
        images = self.data / 'external/fiqs/images'
        originals = self.data / 'external/fiqs/full_size_images'
        images.mkdir(parents=True); originals.mkdir()
        (images/'00135.PNG').write_bytes(b'fixture-resized')
        (originals/'00135.PNG').write_bytes(b'fixture-truncated')
        wb = Workbook(); ws = wb.active
        ws.append(['file_name', 'meanOpinionScore', 'qualityLevel'])
        ws.append(['00135.PNG', 25, 2])
        wb.save(images.parent / 'meta_data.xlsx')
        hygd = self.data / 'external/hygd-1.1.0/Images'
        hygd.mkdir(parents=True)
        (hygd/'65_5.jpg').write_bytes(b'same'); (hygd/'110_0.jpg').write_bytes(b'same')
        common = dict(source='hygd-1.1.0', sha256=hashlib.sha256(b'same').hexdigest(), target='1', original_label='GON+',
                      related_split_group='HYGD:110', train_eligible='False', split='UNASSIGNED',
                      pending_reasons='disc_gradability_pending')
        work = [dict(common, path='external/hygd-1.1.0/Images/65_5.jpg', patient_group='HYGD:65'),
                dict(common, path='external/hygd-1.1.0/Images/110_0.jpg', patient_group='HYGD:110')]
        if conflict:work[1]['target']='0'
        self.write_csv('glaucoma_candidate_worklist.csv', work)
        manifest = [dict(path='external/fiqs/images/00135.PNG', status='full_decode_ok', sha256=hashlib.sha256(b'fixture-resized').hexdigest()),
                    dict(path='external/fiqs/full_size_images/00135.PNG', status='decode_failed', sha256=hashlib.sha256(b'fixture-truncated').hexdigest())]
        manifest += [dict(path=r['path'], status='full_decode_ok', sha256=r['sha256']) for r in work]
        self.write_csv('image_manifest.csv', manifest)

    def test_deduplicates_hygd_without_losing_cross_patient_aliases_or_eligibility(self):
        self.setup_data(); result = self.build()
        selected = self.output_csv('hygd_canonical.csv')
        aliases = self.output_csv('hygd_aliases.csv')
        self.assertEqual(len(selected), 1)
        self.assertEqual(set(aliases[0]['recorded_patient_groups'].split('|')), {'HYGD:65','HYGD:110'})
        self.assertEqual(selected[0]['train_eligible'], 'False')
        self.assertEqual(selected[0]['split'], 'UNASSIGNED')
        self.assertEqual(result['hygd_unique_images'], 1)

    def test_selects_healthy_resized_quality_image_without_deleting_bad_original(self):
        self.setup_data(); self.build()
        selected = self.output_csv('fiqs_quality.csv')
        self.assertEqual(selected[0]['path'], 'external/fiqs/images/00135.PNG')
        self.assertEqual(float(selected[0]['human_mos_normalized']), 0.25)
        self.assertEqual(selected[0]['train_eligible'], 'False')
        self.assertTrue((self.data/'external/fiqs/full_size_images/00135.PNG').exists())
        self.assertEqual(len(self.output_csv('technical_exclusions.csv')), 1)

    def test_rejects_conflicting_labels_in_identical_content(self):
        self.setup_data(conflict=True)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.build()

    def test_rejects_source_changed_since_audit(self):
        self.setup_data()
        (self.data/'external/fiqs/images/00135.PNG').write_bytes(b'changed-after-audit')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()


if __name__ == '__main__':
    unittest.main()
