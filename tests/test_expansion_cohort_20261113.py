"""Checks for transitive image grouping and real-quality group allocation."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'freeze_expansion_training_20261113', ROOT / 'tools/freeze_expansion_training_20261113.py')
FREEZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZER)


class ExpansionCohortTests(unittest.TestCase):
    def test_transitive_patient_and_photo_edges_merge_components(self):
        graph = FREEZER.Groups()
        graph.union('patient_left_photo', 'left_reencoding')
        graph.union('right_photo', 'followup')
        graph.union('patient_left_photo', 'right_photo')
        self.assertEqual(graph.find('left_reencoding'), graph.find('followup'))
        self.assertNotEqual(graph.find('left_reencoding'), graph.find('another_patient'))

    def test_real_quality_strata_keep_whole_groups_in_each_role(self):
        groups = {}
        for index in range(40):
            groups[str(index)] = [dict(source='fives', source_family='joint_family', label='1',
                                       quality_fives_any_poor_dimension='True')] * 2
        roles = FREEZER.allocate_groups(groups)
        self.assertEqual(set(roles.values()), {'train', 'val', 'boundary', 'test'})
        self.assertEqual(len(roles), 40)
        self.assertEqual(roles, FREEZER.allocate_groups(dict(reversed(list(groups.items())))))
        self.assertEqual(sum(role == 'train' for role in roles.values()), 24)
        self.assertEqual(sum(role == 'val' for role in roles.values()), 4)

    def test_quality_grade_interpretation_is_source_specific(self):
        self.assertTrue(FREEZER.poor(dict(source='mshf', label='1', quality_overall_official='0')))
        self.assertFalse(FREEZER.poor(dict(source='mshf', label='1', quality_overall_official='1')))
        self.assertFalse(FREEZER.poor(dict(source='fives', label='0', quality_fives_any_poor_dimension='True')))
        self.assertFalse(FREEZER.poor(dict(source='keh', label='1', quality_overall_official='0')))


if __name__ == '__main__':
    unittest.main()
