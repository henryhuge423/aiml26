"""Regression checks for the calibrated defocus research arm."""
import unittest

import numpy as np
from PIL import Image

from training.calibrated_defocus import CalibratedDefocus
from training.fundus_degradation import FundusDegradation


class CalibratedDefocusTest(unittest.TestCase):
    def test_only_defocus_changes_and_remains_replayable(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[8:72, 8:72] = (130, 80, 40)
        image[16:64:4, 16:64, :] = (195, 130, 70)
        image[1, 1] = (2, 1, 0)
        original = Image.fromarray(image)
        baseline = FundusDegradation(.5)
        calibrated = CalibratedDefocus(.5, [[9., 1.08], [6.5, .9]])
        seen = set()
        for seed in range(60):
            expected, before = baseline.augment(original, seed=seed)
            actual, after = calibrated.augment(original, seed=seed)
            self.assertEqual(before['family'], after['family'])
            self.assertEqual(before['applied'], after['applied'])
            seen.add(before['family'])
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(calibrated.replay(original, after)))
            if before['family'] == 'defocus':
                self.assertFalse(np.array_equal(actual, expected))
                self.assertEqual(tuple(np.asarray(actual)[1, 1]), (2, 1, 0))
                self.assertIn(after['parameters']['radius_at_512'], (9., 6.5))
            else:
                self.assertEqual(before, after)
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(seen, {'original', 'illumination', 'defocus', 'scatter'})

    def test_rejects_nonfinite_profile_and_wrong_replay_source(self):
        with self.assertRaises(ValueError):
            CalibratedDefocus(1, [[1, np.nan]])

        calibrated = CalibratedDefocus(1, [[9., 1.08]], families=['defocus'])
        source = Image.new('RGB', (20, 20), (140, 90, 50))
        _, record = calibrated.augment(source, seed=12)
        different = Image.new('RGB', (20, 20), (141, 90, 50))
        with self.assertRaises(ValueError):
            calibrated.replay(different, record)


if __name__ == '__main__':
    unittest.main()
