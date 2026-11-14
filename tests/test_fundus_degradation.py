import json
import unittest

import numpy as np
from PIL import Image, ImageDraw

from training.fundus_degradation import FAMILIES, FundusDegradation, retinal_context


def fundus(size=128):
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    draw.ellipse((size // 8, size // 8, size * 7 // 8, size * 7 // 8), fill=(155, 70, 45))
    for x in range(size // 4, size * 3 // 4, 4):
        draw.line((x, size // 4, x, size * 3 // 4), fill=(80, 30, 20))
    draw.ellipse((size // 3, size // 3, size // 2, size // 2), fill=(225, 180, 120))
    return image


class FundusDegradationTests(unittest.TestCase):
    def test_replay_all_families_and_preserve_original_padding(self):
        image = fundus()
        before = image.tobytes()
        found = set()
        augmentation = FundusDegradation(1)
        for seed in range(30):
            changed, record = augmentation.augment(image, seed=seed)
            replay = augmentation.replay(image, json.loads(json.dumps(record)))
            self.assertEqual(changed.tobytes(), replay.tobytes())
            self.assertEqual(image.tobytes(), before)
            self.assertEqual(changed.size, image.size)
            self.assertEqual(changed.mode, "RGB")
            background = np.asarray(image).max(2) <= 2
            np.testing.assert_array_equal(np.asarray(changed)[background], np.asarray(image)[background])
            self.assertNotEqual(changed.tobytes(), before)
            found.add(record["family"])
        self.assertEqual(found, set(FAMILIES))

    def test_probability_zero_and_blank_are_exact(self):
        image = fundus()
        changed, record = FundusDegradation(0).augment(image, seed=11)
        self.assertEqual(changed.tobytes(), image.tobytes())
        self.assertFalse(record["applied"])
        blank = Image.new("RGB", (32, 24))
        changed, record = FundusDegradation(1).augment(blank, seed=2)
        self.assertEqual(changed.tobytes(), blank.tobytes())
        self.assertFalse(record["applied"])
        with self.assertRaises(ValueError):
            FundusDegradation(float("nan"))

    def test_seed_probability_nested_without_global_rng_changes(self):
        image = fundus()
        np.random.seed(3)
        expected = np.random.get_state()
        a, b = FundusDegradation(.3), FundusDegradation(.5)
        for seed in range(30):
            x, p = a.augment(image, seed=seed)
            y, q = b.augment(image, seed=seed)
            if p["applied"]:
                self.assertTrue(q["applied"])
                self.assertEqual(p["parameters"], q["parameters"])
                self.assertEqual(x.tobytes(), y.tobytes())
        actual = np.random.get_state()
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2:], expected[2:])

    def test_scale_relative_to_visible_field_and_reject_wrong_replay(self):
        small, large = fundus(128), fundus(256)
        augmentation = FundusDegradation(1)
        for seed in range(30):
            _, a = augmentation.augment(small, seed=seed)
            _, b = augmentation.augment(large, seed=seed)
            if "gaussian_sigma_pixels" in a["parameters"]:
                self.assertEqual(a["parameters"]["gaussian_sigma_at_512"], b["parameters"]["gaussian_sigma_at_512"])
                ratio = b["parameters"]["gaussian_sigma_pixels"] / a["parameters"]["gaussian_sigma_pixels"]
                self.assertAlmostEqual(ratio, retinal_context(large)["field_diameter_pixels"] / retinal_context(small)["field_diameter_pixels"])
        altered = small.copy()
        altered.putpixel((64, 64), (255, 255, 255))
        _, record = augmentation.augment(small, seed=7)
        with self.assertRaisesRegex(ValueError, "source image"):
            augmentation.replay(altered, record)


if __name__ == "__main__":
    unittest.main()
