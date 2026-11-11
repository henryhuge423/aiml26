import importlib.util
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ConfirmationCacheTests(unittest.TestCase):
    def module(self):
        path=ROOT/'tools/continue_eyeq_confirmation.py'
        self.assertTrue(path.exists(),'cached confirmation is missing')
        spec=importlib.util.spec_from_file_location('cached_confirmation',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_cached_compare_agrees_with_original_on_duplicates_and_nonmatches(self):
        m=self.module();rng=np.random.default_rng(12)
        original=rng.integers(0,256,(384,384),dtype=np.uint8)
        for orientation in range(8):
            for candidate in (m.A.orient(original,orientation),rng.integers(0,256,(384,384),dtype=np.uint8)):
                distance=(m.A.phash(m.A.orient(original,orientation))^m.A.phash(candidate)).bit_count()
                expected=m.A.compare(original,candidate,orientation)
                result=m.compare_cached(original,candidate,orientation,distance,m.moments(original),m.moments(candidate))
                self.assertEqual(result['confirmed'],expected['confirmed'])
                self.assertEqual(result['phash_distance'],expected['phash_distance'])
                self.assertAlmostEqual(result['ssim'],expected['ssim'],places=10)

    def test_incomplete_or_unsealed_candidate_generation_is_rejected(self):
        m=self.module()
        self.assertTrue(hasattr(m,'verify_candidate_proof'),'candidate completeness validation is missing')
        m.verify_candidate_proof(dict(stage='confirmation',processed=1000,pairs=1106981),1106981)
        for proof,count in ((dict(stage='candidates',processed=1000,pairs=3),3),
                            (dict(stage='confirmation',processed=0,pairs=3),3),
                            (dict(stage='confirmation',processed=1,pairs=3),2)):
            with self.assertRaises(ValueError):m.verify_candidate_proof(proof,count)

    def test_cached_compare_agrees_near_ssim_confirmation_threshold(self):
        m=self.module();rng=np.random.default_rng(14)
        original=rng.integers(0,256,(384,384),dtype=np.uint8)
        noise=rng.normal(size=original.shape)
        for amplitude in (17.,19.,20.,21.,23.):
            candidate=np.clip(original.astype(float)+amplitude*noise,0,255).astype(np.uint8)
            for orientation in (0,1,4,7):
                b=m.A.orient(candidate,orientation)
                distance=(m.A.phash(m.A.orient(original,orientation))^m.A.phash(b)).bit_count()
                expected=m.A.compare(original,b,orientation)
                result=m.compare_cached(original,b,orientation,distance,m.moments(original),m.moments(b))
                self.assertEqual(result['confirmed'],expected['confirmed'])
                self.assertAlmostEqual(result['ssim'],expected['ssim'],places=10)

    def test_large_cache_size_does_not_overflow_windows_numpy_int32(self):
        m=self.module()
        self.assertTrue(hasattr(m,'array_nbytes'),'large array size guard is missing')
        self.assertEqual(m.array_nbytes((46716,384,384),np.uint8),6888554496)
        self.assertEqual(m.array_nbytes((46716,2),np.float64),747456)


if __name__=='__main__':unittest.main()
