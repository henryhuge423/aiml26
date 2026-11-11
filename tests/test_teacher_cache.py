import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class CacheJournalTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'.test_tmp';scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch)
        assert Path(self.tmp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'cache.sqlite'

    def storage(self):
        self.assertTrue((ROOT/'training/cache.py').exists(),'transactional teacher cache not implemented')
        from training.cache import TeacherCacheStore
        return TeacherCacheStore

    def test_committed_row_survives_reopen_even_without_csv_export(self):
        store=self.storage()(self.path,{'teacher_impl':'one'})
        row={'sample_id':'id1','sha256':'abc','teacher_q':.3}
        store.put(row);store.close()
        reopened=self.storage()(self.path,{'teacher_impl':'one'})
        self.assertEqual(reopened.records(),{'id1':row});reopened.close()

    def test_changed_implementation_rejects_partial_cache(self):
        store=self.storage()(self.path,{'teacher_impl':'one'});store.close()
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            self.storage()(self.path,{'teacher_impl':'changed'})

    def test_duplicate_id_with_different_target_is_rejected(self):
        store=self.storage()(self.path,{'teacher_impl':'one'})
        self.addCleanup(store.close)
        store.put({'sample_id':'id1','sha256':'abc','teacher_q':.3})
        with self.assertRaisesRegex(ValueError,'conflicting'):
            store.put({'sample_id':'id1','sha256':'abc','teacher_q':.9})

    def test_rebinding_requires_identical_allowed_quality_identity_hash_and_split(self):
        self.storage()
        from training import cache
        self.assertTrue(hasattr(cache,'validated_reuse_rows'),'cache rebinding guard not implemented')
        from training.cache import validated_reuse_rows
        new=[{'sample_id':'q1','sha256':'abc','path':'photo.png','split':'train'}]
        old={'q1':dict(new[0],teacher_q='.4',crop_std='.01')}
        self.assertEqual(validated_reuse_rows(new,old),[old['q1']])
        for field,value in [('sha256','different'),('split','test'),('path','different.png')]:
            changed=[dict(new[0],**{field:value})]
            with self.assertRaises(ValueError):validated_reuse_rows(changed,old)


if __name__=='__main__':unittest.main()
