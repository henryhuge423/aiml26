import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))


class CohortTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'tools/freeze_training_cohort.py').exists(),'cohort freezer not implemented')
        import freeze_training_cohort
        return freeze_training_cohort

    def row(self,path,source='airogs-light-v2',label='1',patient='UNKNOWN',sha=None):
        return dict(path=path,source=source,target=label,original_label=label,target_definition='proxy',
            patient_group=patient,sha256=sha or path,related_split_group=patient if patient!='UNKNOWN' else 'sha:'+path)

    def test_transitive_duplicate_family_keeps_external_holdout_out_of_training(self):
        rows=[self.row('a'),self.row('b'),self.row('c','hygd-1.1.0',patient='HYGD:1')]
        edges=[('a','b'),('b','c')]
        selected,excluded=self.module().assign_cohort(rows,[],edges,{},7)
        self.assertEqual([r['path'] for r in selected],['c'])
        self.assertEqual(selected[0]['split'],'external_hygd')
        self.assertEqual(len(excluded),2)

    def test_discordant_eyes_same_patient_allowed_but_group_kept_together(self):
        rows=[self.row('a','DATASET_101/ODIR-5K','1','ODIR:3'),
              self.row('b','DATASET_101/ODIR-5K','0','ODIR:3')]
        selected,_=self.module().assign_cohort(rows,[],[],{},7)
        self.assertEqual(len(selected),2)
        self.assertEqual(len({r['related_group'] for r in selected}),1)
        self.assertEqual(len({r['split'] for r in selected}),1)

    def test_conflicting_labels_same_image_family_are_quarantined(self):
        rows=[self.row('a',label='0'),self.row('b',label='1')]
        selected,excluded=self.module().assign_cohort(rows,[],[('a','b')],{},7)
        self.assertEqual(selected,[])
        self.assertTrue(all('conflicting' in r['reason'] for r in excluded))

    def test_external_clinical_truth_not_removed_by_different_development_endpoint(self):
        rows=[self.row('a',label='0'),self.row('h','hygd-1.1.0','1','HYGD:1')]
        selected,excluded=self.module().assign_cohort(rows,[],[('a','h')],{},7)
        self.assertEqual([r['path'] for r in selected],['h'])
        self.assertEqual(selected[0]['label'],'1')

    def test_quality_image_related_to_disease_external_test_excluded(self):
        rows=[self.row('h','hygd-1.1.0',patient='HYGD:1')]
        quality=[dict(path='q',sha256='q',label='0.4',image_id='q')]
        selected,excluded=self.module().assign_cohort(rows,quality,[('h','q')],{},7)
        self.assertFalse(any(r['task']=='quality' for r in selected))
        self.assertTrue(any('protected' in r['reason'] for r in excluded))

    def test_possible_teacher_training_image_family_never_in_final_disease_test(self):
        module=self.module()
        seed=next(s for s in range(100) if module.stable_fraction('a',s)>=.9)
        rows=[self.row('a')];quality=[dict(path='q',sha256='q',label='0.4',image_id='q')]
        selected,_=module.assign_cohort(rows,quality,[('a','q')],{},seed)
        self.assertTrue(all(r['split'] in {'train','val'} for r in selected))

    def test_external_related_to_teacher_training_corpus_quarantined_before_freeze(self):
        rows=[self.row('h','hygd-1.1.0',patient='HYGD:1')]
        quality=[dict(path='q',sha256='q',label='0.4',image_id='q')]
        selected,excluded=self.module().assign_cohort(rows,quality,[('h','q')],{},7)
        self.assertFalse(any(r['task']=='glaucoma' for r in selected))
        self.assertTrue(any('teacher_pretraining' in r['reason'] for r in excluded))

    def test_unknown_patient_remains_unknown_and_unsupported_augmentation_quarantined(self):
        selected,excluded=self.module().assign_cohort([self.row('a'),self.row('e','EyeAugmented')],[],[],{},7)
        self.assertEqual(selected[0]['patient_id_status'],'unknown')
        self.assertTrue(any('augmentation' in r['reason'] for r in excluded))

    def test_technical_quarantine_does_not_erase_external_low_quality_challenge(self):
        rows=[self.row('a'),self.row('h','hygd-1.1.0',patient='HYGD:1')]
        decisions={'a':'quarantine_severe_technical','h':'retain_external_challenge'}
        selected,excluded=self.module().assign_cohort(rows,[],[],decisions,7)
        self.assertEqual([r['path'] for r in selected],['h'])
        self.assertEqual(excluded[0]['path'],'a')

    def test_annotated_overlay_cannot_reenter_via_cross_source_duplicate(self):
        rows=[self.row('original','DATASET_101/ODIR-5K'),
              self.row('copy','Kaggle-Eye_diseases_classification')]
        selected,excluded=self.module().assign_cohort(rows,[],[('original','copy')],{'original':'quarantine_overlay'},7)
        self.assertEqual(selected,[])
        self.assertTrue(all('technical' in r['reason'] for r in excluded))


if __name__=='__main__':unittest.main()
