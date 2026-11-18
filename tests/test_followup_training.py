import unittest
import torch
from PIL import Image
from training.data import epoch_indices
from training.fundus_degradation import FundusDegradation


class FollowupTrainingTests(unittest.TestCase):
    def test_origin_quota_preserves_prefix_budget_and_replay(self):
        rows=[dict(source='old',label=str(i%2),origin_cohort='formal_v3') for i in range(200)]
        rows += [dict(source='new',label=str(i%2),origin_cohort='expansion_20261113') for i in range(100)]
        order=epoch_indices(rows,19,0,True,'inverse_sqrt',.65)
        self.assertEqual(order,epoch_indices(rows,19,0,True,'inverse_sqrt',.65))
        for prefix in (16,32,119,300):
            self.assertLessEqual(abs(sum(i<200 for i in order[:prefix])-.65*prefix),.5)
        self.assertNotEqual(order,epoch_indices(rows,20,0,True,'inverse_sqrt',.65))
        counts=torch.tensor([100.]*200+[50.]*100,dtype=torch.double)
        historical=torch.multinomial(counts**-.5,300,replacement=True,generator=torch.Generator().manual_seed(19)).tolist()
        self.assertEqual(epoch_indices(rows,19,0,True,'inverse_sqrt'),historical)
        with self.assertRaisesRegex(ValueError,'both recognized'):
            epoch_indices(rows[:200],19,0,True,'inverse_sqrt',.65)
        with self.assertRaisesRegex(ValueError,'fraction'):
            epoch_indices(rows,19,0,True,'inverse_sqrt',1.)

    def test_defocus_only_replays_and_keeps_application_draws(self):
        image=Image.new('RGB',(64,64),(150,80,50))
        mixed=FundusDegradation(.5)
        defocus=FundusDegradation(.5,families=['defocus'])
        applied=0
        for seed in range(30):
            variant,record=defocus.augment(image,seed=seed)
            _,comparison=mixed.augment(image,seed=seed)
            self.assertEqual(record['applied'],comparison['applied'])
            self.assertEqual(variant.tobytes(),defocus.replay(image,record).tobytes())
            if record['applied']:
                applied+=1;self.assertEqual(record['family'],'defocus')
        self.assertGreater(applied,0)
        with self.assertRaisesRegex(ValueError,'families'):
            FundusDegradation(.5,families=['unrecognized'])


if __name__=='__main__':unittest.main()
