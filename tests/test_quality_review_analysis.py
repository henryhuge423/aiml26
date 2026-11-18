import unittest
import numpy as np
import pandas as pd
from tools.analyze_quality_errors_20261115 import review_counts,crossfit,bootstrap_review_capture


class QualityReviewAnalysisTests(unittest.TestCase):
    def test_capacity_and_ties_do_not_consult_outcomes(self):
        data=pd.DataFrame(dict(sample_id=['d','b','a','c','e'],error=[1,0,1,0,0],fn=[1,0,1,0,0]))
        statistics,mask=review_counts(data,np.zeros(5),.3)
        self.assertEqual(statistics['k'],2)
        self.assertEqual(np.flatnonzero(mask).tolist(),[1,2])
        changed=data.assign(error=1-data.error,fn=0)
        _,again=review_counts(changed,np.zeros(5),.3)
        np.testing.assert_array_equal(mask,again)
        self.assertEqual(statistics['remaining_images'],3)

    def test_linked_images_receive_the_same_held_out_fold(self):
        rng=np.random.default_rng(71)
        data=pd.DataFrame(dict(related_group=np.repeat(np.arange(50),2),
            label=np.tile([0,1],50),error=np.repeat(np.arange(50)%2,2),
            prediction=rng.uniform(.01,.99,100),q=rng.uniform(0,1,100),source='s'))
        risk,fold=crossfit(data,'q')
        self.assertTrue(np.isfinite(risk).all())
        for i in range(50):self.assertEqual(fold[2*i],fold[2*i+1])
        again,again_fold=crossfit(data,'q')
        np.testing.assert_array_equal(risk,again);np.testing.assert_array_equal(fold,again_fold)

    def test_bootstrap_capacity_matches_explicit_repeated_images(self):
        data=pd.DataFrame(dict(sample_id=['d','b','a','c'],error=[1,0,1,0],fn=[1,0,1,0]))
        risk=np.array([.5,.5,.9,.1]);weights=np.array([[1,3,2,1],[0,1,2,4]],float)
        captures=bootstrap_review_capture(data,risk,weights,.3)
        for index,w in enumerate(weights):
            indices=np.repeat(np.arange(4),w.astype(int))
            repeated=data.iloc[indices]
            statistics,_=review_counts(repeated,risk[indices],.3)
            self.assertEqual(captures[index],statistics['captured_errors'])


if __name__=='__main__':unittest.main()
