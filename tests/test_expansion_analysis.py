"""Statistical correctness checks for the paired November 13 analysis."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from tools.analyze_expansion_results_20261113 import align_rows, group_weights, weighted_metrics, validate_complete_prediction_rows


def test_weighted_ranking_matches_repeated_observations_with_ties():
    y=np.array([0,1,0,1,1,0]);p=np.array([.2,.2,.5,.8,.8,.8])
    weights=np.array([[1,2,3,2,1,1],[3,0,1,1,2,4]])
    actual=weighted_metrics(y,p,weights)
    for i,w in enumerate(weights):
        repeated_y=np.repeat(y,w);repeated_p=np.repeat(p,w)
        assert actual['auc'][i]==pytest.approx(roc_auc_score(repeated_y,repeated_p))
        assert actual['auprc'][i]==pytest.approx(average_precision_score(repeated_y,repeated_p))
        assert actual['brier'][i]==pytest.approx(np.mean((repeated_y-repeated_p)**2))


def test_group_bootstrap_keeps_linked_images_together():
    weights=group_weights(['patient_a','patient_b','patient_a','patient_c'],repeats=100)
    np.testing.assert_array_equal(weights[:,0],weights[:,2])
    np.testing.assert_array_equal(weights[:,[0,1,3]].sum(axis=1),np.full(100,3))
    np.testing.assert_array_equal(weights,group_weights(['patient_a','patient_b','patient_a','patient_c'],repeats=100))


def test_paired_validation_rejects_changed_label_or_duplicate():
    row=dict(sample_id='a',source='s',label='0',related_group='g',target_definition='t',
             origin_cohort='o',split='boundary',condition='original',real_poor_quality='False',prediction='.4')
    assert align_rows([[row],[{**row,'prediction':'.7'}]])
    with pytest.raises(ValueError,match='metadata differ'):
        align_rows([[row],[{**row,'label':'1'}]])
    with pytest.raises(ValueError,match='duplicate'):
        align_rows([[row,row]])


def test_completeness_rejects_consistently_missing_or_protected_predictions():
    manifest=[dict(sample_id='a',split='val'),dict(sample_id='b',split='boundary'),dict(sample_id='c',split='test')]
    rows=[dict(sample_id='a',condition='original')]+[dict(sample_id='b',condition=c)
        for c in ('original','defocus_2_at512','exposure_070')]
    validate_complete_prediction_rows(rows,manifest)
    with pytest.raises(ValueError,match='incomplete'):
        validate_complete_prediction_rows(rows[1:],manifest)
    with pytest.raises(ValueError,match='incomplete'):
        validate_complete_prediction_rows(rows+[dict(sample_id='c',condition='original')],manifest)
