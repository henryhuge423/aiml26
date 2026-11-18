"""Compare prespecified review-score shapes using grouped development folds."""
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.rehearse_proposal_review_20261116 import (
    SOURCES, counts, data_for_analysis, margin, operating_points, ranked_mask, upper_tail, write_json,
)


def run(dev, threshold):
    dev = dev.sort_values('sample_id').reset_index(drop=True)
    y, p, q = (dev[name].to_numpy() for name in ('label', 'p_a', 'q_eyeq'))
    q = np.clip(q, 0, 1)
    uncertainty = margin(p, threshold)
    risks = {'linear': uncertainty + (1-q), 'quadratic': uncertainty + (1-q)**2,
             'sqrt': uncertainty + np.sqrt(1-q)}
    strata = dev.source.where(dev.source.isin(SOURCES), 'other') + ':' + dev.label.astype(str)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261116).split(dev, strata, dev.related_group))
    out = {}
    order = np.argsort(-uncertainty, kind='stable')
    for name, score in risks.items():
        mask = np.zeros(len(y), dtype=bool)
        cuts = []
        for train, held in folds:
            cut = upper_tail(score[train], .1)
            cuts.append(cut)
            mask[held] = score[held] >= cut
        base = ranked_mask(order, int(mask.sum()))
        out[name] = {'quality': counts(y, p, threshold, mask),
                     'uncertainty': counts(y, p, threshold, base),
                     'fold_cutoffs': cuts, 'full_development_cutoff': upper_tail(score, .1)}
    out['no_review'] = counts(y, p, threshold, np.zeros(len(y), dtype=bool))
    return out


def main():
    val, dev, provenance = data_for_analysis()
    airogs_val = val[val.source.eq(SOURCES[0])]
    airogs_dev = dev[dev.source.eq(SOURCES[0])]
    pooled_t = operating_points(val)[0][0]
    airogs_t = operating_points(airogs_val)[0][0]
    result = {'pooled_95_specificity': run(dev, pooled_t), 'pooled_0_5': run(dev, .5),
              'airogs_95_specificity': run(airogs_dev, airogs_t), 'airogs_0_5': run(airogs_dev, .5),
              'thresholds': {'pooled': pooled_t, 'airogs': airogs_t},
              'method': 'five group-held folds; thresholds from separate model validation; fixed scores, no shape selection',
              'provenance': provenance}
    destination = ROOT / 'docs/progress/2026-11-18/nonlinear-exploration.json'
    write_json(destination, result)
    for cohort in ('pooled_95_specificity', 'pooled_0_5', 'airogs_95_specificity', 'airogs_0_5'):
        print(cohort)
        for name in ('linear', 'quadratic', 'sqrt'):
            a, b = result[cohort][name]['quality'], result[cohort][name]['uncertainty']
            print(name, 'k', a['reviewed'], 'errors', a['errors'], b['errors'], 'misses', a['fn'], b['fn'])


if __name__ == '__main__':
    main()
