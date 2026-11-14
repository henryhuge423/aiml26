"""Plot the matched original rule-development subset; no test data used."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/progress/2026-11-13'
quality = ROOT / 'data/audit/data_expansion_20261113/repaired_boundary_pq.csv'
disease = ROOT / 'runs/expanded_20261113/expanded_degradation50/last_development_predictions.csv'
q = pd.read_csv(quality)
p = pd.read_csv(disease)
p = p[(p['split'] == 'boundary') & (p['condition'] == 'original')]
assert q.sample_id.is_unique and p.sample_id.is_unique
d = q.merge(p, on='sample_id', validate='one_to_one', suffixes=('_q', '_p'))
assert len(d) == len(q) == 1220
assert (d.label_q == d.label_p).all() and (d.source_q == d.source_p).all()
assert d.q.between(0, 1).all() and d.prediction.between(0, 1).all()
d['error'] = (d.prediction > .5) != d.label_p.astype(bool)
d['fn'] = (d.label_p == 1) & (d.prediction <= .5)
d['fp'] = (d.label_p == 0) & (d.prediction > .5)
bins = np.linspace(0, 1, 6)
counts = np.histogram2d(d.q, d.prediction, bins=[bins, bins])[0].T
errors = np.histogram2d(d.q[d.error], d.prediction[d.error], bins=[bins, bins])[0].T
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Arial'], 'axes.unicode_minus':False, 'font.size':9})
for lang in ('zh','en'):
    zh = lang == 'zh'
    fig, ax = plt.subplots(figsize=(10, 3.7), constrained_layout=True)
    for label, color in [(0, '#277892'), (1, '#c7735b')]:
        group = d[(d.label_p == label) & ~d.error]
        ax.scatter(group.prediction.clip(.0001, .9999), group.q, s=12, alpha=.38,
                   color=color, edgecolors='none', label=(f'预测正确：原标签 {label}' if zh else f'Correct: label {label}'), zorder=2)
    for mask, marker, color, label in [
        (d.fn, 'X', '#b53229', '假阴性（漏诊）' if zh else 'False negative'),
        (d.fp, 'D', '#145e9c', '假阳性（误报）' if zh else 'False positive'),
    ]:
        group = d[mask]
        ax.scatter(group.prediction.clip(.0001, .9999), group.q, s=42 if marker == 'X' else 32,
                   marker=marker, color=color, edgecolors='white', linewidths=.55, alpha=.94,
                   label=f'{label} (n={len(group)})', zorder=4)
    ax.set_xscale('logit')
    ax.set_xlim(.00006, .99994)
    ax.set_ylim(-.03, .95)
    ticks = [.0001, .01, .1, .5, .9, .99, .9999]
    ax.set_xticks(ticks, ['0.0001', '0.01', '0.1', '0.5', '0.9', '0.99', '0.9999'])
    ax.axvline(.5, color='#7e8d90', lw=1, ls='--')
    ax.grid(axis='y', color='#e6e6e6', linewidth=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('疾病模型的分数 p（非线性刻度）' if zh else 'Disease-model score p (nonlinear scale)')
    ax.set_ylabel('质量模型分数 q' if zh else 'Quality-model score q')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=.9, fontsize=8)
    fig.savefig(OUT / f'pq-distribution.{lang}.png', dpi=200, facecolor='white')
    plt.close(fig)
result = {'n':len(d), 'positives':int(d.label_p.sum()), 'fn':int(d.fn.sum()), 'fp':int(d.fp.sum()), 'threshold':0.5,
          'counts_by_p_then_q':counts.astype(int).tolist(), 'errors_by_p_then_q':errors.astype(int).tolist(),
          'input_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (quality,disease)}}
(OUT/'pq-distribution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
