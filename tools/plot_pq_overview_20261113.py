"""Plot frozen p and q scores separately by development role; no test data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/progress/2026-11-13'
quality = ROOT/'runs/expanded_20261113/pq_quality_20261114/quality_scores.csv'
disease = ROOT/'runs/expanded_20261113/expanded_degradation50/last_development_predictions.csv'
verification = ROOT/'runs/expanded_20261113/pq_quality_20261114/verification.json'
proof = json.loads(verification.read_text(encoding='utf-8'))
assert hashlib.sha256(quality.read_bytes()).hexdigest() == proof['scores_sha256']
q, p = pd.read_csv(quality), pd.read_csv(disease)
p = p[(p.condition == 'original') & p['split'].isin(['val', 'boundary'])]
assert q.sample_id.is_unique and p.sample_id.is_unique
d = q.merge(p, on='sample_id', validate='one_to_one', suffixes=('_q','_p'))
assert len(d) == len(q) == len(p) == 4099
assert (d.label_q == d.label_p).all() and (d.source_q == d.source_p).all() and (d.split_q == d.split_p).all()
assert d.q.between(0,1).all() and d.prediction.between(0,1).all()
d['error'] = (d.prediction > .5) != d.label_p.astype(bool)
d['fn'] = (d.label_p == 1) & (d.prediction <= .5)
d['fp'] = (d.label_p == 0) & (d.prediction > .5)
pbins = np.array([.0001,.01,.1,.5,.9,.99,.9999])
qticks = np.linspace(0,1,6)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Arial'], 'axes.unicode_minus':False, 'font.size':9})
result = {'threshold':.5, 'p_bin_edges_display':pbins.tolist(), 'q_bin_edges':qticks.tolist(), 'roles':{},
          'input_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (quality,disease,verification)}}
for role, expected in [('boundary',2053),('val',2046)]:
    data = d[d.split_q == role].copy()
    assert len(data) == expected
    # Histogram uses the same display clipping as the scatter; 0.5 falls on a bin edge.
    display_p = data.prediction.clip(.0001,.9999).to_numpy()
    counts = np.histogram2d(data.q, display_p, bins=[qticks,pbins])[0].astype(int)
    errors = np.histogram2d(data.q[data.error], display_p[data.error], bins=[qticks,pbins])[0].astype(int)
    assert counts.sum() == len(data) and errors.sum() == data.error.sum()
    result['roles'][role] = dict(n=len(data), positives=int(data.label_p.sum()), fn=int(data.fn.sum()), fp=int(data.fp.sum()), counts_by_q_then_p=counts.tolist(), errors_by_q_then_p=errors.tolist())
    for lang in ('zh','en'):
        zh = lang == 'zh'
        fig, (ax, heat) = plt.subplots(2, 1, figsize=(10,7.5), constrained_layout=True, gridspec_kw={'height_ratios':[1.15,1]})
        for label,color in [(0,'#277892'),(1,'#c7735b')]:
            group=data[(data.label_p == label) & ~data.error]
            ax.scatter(group.prediction.clip(.0001,.9999), group.q, s=11, alpha=.32, color=color, edgecolors='none', label=(f'预测正确：原标签 {label}' if zh else f'Correct: label {label}'), zorder=2)
        for mask, marker, color, label in [(data.fn,'X','#b53229','假阴性（漏诊）' if zh else 'False negative'),(data.fp,'D','#145e9c','假阳性（误报）' if zh else 'False positive')]:
            group=data[mask]
            ax.scatter(group.prediction.clip(.0001,.9999),group.q,s=38 if marker=='X' else 30,marker=marker,color=color,edgecolors='white',linewidths=.4,alpha=.94,label=f'{label} (n={len(group)})',zorder=4)
        ax.set_xscale('logit'); ax.set_xlim(.00006,.99994); ax.set_ylim(-.03,1.02)
        ax.set_xticks(pbins,['0.0001','0.01','0.1','0.5','0.9','0.99','0.9999'])
        ax.axvline(.5,color='#7e8d90',lw=1,ls='--'); ax.grid(axis='y',color='#e6e6e6',lw=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('疾病分数 p（非线性刻度）' if zh else 'Disease score p (nonlinear scale)')
        ax.set_ylabel('质量分数 q' if zh else 'Quality score q')
        ax.legend(loc='lower right',frameon=True,facecolor='white',framealpha=.92,fontsize=8,ncol=2)
        ax.set_title(('规则开发集' if role=='boundary' else '模型验证集')+f'：{len(data):,} 张图像' if zh else ('Rule-development set' if role=='boundary' else 'Model-validation set')+f': {len(data):,} images')
        rates=np.divide(errors,counts,out=np.full(counts.shape,np.nan),where=counts>0)*100
        image=heat.imshow(rates,origin='lower',aspect='auto',cmap='OrRd',norm=Normalize(0,100),extent=(0,6,0,5))
        for iy in range(5):
            for ix in range(6):
                n=counts[iy,ix]; k=errors[iy,ix]
                if 0<n<5: heat.add_patch(Rectangle((ix,iy),1,1,fill=False,hatch='///',edgecolor='#777',linewidth=0))
                label=(f'{rates[iy,ix]:.0f}%\n{k}/{n}' if n else '—\n0/0')
                heat.text(ix+.5,iy+.5,label,ha='center',va='center',fontsize=8,color='white' if n and rates[iy,ix]>55 else '#222')
        heat.set_xticks(np.arange(7),['0.0001','0.01','0.1','0.5','0.9','0.99','0.9999'])
        heat.set_yticks(np.arange(6),[f'{x:.1f}' for x in qticks])
        heat.set_xlabel('疾病分数 p（非线性分箱）' if zh else 'Disease score p (nonlinear bins)')
        heat.set_ylabel('质量分数 q' if zh else 'Quality score q')
        heat.set_title('每格错判率：错判数 / 图像数；斜线表示少于 5 张' if zh else 'Error rate per cell: errors / images; hatching: fewer than 5 images')
        fig.colorbar(image,ax=heat,pad=.012,label='错判率（%）' if zh else 'Error rate (%)')
        fig.savefig(OUT/f'pq-distribution-{role}.{lang}.png',dpi=200,facecolor='white')
        plt.close(fig)
(OUT/'pq-distribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
