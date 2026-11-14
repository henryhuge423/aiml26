"""Recreate aggregate development analysis; never load final-test predictions."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]

def main():
    manifest = ROOT/'data/prepared/formal_v3/manifest.csv'
    rows = [r for r in read(manifest) if r['task']=='glaucoma']
    repaired = ROOT/'data/prepared/formal_v3_boundary_repaired_20261113/manifest.csv'
    if not repaired.exists():
        repaired = ROOT/'data/prepared/formal_v3_boundary_repaired_20261110/manifest.csv'
    split = {}
    for name in sorted({r['split'] for r in rows}):
        rr = [r for r in rows if r['split']==name]
        if name=='boundary':
            rr = [r for r in read(repaired) if r['task']=='glaucoma']
        split[name] = dict(n=len(rr), positive=sum(r['label']=='1' for r in rr),
            groups=len({r['related_group'] for r in rr}),
            positive_groups=len({r['related_group'] for r in rr if r['label']=='1'}),
            sources={s:dict(n=sum(r['source']==s for r in rr),
                positive=sum(r['source']==s and r['label']=='1' for r in rr))
                for s in sorted({r['source'] for r in rr})})
    train = [r for r in rows if r['split']=='train']
    counts = collections.Counter((r['source'], r['label']) for r in train)
    classes = collections.defaultdict(set)
    for source,label in counts: classes[source].add(label)
    source_count = len(classes)
    sampling = [dict(source=s, label=l, unique_images=n,
        expected_draws_per_epoch=len(train)/source_count/len(classes[s]),
        expected_draws_per_image=len(train)/source_count/len(classes[s])/n)
        for (s,l),n in sorted(counts.items())]
    selected = ROOT/'data/audit/session_20261110/selected_validation/model_val_pq.csv'
    dev = ROOT/'data/audit/data_expansion_20261113/repaired_boundary_pq.csv'
    if not dev.exists():
        dev = ROOT/'runs/formal_v3/boundary_development_repaired_20261110/boundary_pq.csv'
    metrics = {}
    for role,path in [('selected_model_validation', selected), ('boundary_development',dev)]:
        rr=read(path)
        metrics[role] = {}
        for source in sorted({r['source'] for r in rr}):
            ss=[r for r in rr if r['source']==source]
            y=np.array([int(float(r['label'])) for r in ss])
            m={}
            for field in ['p_a','p_b']:
                p=np.array([float(r[field]) for r in ss])
                tp=int(((p>=.5)&(y==1)).sum()); fn=int(((p<.5)&(y==1)).sum())
                tn=int(((p<.5)&(y==0)).sum()); fp=int(((p>=.5)&(y==0)).sum())
                m[field]=dict(auc=float(roc_auc_score(y,p)), brier=float(brier_score_loss(y,p)),
                    binary_cross_entropy=float(log_loss(y,p,labels=[0,1])),
                    display_threshold=.5, tp=tp, fn=fn, tn=tn, fp=fp,
                    sensitivity=tp/(tp+fn), wilson_image_level_sensitivity=wilson(tp,tp+fn))
            metrics[role][source] = dict(n=len(ss), positive=int(y.sum()), models=m)
    precision = []
    for n in [21,50,100,200,384,489]:
        k=round(.95*n)
        precision.append(dict(n_positive=n, observed_tp=k, sensitivity=k/n,
            wilson_95=wilson(k,n)))
    output = dict(date='2026-11-13',inputs={str(p.relative_to(ROOT)):digest(p)
        for p in [manifest,repaired,selected,dev]},splits=split,
        current_source_class_sampling=sampling, development_metrics=metrics,
        precision_examples=precision,
        normal_approx_required_positive={str(e):math.ceil(1.959963984540054**2*.95*.05/e**2)
            for e in [.05,.03,.02]},
        zero_misses_one_sided_95={str(n):1-.05**(1/n) for n in [59,149,299]},
        note='Final-test labels used only for existing split counts. No final-test predictions loaded. '
            'Wilson intervals describe image-level binomial precision, not patient-cluster inference. '
            '0.5 is a display threshold, not a chosen deployment threshold.')
    (OUT/'development_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    names=['AIROGS','ODIR','Kaggle']
    sources=['airogs-light-v2','DATASET_101/ODIR-5K','Kaggle-Eye_diseases_classification']
    x=np.arange(3)
    for i,(role,title) in enumerate([('train','Train'),('val','Model validation'),('boundary','Rule development'),('test','Locked test')]):
        values=[split[role]['sources'][s]['positive'] for s in sources]
        bars=axes[0].bar(x+(i-1.5)*.19,values,.19,label=title)
        axes[0].bar_label(bars,fontsize=7,padding=2)
    axes[0].set_xticks(x,names); axes[0].set_yscale('log')
    axes[0].set_ylim(10,10000); axes[0].set_ylabel('Positive-label images (log scale)')
    axes[0].set_title('Positive labels concentrate in AIROGS')
    axes[0].legend(fontsize=7)
    ns=[p['n_positive'] for p in precision]
    means=[p['sensitivity']*100 for p in precision]
    lower=[(p['sensitivity']-p['wilson_95'][0])*100 for p in precision]
    upper=[(p['wilson_95'][1]-p['sensitivity'])*100 for p in precision]
    axes[1].errorbar(ns,means,yerr=[lower,upper],fmt='o',capsize=4,color='#275dad')
    axes[1].set_ylim(75,101); axes[1].set_xlabel('Independent positive cases')
    axes[1].set_ylabel('Sensitivity and 95% Wilson interval (%)')
    axes[1].set_title('Precision example near 95% sensitivity')
    axes[1].grid(alpha=.2)
    fig.savefig(OUT/'positive_counts_and_precision.png',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(splits=split,sampling=sampling,metrics=metrics),indent=2))

if __name__=='__main__': main()
