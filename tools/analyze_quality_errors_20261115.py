"""Assess quality, source composition, and out-of-fold error ranking on development data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact,spearmanr,beta
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.analyze_expansion_results_20261113 import digest,group_weights,weighted_metrics,write_json
from tools.evaluate_expansion_20261113 import poor
OUT=ROOT/'docs/progress/2026-11-15'
RUN=ROOT/'runs/expanded_20261113'
SEED=20261115
REPEATS=2000


def aligned():
    manifest=ROOT/'data/prepared/expanded_20261113/manifest.csv'
    quality=RUN/'pq_quality_20261114/quality_scores.csv'
    disease=RUN/'expanded_degradation50/last_development_predictions.csv'
    verification=RUN/'pq_quality_20261114/verification.json'
    proof=json.loads(verification.read_text())
    assert digest(quality)==proof['scores_sha256'] and digest(manifest)==proof['manifest_sha256']
    m=pd.read_csv(manifest,keep_default_na=False)
    p=pd.read_csv(disease);p=p[p.condition.eq('original') & p['split'].isin(['val','boundary'])]
    q=pd.read_csv(quality)
    assert p.sample_id.is_unique and q.sample_id.is_unique
    expected=m[m['split'].isin(['val','boundary'])]
    assert set(p.sample_id)==set(q.sample_id)==set(expected.sample_id)
    d=p.merge(q,on=['sample_id','source','split','label'],validate='one_to_one').merge(
        expected.drop(columns=['source','split','label','related_group','target_definition','origin_cohort']),on='sample_id',validate='one_to_one')
    assert len(d)==4099 and d.q.between(0,1).all() and d.prediction.between(0,1).all()
    d['error']=(d.prediction>.5)!=d.label.astype(bool)
    d['fn']=d.label.eq(1)&d.prediction.le(.5);d['fp']=d.label.eq(0)&d.prediction.gt(.5)
    d['poor_human']=[poor(r) for r in d.to_dict('records')]
    d['qbin']=pd.cut(d.q,bins=[0,.2,.4,.6,.8,1],include_lowest=True,right=False,labels=['0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1'])
    d.loc[d.q.eq(1),'qbin']='0.8–1'
    files=[manifest,quality,disease,verification]
    alternative=ROOT/'runs/followup_20261115/eyeq_development/quality_scores.csv'
    if alternative.exists():
        otherproof=json.loads(alternative.with_name('verification.json').read_text())
        assert digest(alternative)==otherproof['scores_sha256']
        other=pd.read_csv(alternative)
        d=d.merge(other,on=['sample_id','source','split','label'],validate='one_to_one')
        assert len(d)==4099
        files+=[alternative,alternative.with_name('verification.json')]
    return m,d,files


def count(data):
    n=len(data);pos=int(data.label.sum());fn=int(data.fn.sum());fp=int(data.fp.sum())
    return dict(n=n,positive=pos,negative=n-pos,groups=int(data.related_group.nunique()),errors=fn+fp,fn=fn,fp=fp,
        error_rate=(fn+fp)/n if n else None,fn_rate=fn/pos if pos else None,fp_rate=fp/(n-pos) if n-pos else None,
        q_mean=float(data.q.mean()) if n else None)


def counted_interval(data):
    result=count(data)
    if len(data):
        weights=group_weights(data.related_group.to_numpy(),REPEATS,SEED)
        rates=(weights@data.error.to_numpy().astype(float))/weights.sum(axis=1)
        result['error_rate_group_interval95']=np.quantile(rates,[.025,.975]).tolist()
    return result


def review_counts(data,risk,fraction):
    k=int(np.ceil(len(data)*fraction))
    # Stable identity resolves equal risks without consulting disease labels.
    order=np.lexsort((data.sample_id.to_numpy(),-np.asarray(risk)))
    reviewed=np.zeros(len(data),dtype=bool);reviewed[order[:k]]=True
    return dict(k=k,actual_review_fraction=k/len(data),captured_errors=int(data.error.to_numpy()[reviewed].sum()),
        captured_fn=int(data.fn.to_numpy()[reviewed].sum()),remaining_errors=int(data.error.to_numpy()[~reviewed].sum()),
        remaining_fn=int(data.fn.to_numpy()[~reviewed].sum()),remaining_images=len(data)-k,
        remaining_error_rate=float(data.error.to_numpy()[~reviewed].mean())),reviewed


def bootstrap_review_capture(data,risk,weights,fraction):
    """Rerank each case bootstrap at exactly the same image review capacity."""
    order=np.lexsort((data.sample_id.to_numpy(),-np.asarray(risk)))
    ranked=weights[:,order]
    capacity=np.ceil(weights.sum(axis=1)*fraction)
    before=np.cumsum(ranked,axis=1)-ranked
    selected=np.minimum(ranked,np.maximum(capacity[:,None]-before,0))
    assert np.array_equal(selected.sum(axis=1),capacity)
    return selected@data.error.to_numpy().astype(float)[order]


def features(data,qkey=None,source_adjustment=False):
    p=data.prediction.to_numpy();z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    x=np.column_stack([z,np.abs(z),z*z])
    if qkey:
        q=data[qkey].to_numpy();x=np.column_stack([x,q,q*z,q*np.abs(z)])
    if source_adjustment:
        x=np.column_stack([x,pd.get_dummies(data.source,dtype=float).to_numpy()])
    return x


def crossfit(data,qkey=None,source_adjustment=False,seed=SEED):
    x=features(data,qkey,source_adjustment);y=data.error.astype(int).to_numpy();groups=data.related_group.to_numpy()
    predictions=np.full(len(data),np.nan);folds=np.full(len(data),-1)
    splitter=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
    stratification=2*data.label.to_numpy()+y
    for fold,(train,validation) in enumerate(splitter.split(x,stratification,groups)):
        assert not set(groups[train])&set(groups[validation])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=seed))
        model.fit(x[train],y[train]);predictions[validation]=model.predict_proba(x[validation])[:,1];folds[validation]=fold
    assert np.isfinite(predictions).all() and (folds>=0).all()
    return predictions,folds


def rank_metrics(y,p):
    return dict(error_auroc=float(roc_auc_score(y,p)),error_auprc=float(average_precision_score(y,p)))


def error_ranking(data):
    y=data.error.astype(int).to_numpy()
    risks={'uncertainty':-abs(data.prediction.to_numpy()-.5),'low_quality':-data.q.to_numpy()}
    folds=None
    for name,qkey,adjust in [('p_model',None,False),('pq_model','q',False),('p_source_model',None,True),('pq_source_model','q',True)]:
        risks[name],folds=crossfit(data,qkey,adjust)
    if 'q_eyeq' in data:
        risks['eyeq_low_quality']=-data.q_eyeq.to_numpy();risks['p_eyeq_model'],_=crossfit(data,'q_eyeq')
    result=dict(n=len(data),errors=int(y.sum()),error_prevalence=float(y.mean()),folds=5,
        grouped_crossfit_seed=SEED,models={},contrasts={},split_sensitivity=[])
    masks={}
    for name,risk in risks.items():
        result['models'][name]={**rank_metrics(y,risk),'review':{}}
        for fraction in (.1,.2,.3):
            statistics,masks[name,fraction]=review_counts(data,risk,fraction)
            result['models'][name]['review'][str(fraction)]=statistics
    weights=group_weights(data.related_group.to_numpy(),REPEATS,SEED)
    estimates={name:weighted_metrics(y,risk,weights) for name,risk in risks.items()}
    captures={(name,fraction):bootstrap_review_capture(data,risk,weights,fraction)
        for name,risk in risks.items() for fraction in (.1,.2,.3)}
    for name in ('uncertainty','low_quality'):
        result['models'][name]['error_auroc_interval']=np.nanquantile(estimates[name]['auc'],[.025,.975]).tolist()
    contrasts=[('p_model','pq_model'),('p_source_model','pq_source_model'),('uncertainty','pq_model')]
    if 'p_eyeq_model' in risks:contrasts.append(('p_model','p_eyeq_model'))
    for left,right in contrasts:
        differences={key:dict(difference=float(rank_metrics(y,risks[right])['error_'+('auroc' if key=='auc' else 'auprc')]-rank_metrics(y,risks[left])['error_'+('auroc' if key=='auc' else 'auprc')]),
            interval_95=np.nanquantile(estimates[right][key]-estimates[left][key],[.025,.975]).tolist()) for key in ('auc','auprc')}
        differences['review']={}
        for fraction in (.1,.2,.3):
            delta=(masks[right,fraction].astype(int)-masks[left,fraction].astype(int))*y
            counts=weights@delta
            differences['review'][str(fraction)]=dict(additional_errors_captured=int(delta.sum()),
                fixed_mask_group_bootstrap_interval_95=np.quantile(counts,[.025,.975]).tolist(),
                matched_capacity_group_bootstrap_interval_95=np.quantile(captures[right,fraction]-captures[left,fraction],[.025,.975]).tolist())
        result['contrasts'][right+' minus '+left]=differences
    for seed in (20261116,20261117):
        p,_=crossfit(data,seed=seed);pq,_=crossfit(data,'q',seed=seed)
        result['split_sensitivity'].append(dict(seed=seed,p=rank_metrics(y,p),pq=rank_metrics(y,pq)))
    output=data[['sample_id','related_group','source','label','error','fn','fp','prediction','q']].copy()
    output['fold']=folds
    for name,risk in risks.items():output[name]=risk
    local=ROOT/'runs/followup_20261115/error_ranking';local.mkdir(parents=True,exist_ok=True)
    output.to_csv(local/'out_of_fold_scores.csv',index=False)
    return result


def quality_metadata(manifest,data):
    result=dict(strata={},quality_target_overlap={},human_quality_score_checks={})
    for role in ('train','val','boundary','test'):
        roledata=manifest[manifest['split'].eq(role)]
        result['strata'][role]={}
        for source in ('fives','mshf'):
            selected=roledata[roledata.source.eq(source)];records=selected.to_dict('records')
            ispoor=np.array([poor(r) for r in records],bool)
            labels=selected.label.astype(int).to_numpy()
            result['strata'][role][source]={quality:dict(positive=int(((labels==1)&mask).sum()),negative=int(((labels==0)&mask).sum()))
                for quality,mask in [('poor',ispoor),('other',~ispoor)]}
    for source in ('fives','mshf'):
        selected=data[data.source.eq(source)]
        target=selected.poor_human.astype(int).to_numpy()
        result['human_quality_score_checks'][source]=dict(n=len(selected),poor=int(target.sum()),
            poor_q_median=float(selected.loc[selected.poor_human,'q'].median()),
            other_q_median=float(selected.loc[~selected.poor_human,'q'].median()) if (~selected.poor_human).any() else None,
            poor_detection_auc=float(roc_auc_score(target,-selected.q)) if len(set(target))==2 else None,
            poor_q_below_point2=int((selected.poor_human&selected.q.lt(.2)).sum()),
            poor_q_below_point4=int((selected.poor_human&selected.q.lt(.4)).sum()))
        if 'q_eyeq' in selected and len(set(target))==2:
            result['human_quality_score_checks'][source]['eyeq_poor_detection_auc']=float(roc_auc_score(target,-selected.q_eyeq))
    for name,relative in [('original_quality','data/prepared/formal_v3/manifest.csv'),('eyeq_quality','data/prepared/eyeq_quality_v1/manifest.csv')]:
        quality=pd.read_csv(ROOT/relative,keep_default_na=False)
        train=quality[quality.task.eq('quality')&quality['split'].eq('train')]
        evaluation=manifest[manifest['split'].isin(['val','boundary'])]
        result['quality_target_overlap'][name]=dict(training_n=len(train),same_file_hash_evaluation_images=int(evaluation.sha256.isin(set(train.sha256)).sum()))
    train=result['strata']['train']['fives']
    contingency=[[train['poor']['positive'],train['poor']['negative']],[train['other']['positive'],train['other']['negative']]]
    odds,pvalue=fisher_exact(contingency)
    result['fives_training_quality_label_association']=dict(table=contingency,odds_ratio=float(odds),fisher_p=float(pvalue))
    fives_book=ROOT/'data/external/fives/Quality Assessment.xlsx'
    official={}
    for sheet,frame in pd.read_excel(fives_book,sheet_name=None).items():
        official[sheet]={}
        for disease,subset in frame.groupby('Disease'):
            official[sheet][str(disease)]=dict(n=len(subset),poor=int(subset[['IC','Blur','LC']].eq(0).any(axis=1).sum()))
    healthy=pd.read_excel(ROOT/'data/external_quality/mshf/MSHF dataset 2.0/MSHF_quality_scores.xlsx',sheet_name='Healthy')
    decisions=pd.read_csv(ROOT/'data/audit/data_expansion_20261113/local_decisions.csv',keep_default_na=False)
    mshfneg=decisions[decisions.source.eq('mshf') & decisions.label.eq('0')]
    result['original_source_composition']=dict(fives_official=official,
        mshf_healthy=dict(n=len(healthy),overall_poor=int(healthy.Overall.eq(0).sum()),
            negative_candidates=len(mshfneg),excluded=len(mshfneg[mshfneg.decision.eq('excluded_from_pooled_training')]),
            reasons=mshfneg.reasons.value_counts().to_dict()))
    # Planning numbers quantify how little zero misses / zero false alarms establishes.
    result['zero_error_precision']=dict(upper_one_sided95_rate={str(n):float(1-.05**(1/n)) for n in (12,30,59,100)},
        needed_zero_error_independent_cases_for_one_sided95_upper_below_point05=59,
        approximate_cases_for_95_halfwidth_point05_at_rate_point10=139)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-crossfit',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);manifest,data,files=aligned()
    result=dict(data_version='expanded_20261113',disease_checkpoint='expanded_degradation50/last.pt',disease_update=1200,
        display_threshold=.5,quality_kind='FTHNet_distilled_raw_regression',protected_test_inference=False,
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},roles={},quality_metadata=quality_metadata(manifest,data))
    for role in ('boundary','val'):
        selected=data[data['split'].eq(role)].copy()
        item=dict(overall=count(selected),quality_bins={str(name):counted_interval(subset) for name,subset in selected.groupby('qbin',observed=False)},
            sources={source:count(subset) for source,subset in selected.groupby('source')},
            by_source_quality={source:{str(name):count(binrows) for name,binrows in subset.groupby('qbin',observed=False)}
                for source,subset in selected.groupby('source')},
            quality_error_ranking=rank_metrics(selected.error,-selected.q),
            confident_errors=count(selected[selected.error&(selected.prediction.le(.1)|selected.prediction.ge(.9))]),
            confident_highq_errors=count(selected[selected.error&(selected.prediction.le(.1)|selected.prediction.ge(.9))&selected.q.ge(.4)]),
            correlation_q_disease=spearmanr(selected.q,selected.prediction).statistic)
        classranking={}
        for label in (0,1):
            subset=selected[selected.label.eq(label)]
            classranking[str(label)]=rank_metrics(subset.error,-subset.q)
        item['quality_error_ranking_by_class']=classranking
        item['quality_error_ranking_by_source']={source:rank_metrics(subset.error,-subset.q)
            for source,subset in selected.groupby('source') if subset.error.nunique()==2}
        result['roles'][role]=item
    boundary=data[data['split'].eq('boundary')].copy()
    if not args.skip_crossfit:result['crossfit_error_ranking']=error_ranking(boundary)
    # Compare all historical arms on exactly the same development quality scores.
    result['historical_arms']={}
    for arm in ('original_clean','expanded_clean','expanded_degradation30','expanded_degradation50'):
        path=RUN/arm/'last_development_predictions.csv';files.append(path)
        p=pd.read_csv(path);p=p[p['split'].eq('boundary')&p.condition.eq('original')]
        d=boundary.drop(columns=['prediction','error','fn','fp']).merge(p[['sample_id','prediction']],on='sample_id',validate='one_to_one')
        d['error']=(d.prediction>.5)!=d.label.astype(bool);d['fn']=d.label.eq(1)&d.prediction.le(.5);d['fp']=d.label.eq(0)&d.prediction.gt(.5)
        result['historical_arms'][arm]=dict(overall=count(d),quality_bins={str(name):count(subset) for name,subset in d.groupby('qbin',observed=False)},
            error_rank_lowq=rank_metrics(d.error,-d.q))
    files += [ROOT/'data/external/fives/Quality Assessment.xlsx',
        ROOT/'data/external_quality/mshf/MSHF dataset 2.0/MSHF_quality_scores.xlsx',
        ROOT/'data/audit/data_expansion_20261113/local_decisions.csv',
        ROOT/'data/prepared/formal_v3/manifest.csv',ROOT/'data/prepared/eyeq_quality_v1/manifest.csv',
        Path(__file__)]
    result['input_sha256']={p.relative_to(ROOT).as_posix():digest(p) for p in set(files)}
    write_json(OUT/'quality-error-analysis.json',result)
    columns=['sample_id','path','source','label','split','prediction','q','poor_human','error','fn','fp','qbin']
    if 'q_eyeq' in data:columns+=['q_eyeq','predicted_eyeq_grade']
    local=ROOT/'runs/followup_20261115/error_ranking';local.mkdir(parents=True,exist_ok=True)
    data[columns].to_csv(local/'development_join.csv',index=False)
    print(json.dumps(dict(roles={role:r['overall'] for role,r in result['roles'].items()},human_quality=result['quality_metadata']['human_quality_score_checks'],
        crossfit=result.get('crossfit_error_ranking',{}).get('models',{})),indent=2))


if __name__=='__main__':main()
