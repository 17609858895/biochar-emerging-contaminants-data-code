"""Summaries for reviewer checks, always with displayed errors in percentage points."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from run_model24_topsis import D,P,ROOT,hierarchy

OLD=ROOT/'evidence/model24_topsis_v1'
OUT=ROOT/'evidence/reviewer_validation_v1'
PAIRS=['PB600 / PB800','GCRB / GCRB-N','PSB / PSBOX-A','Alkali-modified SCG biochars / Pristine SCG biochar','NaOH-activated SCW biochars / Pristine SCW Biochar','AMCB / CB','AMCB / MCB','CB / MCB']

def summarize(y,q):
    rows=[]
    for (seed,track,strategy),g in q.groupby(['seed','track','strategy']):
        v=g[g.prediction.notna()].copy()
        a=y[(y.seed==seed)&(y.track==track)&(y.strategy==strategy)]
        point=100*a.assign(e=abs(a.prediction-a.r)).groupby('source_candidate').e.mean().mean() if len(a) else np.nan
        err=abs(v.prediction-v.observed);wrong=(v.prediction<0)!=(v.observed<0)
        rows.append(dict(seed=seed,track=track,strategy=strategy,point_mae_pp=point,
                         contrast_mae_pp=100*hierarchy(v,err) if len(v) else np.nan,
                         selection_loss_pp=100*hierarchy(v,np.where(wrong,abs(v.observed),0)) if len(v) else np.nan,
                         wrong=int(wrong.sum()),n_queries=len(v),
                         pooled_r2=r2_score(a.r,a.prediction) if len(a) else np.nan))
    return pd.DataFrame(rows)

def main():
    q=pd.read_csv(OLD/'pairs.csv');y=pd.read_csv(OLD/'predictions.csv')
    rows=[]
    for (seed,track,strategy,pair),g in q[q.strategy.isin(['TOPSIS','Contrast','Point','History'])].groupby(['seed','track','strategy','pair_id']):
        g=g[g.prediction.notna()]
        if len(g)==0:continue
        rho=spearmanr(g.observed,g.prediction).statistic if g.prediction.nunique()>1 and g.observed.nunique()>1 else np.nan
        blocks=g.groupby('exact_block')[['observed','prediction']].mean()
        rows.append(dict(seed=seed,track=track,strategy=strategy,pair_number=PAIRS.index(pair)+1,pair=pair,n_queries=len(g),n_blocks=len(blocks),
                         query_r2=r2_score(g.observed,g.prediction),query_spearman=rho,
                         query_mae_pp=100*abs(g.prediction-g.observed).mean(),
                         block_balanced_mae_pp=100*g.assign(e=abs(g.prediction-g.observed)).groupby('exact_block').e.mean().mean(),
                         observed_mean_pp=100*blocks.observed.mean(),prediction_mean_pp=100*blocks.prediction.mean(),
                         observed_min_pp=100*g.observed.min(),observed_max_pp=100*g.observed.max(),
                         prediction_sd_pp=100*g.prediction.std(),observed_sd_pp=100*g.observed.std(),
                         n_positive=int((g.observed>0).sum()),n_negative=int((g.observed<0).sum())))
    pd.DataFrame(rows).to_csv(OUT/'query_agreement.csv',index=False)
    summarize(y,q).to_csv(OUT/'original_rule_metrics_pp.csv',index=False)
    if len(list((OUT/'repeated_fits').glob('*/complete.json')))!=70:
        print('Saved original summaries; additional partitions not complete.');return
    yp=[y];qp=[q];warnings=[]
    for p in sorted((OUT/'repeated_fits').glob('*/complete.json')):
        meta=json.loads(p.read_text());fd=p.parent
        for filename,dest in [('predictions.csv',yp),('pairs.csv',qp)]:
            f=pd.read_csv(fd/filename)
            for k in ['seed','track','fold']:f[k]=meta[k]
            dest.append(f)
        s=pd.read_csv(fd/'split.csv');assert len(s)==len(D) and s.condition_id.equals(D.condition_id)
        group=meta['track']+'_block';tr=~s.outer_test;te=s.outer_test
        assert not set(D.loc[tr,group])&set(D.loc[te,group])
        assert not np.any(te.to_numpy()[P.ia]^te.to_numpy()[P.ib])
        for c in pd.read_csv(fd/'configuration_scores.csv').itertuples():
            if c.warnings!='[]':warnings.append(dict(fold=fd.name,config=c.id,warnings=c.warnings))
    yy=pd.concat(yp,ignore_index=True);qq=pd.concat(qp,ignore_index=True)
    for _,g in yy.groupby(['seed','track','strategy']):assert len(g)==1279 and g.condition_id.nunique()==1279
    for _,g in qq.groupby(['seed','track','strategy']):assert len(g)==559 and not g[['condition_a','condition_b']].duplicated().any()
    mm=summarize(yy,qq);mm.to_csv(OUT/'expanded_rule_metrics_pp.csv',index=False)
    h=mm[mm.strategy=='History'][['seed','track','contrast_mae_pp']].rename(columns={'contrast_mae_pp':'history_contrast_mae_pp'})
    diff=mm[mm.strategy.isin(['Point','Contrast','TOPSIS'])].merge(h,on=['seed','track'])
    diff['improvement_over_history_pp']=diff.history_contrast_mae_pp-diff.contrast_mae_pp
    diff.to_csv(OUT/'repeated_rule_comparisons.csv',index=False)
    checks=dict(original_folds=51,additional_folds=70,total_nested_assessments=121,
                original_estimator_fits=13662,additional_estimator_fits=16100,total_benchmark_fits=29762,
                additional_constant_calculations=350,grouped_partition_repeats=10,source_partition_repeats=1,source_fitting_seeds=3,
                source_groups=7,all_group_boundaries_disjoint=True,all_pairs_kept_together=True,
                all_outer_rows_predicted_once_per_repeat=True,solver_warnings=warnings,
                new_observations=0,scope='Descriptive repeated-partition sensitivity; datasets and training observations overlap')
    (OUT/'repeated_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(diff[diff.track!='source'].groupby(['track','strategy']).improvement_over_history_pp.agg(['min','median','max']).to_string())

if __name__=='__main__':main()
