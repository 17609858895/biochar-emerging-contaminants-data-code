"""Cross-fitted ALE/PDP from the frozen 24-method TOPSIS benchmark.
No retuning. All predictions are from the original outer-training partitions.
ALE uses right-closed intervals at observed operational levels, source weights,
and an empirical weighted zero centre. Repeated-seed ranges are not CIs.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys, json, zipfile, hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from model24_registry import CONFIGS, estimator

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/model24_topsis_v1'
OUT=ROOT/'evidence/adsorption_effects_v1';OUT.mkdir(exist_ok=True)
MAT=['Pyrolysis temperature','O/C','Surface area','Pore volume','Average pore size']
NUM=MAT+['log_time','Solution pH','RPM','Adsorption temperature']
CAT=['Pollutant','Wastewater type','Adsorption type']
D=pd.read_csv(ROOT/'data/condition_data.csv',float_precision='round_trip')
D=D[(D.Adsorbent!='PAC')&(D['Adsorption time']>0)].reset_index(drop=True)
D['log_time']=np.log1p(D['Adsorption time'])
assert len(D)==1279
FEATURES={'Adsorption time':sorted(D['Adsorption time'].unique()),'Solution pH':[3,5,7,9,11],'Adsorption temperature':[15,25,35,45]}
weights=1/D.groupby('source_candidate').condition_id.transform('size').to_numpy(float)

def preprocess():
    return ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)

def accumulate(levels, x, differences, w):
    """differences[k] corresponds to samples with x in (level[k-1],level[k]].
    At the minimum level the initial contribution is zero. Exact observed levels
    avoid interpolation in the weighted centring and double-counted boundaries.
    """
    increments=np.zeros(len(levels));counts=np.zeros(len(levels),int)
    for k,z in enumerate(levels):
        m=x==z;counts[k]=m.sum()
        if k:
            assert m.any() and np.isfinite(differences[k][m]).all()
            increments[k]=np.average(differences[k][m],weights=w[m])
    raw=np.cumsum(increments)
    idx=np.searchsorted(levels,x)
    centred=raw-np.average(raw[idx],weights=w)
    assert abs(np.average(centred[idx],weights=w))<1e-10
    return centred, increments, counts

def self_test():
    # Correlated additive model: ALE must recover the known additive term and
    # remain unchanged under a nuisance offset; the minimum must count once.
    x=np.array([1,1,2,3,3]);z=np.array([1,2,3]);w=np.array([1,2,1,3,1.])
    dif=np.zeros((3,5));dif[1]=2;dif[2]=2
    a,inc,c=accumulate(z,x,dif,w)
    assert np.allclose(a,2*(z-np.average(x,weights=w)))
    assert c.sum()==len(x) and np.array_equal(c,[2,1,2])
    assert np.allclose(np.diff(a),[2,2])
    return True

def run():
    self_test(); curves=[];checks=[];predictions=[]
    for seed in [11,23,47]:
        models=[];pred=np.full(len(D),np.nan)
        for fold in range(5):
            fd=E/f's{seed}_exact_{fold:02d}';split=pd.read_csv(fd/'split.csv')
            assert np.array_equal(split.condition_id,D.condition_id)
            te=np.flatnonzero(split.outer_test.to_numpy(bool));tr=np.flatnonzero(~split.outer_test.to_numpy(bool))
            chosen=json.loads((fd/'complete.json').read_text())['selected']['TOPSIS']
            cfg=next(c for c in CONFIGS if c['id']==chosen)
            model=make_pipeline(preprocess(),estimator(cfg,seed))
            with threadpool_limits(limits=1):model.fit(D.iloc[tr],D.iloc[tr].r.to_numpy());p=np.asarray(model.predict(D.iloc[te])).ravel()
            old=np.load(fd/(chosen+'.npz'))['outer'][te]
            err=float(np.max(np.abs(old-p)));assert err<1e-10,(seed,fold,err)
            pred[te]=p;models.append((model,te))
            checks.append(dict(seed=seed,fold=fold,config=chosen,n_train=len(tr),n_test=len(te),max_abs_cached_difference=err))
            print('REPRODUCED',seed,fold,chosen,err,flush=True)
        assert np.isfinite(pred).all()
        predictions.extend(dict(seed=seed,condition_id=int(c),r_pred=float(p)) for c,p in zip(D.condition_id,pred))
        for feat,levels in FEATURES.items():
            levels=np.asarray(levels,float);x=D[feat].to_numpy(float)
            dp=np.full((len(levels),len(D)),np.nan);local=np.zeros_like(dp)
            column='log_time' if feat=='Adsorption time' else feat
            for model,te in models:
                for k,z in enumerate(levels):
                    ref=D.iloc[te].copy();ref[column]=np.log1p(z) if feat=='Adsorption time' else z
                    with threadpool_limits(limits=1):dp[k,te]=100*(1-model.predict(ref))
                    if k:
                        m=te[x[te]==z]
                        if len(m):
                            a=D.iloc[m].copy();b=a.copy()
                            a[column]=np.log1p(levels[k-1]) if feat=='Adsorption time' else levels[k-1]
                            b[column]=np.log1p(z) if feat=='Adsorption time' else z
                            with threadpool_limits(limits=1):local[k,m]=100*(model.predict(a)-model.predict(b))
            ale,inc,counts=accumulate(levels,x,local,weights)
            pdp=np.average(dp,axis=1,weights=weights)
            pdp-=np.average(pdp[np.searchsorted(levels,x)],weights=weights)
            for k,z in enumerate(levels):
                curves.append(dict(seed=seed,feature=feat,level=z,ale_pp=ale[k],pdp_pp=pdp[k],local_increment_pp=inc[k],n=int(counts[k]),n_sources=int(D.loc[x==z,'source_candidate'].nunique())))
    pd.DataFrame(checks).to_csv(OUT/'model_reproduction.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'effect_curves.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'oof_predictions.csv',index=False)
    # Exact matched observations, distinct from synthetic ALE/PDP evaluations.
    keys=['source_candidate','Adsorbent','Pollutant','Adsorption type','Wastewater type','Adsorption time','Initial concentration','Solution pH','RPM','Volume','Adsorbent dosage','Adsorption temperature','Ion concentration','Humic acid']
    contrasts=[]
    for feat,lo,hi in [('Adsorption time',30,1440),('Solution pH',7,11),('Adsorption temperature',15,35)]:
        kk=[k for k in keys if k!=feat]
        for ident,(key,g) in enumerate(D.groupby(kk,dropna=False,sort=True)):
            low=g[g[feat]==lo];high=g[g[feat]==hi]
            if len(low)==0 or len(high)==0:continue
            assert len(low)==1 and len(high)==1,(feat,ident,len(low),len(high))
            row=dict(zip(kk,key));row.update(feature=feat,low=lo,high=hi,condition_low=int(low.condition_id.iloc[0]),condition_high=int(high.condition_id.iloc[0]),removal_change_pp=100*(low.r.iloc[0]-high.r.iloc[0]))
            contrasts.append(row)
    c=pd.DataFrame(contrasts);c.to_csv(OUT/'observed_operational_contrasts.csv',index=False)
    # Mean within source, material and pollutant, then balanced over materials and
    # sources, prevents a heavily repeated pollutant/material dominating a panel.
    m=c.groupby(['feature','source_candidate','Adsorbent','Pollutant']).removal_change_pp.mean().groupby(level=[0,1,2]).mean().groupby(level=[0,1]).mean().rename('mean_pp').reset_index()
    n=c.groupby(['feature','source_candidate']).size().rename('n_contrasts').reset_index();m=m.merge(n)
    m.to_csv(OUT/'observed_source_means.csv',index=False)
    summary=[]
    cc=pd.DataFrame(curves)
    for feat,lo,hi in [('Adsorption time',30,1440),('Solution pH',7,11),('Adsorption temperature',15,35)]:
        u=cc[cc.feature==feat].pivot(index='seed',columns='level',values='ale_pp')
        v=c[c.feature==feat];mean=m[m.feature==feat].mean_pp.mean()
        summary.append(dict(feature=feat,low=lo,high=hi,n_matched=len(v),n_sources=v.source_candidate.nunique(),observed_source_balanced_pp=mean,ale_delta_median=float((u[hi]-u[lo]).median()),ale_delta_min=float((u[hi]-u[lo]).min()),ale_delta_max=float((u[hi]-u[lo]).max())))
    protocol=dict(analysis='Post hoc cross-fitted operational ALE and brute-force PDP',seeds=[11,23,47],additional_outer_refits=15,new_data=0,selection='frozen inner-only TOPSIS; exact-condition-block track',grid='all observed operational levels; no fitted smoothing',ale_bins='right closed (previous observed level, current level]; minimum level anchors zero increment',centering='source-balanced empirical mean zero',response='predicted removal 100*(1-r); centred effect in percentage points',uncertainty='min/max of 3 seeds is a sensitivity range, not a confidence interval',interpretation='model associations; matched observations are contextual corroboration from reused data, not independent mechanistic validation',time='recorded units only; no globally resolved physical time unit; 1441-1455 retained in computations',model_reproduction_max_error=max(x['max_abs_cached_difference'] for x in checks),self_test=True,summary=summary)
    (OUT/'protocol_and_summary.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':run()
