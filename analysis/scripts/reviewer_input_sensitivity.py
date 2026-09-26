"""Training-only input controls on the original three matched grouped partitions.

Three prespecified methods span linear, bagged-tree and boosted-tree models.
Each retains its original two-configuration inner-point-MAE selection rule.
Whole-profile permutations retain material identity but randomize physical
descriptor assignments; these are diagnostic encodings, not proposed materials.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import json, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from model24_registry import CONFIGS, estimator
from run_model24_topsis import D, P, ROOT, MAT, NUM, CAT, hierarchy

OUT=ROOT/'evidence/reviewer_validation_v1'
CACHE=OUT/'input_fits'
CACHE.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'evidence/model24_topsis_v1'
METHODS=['Ridge','RandomForest','LightGBM']
VARIANTS=['Descriptors','Material_ID','Source_time','Source_time_concentration_dose','Exclude_raw_r_gt1']
RAW=pd.read_csv(ROOT/'data/raw_to_condition.csv')
CLEAN=RAW[RAW.r<=1].groupby('condition_id').r.mean()

def transformed(train,test,variant,mapping=None):
    a,b=train.copy(),test.copy()
    nums,cats=list(NUM),list(CAT)
    if variant=='Material_ID':
        nums=[x for x in nums if x not in MAT];cats+=['Adsorbent']
    if mapping is not None:
        for col in MAT:
            a[col]=a.Adsorbent.map(mapping[col]);b[col]=b.Adsorbent.map(mapping[col])
    if variant.startswith('Source_time'):
        fields=['Adsorption time']
        if variant=='Source_time_concentration_dose':fields+=['Initial concentration','Adsorbent dosage']
        for col in fields:
            positive=a[col].where(a[col]>0)
            med=positive.groupby(a.source_candidate).median()
            fallback=float(positive.median())
            assert np.isfinite(fallback) and fallback>0
            dest='log_time' if col=='Adsorption time' else 'source_scaled_'+col
            for f in [a,b]:
                scale=f.source_candidate.map(med).fillna(fallback)
                f[dest]=np.log1p(f[col]/scale)
            if dest not in nums:nums.append(dest)
    pre=ColumnTransformer([
        ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
    return a,b,pre

def fit_predict(c,train,val,variant,seed,mapping=None):
    if variant=='Exclude_raw_r_gt1':
        train=train[D.iloc[train].condition_id.isin(CLEAN.index).values]
        val=val[D.iloc[val].condition_id.isin(CLEAN.index).values]
    if len(val)==0:return val,np.array([]),[]
    a,b,pre=transformed(D.iloc[train],D.iloc[val],variant,mapping)
    y=a.condition_id.map(CLEAN).values if variant=='Exclude_raw_r_gt1' else a.r.values
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        model=make_pipeline(pre,estimator(c,seed));model.fit(a,y)
        pred=np.asarray(model.predict(b)).ravel()
    return val,pred,sorted(set(str(x.message) for x in ws))

def task(seed,track,fold,method,variant):
    cp=CACHE/f's{seed}_{track}_{fold:02d}_{method}_{variant}.npz'
    if cp.exists():return
    fd=OLD/f's{seed}_{track}_{fold:02d}';s=pd.read_csv(fd/'split.csv')
    te=np.flatnonzero(s.outer_test);tr=np.flatnonzero(~s.outer_test)
    inner=[(np.flatnonzero((s.inner_fold>=0)&(s.inner_fold!=i)),np.flatnonzero(s.inner_fold==i)) for i in sorted(s.loc[s.inner_fold>=0,'inner_fold'].unique())]
    candidates=[c for c in CONFIGS if c['method']==method];scores=[];messages=[]
    for c in candidates:
        p=np.full(len(D),np.nan)
        for a,b in inner:
            v,z,w=fit_predict(c,a,b,variant,seed);p[v]=z;messages+=w
        valid=tr[np.isfinite(p[tr])]
        y=D.iloc[valid].condition_id.map(CLEAN).values if variant=='Exclude_raw_r_gt1' else D.iloc[valid].r.values
        score=pd.DataFrame({'s':D.iloc[valid].source_candidate.values,'e':abs(p[valid]-y)}).groupby('s').e.mean().mean()
        scores.append((float(score),c['id'],c))
    selected=sorted(scores,key=lambda x:(x[0],x[1]))[0][2]
    val,z,w=fit_predict(selected,tr,te,variant,seed);messages+=w
    p=np.full(len(D),np.nan);p[val]=z
    if variant=='Descriptors':
        old=np.load(fd/(selected['id']+'.npz'))['outer']
        assert np.max(abs(p[val]-old[val]))<1e-9
        prior=pd.read_csv(fd/'method_scores.csv').set_index('method').loc[method,'id']
        assert selected['id']==prior
    np.savez_compressed(cp,prediction=p,selected=selected['id'],warnings=json.dumps(sorted(set(messages))))

def metrics(pred,variant):
    valid=np.isfinite(pred);idx=np.flatnonzero(valid)
    obs=D.r.to_numpy().copy()
    if variant=='Exclude_raw_r_gt1':obs=D.condition_id.map(CLEAN).to_numpy()
    q=P[valid[P.ia]&valid[P.ib]].copy()
    o=obs[q.ib]-obs[q.ia];p=pred[q.ib]-pred[q.ia]
    wrong=(o<0)!=(p<0)
    return dict(n_conditions=len(idx),n_queries=len(q),
                point_mae_pp=100*pd.DataFrame({'s':D.iloc[idx].source_candidate.values,'e':abs(pred[idx]-obs[idx])}).groupby('s').e.mean().mean(),
                contrast_mae_pp=100*hierarchy(q,abs(p-o)),
                selection_loss_pp=100*hierarchy(q,np.where(wrong,abs(o),0)),
                pooled_r2=r2_score(obs[idx],pred[idx]))

def perm_task(track,rep):
    cp=CACHE/f'profile_permutation_{track}_{rep:02d}.npz'
    if cp.exists():return
    profiles=D.groupby('Adsorbent')[MAT].first().sort_index()
    assert len(profiles)==14 and len(profiles.drop_duplicates())==14
    rng=np.random.default_rng(20260917+rep)
    mapping=pd.DataFrame(profiles.to_numpy()[rng.permutation(len(profiles))],index=profiles.index,columns=MAT)
    p=np.full(len(D),np.nan)
    for fold in range(5):
        fd=OLD/f's11_{track}_{fold:02d}';s=pd.read_csv(fd/'split.csv')
        tr=np.flatnonzero(~s.outer_test);te=np.flatnonzero(s.outer_test)
        selected=pd.read_csv(fd/'method_scores.csv').set_index('method').loc['LightGBM','id']
        c=next(c for c in CONFIGS if c['id']==selected)
        idx,z,w=fit_predict(c,tr,te,'Descriptors',11,mapping)
        p[idx]=z
    np.savez_compressed(cp,prediction=p,mapping=mapping.to_numpy())

def main():
    tasks=[(seed,track,fold,method,variant) for seed in [11,23,47] for track in ['exact','broad'] for fold in range(5) for method in METHODS for variant in VARIANTS]
    Parallel(n_jobs=4)(delayed(task)(*x) for x in tasks)
    print('INPUT_CONTROLS_COMPLETE',flush=True)
    Parallel(n_jobs=4)(delayed(perm_task)(track,rep) for track in ['exact','broad'] for rep in range(30))
    rows=[]
    for seed in [11,23,47]:
        for track in ['exact','broad']:
            for method in METHODS:
                for variant in VARIANTS:
                    pred=np.full(len(D),np.nan)
                    for fold in range(5):
                        z=np.load(CACHE/f's{seed}_{track}_{fold:02d}_{method}_{variant}.npz');p=z['prediction'];m=np.isfinite(p);pred[m]=p[m]
                    rows.append(dict(seed=seed,track=track,method=method,variant=variant,**metrics(pred,variant)))
    pd.DataFrame(rows).to_csv(OUT/'input_sensitivity.csv',index=False)
    rows=[]
    for track in ['exact','broad']:
        for rep in range(30):
            p=np.load(CACHE/f'profile_permutation_{track}_{rep:02d}.npz')['prediction']
            rows.append(dict(track=track,replicate=rep,method='LightGBM',**metrics(p,'Descriptors')))
    pd.DataFrame(rows).to_csv(OUT/'profile_permutation_metrics.csv',index=False)
    protocol=dict(methods=METHODS,variants=VARIANTS,seeds=[11,23,47],tracks=['exact','broad'],
                  selection='two configurations per method, inner source-balanced point MAE, original splits',
                  time='log1p(time / positive median time of source in current training partition)',
                  additions='same within-training-source positive-median transform for concentration and dosage',
                  unit_scope='invariant to a common multiplicative unit conversion within source; not a verified unit harmonization',
                  excluded_raw_records=int((RAW.r>1).sum()),remaining_raw_records=int((RAW.r<=1).sum()),
                  remaining_analysis_conditions=int(D.condition_id.isin(CLEAN.index).sum()),
                  permutation='30 joint profile bijections across 14 materials; same mapping in train and test; original LightGBM settings fixed',
                  permutation_scope='identity-preserving diagnostic; no chemical interpretation or formal permutation p-value',
                  new_observations=0, baseline_cache_reproduction=True)
    (OUT/'input_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('track').contrast_mae_pp.agg(['min','median','max']),flush=True)

if __name__=='__main__':main()
