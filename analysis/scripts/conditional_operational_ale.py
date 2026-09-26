"""Restrict one-variable ALE/PDP reference conditions to observed joint support.

Version 2 supersedes the pooled operational curves only. Observed contrasts and
original model caches remain unchanged. Each condition uses its held-out model.
"""
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from adsorption_effects import D,E,ROOT,preprocess,accumulate
from model24_registry import CONFIGS,estimator
OUT=ROOT/'evidence/adsorption_effects_v2';OUT.mkdir(exist_ok=True)
def main():
 curves=[];checks=[]
 for seed in [11,23,47]:
  models=[]
  for fold in range(5):
   fd=E/f's{seed}_exact_{fold:02d}';sp=pd.read_csv(fd/'split.csv');tr=np.flatnonzero(~sp.outer_test);te=np.flatnonzero(sp.outer_test)
   chosen=json.loads((fd/'complete.json').read_text())['selected']['TOPSIS'];c=next(c for c in CONFIGS if c['id']==chosen)
   model=make_pipeline(preprocess(),estimator(c,seed))
   with threadpool_limits(limits=1):model.fit(D.iloc[tr],D.iloc[tr].r);pred=model.predict(D.iloc[te])
   err=float(np.max(abs(pred-np.load(fd/(chosen+'.npz'))['outer'][te])))
   assert err<1e-9,(chosen,err)
   checks.append(dict(seed=seed,fold=fold,selected=chosen,max_abs_cached_difference=err));models.append((model,te))
  for feature,other,fixed,levels in [('Solution pH','Adsorption temperature',25,[3,5,7,9,11]),('Adsorption temperature','Solution pH',7,[15,25,35,45])]:
   levels=np.array(levels,float);mask=D[other].eq(fixed).to_numpy();ids=np.flatnonzero(mask);ref=D.iloc[ids]
   x=D[feature].to_numpy();local=np.zeros((len(levels),len(D)));dp=np.full_like(local,np.nan)
   weights=1/ref.groupby('source_candidate').condition_id.transform('size').to_numpy(float)
   for model,test in models:
    te=test[mask[test]]
    for k,z in enumerate(levels):
     a=D.iloc[te].copy();a[feature]=z
     with threadpool_limits(limits=1):dp[k,te]=100*(1-model.predict(a))
     if k:
      m=te[x[te]==z]
      if len(m):
       a=D.iloc[m].copy();b=a.copy();a[feature]=levels[k-1];b[feature]=z
       with threadpool_limits(limits=1):local[k,m]=100*(model.predict(a)-model.predict(b))
   ale,inc,n=accumulate(levels,x[ids],local[:,ids],weights)
   pdp=np.average(dp[:,ids],axis=1,weights=weights);pdp-=np.average(pdp[np.searchsorted(levels,x[ids])],weights=weights)
   for k,z in enumerate(levels):curves.append(dict(seed=seed,feature=feature,level=z,ale_pp=ale[k],pdp_pp=pdp[k],local_increment_pp=inc[k],n=int(n[k]),n_sources=int(ref.loc[ref[feature]==z,'source_candidate'].nunique()),fixed_feature=other,fixed_value=fixed,n_reference=len(ids)))
  print('CONDITIONAL ALE',seed,flush=True)
 c=pd.DataFrame(curves);c.to_csv(OUT/'effect_curves.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'model_reproduction.csv',index=False)
 summary=[]
 for f,lo,hi in [('Solution pH',7,11),('Adsorption temperature',15,35)]:
  g=c[c.feature==f];a=g.pivot(index='seed',columns='level',values='ale_pp');p=g.pivot(index='seed',columns='level',values='pdp_pp');d=a[hi]-a[lo]
  summary.append(dict(feature=f,low=lo,high=hi,n_reference=int(g.n_reference.iloc[0]),ale_delta_median=float(d.median()),ale_delta_min=float(d.min()),ale_delta_max=float(d.max()),pdp_delta_median=float((p[hi]-p[lo]).median())))
 protocol=dict(analysis='Cross-fitted conditional-reference operational ALE/PDP',version=2,supersedes='adsorption_effects_v1/effect_curves.csv for pH and temperature only',additional_outer_refits=15,
     new_data=0,reference_slices='pH: T=25 C, n=1068; temperature: pH=7, n=1174',
     scope='Observed pH-temperature cross retained; other dependencies still preclude causal interpretation',
     weights='equal sources and equal reference conditions within source, renormalized separately for each slice',
     unchanged='original outcomes, splits, model selections, held-out predictions and matched observed comparisons',
     max_cached_prediction_error=max(x['max_abs_cached_difference'] for x in checks),summary=summary)
 (OUT/'protocol_and_summary.json').write_text(json.dumps(protocol,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
