"""Held-out, path-dependent TreeSHAP for the existing LightGBM controls.

No new parameter search, selection, observations or validation partitions.
Explain the regressor, not the heterogeneous TOPSIS selection rule.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import json,itertools,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from reviewer_input_sensitivity import transformed, metrics
from run_model24_topsis import D,P,ROOT,MAT,NUM,CAT,hierarchy
from model24_registry import CONFIGS,estimator

OUT=ROOT/'evidence/paired_shap_v1';OUT.mkdir(exist_ok=True)
OLD=ROOT/'evidence/model24_topsis_v1'
CACHE=ROOT/'evidence/reviewer_validation_v1/input_fits'
VARIANTS=['Descriptors','Source_time_concentration_dose']
SEEDS=[11,23,47]
GROUPS=['Material profile','Operating conditions','Chemical context','Loading variables']
def group(f):
    if f in MAT:return GROUPS[0]
    if f in CAT:return GROUPS[2]
    if f.startswith('source_scaled_'):return GROUPS[3]
    return GROUPS[1]
def point_mean(x):
    return pd.DataFrame({'v':x,'s':D.source_candidate}).groupby('s').v.mean().mean()

def main():
    importance=[];group_rows=[];pair_rows=[];checks=[];model_metrics=[]
    for variant in VARIANTS:
      features=NUM+CAT+(['source_scaled_Initial concentration','source_scaled_Adsorbent dosage'] if variant!=VARIANTS[0] else [])
      for seed in SEEDS:
        values=np.full((len(D),len(features)),np.nan);pred=np.full(len(D),np.nan);expected=pred.copy();foldids=np.full(len(D),-1)
        for fold in range(5):
          fd=OLD/f's{seed}_broad_{fold:02d}';split=pd.read_csv(fd/'split.csv')
          assert np.array_equal(split.condition_id,D.condition_id)
          tr=np.flatnonzero(~split.outer_test);te=np.flatnonzero(split.outer_test)
          cached=np.load(CACHE/f's{seed}_broad_{fold:02d}_LightGBM_{variant}.npz')
          ident=str(cached['selected']);config=next(c for c in CONFIGS if c['id']==ident)
          a,b,pre=transformed(D.iloc[tr],D.iloc[te],variant)
          with threadpool_limits(limits=1):
            x=pre.fit_transform(a);xt=pre.transform(b);model=estimator(config,seed);model.fit(x,a.r)
            y=model.predict(xt);sh=np.asarray(model.booster_.predict(xt,pred_contrib=True))
          err=float(np.max(abs(y-cached['prediction'][te])))
          assert err<1e-9,(seed,fold,variant,err)
          names=pre.get_feature_names_out();agg=np.zeros((len(te),len(features)))
          for j,name in enumerate(names):
            prefix,raw=name.split('__',1)
            if prefix=='cat':raw=next(c for c in CAT if raw.startswith(c+'_'))
            agg[:,features.index(raw)]+=sh[:,j]
          add=float(np.max(abs(agg.sum(axis=1)+sh[:,-1]-y)))
          assert add<1e-8
          values[te]=agg;pred[te]=y;expected[te]=sh[:,-1];foldids[te]=fold
          checks.append(dict(seed=seed,fold=fold,variant=variant,selected=ident,cache_max_error=err,additivity_max_error=add,n_train=len(tr),n_test=len(te)))
        assert np.isfinite(values).all() and np.array_equal(foldids[P.ia],foldids[P.ib])
        removal=-100*values
        pair=100*(values[P.ib]-values[P.ia]);advantage=100*(pred[P.ib]-pred[P.ia])
        assert np.max(abs(expected[P.ib]-expected[P.ia]))<1e-12
        assert np.max(abs(pair.sum(axis=1)-advantage))<1e-6
        np.savez_compressed(OUT/f's{seed}_{variant}.npz',phi_remaining_fraction=values,phi_removal_pp=removal,phi_advantage_pp=pair,
                            expected_r=expected,prediction_r=pred,fold=foldids,features=np.array(features),condition_id=D.condition_id.to_numpy())
        model_metrics.append(dict(seed=seed,variant=variant,**metrics(pred,variant)))
        for j,feature in enumerate(features):
          importance.append(dict(seed=seed,variant=variant,feature=feature,group=group(feature),
                           point_mean_abs_pp=point_mean(abs(removal[:,j])),contrast_mean_abs_pp=hierarchy(P,abs(pair[:,j]))))
        # Sum signed original-column contributions before taking the absolute value.
        for g in GROUPS:
          js=[j for j,f in enumerate(features) if group(f)==g]
          pointg=removal[:,js].sum(axis=1);pairg=pair[:,js].sum(axis=1)
          group_rows.append(dict(seed=seed,variant=variant,group=g,point_mean_abs_pp=point_mean(abs(pointg)),contrast_mean_abs_pp=hierarchy(P,abs(pairg))))
          for name,indices in P.groupby('pair_id').groups.items():
            q=P.loc[indices];v=pairg[indices]
            signed=pd.DataFrame({'v':v,'b':q.exact_block}).groupby('b').v.mean().mean()
            pair_rows.append(dict(seed=seed,variant=variant,pair_id=name,group=g,signed_mean_pp=signed,n_queries=len(q)))
        print('DONE',variant,seed,flush=True)
    imp=pd.DataFrame(importance);imp.to_csv(OUT/'feature_importance.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'group_importance.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(OUT/'pair_group_contributions.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'reconstruction_checks.csv',index=False)
    pd.DataFrame(model_metrics).to_csv(OUT/'model_metrics.csv',index=False)
    stability=[]
    for variant in VARIANTS:
      for metric in ['point_mean_abs_pp','contrast_mean_abs_pp']:
        t=imp[imp.variant==variant].pivot(index='feature',columns='seed',values=metric)
        for a,b in itertools.combinations(SEEDS,2):stability.append(dict(variant=variant,metric=metric,seed_a=a,seed_b=b,spearman=float(spearmanr(t[a],t[b]).statistic)))
    pd.DataFrame(stability).to_csv(OUT/'rank_stability.csv',index=False)
    joint=pd.crosstab(D['Solution pH'],D['Adsorption temperature']);joint.to_csv(OUT/'ph_temperature_joint_support.csv')
    protocol=dict(method='LightGBM native path-dependent TreeSHAP pred_contrib',scope='context holdout',seeds=SEEDS,n_fits=len(checks),
       selection='reuse selected inner-point-MAE configuration separately for each existing input variant',
       background='training tree-path cover statistics; no external interventional background',
       feature_aggregation='sum encoded category-column SHAP values by original field; not a separately recomputed coalition SHAP game',
       group_aggregation='sum signed feature contributions by prespecified semantic group before weighted mean absolute summary; not coalition SHAP',
       removal='phi_eta=-100*phi_r; expected_eta=100*(1-expected_r)',
       paired_advantage='psi_j=100*(phi_r_bj-phi_r_aj); paired model and expected value identical, so sum psi=predicted A',
       point_weights='equal sources, equal conditions within source',pair_weights='equal source/pair/block/query hierarchy',
       stability='three pairwise rank correlations over three partitions; descriptive, not confidence intervals',
       new_observations=0,post_hoc=True,not_explanation_of_all_24_methods=True,
       max_cached_prediction_error=max(r['cache_max_error'] for r in checks),max_additivity_error_fraction=max(r['additivity_max_error'] for r in checks),
       joint_noncentral_support=int(joint.loc[joint.index!=7,joint.columns!=25].values.sum()),
       data_sha256=hashlib.sha256((ROOT/'data/condition_data.csv').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(pd.DataFrame(stability).to_string(index=False),flush=True)
    print(pd.DataFrame(group_rows).groupby(['variant','group']).contrast_mean_abs_pp.agg(['min','median','max']).to_string(),flush=True)

if __name__=='__main__':main()
