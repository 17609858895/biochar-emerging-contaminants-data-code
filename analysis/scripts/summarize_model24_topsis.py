from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import r2_score
from model24_registry import CONFIGS,METHODS,topsis
from run_model24_topsis import D,P,OUT,ROOT,hierarchy

def main():
    dirs=sorted(p.parent for p in OUT.glob('*/complete.json'))
    assert len(dirs)==51,len(dirs)
    preds=[];pairs=[];scores=[];weights=[];removal=[];checks=[];warnings=[]
    for fd in dirs:
        info=json.loads((fd/'complete.json').read_text());meta={k:info[k] for k in ['seed','track','fold']}
        for file,dest in [('predictions.csv',preds),('pairs.csv',pairs),('method_scores.csv',scores),('weight_sensitivity.csv',weights),('candidate_removal.csv',removal)]:
            f=pd.read_csv(fd/file);[f.__setitem__(k,v) for k,v in meta.items()];dest.append(f)
        ss=pd.read_csv(fd/'method_scores.csv');assert len(ss)==24
        assert np.allclose(topsis(ss[['point','contrast','loss']]),ss.topsis,rtol=1e-10)
        split=pd.read_csv(fd/'split.csv');assert len(split)==1279
        if info['seed']==11:
            old=pd.read_csv(ROOT/f'evidence/benchmark_v2/{info["track"]}_{info["fold"]:02d}/split.csv')
            assert split.equals(old),'Original splits changed'
        cfg=pd.read_csv(fd/'configuration_scores.csv');assert len(cfg)==47
        for row in cfg.itertuples():
            if row.warnings!='[]':warnings.append({**meta,'id':row.id,'warnings':row.warnings})
        for npz in fd.glob('*.npz'):
            z=np.load(npz);tr=split.inner_fold>=0;te=split.outer_test
            assert np.isfinite(z['inner'][tr]).all() and np.isnan(z['inner'][te]).all()
            assert np.isfinite(z['outer'][te]).all() and np.isnan(z['outer'][tr]).all()
        checks.append(meta)
    A=pd.concat(preds,ignore_index=True);Q=pd.concat(pairs,ignore_index=True);S=pd.concat(scores,ignore_index=True);W=pd.concat(weights,ignore_index=True);R=pd.concat(removal,ignore_index=True)
    metrics=[]
    for (seed,track,strategy),g in A.groupby(['seed','track','strategy']):
        assert len(g)==1279 and g.condition_id.nunique()==1279
        err=abs(g.prediction-g.r)
        metrics.append(dict(seed=seed,track=track,strategy=strategy,point_mae=float(g.assign(err=err).groupby('source_candidate').err.mean().mean()),pooled_mae=err.mean(),pooled_rmse=np.sqrt(np.mean(err**2)),pooled_r2=r2_score(g.r,g.prediction),pred_outside_0_1=int(((g.prediction<0)|(g.prediction>1)).sum())))
    M=pd.DataFrame(metrics);pm=[];bp=[];atlas=[]
    for (seed,track,strategy),q in Q.groupby(['seed','track','strategy']):
        assert len(q)==559 and not q[['condition_a','condition_b']].duplicated().any()
        valid=q.prediction.notna();v=q[valid].copy();e=abs(v.prediction-v.observed);wrong=(v.prediction<0)!=(v.observed<0);loss=np.where(wrong,abs(v.observed),0)
        pm.append(dict(seed=seed,track=track,strategy=strategy,contrast_mae=hierarchy(v,e) if len(v) else np.nan,selection_loss=hierarchy(v,loss) if len(v) else np.nan,wrong=int(wrong.sum()),coverage=valid.mean(),n=int(valid.sum())))
        for pair,b in v.groupby('pair_id'):
            b=b.copy();b['err']=abs(b.prediction-b.observed);b['wrong']=(b.prediction<0)!=(b.observed<0)
            bp.append(dict(seed=seed,track=track,strategy=strategy,pair=pair,mae=b.groupby('exact_block').err.mean().mean(),wrong=int(b.wrong.sum()),n=len(b)))
            if strategy in ['Point','TOPSIS','History']:
                block=b.groupby('exact_block')[['observed','prediction']].mean().mean()
                # Positive difference r_b-r_a equals removal_a-removal_b.
                atlas.append(dict(seed=seed,track=track,strategy=strategy,pair=pair,observed_advantage_pp=100*block.observed,predicted_advantage_pp=100*block.prediction,n=len(b),pollutants=';'.join(sorted(b.Pollutant.unique())),water_types=';'.join(sorted(b['Wastewater type'].unique()))))
    PM=pd.DataFrame(pm);M=M.merge(PM,on=['seed','track','strategy'],how='outer')
    for name,f in [('predictions',A),('pairs',Q),('method_inner_scores',S),('weight_sensitivity',W),('candidate_removal',R),('metrics',M),('pair_metrics',pd.DataFrame(bp)),('material_advantage',pd.DataFrame(atlas)),('solver_warnings',pd.DataFrame(warnings))]:f.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    assert topsis([[1,1,1],[2,2,2]])[0]==1
    assert np.allclose(topsis([[1,1,0],[1,1,0]]),.5)
    reference=S.sort_values(['seed','track','fold','topsis','point','id'],ascending=[True,True,True,False,True,True]).groupby(['seed','track','fold']).first().reset_index()[['seed','track','fold','id']].rename(columns={'id':'reference'})
    stability=W.merge(reference,on=['seed','track','fold']);stability['same']=stability.id==stability.reference
    stability.groupby('track').same.agg(['mean','count']).to_csv(OUT/'weight_stability.csv')
    R['winner_available']=~R.removed.eq(R.original.str.rsplit('_',n=1).str[0]);R['same']=R.winner==R.original
    R[R.winner_available].groupby('track').same.agg(['mean','count']).to_csv(OUT/'removal_stability.csv')
    report=dict(completed_folds=51,methods=24,configurations=45,baselines=2,estimator_fit_count=46*(30*5+21*7),constant_mean_evaluations=30*5+21*7,conditions=1279,pairs=559,all_primary_splits_equal_previous=True,all_outer_rows_predicted_once=True,all_inner_outer_arrays_disjoint=True,all_topsis_scores_recomputed=True,solver_warning_config_folds=len(warnings),data_sha256=hashlib.sha256((ROOT/'data/condition_data.csv').read_bytes()).hexdigest())
    (OUT/'scientific_checks.json').write_text(json.dumps(report,indent=2),'utf-8')
    (OUT/'complete.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(M[(M.seed==11)&M.strategy.isin(['Point','Contrast','TOPSIS','History','Constant'])].to_string(index=False))
    print(json.dumps(report))
if __name__=='__main__':main()
