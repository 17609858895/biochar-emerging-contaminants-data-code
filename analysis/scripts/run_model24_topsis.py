"""Versioned extension: 24 methods, inner-only selection and held-out material decisions."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,argparse,time,warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold,LeaveOneGroupOut
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from joblib import Parallel,delayed
from model24_registry import CONFIGS,METHODS,estimator,topsis

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/model24_topsis_v1'
MAT=['Pyrolysis temperature','O/C','Surface area','Pore volume','Average pore size']
NUM=MAT+['log_time','Solution pH','RPM','Adsorption temperature']
CAT=['Pollutant','Wastewater type','Adsorption type']
CONTEXT=['source_candidate','Pollutant','Initial concentration','Solution pH','RPM','Volume','Adsorbent dosage','Adsorption temperature','Ion concentration','Humic acid','Wastewater type','Adsorption type']
D=pd.read_csv(ROOT/'data/condition_data.csv',float_precision='round_trip')
D=D[(D.Adsorbent!='PAC')&(D['Adsorption time']>0)].reset_index(drop=True)
D['log_time']=np.log1p(D['Adsorption time'])
D['exact_block']=D.groupby(CONTEXT,dropna=False,sort=True).ngroup()
D['broad_block']=D.groupby(['source_candidate']+CAT,dropna=False,sort=True).ngroup()
lookup={int(c):i for i,c in enumerate(D.condition_id)}
P=pd.read_csv(ROOT/'evidence/supported_material_contrasts.csv',float_precision='round_trip')
P=P[P.condition_a.isin(lookup)&P.condition_b.isin(lookup)].reset_index(drop=True)
P['ia']=P.condition_a.map(lookup);P['ib']=P.condition_b.map(lookup)
P['pair_id']=P.material_a+' / '+P.material_b
for k in ['exact_block','broad_block','Pollutant','Wastewater type','Adsorption time']:P[k]=P.ia.map(D[k])
P['observed']=D.iloc[P.ib].r.values-D.iloc[P.ia].r.values
assert len(D)==1279 and len(P)==559

def folds(indices,track,seed,n=5):
    f=D.iloc[indices]
    if track=='source':parts=LeaveOneGroupOut().split(f,groups=f.source_candidate)
    else:parts=StratifiedGroupKFold(n_splits=min(n,f[f'{track}_block'].nunique()),shuffle=True,random_state=seed).split(f,f.source_candidate,f[f'{track}_block'])
    for a,b in parts:yield indices[a],indices[b]

def preprocess(operations=False):
    return ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),NUM[5:] if operations else NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)

def hierarchy(q,values):
    q=q[['source','pair_id','exact_block']].copy();q['v']=values
    return float(q.groupby(['source','pair_id','exact_block']).v.mean().groupby(level=[0,1]).mean().groupby(level=0).mean().mean())

def metrics(pred,idx):
    mask=np.zeros(len(D),bool);mask[idx]=True;q=P[mask[P.ia]&mask[P.ib]]
    delta=pred[q.ib]-pred[q.ia]
    point=float(pd.DataFrame({'e':abs(D.iloc[idx].r.values-pred[idx]),'s':D.iloc[idx].source_candidate.values}).groupby('s').e.mean().mean())
    return [point,hierarchy(q,abs(delta-q.observed)),hierarchy(q,np.where((delta<0)!=(q.observed<0),abs(q.observed),0))]

def evaluate(c,tr,te,inner,seed,fd):
    checkpoint=fd/(c['id']+'.npz')
    if checkpoint.exists():
        z=np.load(checkpoint);return {'config':c,**{k:z[k] for k in ['inner','outer','criteria']},'warnings':str(z['warnings'])}
    begin=time.time();oof=np.full(len(D),np.nan);out=np.full(len(D),np.nan)
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for train,val in inner+[(tr,te)]:
            if c['method']=='Constant':p=np.full(len(val),D.iloc[train].r.mean())
            else:
                est=Ridge(alpha=10) if c['method']=='Operations' else estimator(c,seed)
                model=make_pipeline(preprocess(c['method']=='Operations'),est)
                model.fit(D.iloc[train],D.iloc[train].r.values);p=np.asarray(model.predict(D.iloc[val])).reshape(-1)
            if np.array_equal(val,te):out[val]=p
            else:oof[val]=p
        msgs=sorted(set(str(w.message) for w in caught))
    assert np.isfinite(oof[tr]).all() and np.isfinite(out[te]).all(),c
    crit=np.asarray(metrics(oof,tr))
    np.savez_compressed(checkpoint,inner=oof,outer=out,criteria=crit,warnings=json.dumps(msgs),elapsed=time.time()-begin)
    return dict(config=c,inner=oof,outer=out,criteria=crit,warnings=json.dumps(msgs))

def run(seed,tracks,jobs):
    for track in tracks:
        for fold,(tr,te) in enumerate(folds(np.arange(len(D)),track,seed)):
            fd=OUT/f's{seed}_{track}_{fold:02d}';fd.mkdir(exist_ok=True)
            if (fd/'complete.json').exists():continue
            inner=list(folds(tr,track,seed,4));split=np.full(len(D),-1)
            group='source_candidate' if track=='source' else f'{track}_block'
            assert not set(D.iloc[tr][group])&set(D.iloc[te][group])
            for i,(a,b) in enumerate(inner):
                assert not set(D.iloc[a][group])&set(D.iloc[b][group]);split[b]=i
            mask=np.zeros(len(D),bool);mask[te]=True
            assert not np.any(mask[P.ia]^mask[P.ib])
            trainmask=np.zeros(len(D),bool);trainmask[tr]=True
            pt=P[trainmask[P.ia]&trainmask[P.ib]]
            assert np.array_equal(split[pt.ia],split[pt.ib])
            pd.DataFrame({'condition_id':D.condition_id,'outer_test':mask,'inner_fold':split}).to_csv(fd/'split.csv',index=False)
            baseline=[dict(id=m,method=m,family='Baseline',parameters={}) for m in ['Constant','Operations']]
            print('START',seed,track,fold,len(tr),len(te),flush=True)
            results=Parallel(n_jobs=jobs,backend='loky')(delayed(evaluate)(c,tr,te,inner,seed,fd) for c in CONFIGS+baseline)
            rows=[]
            for r in results:
                c=r['config'];rows.append(dict(id=c['id'],method=c['method'],family=c['family'],point=r['criteria'][0],contrast=r['criteria'][1],loss=r['criteria'][2],warnings=r['warnings']))
            scores=pd.DataFrame(rows);scores.to_csv(fd/'configuration_scores.csv',index=False)
            # One representative per method, selected by inner point MAE before TOPSIS.
            candidates=scores[~scores.method.isin(['Constant','Operations'])].sort_values(['point','id']).groupby('method',sort=True).head(1).sort_values('method').copy()
            candidates['topsis']=topsis(candidates[['point','contrast','loss']]);candidates.to_csv(fd/'method_scores.csv',index=False)
            selected={'Point':candidates.sort_values(['point','id']).iloc[0].id,'Contrast':candidates.sort_values(['contrast','point','id']).iloc[0].id,'TOPSIS':candidates.sort_values(['topsis','point','id'],ascending=[False,True,True]).iloc[0].id}
            sensitivities=[]
            for i in range(1,9):
                for j in range(1,10-i):
                    w=(i/10,j/10,(10-i-j)/10);v=topsis(candidates[['point','contrast','loss']],w)
                    ix=sorted(range(len(v)),key=lambda k:(-v[k],candidates.iloc[k].point,candidates.iloc[k].id))[0]
                    sensitivities.append(dict(w_point=w[0],w_contrast=w[1],w_loss=w[2],id=candidates.iloc[ix].id))
            pd.DataFrame(sensitivities).to_csv(fd/'weight_sensitivity.csv',index=False)
            pool=[]
            for removed in candidates.method:
                c=candidates[candidates.method!=removed].copy();c['topsis']=topsis(c[['point','contrast','loss']]);winner=c.sort_values(['topsis','point','id'],ascending=[False,True,True]).iloc[0]
                pool.append(dict(removed=removed,winner=winner.id,original=selected['TOPSIS']))
            pd.DataFrame(pool).to_csv(fd/'candidate_removal.csv',index=False)
            byid={r['config']['id']:r for r in results}
            mappings={**{r.method:r.id for r in candidates.itertuples()},**selected,'Constant':'Constant','Operations':'Operations'}
            preds=[];pairs=[]
            for label,ident in mappings.items():
                pred=byid[ident]['outer'];a=D.iloc[te][['condition_id','source_candidate','Adsorbent','Pollutant','Wastewater type','r']].copy();a['prediction']=pred[te];a['strategy']=label;a['id']=ident;preds.append(a)
                q=P[mask[P.ia]&mask[P.ib]].copy();q['prediction']=pred[q.ib]-pred[q.ia];q['strategy']=label;q['id']=ident;pairs.append(q)
            hist=pt.groupby(['pair_id','exact_block'])['observed'].mean().groupby(level=0).mean()
            q=P[mask[P.ia]&mask[P.ib]].copy();q['prediction']=q.pair_id.map(hist);q['strategy']='History';q['id']='History';pairs.append(q)
            pd.concat(preds).to_csv(fd/'predictions.csv',index=False)
            pd.concat(pairs).to_csv(fd/'pairs.csv',index=False)
            (fd/'complete.json').write_text(json.dumps(dict(seed=seed,track=track,fold=fold,selected=selected,methods=len(candidates)),indent=2))
            print('DONE',seed,track,fold,selected,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=11);ap.add_argument('--tracks',default='exact,broad,source');ap.add_argument('--jobs',type=int,default=6);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol=dict(version='1.0',extension_to='benchmark_v2',created_before_fitting=True,seeds=[11,23,47],tracks=['exact','broad','source'],method_count=24,configuration_count=len(CONFIGS),configs=CONFIGS,features=NUM+CAT,primary_weights=[1/3]*3,criterion_directions=['cost']*3,criteria=['source balanced point MAE','hierarchical contrast MAE','hierarchical selection loss'],method_hyperparameter_selection='inner point MAE',topsis_ties='lower inner point MAE then lexical config ID',normalization='vector norm, drop range <=1e-12 criteria',weight_sensitivity='36 strictly positive simplex weights at 0.1 spacing',candidate_sensitivity='leave one method out',new_observations=0,clipping=False,preprocessing='training partition only; standardization and dense one-hot; spline additionally expands encoded columns',data_sha256=hashlib.sha256((ROOT/'data/condition_data.csv').read_bytes()).hexdigest())
    pp=OUT/'protocol.json';blob=json.dumps(protocol,ensure_ascii=False,indent=2)
    if pp.exists():assert json.loads(pp.read_text())==json.loads(blob),'Protocol changed; use a new output version'
    else:pp.write_text(blob,encoding='utf-8')
    D.to_csv(OUT/'analysis_conditions.csv',index=False);P.to_csv(OUT/'fixed_pair_queries.csv',index=False)
    run(args.seed,args.tracks.split(','),args.jobs)
