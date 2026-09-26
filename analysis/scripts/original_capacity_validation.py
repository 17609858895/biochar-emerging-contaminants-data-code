"""Validation sensitivity of the supplied Capacity target using reported CatBoost settings.

This is a protocol reconstruction, not an exact reproduction: original seeds,
fitted objects, CatBoost version and iteration count were not provided. No score
from this analysis is compared numerically with concentration-ratio errors.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit,LeaveOneGroupOut
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,r2_score
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/original_capacity_validation_v2';OUT.mkdir(exist_ok=True)
RAW=ROOT.parent/'dataset/Raw_data.xlsx'
if not RAW.exists():RAW=next((ROOT.parent/'dataset').glob('*.xlsx'))
D=pd.read_excel(RAW);D.columns=D.columns.str.strip()
LINE=pd.read_csv(ROOT/'data/raw_to_condition.csv').sort_values('raw_row_excel')
assert len(D)==len(LINE)==3757
D['condition_id']=LINE.condition_id.to_numpy();D['source_candidate']=LINE.source_candidate.to_numpy()
CAT=['Adsorbent','Pollutant','Wastewater type','Adsorption type']
FEATURES=[c for c in D.columns if c not in ['H','O','N','Final concentration','Capacity','condition_id','source_candidate']]
assert len(FEATURES)==24
NUM=[c for c in FEATURES if c not in CAT]
KEYS=['source_candidate','Pollutant','Initial concentration','Solution pH','RPM','Volume','Adsorbent dosage','Adsorption temperature','Ion concentration','Humic acid','Wastewater type','Adsorption type']
G=D.copy()
for col in G.select_dtypes(include='object'):G[col]=G[col].str.strip()
G['Pollutant']=G.Pollutant.replace({'NXP':'NPX','IBF':'IBU'})
D['operation_block']=G.groupby(KEYS,dropna=False,sort=True).ngroup()
D['context']=G.groupby(['source_candidate','Pollutant','Wastewater type','Adsorption type'],dropna=False,sort=True).ngroup()
SEEDS=[11,23,47,91,131,173,211,257,307,353]
PARAMS=dict(iterations=1000,learning_rate=0.7787452095810325,depth=12,l2_leaf_reg=10,rsm=0.9367412792865765,
            loss_function='RMSE',verbose=False,allow_writing_files=False,thread_count=1)

def task(track,seed,train,test):
    cp=OUT/f'{track}_{seed}.npz'
    if cp.exists():return
    start=time.time()
    pre=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)
    model=make_pipeline(pre,CatBoostRegressor(random_seed=seed,**PARAMS))
    with threadpool_limits(limits=1):model.fit(D.iloc[train][FEATURES],D.iloc[train].Capacity);pred=model.predict(D.iloc[test][FEATURES])
    np.savez_compressed(cp,train=train,test=test,prediction=pred,elapsed=time.time()-start)
    print('DONE',track,seed,len(train),len(test),round(r2_score(D.iloc[test].Capacity,pred),4),flush=True)

def main():
    # Unchanged designs reuse exact prior predictions; only alias-harmonized
    # group designs are refitted. The 20 superseded fits remain traceable.
    import shutil
    prior=ROOT/'evidence/original_capacity_validation_v1'
    for track in ['Row random','Condition group','Source holdout']:
        for p in prior.glob(track+'_*.npz'):
            if not (OUT/p.name).exists():shutil.copy2(p,OUT/p.name)
    jobs=[];idx=np.arange(len(D))
    for seed in SEEDS:
        a,b=train_test_split(idx,test_size=.3,random_state=seed);jobs.append(('Row random',seed,a,b))
        for track,key in [('Condition group','condition_id'),('Operating group','operation_block'),('Context group','context')]:
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(D,groups=D[key]))
            assert not set(D.iloc[a][key])&set(D.iloc[b][key])
            assert not set(D.iloc[a].condition_id)&set(D.iloc[b].condition_id)
            jobs.append((track,seed,a,b))
    for a,b in LeaveOneGroupOut().split(D,groups=D.source_candidate):
        source=str(D.iloc[b].source_candidate.iloc[0]);jobs.append(('Source holdout',int(source.rsplit('_',1)[1]),a,b))
    (OUT/'protocol.json').write_text(json.dumps(dict(response='Supplied Capacity column, reported mg/g',features=FEATURES,
        reported_parameters={k:PARAMS[k] for k in ['learning_rate','depth','l2_leaf_reg','rsm']},explicit_completion_of_unreported_settings=PARAMS,
        original_reported_test_r2=.9433,original_reported_test_mae=4.95,seeds=SEEDS,
        grouping_aliases='Strip grouping strings; harmonize NXP to NPX and IBF to IBU for group assignment only; original model inputs unchanged',
        fit_accounting='47 current design fits, of which 27 reused unchanged; 20 superseded raw-label group fits remain in v1, 67 executed fits across versions',
        preparation='Numerical scaling and dense one-hot fitted on training data only; no tuning on these held-out results',
        target_population='All 3757 supplied rows including PAC and zero-time records',
        raw_material_labels=int(D.Adsorbent.nunique()),
        comparability='Same rows, capacity response, 24 input fields and fixed estimator settings across split types; group holdout has varying test sizes',
        reconstruction_boundary='Original seed, exact software, iteration count, fitted model and optimization history unavailable; not an exact published-score reproduction',
        source_holdout='One held-out source at a time; stochastic seed is its source number; pooled summary is separate from fold scores',
        new_observations=0,raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
    Parallel(n_jobs=3)(delayed(task)(*x) for x in jobs)
    rows=[];predictions=[]
    for track,seed,a,b in jobs:
        z=np.load(OUT/f'{track}_{seed}.npz');pred=z['prediction'];obs=D.iloc[b].Capacity.to_numpy()
        row=dict(track=track,seed=seed,n_train=len(a),n_test=len(b),r2=r2_score(obs,pred),mae=mean_absolute_error(obs,pred),
                 condition_overlap_pct=100*D.iloc[b].condition_id.isin(set(D.iloc[a].condition_id)).mean(),
                 operating_overlap_pct=100*D.iloc[b].operation_block.isin(set(D.iloc[a].operation_block)).mean(),
                 context_overlap_pct=100*D.iloc[b].context.isin(set(D.iloc[a].context)).mean(),
                 material_overlap_pct=100*D.iloc[b].Adsorbent.isin(set(D.iloc[a].Adsorbent)).mean(),
                 source_overlap_pct=100*D.iloc[b].source_candidate.isin(set(D.iloc[a].source_candidate)).mean())
        rows.append(row)
        predictions.append(pd.DataFrame(dict(track=track,seed=seed,raw_row_excel=b+2,observed=obs,prediction=pred)))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
    print(pd.DataFrame(rows).groupby('track')[['r2','mae','condition_overlap_pct']].median().to_string(),flush=True)

if __name__=='__main__':main()
