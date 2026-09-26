"""Independent scalar selection and full-array metric checks for the dense grid."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from model24_registry import topsis
from run_model24_topsis import D,P,ROOT,metrics
O=ROOT/'evidence/topsis_surfaces_v1';E=ROOT/'evidence/model24_topsis_v1'
g=np.load(O/'grid_and_selections.npz');xy=np.vstack([g['vertices'],g['centres'],[[1/3,1/3]]]);w=np.c_[xy,1-xy.sum(axis=1)]
rng=np.random.default_rng(20260917);idx=np.r_[rng.choice(len(w)-1,24,replace=False),len(w)-1]
allm=pd.read_csv(O/'all_seed_metrics.csv');checks=[];wins=0
for track in ['exact','broad','source']:
 for si,seed in enumerate([11,23,47]):
  pred=np.full((len(idx),len(D)),np.nan);fds=sorted(p for p in E.glob(f's{seed}_{track}_*')if p.is_dir())
  for fi,fd in enumerate(fds):
   c=pd.read_csv(fd/'method_scores.csv').sort_values(['point','id']).reset_index(drop=True);te=pd.read_csv(fd/'split.csv').outer_test.to_numpy(bool)
   for j,k in enumerate(idx):
    winner=int(np.argmax(topsis(c[['point','contrast','loss']].to_numpy(),w[k])))
    assert winner==g[track+'_winners'][k,si*len(fds)+fi];wins+=1
    pred[j,te]=np.load(fd/(c.iloc[winner].id+'.npz'))['outer'][te]
  ref=allm[(allm.track==track)&(allm.seed==seed)].reset_index(drop=True)
  for j,k in enumerate(idx):
   got=metrics(pred[j],np.arange(len(D)));want=ref.iloc[k][['point_mae','contrast_mae','selection_loss']].to_numpy(float)
   diff=float(np.max(abs(got-want)));assert diff<1e-10
   checks.append(dict(track=track,seed=seed,grid_index=int(k),max_metric_difference=diff))
tri=g['vertices'][g['triangles']];area=abs(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))/2
assert np.isclose(area.sum(),.7*.7/2) and np.all(area>0)
for n,h in json.loads((O/'input_sha256.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
report=dict(scalar_winners_checked=wins,full_prediction_metric_reassemblies=len(checks),max_metric_difference=max(r['max_metric_difference']for r in checks),grid_total_area=float(area.sum()),all_input_hashes_unchanged=True,checks=checks)
(O/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print({k:v for k,v in report.items()if k!='checks'})
