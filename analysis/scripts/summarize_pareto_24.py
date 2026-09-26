"""Descriptive Pareto sets of the 24 methods, independent of TOPSIS selection."""
from pathlib import Path
import ast,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/pareto_v1';OUT.mkdir(exist_ok=True)
node=next(n for n in ast.parse((ROOT/'scripts/model24_registry.py').read_text(encoding='utf8')).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='METHODS'for t in n.targets))
order=list(ast.literal_eval(node.value));assert len(order)==24
src=ROOT/'evidence/model24_topsis_v1/metrics.csv';m=pd.read_csv(src);cols=['point_mae','contrast_mae','selection_loss'];parts=[]
for track in ['exact','broad','source']:
 g=m[(m.track==track)&m.strategy.isin(order)].groupby('strategy')[cols].median().reindex(order);x=g.to_numpy();assert np.isfinite(x).all()
 dom=np.array([np.any(np.all(x<=r+1e-12,axis=1)&np.any(x<r-1e-12,axis=1))for r in x])
 for i in range(24):assert dom[i]==any(all(x[j,k]<=x[i,k]+1e-12 for k in range(3))and any(x[j,k]<x[i,k]-1e-12 for k in range(3))for j in range(24))
 g['nondominated']=~dom;g['track']=track;g['model_number']=np.arange(1,25);parts.append(g.reset_index())
out=pd.concat(parts,ignore_index=True)
if (OUT/'method_medians.csv').exists():pd.testing.assert_frame_equal(out,pd.read_csv(OUT/'method_medians.csv'),check_exact=False,check_dtype=False,rtol=1e-12,atol=1e-15)
out.to_csv(OUT/'method_medians.csv',index=False)
report=dict(methods=24,tracks=3,rows=len(out),aggregation='Separate metric medians across seeds 11,23,47',objectives='All three minimized in untransformed fraction units',tolerance=1e-12,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),nondominated_methods={t:out.loc[(out.track==t)&out.nondominated,'strategy'].tolist()for t in ['exact','broad','source']},used_for_selection=False,independent_pairwise_dominance_check=True,new_observations=0,new_fits=0)
(OUT/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report)
