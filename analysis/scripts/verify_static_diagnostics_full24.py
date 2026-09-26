"""Recompute the active data-only diagnostics without importing retired models."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.linalg import null_space

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'evidence/supplement_v1';V=ROOT/'evidence/full24_v5'
D=pd.read_csv(ROOT/'data/condition_data.csv',float_precision='round_trip')
B=D[D.Adsorbent!='PAC']
PROPS=['Pyrolysis temperature','Pyrolysis time','C','H','O','N','(O+N)/C','Ash','H/C','O/C','N/C','Surface area','Pore volume','Average pore size']
SETS={'Full 14':PROPS,'Non-ratio 10':['Pyrolysis temperature','Pyrolysis time','C','H','O','N','Ash','Surface area','Pore volume','Average pore size'],'Core 6':['Pyrolysis temperature','Pyrolysis time','O/C','Surface area','Pore volume','Average pore size'],'Core 5':['Pyrolysis temperature','O/C','Surface area','Pore volume','Average pore size'],'Pore 3':['Surface area','Pore volume','Average pore size']}
raw=pd.read_excel(ROOT.parent/'先前的论文/Raw_data.xlsx');raw.columns=raw.columns.str.strip()
for c in raw.select_dtypes('object'):raw[c]=raw[c].str.strip()
dup=raw.duplicated();mapping=pd.read_csv(ROOT/'data/raw_to_condition.csv')
raw['r']=raw['Final concentration']/raw['Initial concentration'];raw['condition_id']=mapping.condition_id;raw['source_candidate']=mapping.source_candidate;raw['complete_duplicate']=dup;raw['raw_row_excel']=mapping.raw_row_excel
saved=pd.read_csv(T/'audited_raw_records.csv',float_precision='round_trip')
pd.testing.assert_frame_equal(raw,saved,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
counts=pd.DataFrame([{'unit':n,'all':len(g),'biochar':int(g.Adsorbent.ne('PAC').sum())} for n,g in [('Raw rows',raw),('Full deduplication',raw[~dup]),('Conditions',D)]])
pd.testing.assert_frame_equal(counts,pd.read_csv(T/'unit_counts.csv'))
flags=raw.assign(zero_time=raw['Adsorption time'].eq(0),r_gt1=raw.r.gt(1),PB600_flag=raw.Adsorbent.eq('PB600')&np.isclose(raw.N,2.77)).groupby('source_candidate')[['zero_time','r_gt1','PB600_flag']].sum()
pd.testing.assert_frame_equal(flags,pd.read_csv(T/'raw_flag_counts.csv',index_col=0),check_dtype=False)
ratio=[];masses={'H':1.008,'C':12.011,'O':15.999,'N':14.007}
for _,r in B[['source_candidate','Adsorbent']+PROPS].drop_duplicates().iterrows():
 for el in ['H','O','N']:ratio.append({'material':r.Adsorbent,'source':r.source_candidate,'ratio':el+'/C','recorded':r[el+'/C'],'atomic_calculated':(r[el]/masses[el])/(r.C/masses['C']),'mass_calculated':r[el]/r.C})
pd.testing.assert_frame_equal(pd.DataFrame(ratio),pd.read_csv(T/'ratio_definition_diagnostic.csv'),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
ranks=[]
for variant,frame in {'All':D,'Biochar':B,'Flag excluded':B[~(B.Adsorbent.eq('PB600')&np.isclose(B.N,2.77))]}.items():
 for label,cols in SETS.items():
  m=frame[['source_candidate','Adsorbent']+cols].drop_duplicates();z=(m[cols]-m[cols].mean())/m[cols].std(ddof=0).replace(0,1)
  s=pd.get_dummies(m.source_candidate).to_numpy(float);b=np.c_[s,z];w=z-z.groupby(m.source_candidate).transform('mean');n=null_space(b,rcond=1e-10)
  for tol in np.logspace(-14,-4,6):ranks.append({'variant':variant,'set':label,'tolerance':tol,'rank':np.linalg.matrix_rank(w,tol=tol),'features':len(cols),'profiles':len(m),'nullity_at_1e10':n.shape[1]})
pd.testing.assert_frame_equal(pd.DataFrame(ranks),pd.read_csv(T/'full_rank_diagnostic.csv'),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-16)
# Rebuild every eligible unordered pair directly from the condition table.
K=['source_candidate','Pollutant','Initial concentration','Solution pH','RPM','Volume','Adsorbent dosage','Adsorption temperature','Ion concentration','Humic acid','Wastewater type','Adsorption type','Adsorption time']
eligible=set()
for _,g in B[B['Adsorption time']>0].groupby(K,dropna=False,sort=True):
 for a,b in combinations(g.itertuples(),2):
  if a.Adsorbent!=b.Adsorbent:eligible.add(tuple(sorted((a.condition_id,b.condition_id))))
P=pd.read_csv(ROOT/'evidence/model24_topsis_v1/fixed_pair_queries.csv')
actual={tuple(sorted((int(r.condition_a),int(r.condition_b)))) for r in P.itertuples()}
assert actual==eligible and len(actual)==559,(len(actual),len(eligible))
report=dict(audited_original_rows=len(raw),count_table_verified=True,flags_verified=True,ratio_records_verified=len(ratio),descriptor_rank_rows_verified=len(ranks),all_eligible_pairs_rebuilt=True,matched_queries=len(eligible),new_fits=0)
(V/'static_diagnostic_check.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps(report,indent=2))
