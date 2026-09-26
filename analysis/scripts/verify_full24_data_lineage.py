"""Independently verify aggregation against the preserved original workbook."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT.parent/'先前的论文/Raw_data.xlsx';OUT=ROOT/'evidence/full24_v5'
r=pd.read_excel(RAW);r.columns=r.columns.str.strip()
for c in r.select_dtypes(include='object'):r[c]=r[c].str.strip()
distinct=len(r.drop_duplicates());original_columns=list(r.columns)
r['Pollutant']=r.Pollutant.replace({'IBF':'IBU','NXP':'NPX'})
r['r']=r['Final concentration']/r['Initial concentration']
lineage=pd.read_csv(ROOT/'data/raw_to_condition.csv');assert len(r)==len(lineage)==3757
assert np.allclose(r.r,lineage.r,atol=1e-14,rtol=1e-14)
r['condition_id']=lineage.condition_id.values
cols=[c for c in original_columns if c not in ['Final concentration','Capacity']]
assert r.groupby('condition_id')[cols].nunique(dropna=False).max().max()==1
assert r[cols].drop_duplicates().shape[0]==1348
check=r.groupby('condition_id').r.agg(['mean','std','size'])
saved=pd.read_csv(ROOT/'data/condition_data.csv').set_index('condition_id')
assert np.allclose(check['mean'],saved.r,rtol=1e-13,atol=1e-13)
assert np.array_equal(check['size'],saved.raw_n)
assert np.allclose(check['std'].fillna(0),saved.r_sd.fillna(0),rtol=1e-11,atol=1e-13)
summary=dict(original_rows=len(r),original_fields=len(original_columns),full_field_distinct_before_aliases=distinct,condition_count=len(check),ratio_lineage_verified=True,all_grouping_fields_constant=True,mean_and_count_verified=True,max_mean_error=float(abs(check['mean']-saved.r).max()),raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),condition_sha256=hashlib.sha256((ROOT/'data/condition_data.csv').read_bytes()).hexdigest(),grouping_columns=cols,aliases={'IBF':'IBU','NXP':'NPX'})
(OUT/'data_lineage_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
