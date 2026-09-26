"""Observed chemistry panels for Fig. 6; no fitted effects or new observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'evidence/adsorption_effects_v1'
OUT=ROOT/'evidence/adsorption_figure6_v2';OUT.mkdir(parents=True,exist_ok=True)
C=pd.read_csv(OLD/'observed_operational_contrasts.csv',float_precision='round_trip')
W=pd.read_csv(OLD/'matched_water_contrasts.csv',float_precision='round_trip')
D=pd.read_csv(ROOT/'evidence/model24_topsis_v1/analysis_conditions.csv',float_precision='round_trip').set_index('condition_id')
P=pd.read_csv(ROOT/'evidence/model24_topsis_v1/fixed_pair_queries.csv',float_precision='round_trip')

T=C[C.feature=='Adsorption temperature'].copy()
TEMP=T.groupby(['Pollutant','source_candidate','Adsorbent']).removal_change_pp.mean().groupby(level=[0,1]).mean().groupby(level=0).mean().rename('mean_pp').reset_index()
TEMP=TEMP.merge(T.groupby('Pollutant').removal_change_pp.agg(n='size',minimum='min',maximum='max').reset_index())
TEMP.to_csv(OUT/'pollutant_temperature_responses.csv',index=False)
WATER=W.groupby(['source_candidate','Adsorbent','Pollutant','water_a','water_b'],sort=False).removal_change_pp.agg(mean_pp='mean',n='size').reset_index()
WATER.to_csv(OUT/'pollutant_water_responses.csv',index=False)

keys=['source_candidate','Pollutant','Adsorption type','Adsorption time','Initial concentration','Solution pH','RPM','Volume','Adsorbent dosage','Adsorption temperature','Ion concentration','Humic acid']
quads=[]
for k,(pair,q) in enumerate(P.groupby('pair_id',sort=False),1):
 a,b=q.material_a.iloc[0],q.material_b.iloc[0]
 aa=W[W.Adsorbent==a];bb=W[W.Adsorbent==b]
 m=aa.merge(bb,on=keys+['water_a','water_b'],suffixes=('_ma','_mb'),validate='one_to_one')
 for _,r in m.iterrows():
  ids=[int(r.condition_a_ma),int(r.condition_b_ma),int(r.condition_a_mb),int(r.condition_b_mb)]
  assert len(set(ids))==4
  removal=100*(1-D.loc[ids,'r'].to_numpy())
  advantage_first=removal[0]-removal[2];advantage_second=removal[1]-removal[3]
  interaction=advantage_second-advantage_first
  assert np.isclose(interaction,r.removal_change_pp_ma-r.removal_change_pp_mb,rtol=0,atol=1e-10)
  values={key:r[key] for key in keys}
  values.update(pair=pair,pair_label='P'+str(k),material_a=a,material_b=b,water_first=r.water_a,water_second=r.water_b,
   condition_a_first=ids[0],condition_a_second=ids[1],condition_b_first=ids[2],condition_b_second=ids[3],
   advantage_first_pp=advantage_first,advantage_second_pp=advantage_second,interaction_pp=interaction)
  quads.append(values)
Q=pd.DataFrame(quads)
assert len(Q)==119 and Q.source_candidate.nunique()==2
blockkeys=['pair']+[k for k in keys if k!='Adsorption time']
Q['context_block']=pd.factorize(pd.MultiIndex.from_frame(Q[blockkeys]),sort=True)[0]
Q.to_csv(OUT/'material_water_four_condition_comparisons.csv',index=False)
BLOCK=Q.groupby(['pair','pair_label','source_candidate','Pollutant','context_block','water_first','water_second'],sort=False)[['advantage_first_pp','advantage_second_pp','interaction_pp']].mean().reset_index()
BLOCK.to_csv(OUT/'material_water_context_means.csv',index=False)
INTER=BLOCK.groupby(['pair','pair_label','source_candidate','Pollutant','water_first','water_second'],sort=False).agg(mean_pp=('interaction_pp','mean'),minimum=('interaction_pp','min'),maximum=('interaction_pp','max'),blocks=('context_block','size'),advantage_first_pp=('advantage_first_pp','mean'),advantage_second_pp=('advantage_second_pp','mean')).reset_index()
INTER=INTER.merge(Q.groupby(['pair','Pollutant']).size().rename('n').reset_index(),on=['pair','Pollutant'],validate='one_to_one')
INTER.to_csv(OUT/'material_water_interaction_summary.csv',index=False)
assert np.allclose(INTER.mean_pp,INTER.advantage_second_pp-INTER.advantage_first_pp)
assert WATER.n.sum()==239 and TEMP.n.sum()==99
report=dict(new_observations=0,new_estimator_fits=0,pollutants_with_temperature_comparisons=len(TEMP),temperature_contrasts=99,water_contrasts=239,water_material_pollutant_groups=len(WATER),four_condition_comparisons=len(Q),four_condition_sources=2,four_condition_material_pairs=Q.pair.nunique(),four_condition_context_blocks=len(BLOCK),interaction_groups=len(INTER),interaction_definition='(removal_a_second - removal_b_second) - (removal_a_first - removal_b_first)',interaction_identity_verified=True,interaction_aggregation='mean across recorded times within matched context blocks, then equal block means within pair-pollutant',interpretation='Descriptive observed interaction contrast; no causal identification or new experiment',temperature_weights='mean within source-material-pollutant, then equal materials and sources within pollutant',water_weights='mean within each source-material-pollutant group',intervals='No confidence intervals; temperature marks retain all recorded differences, and interaction lines span matched context-block means')
(OUT/'protocol.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(INTER[['pair_label','Pollutant','n','blocks','mean_pp','minimum','maximum']].to_string(index=False),flush=True)
