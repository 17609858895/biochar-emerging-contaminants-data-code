"""Observed, exactly matched structure and water-context comparisons.
Descriptor changes are joint material-profile changes, not independent effects.
"""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from compact_figure_layout import *
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/adsorption_effects_v1'
D=pd.read_csv(ROOT/'evidence/model24_topsis_v1/analysis_conditions.csv',float_precision='round_trip')
P=pd.read_csv(ROOT/'evidence/model24_topsis_v1/fixed_pair_queries.csv',float_precision='round_trip')
C=pd.read_csv(E/'observed_operational_contrasts.csv',float_precision='round_trip')
FEAT=['Pyrolysis temperature','Surface area','Pore volume','Average pore size','O/C']
PAIR=list(P.pair_id.drop_duplicates())
profiles=[]
for i,pair in enumerate(PAIR):
 q=P[P.pair_id==pair];row=dict(pair=pair,pair_label='P'+str(i+1),material_a=q.material_a.iloc[0],material_b=q.material_b.iloc[0])
 for f in FEAT:
  delta=D.iloc[q.ia][f].to_numpy()-D.iloc[q.ib][f].to_numpy()
  assert np.ptp(delta)<1e-10
  row[f]=float(delta[0]);row[f+'_scaled']=float(delta[0]/np.ptp(D[f]))
 profiles.append(row)
PROF=pd.DataFrame(profiles);PROF.to_csv(E/'paired_descriptor_changes.csv',index=False)
RESP=P.groupby(['pair_id','Pollutant','exact_block'])['observed'].mean().groupby(level=[0,1]).mean().mul(100).rename('removal_advantage_pp').reset_index()
RESP.to_csv(E/'pair_pollutant_advantages.csv',index=False)
ph=C[C.feature=='Solution pH']
PH=ph.groupby(['Pollutant','source_candidate','Adsorbent']).removal_change_pp.mean().groupby(level=[0,1]).mean().groupby(level=0).mean().rename('mean_pp').reset_index()
PH=PH.merge(ph.groupby('Pollutant').size().rename('n').reset_index())
PH.to_csv(E/'pollutant_ph_responses.csv',index=False)
keys=['source_candidate','Adsorbent','Pollutant','Adsorption type','Adsorption time','Initial concentration','Solution pH','RPM','Volume','Adsorbent dosage','Adsorption temperature','Ion concentration','Humic acid']
rows=[]
for key,g in D.groupby(keys,dropna=False,sort=True):
 for a,b in combinations(sorted(g['Wastewater type'].unique()),2):
  aa=g[g['Wastewater type']==a];bb=g[g['Wastewater type']==b]
  if len(aa)==len(bb)==1:rows.append(dict(zip(keys,key),water_a=a,water_b=b,condition_a=int(aa.condition_id.iloc[0]),condition_b=int(bb.condition_id.iloc[0]),removal_change_pp=100*(aa.r.iloc[0]-bb.r.iloc[0])))
WATER=pd.DataFrame(rows);WATER.to_csv(E/'matched_water_contrasts.csv',index=False)
WM=WATER.groupby(['source_candidate','Adsorbent','Pollutant']).removal_change_pp.mean().groupby(level=[0,1]).mean().rename('mean_pp').reset_index()
WM=WM.merge(WATER.groupby(['source_candidate','Adsorbent']).size().rename('n').reset_index())
WM.to_csv(E/'material_water_responses.csv',index=False)
summary={'matched_water_contrasts':len(WATER),'water_sources':int(WATER.source_candidate.nunique()),'water_materials':int(WATER.Adsorbent.nunique()),'water_source_means':WM.groupby('source_candidate').mean_pp.mean().to_dict(),'pollutant_ph':PH.to_dict('records'),'pair_pollutant_responses':RESP.to_dict('records'),'matched_descriptor_profiles':14,'descriptor_interpretation':'joint measured changes only; no independent descriptor effects or causal slopes','weighting':'condition contrasts averaged within material-pollutant, then pollutants/materials/sources as relevant; no new observations'}
(E/'context_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

def descriptor_panel(ax):
 v=PROF[[f+'_scaled' for f in FEAT]].to_numpy()
 cm=LinearSegmentedColormap.from_list('nature_new_signed',[CORAL,'white',BLUE])
 im=ax.imshow(v,cmap=cm,vmin=-1,vmax=1,aspect='auto')
 for i in range(len(PROF)):
  for j,f in enumerate(FEAT):
   val=PROF.iloc[i][f];label=f'{val:+.0f}' if j<2 else (f'{val:+.2f}' if j==3 else f'{val:+.3f}')
   if abs(val)<1e-12:label='0'
   ax.text(j,i,label,ha='center',va='center',fontsize=10.5,color=INK)
 ax.set_xticks(range(5),['ΔT\n(°C)','ΔBET\n($m^2$/$g$)','ΔV\n($cm^3$/$g$)','Δd\n(nm)','ΔO/C'])
 ax.set_yticks(range(8),PROF.pair_label);axis(ax,ticks=11.5)
 colorbar(ax,im,'Joint descriptor change / range',[-1,0,1])
 # Numeric annotations retain physical units; colour encodes signed fraction
 # of the feature's observed range, allowing unlike descriptors to share a map.
 matrix_dividers(ax,8,5)
def compound_range_panel(ax):
 ax.axvline(0,c=INK,lw=.7,ls=':')
 for i,pair in enumerate(PAIR):
  q=RESP[RESP.pair_id==pair].removal_advantage_pp
  ax.plot([q.min(),q.max()],[i,i],c=TEAL,lw=2.2)
  ax.scatter(q,np.full(len(q),i),c=TEAL,s=35,edgecolors=INK,lw=.5,zorder=2)
 ax.set_yticks(range(8),['P'+str(i+1)+f'  (k = {sum(RESP.pair_id==p)})' for i,p in enumerate(PAIR)])
 ax.set_ylim(7.7,-.7);ax.set_xlim(-95,95);ax.set_xticks([-80,-40,0,40,80])
 axis(ax,'Pollutant-specific advantage (pp)',ticks=11.5)
 ax.set_title('Each point is one represented pollutant',fontsize=12,pad=8)

def compound_heatmap_panel(ax):
 """Expose compound identities while retaining the original block-balanced values."""
 compounds=['IBU','CBZ','EE2','DCF','NPX','ALA','DIU','SIM','CAR','PYR','TEB']
 matrix=RESP.pivot(index='Pollutant',columns='pair_id',values='removal_advantage_pp').reindex(index=compounds,columns=PAIR)
 assert int(matrix.notna().sum().sum())==len(RESP)==17
 cm=LinearSegmentedColormap.from_list('nature_new_advantage',[CORAL,'white',BLUE])
 cm.set_bad('white')
 im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),cmap=cm,vmin=-90,vmax=90,aspect='auto')
 for i in range(len(compounds)):
  for j in range(len(PAIR)):
   val=matrix.iloc[i,j]
   ax.text(j,i,f'{val:.1f}' if pd.notna(val) else '–',ha='center',va='center',fontsize=10.4,color=INK)
 ax.set_xticks(range(len(PAIR)),PROF.pair_label);ax.set_yticks(range(len(compounds)),compounds)
 axis(ax,ticks=10.4)
 colorbar(ax,im,'Observed material advantage (pp)',[-90,0,90])
 matrix_dividers(ax,len(compounds),len(PAIR))
 matrix.to_csv(E/'pair_pollutant_advantage_matrix.csv')
def pollutant_ph_panel(ax):
 q=PH.sort_values('mean_pp');y=np.arange(len(q))
 ax.axvline(0,c=INK,lw=.7,ls=':');ax.hlines(y,0,q.mean_pp,color=PURPLE,lw=1.6)
 ax.scatter(q.mean_pp,y,c=PURPLE,s=42,edgecolors=INK,lw=.5)
 ax.set_yticks(y,[f'{r.Pollutant} (n = {r.n})' for r in q.itertuples()])
 ax.set_title('Compound-specific pH response',fontsize=14,pad=9)
 ax.set_ylim(len(q)-.3,-.7);ax.set_xlim(-11,6);ax.set_xticks([-10,-5,0,5]);axis(ax,'Observed pH 7 → 11 change (pp)',ticks=12.5)
def water_panel(ax):
 q=WM.sort_values(['source_candidate','Adsorbent']);labs=[]
 short={'NaOH-activated SCW biochars':'NaOH-SCW','Pristine SCW Biochar':'Pristine SCW'}
 ax.axvline(0,c=INK,lw=.7,ls=':')
 for i,r in enumerate(q.itertuples()):
  col=TEAL if r.source_candidate=='SI_ref_18' else GOLD
  ax.plot([0,r.mean_pp],[i,i],c=col,lw=2.3)
  ax.scatter(r.mean_pp,i,c=col,s=65,edgecolors=INK,lw=.5,label=('Effluent − lake' if i==0 else 'Lake − ground' if i==2 else None))
  labs.append(short.get(r.Adsorbent,r.Adsorbent)+f' (n = {r.n})')
 ax.set_yticks(range(len(q)),labs);ax.set_ylim(len(q)-.3,-.7);ax.set_xlim(min(-6,q.mean_pp.min()-1),max(2,q.mean_pp.max()+1))
 axis(ax,'Observed water-matrix change (pp)',ticks=12.5);ax.xaxis.set_major_locator(MaxNLocator(4))
 legend(ax,ncol=2,fontsize=10.5,handlelength=.7,columnspacing=.6)
if __name__=='__main__':print(json.dumps(summary,indent=2))

