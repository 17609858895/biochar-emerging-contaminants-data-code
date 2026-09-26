"""Operational adsorption interpretation, exact nature_new palette."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from compact_figure_layout import *
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/adsorption_effects_v1'
F=ROOT/'figures/adsorption_effects_v1/Fig07'
C=pd.read_csv(ROOT/'evidence/adsorption_effects_v2/effect_curves.csv');S=pd.read_csv(E/'observed_source_means.csv')
Q=pd.read_csv(E/'observed_operational_contrasts.csv')
def effects(feat,label):
 def draw(ax):
  g=C[C.feature==feat].groupby('level')
  z=g.ale_pp.agg(['min','median','max']);p=g.pdp_pp.median()
  ax.fill_between(z.index,z['min'],z['max'],color=BLUE,alpha=.3,lw=0)
  ax.plot(z.index,z['median'],'o-',c=BLUE,lw=2.5,ms=6,label='ALE')
  ax.plot(p.index,p.values,'s--',c=CORAL,lw=2.1,ms=5,label='PDP')
  ax.axhline(0,color=INK,lw=.7,ls=':',zorder=0)
  ax.set_xticks(z.index)
  axis(ax,label,'Centred removal effect (pp)',ticks=14)
  ax.yaxis.set_major_locator(MaxNLocator(5))
  legend(ax,ncol=2,fontsize=13)
  ax.set_xlim(z.index.min()-.06*np.ptp(z.index),z.index.max()+.06*np.ptp(z.index))
  if feat=='Solution pH':ax.set_ylim(-8.5,8.5)
  else:ax.set_ylim(-3.4,3.4)
 return draw
def matched(feat,caption):
 def draw(ax):
  s=S[S.feature==feat].sort_values('source_candidate');y=np.arange(len(s))
  # The points are descriptive hierarchical means, not fitted regression effects.
  ax.axvline(0,color=INK,lw=.7,ls=':',zorder=0)
  for k,row in enumerate(s.itertuples()):
   ax.plot([0,row.mean_pp],[k,k],color=TEAL,lw=2.3,zorder=1)
   ax.scatter(row.mean_pp,k,s=72,c=TEAL,edgecolors=INK,linewidths=.6,zorder=2)
  pooled=s.mean_pp.mean()
  ax.scatter(pooled,len(s)+.15,c=PURPLE,s=100,marker='D',edgecolors=INK,linewidths=.6,zorder=3)
  labels=[r.source_candidate.replace('SI_ref_','S')+f'  (n = {r.n_contrasts})' for r in s.itertuples()]+['Source mean']
  ax.set_yticks(list(y)+[len(s)+.15],labels);ax.invert_yaxis()
  ax.set_ylim(len(s)+.8,-.7)
  axis(ax,'Observed removal change (pp)',ticks=13.5)
  ax.set_title(caption,fontsize=14,loc='center',pad=9,weight='normal')
  if feat=='Solution pH':ax.set_xlim(-7.5,1);ax.set_xticks([-6,-4,-2,0])
  else:ax.set_xlim(-1,11);ax.set_xticks([0,5,10])
  ax.text(.97,.04,f'{pooled:+.2f} pp',transform=ax.transAxes,ha='right',va='bottom',fontsize=14)
 return draw
from matplotlib.ticker import MaxNLocator
from adsorption_context import pollutant_ph_panel, water_panel, PH, WM
NEW=ROOT/'evidence/adsorption_figure6_v2'
TEMP=pd.read_csv(NEW/'pollutant_temperature_responses.csv')
WATER=pd.read_csv(NEW/'pollutant_water_responses.csv')
INTER=pd.read_csv(NEW/'material_water_interaction_summary.csv')
BLOCK=pd.read_csv(NEW/'material_water_context_means.csv')

def water_heatmap(ax):
 materials=['NaOH-activated SCW biochars','Pristine SCW Biochar','AMCB','CB','MCB']
 drugs=['DCF','IBU','NPX']
 v=WATER.pivot(index='Adsorbent',columns='Pollutant',values='mean_pp').reindex(index=materials,columns=drugs)
 n=WATER.pivot(index='Adsorbent',columns='Pollutant',values='n').reindex(index=materials,columns=drugs)
 cm=LinearSegmentedColormap.from_list('nature_new_water',[CORAL,'white',BLUE]);cm.set_bad('white')
 lim=max(10,float(np.ceil(np.nanmax(np.abs(v.to_numpy()))/5)*5))
 im=ax.imshow(v,cmap=cm,vmin=-lim,vmax=lim,aspect='auto')
 for i in range(5):
  for j in range(3):
   x=v.iloc[i,j]
   s='—' if pd.isna(x) else f'{x:+.1f}\n({int(n.iloc[i,j])})'
   ax.text(j,i,s,ha='center',va='center',fontsize=13,color=INK)
 ax.set_xticks(range(3),drugs)
 ax.set_yticks(range(5),['NaOH-SCW','Pristine SCW','AMCB','CB','MCB'])
 axis(ax,'Emerging contaminant',ticks=13)
 cb=colorbar(ax,im,'Water-matrix change (pp)',[-lim,0,lim]);cb.set_label('Water-matrix change (pp)',fontsize=14.5);cb.ax.tick_params(labelsize=13.5)
 matrix_dividers(ax,5,3)

def temperature_pollutants(ax):
 q=TEMP.sort_values('mean_pp');rng=np.random.default_rng(1609)
 ax.axvline(0,color=INK,lw=.7,ls=':',zorder=0)
 for i,r in enumerate(q.itertuples()):
  vals=Q[(Q.feature=='Adsorption temperature')&(Q.Pollutant==r.Pollutant)].removal_change_pp.to_numpy()
  ax.scatter(vals,i+rng.uniform(-.12,.12,len(vals)),s=22,c=BLUE,edgecolors=BLUE,lw=.4,alpha=.65,zorder=2,label='Contrasts' if i==0 else None)
  ax.scatter(r.mean_pp,i,s=60,c=PURPLE,marker='D',edgecolors=INK,lw=.6,zorder=3,label='Mean' if i==0 else None)
 ax.set_yticks(range(len(q)),[f'{r.Pollutant} (n = {r.n})' for r in q.itertuples()])
 ax.set_ylim(len(q)-.3,-.7);ax.set_xlim(-15,23);ax.set_xticks([-10,0,10,20])
 assert Q.loc[Q.feature=='Adsorption temperature','removal_change_pp'].between(-15,23).all()
 axis(ax,'Observed 15 → 35 °C change (pp)',ticks=12.5)
 legend(ax,ncol=2,fontsize=12.5)

def material_water_interaction(ax):
 q=INTER.sort_values(['pair_label','Pollutant']);rng=np.random.default_rng(1610)
 ax.axvline(0,color=INK,lw=.7,ls=':',zorder=0)
 for i,r in enumerate(q.itertuples()):
  vals=BLOCK[(BLOCK.pair==r.pair)&(BLOCK.Pollutant==r.Pollutant)].interaction_pp.to_numpy()
  ax.plot([r.minimum,r.maximum],[i,i],color=TEAL,lw=1.8,zorder=1)
  ax.scatter(vals,i+rng.uniform(-.10,.10,len(vals)),s=30,c=TEAL,edgecolors=TEAL,lw=.5,zorder=2,label='Contexts' if i==0 else None)
  ax.scatter(r.mean_pp,i,s=65,c=PURPLE,marker='D',edgecolors=INK,lw=.6,zorder=3,label='Mean' if i==0 else None)
 ax.set_yticks(range(len(q)),[f'{r.pair_label} · {r.Pollutant}\n(n = {r.n})' for r in q.itertuples()])
 ax.set_ylim(len(q)-.35,-.7);ax.set_xlim(-17,2);ax.set_xticks([-15,-10,-5,0])
 axis(ax,'Advantage change (pp)',ticks=13)
 legend(ax,ncol=2,fontsize=12.5)

def compact_water_panel(ax):
 water_panel(ax)
 ax.set_xlabel('Water change (pp)',fontsize=15.5)
 ax.set_yticklabels([t.get_text().replace(' (n = ',' (').replace('Pristine SCW','SCW') for t in ax.get_yticklabels()])
 ax._wide_legend=True
 ax.get_legend().set_loc('lower right');ax.get_legend().set_bbox_to_anchor((1,1.025))

builders=[effects('Solution pH','Solution pH (25 °C)'),effects('Adsorption temperature','Temperature (°C; pH 7)'),compact_water_panel,
 matched('Solution pH','pH 7 → 11'),matched('Adsorption temperature','15 → 35 °C'),water_heatmap,
 pollutant_ph_panel,temperature_pollutants,material_water_interaction]
def readable(builder):
 def draw(ax):
  builder(ax)
  ax.set_xlabel(ax.get_xlabel().replace('Observed ',''))
  ax.tick_params(axis='both',labelsize=14.5)
  ax.xaxis.label.set_fontsize(16);ax.yaxis.label.set_fontsize(16)
  ax.title.set_fontsize(14.5)
  if ax.get_legend():
   for t in ax.get_legend().get_texts():t.set_fontsize(13 if getattr(ax,'_wide_legend',False) else 14)
  for t in ax.texts:t.set_fontsize(14.5)
 return draw
builders=[readable(b) for b in builders]
render_figure(F,'Fig07_adsorption_responses',builders,3,3,(12.6,11.9),{'effect_curves':C[C.feature!='Adsorption time'],'matched_observations':Q[Q.feature!='Adsorption time'],'source_means':S[S.feature!='Adsorption time'],'pollutant_ph':PH,'material_water':WM,'pollutant_temperature':TEMP,'pollutant_water':WATER,'water_interaction':INTER,'water_contexts':BLOCK},panel_letters=True)
audit=json.loads((F/'layout_audit.json').read_text());audit.update(axis_label_pt=16,tick_label_pt=14.5,colorbar_tick_pt=13.5,canvas_at_word_width_inches=6.1,axis_label_at_word_width_pt=16*6.1/12.6,tick_label_at_word_width_pt=14.5*6.1/12.6)
(F/'layout_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
# Metadata on both composite and individually redrawn vector deliverables.
import fitz
for p in F.rglob('*.pdf'):
 with fitz.open(p) as d:
  meta=d.metadata;meta.update(author='CHONG LIU',creator='CHONG LIU');d.set_metadata(meta);d.saveIncr()
print('Nine-panel figure complete',flush=True)
