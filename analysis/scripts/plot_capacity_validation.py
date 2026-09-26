"""Same-target validation reconstruction; raw capacity units, never r or pp."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
import plot_full24_revision as base
from compact_figure_layout import axis,legend,INK,BLUE,CORAL,TEAL,PURPLE,GOLD
E=base.ROOT/'evidence/original_capacity_validation_v2'
M=pd.read_csv(E/'metrics.csv');P=pd.read_csv(E/'predictions.csv')
TRACKS=['Row random','Condition group','Operating group','Context group','Source holdout']
LABELS=['Row\nrandom','Full\ncondition','Operating\ncondition','Context','Source']
COLS=[BLUE,TEAL,GOLD,PURPLE,CORAL]

def metrics(col,label,reported):
 def draw(ax):
  for j,t in enumerate(TRACKS):
   v=M.loc[M.track==t,col].to_numpy()
   ax.scatter(j+np.linspace(-.12,.12,len(v)),v,c=COLS[j],s=38,ec=INK,lw=.4,zorder=3)
   ax.plot([j-.22,j+.22],[np.median(v)]*2,c=COLS[j],lw=2.7)
  ax.axhline(reported,c='#C7A3CC',lw=1.2,ls='--',label='Published random-split score')
  ax.set_xticks(range(5),LABELS);axis(ax,y=label)
  if col=='r2':
   ax.set_yscale('symlog',linthresh=1);ax.set_ylim(-120,1.12);ax.set_yticks([-100,-10,-1,0,1],['−100','−10','−1','0','1'])
  ax.tick_params(axis='x',labelsize=10);legend(ax,fontsize=10,ncol=1)
 return draw

def overlaps(ax):
 keys=['condition_overlap_pct','operating_overlap_pct','context_overlap_pct','material_overlap_pct','source_overlap_pct']
 a=M.groupby('track')[keys].median().reindex(TRACKS).to_numpy()
 from matplotlib.colors import LinearSegmentedColormap
 im=ax.imshow(a,cmap=LinearSegmentedColormap.from_list('nature_new',[COLS[0],COLS[4]]),vmin=0,vmax=100,aspect='auto')
 ax.set_xticks(range(5),['Full\ncondition','Operating\nblock','Context','Material','Source'],fontsize=10)
 ax.set_yticks(range(5),['Row random','Full condition','Operating condition','Context','Source'],fontsize=11)
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{a[i,j]:.0f}',ha='center',va='center',color=INK,fontsize=11)
 ax.set_xlabel('Test rows with a training counterpart (%)',fontsize=12)
 ax.tick_params(length=0)

def parity(track,pooled=False):
 def draw(ax):
  g=P[P.track==track]
  if not pooled:g=g[g.seed==11]
  x=g.observed.to_numpy();y=g.prediction.to_numpy();idx=TRACKS.index(track)
  ax.scatter(x,y,c=COLS[idx],s=12,alpha=.45,ec='none',rasterized=True)
  lo=min(0,x.min(),y.min());hi=max(x.max(),y.max());pad=(hi-lo)*.04
  ax.plot([lo,hi],[lo,hi],c=INK,lw=.8,ls='--');ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
  axis(ax,x='Observed q (mg g$^{-1}$)',y='Predicted q (mg g$^{-1}$)')
  ax.text(.04,.96,f'{track}'+(' · pooled' if pooled else ' · seed 11')+f'\nR² = {r2_score(x,y):.3f}\nMAE = {mean_absolute_error(x,y):.2f} mg/g\nn = {len(g):,}',transform=ax.transAxes,va='top',fontsize=11)
 return draw

if __name__=='__main__':
 base.render('FigS06','capacity_validation_reconstruction',
 [metrics('r2','Test R²',.9433),metrics('mae','Capacity MAE (mg g$^{-1}$)',4.95),overlaps,
 parity('Row random'),parity('Context group'),parity('Source holdout',True)],3,2,(12.1,14.4),{'metrics':M,'predictions':P})

