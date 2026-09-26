"""Six evidence panels; native path-dependent SHAP, not mechanistic effects."""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import plot_full24_revision as base
from compact_figure_layout import axis,legend,INK,BLUE,CORAL,TEAL,GOLD,PURPLE
from paired_shap import OUT,VARIANTS,GROUPS,group,D,P
I=pd.read_csv(OUT/'feature_importance.csv')
G=pd.read_csv(OUT/'group_importance.csv')
Q=pd.read_csv(OUT/'pair_group_contributions.csv')
COL=dict(zip(GROUPS,[BLUE,TEAL,PURPLE,CORAL]))
LABEL={'Pyrolysis temperature':'Pyrolysis temperature','O/C':'O/C','Surface area':'BET surface area',
       'Pore volume':'Pore volume','Average pore size':'Pore size','log_time':'Recorded time',
       'Solution pH':'pH','RPM':'Agitation speed','Adsorption temperature':'Adsorption temperature',
       'Pollutant':'Pollutant','Wastewater type':'Water type','Adsorption type':'Adsorption type'}
ORDER=I[I.variant==VARIANTS[0]].groupby('feature').contrast_mean_abs_pp.median().sort_values(ascending=False).index.tolist()

def importance(metric,label):
 def draw(ax):
    t=I[I.variant==VARIANTS[0]].groupby('feature')[metric].agg(['min','median','max']).reindex(ORDER)
    for j,f in enumerate(ORDER):
      r=t.loc[f];ax.barh(j,r['median'],color=COL[group(f)],edgecolor=INK,lw=.45,height=.70)
      ax.errorbar(r['median'],j,xerr=[[r['median']-r['min']],[r['max']-r['median']]],fmt='none',ecolor=INK,capsize=2,lw=.9)
    ax.set_yticks(range(len(ORDER)),[LABEL[f] for f in ORDER]);ax.set_ylim(len(ORDER)-.45,-.65)
    ax.set_xlim(left=0);axis(ax,x=label,ticks=11)
 return draw

def swarm(ax):
 z=np.load(OUT/'s11_Descriptors.npz');features=list(z['features']);v=z['phi_advantage_pp']
 rng=np.random.default_rng(19)
 for j,f in enumerate(ORDER):
    x=v[:,features.index(f)];bins=np.floor((x-x.min())/(np.ptp(x)+1e-9)*38).astype(int);y=np.zeros(len(x))
    for b in np.unique(bins):
      ix=np.flatnonzero(bins==b);rng.shuffle(ix);offset=np.arange(len(ix));offset=(offset//2+1)*np.where(offset%2==0,1,-1);offset[0]=0;y[ix]=offset
    y=.32*y/max(1,max(abs(y)))
    ax.scatter(x,j+y,s=9,c=COL[group(f)],alpha=.70,ec=INK,lw=.12,rasterized=True)
 ax.axvline(0,c=INK,lw=.7,ls='--');ax.set_yticks(range(len(ORDER)),[LABEL[f] for f in ORDER]);ax.set_ylim(len(ORDER)-.45,-.65)
 axis(ax,x='Contribution to predicted\nadvantage (pp)',ticks=11)

def pairs(ax):
 # One reference partition preserves the exact additive pair-mean identity.
 q=Q[(Q.variant==VARIANTS[0])&(Q.seed==11)].pivot(index='pair_id',columns='group',values='signed_mean_pp').reindex(base.PAIR)
 neg=np.zeros(8);pos=np.zeros(8)
 for g in GROUPS[:3]:
    v=q[g].values;left=np.where(v>=0,pos,neg);ax.barh(range(8),v,left=left,height=.62,color=COL[g],ec=INK,lw=.4,label=g)
    pos+=np.maximum(v,0);neg+=np.minimum(v,0)
 ax.scatter(q.sum(axis=1),range(8),marker='D',c=GOLD,s=38,ec=INK,lw=.65,zorder=4,label='Predicted total')
 ax.axvline(0,c=INK,lw=.7);ax.set_yticks(range(8),['P'+str(i+1) for i in range(8)]);ax.set_ylim(7.6,-.7)
 axis(ax,x='Mean predicted advantage (pp)',ticks=11)
 handles=[Patch(facecolor=COL[g],edgecolor=INK,label=l) for g,l in zip(GROUPS[:3],['Material','Operations','Context'])]+[Line2D([],[],color=GOLD,marker='D',ls='',mec=INK,label='Total')]
 legend(ax,ncol=4,fontsize=9,handles=handles,columnspacing=.7)

def groups(ax):
 for k,v in enumerate(VARIANTS):
    t=G[G.variant==v].set_index('group')
    for j,g in enumerate(GROUPS):
      if k==0 and g=='Loading variables':continue
      vals=t.loc[g,'contrast_mean_abs_pp'].values;y=j+(k-.5)*.22
      ax.plot([min(vals),max(vals)],[y,y],c=COL[g],lw=1.5)
      ax.scatter(np.median(vals),y,c=COL[g],marker=['o','s'][k],s=48,ec=INK,lw=.6)
 ax.set_yticks(range(4),['Material profile','Operating conditions','Chemical context','Loading variables']);ax.set_ylim(3.65,-.6);ax.set_xlim(left=-1)
 axis(ax,x='Mean absolute group\ncontribution (pp)',ticks=11)
 # The markers encode input variants; both remain in the selected palette.
 h=[Line2D([],[],color=BLUE,marker=m,ls='',mec=INK,label=l) for m,l in zip(['o','s'],['12 inputs','14 inputs'])]
 legend(ax,ncol=2,fontsize=10,handles=h)

def support(ax):
 joint=pd.read_csv(OUT/'ph_temperature_joint_support.csv',index_col=0)
 base.heat(ax,joint,'Observed conditions',fmt='.0f',count=True)
 for i in range(len(joint)):
  for j in range(len(joint.columns)):
   if joint.iloc[i,j]==0:ax.text(j,i,'0',ha='center',va='center',fontsize=12,color=INK)
 axis(ax,x='Adsorption temperature (°C)',y='pH',ticks=11)

if __name__=='__main__':
 base.render('FigS05','paired_SHAP_and_support',
  [importance('point_mean_abs_pp','Mean absolute removal\nSHAP (pp)'),
   importance('contrast_mean_abs_pp','Mean absolute advantage\ncontribution (pp)'),swarm,pairs,groups,support],
  3,2,(12.8,15.6),{'feature_importance':I,'group_importance':G,'pair_contributions':Q},
  standalone_sizes={0:(6.4,5.3),1:(6.4,5.3),2:(6.4,5.3),3:(6.4,5.3),4:(6.4,5.3),5:(6.4,5.3)})
