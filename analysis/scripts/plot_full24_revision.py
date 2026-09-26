"""Unified 24-method figure portfolio; cached held-out predictions only."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from compact_figure_layout import *
from model24_registry import METHODS

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/model24_topsis_v1';V=ROOT/'evidence/full24_v5';T=ROOT/'evidence/supplement_v1';F=ROOT/'figures/revision_full24_20260912'
M=pd.read_csv(E/'metrics.csv');Y=pd.read_csv(E/'predictions.csv');Q=pd.read_csv(E/'pairs.csv');D=pd.read_csv(E/'analysis_conditions.csv');P=pd.read_csv(E/'fixed_pair_queries.csv');I=pd.read_csv(E/'method_inner_scores.csv')
ORDER=list(METHODS);TR=['exact','broad','source'];TL=['Condition','Context','Source'];TC=[BLUE,CORAL,PURPLE]
SC={'History':GOLD,'Point':BLUE,'Contrast':PURPLE,'TOPSIS':TEAL,'Constant':GREEN,'Operations':LILAC}
LAB={'HistGradientBoosting':'Hist. gradient boosting','GradientBoosting':'Gradient boosting','RandomForest':'Random forest','BayesianRidge':'Bayesian ridge','DecisionTree':'Decision tree','KernelRidge':'Kernel ridge','SplineRidge':'Spline ridge'}
PAIR=['PB600 / PB800','GCRB / GCRB-N','PSB / PSBOX-A','Alkali-modified SCG biochars / Pristine SCG biochar','NaOH-activated SCW biochars / Pristine SCW Biochar','AMCB / CB','AMCB / MCB','CB / MCB']
SHORT=['PB600 / PB800','GCRB / GCRB-N','PSB / PSBOX-A','Alkali-SCG / SCG','NaOH-SCW / SCW','AMCB / CB','AMCB / MCB','CB / MCB']
def src(s):return str(s).replace('SI_ref_','S')
def read(f):return pd.read_csv(V/f)
def heat(ax,d,label,fmt='.2f',count=False,vmin=0,vmax=None):
 v=d.to_numpy(float);vmax=max(float(np.nanmax(v)),1) if vmax is None else vmax
 im=ax.imshow(np.ma.masked_equal(v,0) if count else v,aspect='auto',cmap=palette_sequential(BLUE),norm=LogNorm(1,vmax) if count else None,vmin=None if count else vmin,vmax=None if count else vmax)
 for i in range(len(d)):
  for j in range(len(d.columns)):
   if np.isfinite(v[i,j]) and (not count or v[i,j]>0):ax.text(j,i,format(v[i,j],fmt),ha='center',va='center',fontsize=11 if len(d)<15 else 10,color=cell_ink(im.cmap(im.norm(v[i,j]))))
 ax.set_xticks(range(len(d.columns)),d.columns);ax.set_yticks(range(len(d)),d.index);axis(ax,ticks=11.5);matrix_dividers(ax,len(d),len(d.columns));colorbar(ax,im,label,[1,10,100] if count else [vmin,(vmin+vmax)/2,vmax])
def bars(metric,strategies=('History','Point','Contrast','TOPSIS'),tracks=('exact','broad'),scale=1,label=None):
 def draw(ax):
  n=len(strategies);width=.8/n
  for i,s in enumerate(strategies):
   for j,t in enumerate(tracks):
    v=M[(M.strategy==s)&(M.track==t)][metric].dropna().values*scale;x=j+(i-(n-1)/2)*width
    if not len(v):continue
    ax.bar(x,np.median(v),width=width*.88,color=SC[s],ec=INK,lw=.5,label=s if j==0 else None)
    ax.scatter(x+np.linspace(-width*.18,width*.18,len(v)),v,s=23,color=SC[s],ec=INK,lw=.5,zorder=3)
  ax.set_xticks(range(len(tracks)),[TL[TR.index(t)] for t in tracks]);ax.set_ylim(bottom=0);axis(ax,y=label or metric);ax.yaxis.set_major_locator(MaxNLocator(4));legend(ax,ncol=len(strategies),fontsize=10,handlelength=.65,columnspacing=.6)
 return draw
def methodpanel(metric,label):
 def draw(ax):
  for k,t in enumerate(TR):
   g=M[(M.track==t)&M.strategy.isin(ORDER)].groupby('strategy')[metric].agg(['min','median','max']).reindex(ORDER);y=np.arange(24)+(k-1)*.21
   ax.hlines(y,100*g['min'],100*g['max'],color=TC[k],lw=1.1);ax.scatter(100*g['median'],y,c=TC[k],marker=['o','s','^'][k],s=22,ec=INK,lw=.35,label=TL[k].replace('\n',' '))
  ax.set_yticks(range(24),[LAB.get(m,m) for m in ORDER]);ax.set_ylim(23.7,-.7);axis(ax,label,ticks=11.5);ax.xaxis.set_major_locator(MaxNLocator(4));legend(ax,ncol=3,fontsize=9.5,handlelength=.6,columnspacing=.45)
  if metric in ['point_mae','contrast_mae','pooled_rmse','selection_loss']:ax.set_xlim(left=0)
 return draw
def render(n,stem,builders,rows,cols,size,tables,**kwargs):
 def enlarged(builder):
  def draw(ax):
   builder(ax)
   wraps={'Test conditions with training support (%)':'Test conditions with\ntraining support (%)','TOPSIS–hindsight point MAE gap (pp)':'TOPSIS–hindsight\npoint MAE gap (pp)','Predicted remaining fraction':'Predicted remaining\nfraction (r)','Prediction minus observation (pp)':'Prediction minus\nobservation (pp)','Paired mean squared error (pp²)':'Paired mean\nsquared error','Observed advantage at wrong choices (pp)':'Observed advantage at\nwrong choices (pp)','Inner–outer rank agreement (ρ)':'Inner–outer rank\nagreement (ρ)'}
   wraps['Selection loss (pp; log scale)']='Selection loss\n(pp; log scale)'
   for label in [ax.xaxis.label,ax.yaxis.label]:
    if label.get_text() in wraps:label.set_text(wraps[label.get_text()])
   for txt in ax.findobj(match=Text):
    txt.set_text(txt.get_text().replace('History','Pair mean'))
    txt.set_fontsize(max(13.5,txt.get_fontsize()*1.30))
   if ax.get_legend() is not None:
    ax.get_legend().set_loc('lower right');ax.get_legend().set_bbox_to_anchor((1,1.025))
   for collection in ax.collections:
    if hasattr(collection,'get_sizes') and len(collection.get_sizes()):collection.set_sizes(collection.get_sizes()*1.45)
  return draw
 render_figure(F/n,n+'_'+stem,[enlarged(b) for b in builders],rows,cols,size,tables,panel_letters=True,**kwargs)
 auditfile=F/n/'layout_audit.json';audit=json.loads(auditfile.read_text());audit.update(font_scale_post_builder=1.30,minimum_text_pt=13.5,axis_label_pt=20.15,tick_label_pt='minimum 13.5; role-dependent');auditfile.write_text(json.dumps(audit,indent=2))

def fig1():
 a=pd.crosstab(D.Pollutant,D.source_candidate).rename(columns=src);b=pd.crosstab(D['Wastewater type'],D.source_candidate).rename(columns=src);b.index=[x.replace('Secondary effluent','Secondary\neffluent').replace('Ground water','Ground\nwater') for x in b.index]
 def flow(ax):
  vals=[3757,1348,1297,1279];labels=['Original records','Condition means','Biochar conditions','Positive-time conditions']
  ax.barh(range(4),vals,color=[BLUE,PURPLE,CORAL,TEAL],ec=INK,lw=.6)
  for j,v in enumerate(vals):ax.text(v+50,j,f'{v:,}',va='center',fontsize=13)
  ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,4350);axis(ax,'Count');ax.set_xticks([0,2000,4000])
 def pairs(ax):
  g=P.pair_id.value_counts().reindex(PAIR);ax.barh(range(8),g,color=TEAL,ec=INK,lw=.5)
  for j,v in enumerate(g):ax.text(v+2,j,str(v),va='center',fontsize=12)
  ax.set_yticks(range(8),[f'P{i+1}  {p}' for i,p in enumerate(SHORT)]);ax.invert_yaxis();ax.set_xlim(0,165);axis(ax,'Matched queries',ticks=11.5);ax.set_xticks([0,50,100,150])
 render('Fig01','data_and_comparisons',[flow,lambda ax:heat(ax,a,'Positive-time conditions',fmt='.0f',count=True),lambda ax:heat(ax,b,'Positive-time conditions',fmt='.0f',count=True),pairs],2,2,(11.4,10.2),{'conditions':D,'fixed_pairs':P,'pollutants':a,'water':b})
def fig2():
 render('Fig02','24_method_benchmark',[methodpanel('point_mae','Point MAE (pp)'),methodpanel('contrast_mae','Contrast MAE (pp)'),bars('point_mae',('Constant','Operations','Point','TOPSIS'),tuple(TR),scale=100,label='Point MAE (pp)'),bars('contrast_mae',tracks=tuple(TR),scale=100,label='Contrast MAE (pp)')],2,2,(11.4,12.1),{'metrics_all_seeds':M},row_ratios=[1.8,1],standalone_sizes={0:(6.0,8),1:(6.0,8)})
def fig4():
 def trade(ax,x,y,xl,yl,sc=1):
  for k,t in enumerate(TR):
   g=M[(M.track==t)&M.strategy.isin(ORDER)].groupby('strategy')[[x,y]].median();ax.scatter(100*g[x],g[y]*sc,c=TC[k],s=40,marker=['o','s','^'][k],ec=INK,lw=.45,label=TL[k].replace('\n',' '))
  axis(ax,xl,yl);ax.set_xlim(left=0)
  if y=='selection_loss':ax.set_yscale('log');ax.set_ylim(.006,30)
  else:ax.set_ylim(bottom=0)
  legend(ax,ncol=3,fontsize=9.5,handlelength=.6,columnspacing=.4)
 decomp=read('paired_error_decomposition.csv')
 def decompplot(ax):
  labs=[]
  for j,(t,s) in enumerate([(t,s) for t in ['exact','broad'] for s in ['Point','Contrast','TOPSIS']]):
   g=decomp[(decomp.seed==11)&(decomp.track==t)&(decomp.strategy==s)].iloc[0];ax.barh(j,10000*g.common_mse,color=BLUE,ec=INK,lw=.5,label='Common' if j==0 else None);ax.barh(j,10000*g.differential_mse,left=10000*g.common_mse,color=CORAL,ec=INK,lw=.5,label='Differential' if j==0 else None);labs.append(('Block: ' if t=='exact' else 'Context: ')+s)
  ax.set_yticks(range(6),labs);axis(ax,x='Paired mean squared error (pp²)',ticks=10.5);legend(ax,ncol=2);ax.invert_yaxis();ax.set_xlim(left=0);ax.xaxis.set_major_locator(MaxNLocator(4))
 def margins(ax):
  z=read('wrong_choice_details.csv');z=z[z.track.isin(['exact','broad'])&(z.seed==11)]
  for j,s in enumerate(['History','Point','Contrast','TOPSIS']):
   g=z[z.strategy==s];v=g.observed_pp.abs();ax.scatter(v,np.full(len(v),j)+np.linspace(-.14,.14,len(v)),c=SC[s],s=35,ec=INK,lw=.4)
  ax.set_yticks(range(4),['History','Point','Contrast','TOPSIS']);ax.invert_yaxis();axis(ax,'Observed advantage at wrong choices (pp)');ax.set_xlim(left=0)
 def lossplot(ax):
  for k,s in enumerate(['History','Point','Contrast','TOPSIS']):
   for j,t in enumerate(['exact','broad']):
    v=M[(M.track==t)&(M.strategy==s)].selection_loss*100;x=j+(k-1.5)*.19;ax.plot([x,x],[v.min(),v.max()],c=SC[s],lw=1.3);ax.scatter(x+np.linspace(-.035,.035,len(v)),v,c=SC[s],s=40,ec=INK,lw=.4,label=s if j==0 else None);ax.plot([x-.065,x+.065],[v.median()]*2,c=SC[s],lw=2)
  ax.set_xticks([0,1],TL[:2]);ax.set_yscale('log');ax.set_ylim(.007,.5);axis(ax,y='Selection loss (pp; log scale)');legend(ax,ncol=4,fontsize=10,handlelength=.6,columnspacing=.5)
 render('Fig04','prediction_and_selection',[lambda ax:trade(ax,'point_mae','contrast_mae','Point MAE (pp)','Contrast MAE (pp)',100),lambda ax:trade(ax,'contrast_mae','selection_loss','Contrast MAE (pp)','Selection loss (pp; log scale)',100),lossplot,bars('wrong',label='Wrong choices / 559'),decompplot,margins],3,2,(11.4,12.9),{'metrics':M,'decomposition':decomp,'wrong_choices':read('wrong_choice_details.csv'),'rank_correlations':read('metric_rank_correlations.csv')})
def fig5():
 w=read('outer_weight_and_ablation_metrics.csv');c=read('inner_criteria_correlations.csv');diag=read('fold_selection_diagnostics.csv')
 def weights(ax):
  for k,t in enumerate(TR):
   g=w[(w.track==t)&(w.seed==11)&w.rule.str.startswith('W')];ax.scatter(100*g.contrast_mae,g.selection_loss*100,c=TC[k],s=44,ec=INK,lw=.45,label=TL[k].replace('\n',' '))
   h=w[(w.track==t)&(w.seed==11)&(w.rule=='Full')].iloc[0];ax.scatter(100*h.contrast_mae,h.selection_loss*100,c=TC[k],marker='*',s=150,ec=INK,lw=.8,zorder=5)
  axis(ax,'Contrast MAE (pp)','Selection loss (pp)');ax.xaxis.set_major_locator(MaxNLocator(4));legend(ax,ncol=3,fontsize=9.5,handlelength=.6,columnspacing=.4)
 def retention(ax):
  for j,(f,col,lab) in enumerate([('weight_stability.csv',TEAL,'Weight grid'),('removal_stability.csv',GOLD,'Non-winner removal')]):
   g=pd.read_csv(E/f).set_index('track').reindex(TR)['mean']*100;x=np.arange(3)+(j-.5)*.32;ax.bar(x,g,width=.28,color=col,ec=INK,lw=.5,label=lab)
   for xx,yy in zip(x,g):ax.text(xx,yy+2,f'{yy:.1f}',ha='center',fontsize=10)
  ax.set_xticks(range(3),TL);ax.set_ylim(0,115);axis(ax,y='Unchanged winner (%)');legend(ax,ncol=2,fontsize=10.5)
 def ablation(ax):
  rules=['Full','No_point','No_contrast','No_loss'];lab=['All three','No point','No contrast','No loss']
  for k,t in enumerate(TR):
   g=w[(w.track==t)&w.rule.isin(rules)].groupby('rule').contrast_mae.agg(['min','median','max']).reindex(rules);x=np.arange(4)+(k-1)*.18;ax.errorbar(x,100*g['median'],yerr=[100*(g['median']-g['min']),100*(g['max']-g['median'])],fmt=['o','s','^'][k],c=TC[k],ms=5,capsize=2,label=TL[k].replace('\n',' '))
  ax.set_xticks(range(4),lab);axis(ax,y='Contrast MAE (pp)',ticks=10.5);legend(ax,ncol=3,fontsize=9.5,handlelength=.6,columnspacing=.4)
 def corr(ax):
  # Correlations among candidate criteria, median across all 51 training matrices.
  cols=[x for x in c.columns];m=c.groupby(['a','b']).rho.median().unstack().reindex(index=['point','contrast','loss'],columns=['point','contrast','loss']);m.index=['Point','Contrast','Loss'];m.columns=m.index;heat(ax,m,'Median inner Spearman ρ',vmin=-1,vmax=1)
 def selected(ax):
  order=[m for m in ORDER if m in diag.winner.values]
  for k,t in enumerate(TR):
   g=diag[diag.track==t].winner.value_counts().reindex(order,fill_value=0);v=100*g/(15 if t!='source' else 21);ax.barh(np.arange(len(order))+(k-1)*.22,v,height=.20,color=TC[k],ec=INK,lw=.4,label=TL[k].replace('\n',' '))
  ax.set_yticks(range(len(order)),[LAB.get(m,m) for m in order]);ax.invert_yaxis();axis(ax,'Selected outer fits (%)',ticks=10.5);legend(ax,ncol=3,fontsize=9.5,handlelength=.6,columnspacing=.4)
 def frontier(ax):
  for k,t in enumerate(TR):
   v=diag[diag.track==t].pareto_methods;ax.scatter(np.full(len(v),k)+np.linspace(-.16,.16,len(v)),v,c=TC[k],s=35,ec=INK,lw=.4);ax.plot([k-.22,k+.22],[v.median()]*2,c=TC[k],lw=2.5)
  ax.set_xticks(range(3),TL);axis(ax,y='Nondominated inner methods');ax.set_ylim(0,12);ax.set_yticks([0,4,8,12])
 render('Fig05','TOPSIS_robustness',[corr,selected,weights,retention,ablation,frontier],3,2,(11.4,13.1),{'weights_and_ablations':w,'inner_criteria':c,'fold_diagnostics':diag})
def fig6():
 a=read('material_pair_summary.csv');b=read('conditional_block_resampling.csv')
 def advantage(ax):
  for j,p in enumerate(PAIR):
   g=a[(a.track=='broad')&(a.strategy=='TOPSIS')&(a.pair==p)];obs=g.observed_pp.iloc[0];v=g.predicted_pp;ax.plot([obs,v.median()],[j,j],c=INK,lw=.6);ax.scatter(obs,j,c=GOLD,marker='s',s=45,ec=INK,lw=.45,label='Observed' if j==0 else None);ax.hlines(j+.16,v.min(),v.max(),color=TEAL,lw=1.6);ax.scatter(v.median(),j+.16,c=TEAL,s=40,ec=INK,lw=.45,label='TOPSIS' if j==0 else None)
  ax.axvline(0,c=INK,ls='--',lw=.7);ax.set_yticks(range(8),[f'P{j+1}' for j in range(8)]);ax.invert_yaxis();axis(ax,'Removal advantage of material a (pp)');ax.set_xlim(-95,95);ax.set_xticks([-80,-40,0,40,80]);legend(ax)
 def errors(ax):
  for k,s in enumerate(['History','Point','TOPSIS']):
   g=a[(a.track=='broad')&(a.strategy==s)].groupby('pair').error_pp.agg(['min','median','max']).reindex(PAIR);y=np.arange(8)+(k-1)*.20;ax.hlines(y,g['min'],g['max'],color=SC[s],lw=1.2);ax.scatter(g['median'],y,c=SC[s],s=32,ec=INK,lw=.4,label=s)
  ax.set_yticks(range(8),[f'P{j+1}' for j in range(8)]);ax.invert_yaxis();axis(ax,'Pair contrast MAE (pp)');ax.set_xlim(left=0);legend(ax,ncol=3)
 def boot(ax,metric,label):
  j=0
  for t in ['exact','broad']:
   for s in ['Point','Contrast','TOPSIS']:
    r=b[(b.track==t)&(b.strategy==s)&(b.metric==metric)].iloc[0];ax.plot([100*r.lower,100*r.upper],[j,j],c=SC[s],lw=1.7);ax.scatter(100*r.estimate,j,c=SC[s],s=40,ec=INK,lw=.45);j+=1
  ax.axvline(0,c=INK,ls='--',lw=.8);ax.set_yticks(range(6),['Block: Point','Block: Contrast','Block: TOPSIS','Context: Point','Context: Contrast','Context: TOPSIS']);ax.invert_yaxis();axis(ax,label,ticks=11)
 from adsorption_context import descriptor_panel, compound_heatmap_panel, PROF, RESP
 render('Fig06','material_comparisons',[advantage,errors,lambda ax:boot(ax,'error','Contrast MAE minus history (pp)'),lambda ax:boot(ax,'loss','Selection loss minus history (pp)'),descriptor_panel,compound_heatmap_panel],3,2,(11.4,13.4),{'pairs':a,'conditional_resampling':b,'paired_descriptors':PROF,'pollutant_advantages':RESP})
def fig8():
 source=read('source_point_metrics.csv');support=read('condition_support.csv');r=read('inner_outer_rank_agreement.csv');diag=read('fold_selection_diagnostics.csv')
 def sources(ax):
  m=source[(source.seed==11)&(source.track=='source')&source.strategy.isin(['Constant','Point','TOPSIS','SVR','ExtraTrees'])].pivot(index='source',columns='strategy',values='mae').reindex(columns=['Constant','Point','TOPSIS','SVR','ExtraTrees']);m.index=m.index.map(src);m.columns=['Constant','Point','TOPSIS','SVR','Extra\ntrees'];heat(ax,m*100,'Held-source point MAE (pp)',vmax=100)
 def covered(ax):
  for j,(key,col,lab) in enumerate([('profile_known',TEAL,'Material profile'),('pollutant_known',BLUE,'Pollutant'),('water_known',CORAL,'Water type')]):
   v=support.groupby('track')[key].mean().reindex(TR)*100;ax.bar(np.arange(3)+(j-1)*.22,v,width=.20,color=col,ec=INK,lw=.45,label=lab)
  ax.set_xticks(range(3),TL);ax.set_ylim(0,110);axis(ax,y='Test conditions with training support (%)');legend(ax,ncol=3,fontsize=9.5,handlelength=.65,columnspacing=.5)
 def ranks(ax):
  for j,(k,col,lab) in enumerate([('point',BLUE,'Point'),('contrast',CORAL,'Contrast'),('loss',TEAL,'Loss')]):
   for i,t in enumerate(TR):
    v=r[(r.track==t)&(r.criterion==k)].rho.dropna();x=i+(j-1)*.23;ax.scatter(x+np.linspace(-.055,.055,len(v)),v,c=col,s=20,alpha=.7,ec=INK,lw=.25,label=lab if i==0 else None);ax.plot([x-.08,x+.08],[v.median()]*2,c=col,lw=2.5)
  ax.axhline(0,c=INK,lw=.7,ls='--');ax.set_xticks(range(3),TL);ax.set_ylim(-1.1,1.1);axis(ax,y='Inner–outer rank agreement (ρ)');legend(ax,ncol=3)
 def gap(ax):
  for k,t in enumerate(TR):
   v=100*diag[diag.track==t].outer_point_gap;ax.scatter(np.full(len(v),k)+np.linspace(-.16,.16,len(v)),v,c=TC[k],s=37,ec=INK,lw=.4);ax.plot([k-.22,k+.22],[v.median()]*2,c=TC[k],lw=2.5)
  ax.set_xticks(range(3),TL);ax.set_ylim(bottom=0);axis(ax,y='TOPSIS–hindsight point MAE gap (pp)')
 render('Fig08','transfer_limits',[sources,covered,ranks,gap],2,2,(11.4,10.5),{'source_errors':source,'training_support':support,'rank_agreement':r,'selection_diagnostics':diag})
def figs1():
 raw=pd.read_csv(T/'audited_raw_records.csv');flag=pd.read_csv(T/'raw_flag_counts.csv',index_col=0);rat=pd.read_csv(T/'ratio_definition_diagnostic.csv');counts=pd.read_csv(T/'unit_counts.csv').set_index('unit')
 def countsplot(ax):
  for k,(s,col) in enumerate([('all',GREEN),('biochar',TEAL)]):
   v=counts[s];ax.barh(np.arange(len(v))+(k-.5)*.30,v,height=.27,color=col,ec=INK,lw=.5,label=s.title())
   for j,n in enumerate(v):ax.text(n+40,j+(k-.5)*.30,str(n),va='center',fontsize=10.5)
  ax.set_yticks(range(3),['Original','Full deduplication','Condition means']);ax.invert_yaxis();ax.set_xlim(0,4400);axis(ax,'Count');legend(ax)
 def ecdf(ax):
  d=raw[(raw.Adsorbent!='PAC')&(raw['Adsorption time']>0)]
  for lab,v,col in [('Original',d.r,GREEN),('Deduplicated',d.loc[~d.complete_duplicate,'r'],PURPLE),('Aggregated',D.r,TEAL)]:
   v=np.sort(v);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',c=col,label=lab)
  axis(ax,'Remaining fraction (r)','Cumulative fraction');legend(ax,ncol=3,fontsize=10);ax.set_ylim(0,1.02)
 def flags(ax):
  m=flag.copy();m.index=m.index.map(src);m.columns=['Time = 0','r > 1','PB600\nN flag'];heat(ax,m,'Flagged original rows',fmt='.0f',vmax=40)
 def ratios(ax):
  for s,col,mark in [('H/C',BLUE,'o'),('O/C',TEAL,'s'),('N/C',CORAL,'^')]:
   g=rat[rat.ratio==s];ax.scatter(g.atomic_calculated,g.recorded,c=col,marker=mark,s=45,ec=INK,lw=.45,label=s)
  upper=rat[['atomic_calculated','recorded']].max().max()*1.06;ax.plot([0,upper],[0,upper],c=INK,ls='--',lw=.7);axis(ax,'Calculated atomic ratio','Recorded ratio');legend(ax,ncol=3)
 render('FigS01','data_checks',[countsplot,ecdf,flags,ratios],2,2,(11.4,10.3),{'counts':counts,'flags':flag,'ratio_checks':rat})
def figs2():
 r=pd.read_csv(T/'full_rank_diagnostic.csv');sets=['Full 14','Non-ratio 10','Core 6','Core 5','Pore 3'];d=D.drop_duplicates(['source_candidate','Adsorbent']+['Pyrolysis temperature','O/C','Surface area','Pore volume','Average pore size'])
 def rank(ax):
  m=r[np.isclose(r.tolerance,1e-10,rtol=1e-6,atol=0)].pivot(index='set',columns='variant',values='rank').reindex(index=sets,columns=['All','Biochar','Flag excluded']);m.columns=['All','Biochar','Flag\nexcluded'];heat(ax,m,'Within-source rank',fmt='.0f',vmax=10)
 def tol(ax):
  m=r[r.variant=='Biochar'].pivot(index='set',columns='tolerance',values='rank').reindex(sets);m.columns=[f'$10^{{{int(np.log10(x))}}}$' for x in m.columns];heat(ax,m,'Within-source rank',fmt='.0f',vmax=10)
 def desc(ax):
  for k,s in enumerate(sorted(d.source_candidate.unique())):
   g=d[d.source_candidate==s];ax.scatter(g['Surface area'],g['O/C'],c=PAL[k],s=55,ec=INK,lw=.5,label=src(s))
  axis(ax,'BET surface area (m$^2$ g$^{-1}$)','Recorded O/C');legend(ax,ncol=4,fontsize=10)
 def prof(ax):
  cols=['Pyrolysis temperature','O/C','Surface area','Pore volume','Average pore size'];m=d.set_index('Adsorbent')[cols];m=(m-m.min())/(m.max()-m.min());m.index=[s.replace('Alkali-modified SCG biochars','Alkali-SCG').replace('NaOH-activated SCW biochars','NaOH-SCW').replace('Pristine SCG biochar','SCG').replace('Pristine SCW Biochar','SCW') for s in m.index];m.columns=['Pyrolysis\nT','O/C','BET','Pore\nvolume','Pore\nsize'];heat(ax,m,'Min–max display scale',fmt='.1f',vmax=1)
 render('FigS02','descriptor_support',[rank,tol,desc,prof],2,2,(11.4,11.3),{'rank_diagnostic':r,'profiles':d})
def fig3():
 def parity(t):
  def draw(ax):
   g=Y[(Y.seed==11)&(Y.track==t)&(Y.strategy=='TOPSIS')];ax.scatter(g.r,g.prediction,c=TC[TR.index(t)],s=12,alpha=.55,ec='none');lo=min(0,g.prediction.min());hi=max(1.1,g.prediction.max());ax.plot([lo,hi],[lo,hi],c=INK,ls='--',lw=.8);axis(ax,'Observed remaining fraction (r)','Predicted remaining fraction');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);m=M[(M.seed==11)&(M.track==t)&(M.strategy=='TOPSIS')].iloc[0];ax.text(.03,.97,TL[TR.index(t)].replace('\n',' ')+f'\nPooled R² = {m.pooled_r2:.3f}',ha='left',va='top',transform=ax.transAxes,fontsize=12)
  return draw
 def residual(t):
  def draw(ax):
   g=Y[(Y.seed==11)&(Y.track==t)&(Y.strategy=='TOPSIS')];ax.scatter(g.prediction,100*(g.prediction-g.r),c=TC[TR.index(t)],s=12,alpha=.55,ec='none');ax.axhline(0,c=INK,ls='--',lw=.8);axis(ax,'Predicted remaining fraction','Prediction minus observation (pp)')
  return draw
 render('Fig03','parity_and_residuals',[f(t) for t in TR for f in [parity,residual]],3,2,(11.4,12.9),{'TOPSIS_predictions':Y[(Y.seed==11)&(Y.strategy=='TOPSIS')]})
def figs3():
 def extra(metric,label):
  def draw(ax):
   m=M[M.strategy.isin(ORDER)].groupby(['strategy','track'])[metric].median().unstack().reindex(index=ORDER,columns=TR);m.index=[LAB.get(x,x) for x in m.index];m.columns=TL
   if metric in ['selection_loss','pooled_rmse']:m*=100
   if metric=='pred_outside_0_1':m*=100/1279
   limits={'pooled_rmse':(0,70),'pooled_r2':(-3,1),'selection_loss':(0,20),'pred_outside_0_1':(0,40)};lo,hi=limits[metric];assert m.min().min()>=lo and m.max().max()<=hi
   heat(ax,m,label,fmt='.3f' if metric=='selection_loss' else '.2f',vmin=lo,vmax=hi)
  return draw
 render('FigS03','complete_model_diagnostics',[extra('pooled_rmse','Pooled RMSE (pp)'),extra('pooled_r2','Pooled R²'),extra('selection_loss','Selection loss (pp)'),extra('pred_outside_0_1','Predictions outside [0, 1] (%)')],2,2,(11.4,16.3),{'complete_metrics':M},standalone_sizes={i:(6,8.2) for i in range(4)})
def figs4():
 agreement=pd.read_csv(ROOT/'evidence/reviewer_validation_v1/query_agreement.csv')
 def drawpair(p,j):
  def draw(ax):
   for s,mark in [('History','s'),('TOPSIS','o')]:
    g=Q[(Q.seed==11)&(Q.track=='broad')&(Q.strategy==s)&(Q.pair_id==p)];ax.scatter(g.observed*100,g.prediction*100,c=SC[s],marker=mark,s=29,alpha=.7,ec=INK,lw=.3,label=s)
   g=Q[(Q.seed==11)&(Q.track=='broad')&(Q.strategy=='TOPSIS')&(Q.pair_id==p)];v=np.r_[g.observed*100,g.prediction*100];lo,hi=v.min(),v.max();pad=max((hi-lo)*.08,.1);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],c=INK,ls='--',lw=.7);axis(ax,'Observed advantage (pp)','Predicted advantage (pp)',ticks=11);ax.set_title(f'P{j+1}: {SHORT[j]}',fontsize=12,pad=37);legend(ax,ncol=2,fontsize=10.5)
   a=agreement[(agreement.seed==11)&(agreement.track=='broad')&(agreement.strategy=='TOPSIS')&(agreement.pair_number==j+1)].iloc[0]
   ax.text(.97,.04,f'R² = {a.query_r2:.2f}\nρ = {a.query_spearman:.2f}; n = {a.n_queries}',ha='right',va='bottom',transform=ax.transAxes,fontsize=10.5)
  return draw
 render('FigS04','all_material_pair_predictions',[drawpair(p,j) for j,p in enumerate(PAIR)],4,2,(11.4,17.1),{'paired_predictions':Q[(Q.seed==11)&(Q.track=='broad')&Q.strategy.isin(['History','TOPSIS'])]})

if __name__ == '__main__':
 import runpy
 runpy.run_path(str(Path(__file__).with_name('plot_panel_integration.py')), run_name='__main__')
