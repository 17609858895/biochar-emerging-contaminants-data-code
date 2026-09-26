"""Compact evidence panels for repeated validation and input-identity controls."""
import numpy as np
import pandas as pd
import plot_full24_revision as base
from compact_figure_layout import axis,legend,INK,BLUE,CORAL,TEAL,PURPLE,GOLD
E=base.ROOT/'evidence/reviewer_validation_v1'
M=pd.read_csv(E/'expanded_rule_metrics_pp.csv')
A=pd.read_csv(E/'input_sensitivity.csv')
P=pd.read_csv(E/'profile_permutation_metrics.csv')
names=['Ridge','RandomForest','LightGBM'];labels=['Ridge','Random\nforest','LightGBM']

def repeated(metric,label):
    def draw(ax):
        hist=M[M.strategy=='History'].set_index(['seed','track'])[metric]
        for k,s in enumerate(['Point','Contrast','TOPSIS']):
            for j,t in enumerate(['exact','broad']):
                g=M[(M.strategy==s)&(M.track==t)].set_index(['seed','track'])
                v=(hist.reindex(g.index)-g[metric]).to_numpy();x=j+(k-1)*.22
                ax.scatter(x+np.linspace(-.035,.035,len(v)),v,s=32,c=base.SC[s],ec=INK,lw=.35,label=s if j==0 else None)
                ax.plot([x-.08,x+.08],[np.median(v)]*2,c=base.SC[s],lw=2.2)
        ax.axhline(0,c=INK,lw=.8,ls='--');ax.set_xticks([0,1],['Operating\ncondition','Context\nholdout']);axis(ax,y=label)
        legend(ax,ncol=3,fontsize=10)
    return draw

def identity(ax):
    for j,v in enumerate(['Descriptors','Material_ID']):
        g=A[(A.track=='broad')&(A.variant==v)].groupby('method').contrast_mae_pp.agg(['min','median','max']).reindex(names)
        x=np.arange(3)+(j-.5)*.22;col=[BLUE,CORAL][j]
        ax.errorbar(x,g['median'],yerr=[g['median']-g['min'],g['max']-g['median']],fmt=['o','s'][j],color=col,ms=6,capsize=3,label=['Descriptors','Material ID'][j])
    ax.set_xticks(range(3),labels);axis(ax,y='Contrast MAE (pp)');legend(ax,ncol=2,fontsize=10)

def permutation(ax):
    for i,t in enumerate(['exact','broad']):
        v=P[P.track==t].contrast_mae_pp.to_numpy()
        ax.scatter(i+np.linspace(-.16,.16,len(v)),v,c=BLUE,s=27,ec=INK,lw=.4,label='Permuted profiles' if i==0 else None)
        ax.plot([i-.19,i+.19],[np.median(v)]*2,c=BLUE,lw=2)
        ref=A[(A.track==t)&(A.seed==11)&(A.method=='LightGBM')&(A.variant=='Descriptors')].contrast_mae_pp.iloc[0]
        ax.scatter(i,ref,c=CORAL,marker='*',s=160,ec=INK,lw=.6,zorder=5,label='Measured profiles' if i==0 else None)
    ax.set_xticks([0,1],['Operating\ncondition','Context\nholdout']);axis(ax,y='Contrast MAE (pp)');legend(ax,ncol=2,fontsize=10)

def inputs(ax):
    reference=A[(A.track=='broad')&(A.variant=='Descriptors')].set_index(['seed','method']).contrast_mae_pp
    for j,v in enumerate(['Source_time','Source_time_concentration_dose']):
        g=A[(A.track=='broad')&(A.variant==v)].set_index(['seed','method'])
        delta=(reference-g.contrast_mae_pp).rename('improvement').reset_index().groupby('method').improvement.agg(['min','median','max']).reindex(names)
        col=[TEAL,PURPLE][j];x=np.arange(3)+(j-.5)*.22
        ax.errorbar(x,delta['median'],yerr=[delta['median']-delta['min'],delta['max']-delta['median']],fmt=['o','s'][j],color=col,ms=6,capsize=3,label=['Time scaling','+ Concentration/dose'][j])
    ax.axhline(0,c=INK,ls='--',lw=.8);ax.set_xticks(range(3),labels);axis(ax,y='Contrast MAE gain (pp)');legend(ax,ncol=2,fontsize=9.5,handlelength=.6,columnspacing=.5)

def exclusion(ax):
    for j,metric in enumerate(['point_mae_pp','contrast_mae_pp']):
        a=A[(A.track=='broad')&(A.variant=='Descriptors')].set_index(['seed','method'])[metric]
        b=A[(A.track=='broad')&(A.variant=='Exclude_raw_r_gt1')].set_index(['seed','method'])[metric]
        d=(b-a).rename('difference').reset_index().groupby('method').difference.agg(['min','median','max']).reindex(names)
        col=[BLUE,GOLD][j];x=np.arange(3)+(j-.5)*.22
        ax.errorbar(x,d['median'],yerr=[d['median']-d['min'],d['max']-d['median']],fmt=['o','s'][j],color=col,ms=6,capsize=3,label=['Point MAE','Contrast MAE'][j])
    ax.axhline(0,c=INK,ls='--',lw=.8);ax.set_xticks(range(3),labels);axis(ax,y='Error change (pp)');legend(ax,ncol=2,fontsize=10)

if __name__=='__main__':
    base.render('FigS04','validation_and_input_controls',
                [repeated('contrast_mae_pp','Contrast MAE gain (pp)'),
                 identity,permutation,inputs,exclusion,
                 repeated('selection_loss_pp','Loss reduction (pp)')],
                3,2,(11.8,14.0),{'repeated_metrics_pp':M,'input_controls_pp':A,'profile_permutations_pp':P})

