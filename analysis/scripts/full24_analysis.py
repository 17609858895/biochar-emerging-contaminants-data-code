"""Reanalyse fixed outer predictions; no new observations or estimator fitting."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from model24_registry import METHODS,CONFIGS,topsis
from run_model24_topsis import D,P,ROOT,hierarchy

OLD=ROOT/'evidence/model24_topsis_v1';OUT=ROOT/'evidence/full24_v5';OUT.mkdir(exist_ok=True)
M=pd.read_csv(OLD/'metrics.csv');Q=pd.read_csv(OLD/'pairs.csv');Y=pd.read_csv(OLD/'predictions.csv');S=pd.read_csv(OLD/'method_inner_scores.csv')
STRATEGIES=['History','Point','Contrast','TOPSIS'];TRACKS=['exact','broad','source']
PAIRS=['PB600 / PB800','GCRB / GCRB-N','PSB / PSBOX-A','Alkali-modified SCG biochars / Pristine SCG biochar','NaOH-activated SCW biochars / Pristine SCW Biochar','AMCB / CB','AMCB / MCB','CB / MCB']
def save(name,x):x.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def pmetric(q):
    q=q[np.isfinite(q.prediction)].copy()
    if len(q)==0:return dict(contrast_mae=np.nan,selection_loss=np.nan,wrong=np.nan,n=0)
    wrong=(q.prediction<0)!=(q.observed<0)
    return dict(contrast_mae=hierarchy(q,abs(q.prediction-q.observed)),selection_loss=hierarchy(q,np.where(wrong,abs(q.observed),0)),wrong=int(wrong.sum()),n=len(q))
def pointmetric(q):return float(q.assign(error=abs(q.prediction-q.r)).groupby('source_candidate').error.mean().mean())

records=[]
for (seed,track),g in M[M.strategy.isin(METHODS)].groupby(['seed','track']):
    for a,b in [('point_mae','contrast_mae'),('point_mae','selection_loss'),('contrast_mae','selection_loss')]:
        records.append(dict(seed=seed,track=track,a=a,b=b,rho=spearmanr(g[a],g[b]).statistic))
save('metric_rank_correlations',pd.DataFrame(records))
fixed=M[M.strategy.isin(METHODS)].copy()
for metric in ['point_mae','contrast_mae','selection_loss']:
    fixed[metric+'_rank']=fixed.groupby(['seed','track'])[metric].rank(method='average')
save('method_metrics_and_ranks',fixed)

records=[]
for (seed,track,strategy,source),g in Y.groupby(['seed','track','strategy','source_candidate']):
    records.append(dict(seed=seed,track=track,strategy=strategy,source=source,n=len(g),mae=abs(g.prediction-g.r).mean(),rmse=np.sqrt(np.mean((g.prediction-g.r)**2)),r2=r2_score(g.r,g.prediction),out_of_bounds=int(((g.prediction<0)|(g.prediction>1)).sum())))
save('source_point_metrics',pd.DataFrame(records))

# Robustness of group-weighted contrasts to the composition of evaluated sources.
records=[]
for (seed,track,strategy),g in Q[(Q.track!='source')&Q.strategy.isin(STRATEGIES)].groupby(['seed','track','strategy']):
    for omit in sorted(g.source.unique()):records.append(dict(seed=seed,track=track,strategy=strategy,omitted=omit,**pmetric(g[g.source!=omit])))
save('source_omission_metrics',pd.DataFrame(records))

# Each weighting/ablation is defined before inspecting its newly assembled outer results.
weight_sets={f'W{i}{j}{10-i-j}':(i/10,j/10,(10-i-j)/10) for i in range(1,9) for j in range(1,10-i)}
weight_sets.update({'Full':(1/3,1/3,1/3),'No_point':(0,.5,.5),'No_contrast':(.5,0,.5),'No_loss':(.5,.5,0)})
meta=[];sensitivity=[];ranks=[];criteria=[];supports=[];selection_rows=[]
for seed in [11,23,47]:
    for track in TRACKS:
        bag={key:[] for key in weight_sets};pairbag={key:[] for key in weight_sets}
        for fd in sorted(OLD.glob(f's{seed}_{track}_*')):
            if not fd.is_dir():continue
            split=pd.read_csv(fd/'split.csv');train=split.inner_fold.ge(0).values;test=split.outer_test.values;te=np.flatnonzero(test);tr=np.flatnonzero(train)
            c=pd.read_csv(fd/'method_scores.csv').sort_values('method').reset_index(drop=True);fold=int(fd.name.rsplit('_',1)[1]);mat=c[['point','contrast','loss']].values
            inner_corr=c[['point','contrast','loss']].corr(method='spearman')
            for a in ['point','contrast','loss']:
                for b in ['point','contrast','loss']:criteria.append(dict(seed=seed,track=track,fold=fold,a=a,b=b,rho=inner_corr.loc[a,b]))
            allpred={r.id:np.load(fd/(r.id+'.npz'))['outer'] for r in c.itertuples()}
            q=P[test[P.ia]&test[P.ib]].copy()
            outer_scores=[]
            for r in c.itertuples():
                pred=allpred[r.id];pq=q.copy();pq['prediction']=pred[pq.ib]-pred[pq.ia]
                ym=D.iloc[te][['source_candidate','r']].copy();ym['prediction']=pred[te]
                outer_scores.append(dict(id=r.id,outer_point=pointmetric(ym),**pmetric(pq)))
            o=pd.DataFrame(outer_scores).set_index('id');c=c.join(o,on='id')
            for metric,col in [('point','outer_point'),('contrast','contrast_mae'),('loss','selection_loss')]:
                mask=np.isfinite(c[col])
                rho=spearmanr(c.loc[mask,metric],c.loc[mask,col]).statistic if mask.sum()>1 and c.loc[mask,col].nunique()>1 else np.nan
                ranks.append(dict(seed=seed,track=track,fold=fold,criterion=metric,rho=rho,n_methods=int(mask.sum())))
            frontier=np.array([not np.any(np.all(mat<=v+1e-12,axis=1)&np.any(mat<v-1e-12,axis=1)) for v in mat])
            full_winner=c.sort_values(['topsis','point','id'],ascending=[False,True,True]).iloc[0]
            meta.append(dict(seed=seed,track=track,fold=fold,pareto_methods=int(frontier.sum()),point_range=np.ptp(mat[:,0]),contrast_range=np.ptp(mat[:,1]),loss_range=np.ptp(mat[:,2]),winner=full_winner.method,winner_dominated=not frontier[c.index[c.id==full_winner.id][0]],outer_selected_point=full_winner.outer_point,outer_best_point=c.outer_point.min(),outer_point_gap=full_winner.outer_point-c.outer_point.min()))
            for name,weights in weight_sets.items():
                score=topsis(mat,weights);cc=c.assign(score=score);win=cc.sort_values(['score','point','id'],ascending=[False,True,True]).iloc[0];pred=allpred[win.id]
                py=D.iloc[te][['condition_id','source_candidate','r']].copy();py['prediction']=pred[te];bag[name].append(py)
                pq=q.copy();pq['prediction']=pred[pq.ib]-pred[pq.ia];pairbag[name].append(pq)
                selection_rows.append(dict(seed=seed,track=track,fold=fold,rule=name,id=win.id,method=win.method))
            if seed==11:
                z=D.iloc[tr];zt=D.iloc[te]
                mcols=['Pyrolysis temperature','O/C','Surface area','Pore volume','Average pore size'];cats=['Pollutant','Wastewater type','Adsorption type']
                profiles=set(map(tuple,z[mcols].values));joint=set(map(tuple,z[cats].values))
                for i,row in zt.iterrows():
                    supports.append(dict(track=track,condition_id=row.condition_id,profile_known=tuple(row[mcols]) in profiles,pollutant_known=row.Pollutant in set(z.Pollutant),water_known=row['Wastewater type'] in set(z['Wastewater type']),joint_known=tuple(row[cats]) in joint))
        for name,weights in weight_sets.items():
            yy=pd.concat(bag[name]);qq=pd.concat(pairbag[name]);sensitivity.append(dict(seed=seed,track=track,rule=name,w_point=weights[0],w_contrast=weights[1],w_loss=weights[2],point_mae=pointmetric(yy),**pmetric(qq)))
        print('SENSITIVITY',seed,track,flush=True)
save('outer_weight_and_ablation_metrics',pd.DataFrame(sensitivity));save('weight_and_ablation_selections',pd.DataFrame(selection_rows));save('inner_criteria_correlations',pd.DataFrame(criteria));save('inner_outer_rank_agreement',pd.DataFrame(ranks));save('fold_selection_diagnostics',pd.DataFrame(meta));save('condition_support',pd.DataFrame(supports))

# Paired clustered resampling of fixed held-out errors; preserve shared source/block indices.
B=2000;boot_records=[];rng=np.random.default_rng(20260912)
for track in ['exact','broad']:
    q=Q[(Q.seed==11)&(Q.track==track)&Q.strategy.isin(STRATEGIES)].copy()
    q['loss']=np.where((q.prediction<0)!=(q.observed<0),abs(q.observed),0);q['error']=abs(q.prediction-q.observed)
    block=q.groupby(['source','exact_block','pair_id','strategy'])[['error','loss']].mean().reset_index()
    samples={}
    for source,g in block.groupby('source'):
        ids=sorted(g.exact_block.unique());samples[source]=(ids,rng.multinomial(len(ids),[1/len(ids)]*len(ids),size=B))
    for metric in ['error','loss']:
        pairdraws={}
        for (source,pair,strategy),g in block.groupby(['source','pair_id','strategy']):
            ids,weights=samples[source];series=g.set_index('exact_block')[metric].reindex(ids);valid=series.notna().values
            denominator=weights[:,valid].sum(axis=1);numerator=weights[:,valid]@series.values[valid]
            draw=np.divide(numerator,denominator,out=np.full(B,np.nan),where=denominator>0)
            pairdraws[(source,pair,strategy)]=draw
        strategy_draws={}
        for strategy in STRATEGIES:
            source_draws=[]
            for source in sorted(block.source.unique()):
                source_draws.append(np.mean([v for (ss,pp,st),v in pairdraws.items() if ss==source and st==strategy],axis=0))
            strategy_draws[strategy]=np.mean(source_draws,axis=0)
        for strategy in ['Point','Contrast','TOPSIS']:
            draw=strategy_draws[strategy]-strategy_draws['History'];valid=np.isfinite(draw)
            estimate=pmetric(q[q.strategy==strategy])['contrast_mae' if metric=='error' else 'selection_loss']-pmetric(q[q.strategy=='History'])['contrast_mae' if metric=='error' else 'selection_loss']
            lo,hi=np.quantile(draw[valid],[.025,.975]);boot_records.append(dict(track=track,seed=11,strategy=strategy,metric=metric,estimate=estimate,lower=lo,upper=hi,bootstrap_replicates=B,valid_replicates=int(valid.sum()),scope='paired source-stratified exact-block resampling of fixed outer predictions'))
save('conditional_block_resampling',pd.DataFrame(boot_records))

# Pair-specific summaries and loss-magnitude identities expose the material meaning.
summ=[];mistakes=[];identity=[]
for (seed,track,strategy),g in Q[Q.strategy.isin(STRATEGIES)].groupby(['seed','track','strategy']):
    if not np.isfinite(g.prediction).any():continue
    for pair,part in g.groupby('pair_id'):
        blocks=part.assign(error=abs(part.prediction-part.observed)).groupby('exact_block')[['observed','prediction','error']].mean()
        summ.append(dict(seed=seed,track=track,strategy=strategy,pair=pair,pair_number=PAIRS.index(pair)+1,queries=len(part),blocks=len(blocks),observed_pp=blocks.observed.mean()*100,predicted_pp=blocks.prediction.mean()*100,error_pp=blocks.error.mean()*100,positive=int((part.observed>0).sum()),negative=int((part.observed<0).sum()),min_pp=part.observed.min()*100,max_pp=part.observed.max()*100,wrong=int(((part.prediction<0)!=(part.observed<0)).sum())))
    wrong=(g.prediction<0)!=(g.observed<0)
    for r in g[wrong].itertuples():mistakes.append(dict(seed=seed,track=track,strategy=strategy,pair=r.pair_id,condition_a=r.condition_a,condition_b=r.condition_b,source=r.source,observed_pp=100*r.observed,predicted_pp=100*r.prediction,absolute_margin_pp=100*abs(r.observed)))
    if strategy!='History':
        yy=Y[(Y.seed==seed)&(Y.track==track)&(Y.strategy==strategy)].set_index('condition_id');ea=yy.loc[g.condition_a].prediction.values-yy.loc[g.condition_a].r.values;eb=yy.loc[g.condition_b].prediction.values-yy.loc[g.condition_b].r.values
        common=(ea+eb)/2;diff=(eb-ea)/2;pairmse=hierarchy(g,(ea**2+eb**2)/2)
        identity.append(dict(seed=seed,track=track,strategy=strategy,common_mse=hierarchy(g,common**2),differential_mse=hierarchy(g,diff**2),paired_mse=pairmse))
save('material_pair_summary',pd.DataFrame(summ));save('wrong_choice_details',pd.DataFrame(mistakes));save('paired_error_decomposition',pd.DataFrame(identity))
ident=pd.DataFrame(identity);assert np.allclose(ident.common_mse+ident.differential_mse,ident.paired_mse)
sen=pd.DataFrame(sensitivity);full=sen[sen.rule=='Full'].merge(M[M.strategy=='TOPSIS'],on=['seed','track'],suffixes=('_new','_old'))
for v in ['point_mae','contrast_mae','selection_loss']:assert np.allclose(full[v+'_new'],full[v+'_old'])
assert not pd.DataFrame(meta).winner_dominated.any()
checks=dict(original_data_sha256=hashlib.sha256((ROOT/'data/condition_data.csv').read_bytes()).hexdigest(),fitted_models_unchanged=True,new_estimator_fits=0,method_count=len(METHODS),configuration_count=len(CONFIGS),outer_folds=len(meta),weight_rules=len(weight_sets),full_topsis_reproduces_original=True,error_decomposition_verified=True,topsis_winners_on_inner_pareto_front=True,resampling_replicates=B,resampling_seed=20260912)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
print(json.dumps(checks,indent=2),flush=True)
