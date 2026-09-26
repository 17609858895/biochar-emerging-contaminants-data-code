"""Dense retrospective TOPSIS preference grid; frozen inner scores and outer caches.

No estimators are fitted. Outer outcomes describe the preference grid and are not
used to choose weights. Flat triangles show actual centroid evaluations; no
interpolation of outer errors/losses across a change of selected model.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model24_registry import topsis
from run_model24_topsis import D,P,ROOT,hierarchy
E=ROOT/'evidence/model24_topsis_v1'
O=ROOT/'evidence/topsis_surfaces_v1'
SEEDS=[11,23,47];TRACKS=['exact','broad','source']
CRITERIA=['point','contrast','loss'];METRICS=['point_mae','contrast_mae','selection_loss','wrong']

def grid(step=.005):
 n=int(round(.7/step));assert abs(n*step-.7)<1e-10
 points=[];lookup={}
 for i in range(n+1):
  for j in range(n+1-i):lookup[i,j]=len(points);points.append((.1+i*step,.1+j*step))
 xy=np.array(points);tri=[]
 for i in range(n):
  for j in range(n-i):
   tri.append([lookup[i,j],lookup[i+1,j],lookup[i,j+1]])
   if i+j<=n-2:tri.append([lookup[i+1,j],lookup[i+1,j+1],lookup[i,j+1]])
 tri=np.array(tri);centres=xy[tri].mean(axis=1)
 return xy,tri,centres

def winners(c,weights):
 # c is ordered by the exact published tie breakers: lower point MAE, then ID.
 x=c[CRITERIA].to_numpy(float);active=np.ptp(x,axis=0)>1e-12
 v=np.zeros_like(x);v[:,active]=x[:,active]/np.linalg.norm(x[:,active],axis=0)
 near=(v-v.min(axis=0))**2;far=(v-v.max(axis=0))**2
 dp=np.sqrt((weights**2)@near.T);dm=np.sqrt((weights**2)@far.T)
 scores=np.divide(dm,dp+dm,out=np.full_like(dm,.5),where=dp+dm>1e-15)
 return scores.argmax(axis=1)

def run():
 O.mkdir(exist_ok=True)
 xy,tri,centres=grid();allxy=np.vstack([xy,centres,[[1/3,1/3]]])
 w=np.c_[allxy,1-allxy.sum(axis=1)]
 assert np.all(w>=.1-1e-12)and np.allclose(w.sum(axis=1),1)
 nv,nc=len(xy),len(centres);assert(nv,nc)==(10011,19600)
 pointw=1/D.groupby('source_candidate').r.transform('size').to_numpy()/D.source_candidate.nunique()
 p=P.copy()
 nw=p.groupby(['source','pair_id','exact_block'])['observed'].transform('size')
 nb=p.groupby(['source','pair_id']).exact_block.transform('nunique')
 npair=p.groupby('source').pair_id.transform('nunique')
 pairw=1/nw/nb/npair/p.source.nunique();assert np.isclose(pairw.sum(),1)
 assert np.isclose(np.dot(pairw,p.observed),hierarchy(p,p.observed))
 old=pd.read_csv(ROOT/'evidence/full24_v5/outer_weight_and_ablation_metrics.csv')
 oldselections=pd.read_csv(ROOT/'evidence/full24_v5/weight_and_ablation_selections.csv')
 outputs=[];signatures={};foldmeta=[];checks=[];input_hashes={}
 hashfile=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
 for path in [ROOT/'data/condition_data.csv',E/'metrics.csv',ROOT/'evidence/supported_material_contrasts.csv']:input_hashes[path.relative_to(ROOT).as_posix()]=hashfile(path)
 for track in TRACKS:
  seedvalues=[];seedret=[];allsig=[]
  for seed in SEEDS:
   total=np.zeros((len(w),4));ret=np.zeros(len(w));sig=[];seen=np.zeros(len(D),int);qseen=np.zeros(len(P),int)
   for fd in sorted(E.glob(f's{seed}_{track}_*')):
    if not fd.is_dir():continue
    c=pd.read_csv(fd/'method_scores.csv').sort_values(['point','id']).reset_index(drop=True)
    split=pd.read_csv(fd/'split.csv');assert np.array_equal(split.condition_id,D.condition_id)
    te=split.outer_test.to_numpy(bool);seen+=te;qt=te[P.ia]&te[P.ib];qseen+=qt
    q=P[qt];observed=q.observed.to_numpy();fold=int(fd.name.rsplit('_',1)[1])
    contrib=[]
    for row in c.itertuples():
     predpath=fd/(row.id+'.npz');pred=np.load(predpath)['outer'];assert np.isfinite(pred[te]).all()
     delta=pred[q.ib]-pred[q.ia];wrong=(delta<0)!=(observed<0)
     contrib.append([np.dot(pointw[te],abs(pred[te]-D.loc[te,'r'])),np.dot(pairw[qt],abs(delta-observed)),np.dot(pairw[qt],np.where(wrong,abs(observed),0)),wrong.sum()])
     input_hashes[predpath.relative_to(ROOT).as_posix()]=hashfile(predpath)
    contrib=np.array(contrib);win=winners(c,w);sig.append(win.astype('int8'))
    total+=contrib[win];ret+=win==win[-1]
    for row in old[(old.seed==seed)&(old.track==track)&(old.rule.str.startswith('W')|(old.rule=='Full'))].itertuples():
     ww=np.array([row.w_point,row.w_contrast,row.w_loss]);ii=int(np.argmin(np.max(abs(w-ww),axis=1)));assert np.max(abs(w[ii]-ww))<1e-10
     scalar=topsis(c[CRITERIA].to_numpy(),ww);scalarwin=int(np.argmax(scalar));assert win[ii]==scalarwin
     orig=oldselections[(oldselections.seed==seed)&(oldselections.track==track)&(oldselections.fold==fold)&(oldselections.rule==row.rule)].iloc[0]
     assert c.iloc[win[ii]].id==orig.id
    foldmeta.append(dict(track=track,seed=seed,fold=fold,ordered_ids=c.id.tolist(),ordered_methods=c.method.tolist(),equal_weight_id=c.iloc[win[-1]].id))
    for path in [fd/'method_scores.csv',fd/'split.csv']:input_hashes[path.relative_to(ROOT).as_posix()]=hashfile(path)
   assert np.all(seen==1)and np.all(qseen==1)
   sig=np.stack(sig,axis=1);nf=sig.shape[1];ret=100*ret/nf
   for row in old[(old.seed==seed)&(old.track==track)&(old.rule.str.startswith('W')|(old.rule=='Full'))].itertuples():
    ww=np.array([row.w_point,row.w_contrast,row.w_loss]);ii=int(np.argmin(np.max(abs(w-ww),axis=1)))
    err=np.max(abs(total[ii]-np.array([getattr(row,m)for m in METRICS])))
    assert err<1e-10,(seed,track,row.rule,err)
    checks.append(dict(seed=seed,track=track,rule=row.rule,max_absolute_error=float(err)))
   seedvalues.append(total);seedret.append(ret);allsig.append(sig)
   table=pd.DataFrame(w,columns=['w_point','w_contrast','w_loss']);table[METRICS]=total
   table['winner_retention_pct']=ret;table['seed']=seed;table['track']=track
   table['sample_type']=['vertex']*nv+['triangle_centroid']*nc+['equal_weight']
   table['sample_index']=np.r_[np.arange(nv),np.arange(nc),[-1]];outputs.append(table)
   print('REASSEMBLED',track,seed,'weights',len(w),'folds',nf,flush=True)
  vals=np.stack(seedvalues);retain=np.stack(seedret);signatures[track]=np.concatenate(allsig,axis=1)
  summary=pd.DataFrame(w,columns=['w_point','w_contrast','w_loss'])
  for k,m in enumerate(METRICS):
   for name,fn in [('median',np.median),('min',np.min),('max',np.max)]:summary[m+'_'+name]=fn(vals[:,:,k],axis=0)
  summary['winner_retention_pct']=retain.mean(axis=0)
  summary.iloc[:nv].to_csv(O/(track+'_vertices.csv'),index=False)
  summary.iloc[nv:nv+nc].to_csv(O/(track+'_triangles.csv'),index=False)
  summary.iloc[[-1]].to_csv(O/(track+'_equal_weight.csv'),index=False)
 pd.concat(outputs,ignore_index=True).to_csv(O/'all_seed_metrics.csv',index=False)
 np.savez_compressed(O/'grid_and_selections.npz',vertices=xy,triangles=tri,centres=centres,**{t+'_winners':a for t,a in signatures.items()})
 (O/'fold_model_ids.json').write_text(json.dumps(foldmeta,indent=2),encoding='utf8')
 # Shared edges between centroid-evaluated triangles with different fold signatures.
 edges={}
 for ti,t in enumerate(tri):
  for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:edges.setdefault(tuple(sorted([int(a),int(b)])),[]).append(ti)
 boundary=[];sigs=signatures['broad'][nv:nv+nc]
 for (a,b),ts in edges.items():
  if len(ts)==2 and np.any(sigs[ts[0]]!=sigs[ts[1]]):boundary.append([a,b])
 np.save(O/'broad_boundary_edges.npy',np.array(boundary,dtype=int))
 ranges={}
 for t in TRACKS:
  q=pd.read_csv(O/(t+'_triangles.csv'));eq=pd.read_csv(O/(t+'_equal_weight.csv')).iloc[0]
  ranges[t]={m:{'min_pp':float(q[m+'_median'].min()*100),'max_pp':float(q[m+'_median'].max()*100),'equal_weight_pp':float(eq[m+'_median']*100)}for m in METRICS[:3]}
  ranges[t]['retention_pct_range']=[float(q.winner_retention_pct.min()),float(q.winner_retention_pct.max())]
 report=dict(date='2026-09-17',new_observations=0,new_model_fits=0,domain='each weight >= 0.1; weights sum to one',grid_step=.005,vertex_count=nv,triangle_centroid_count=nc,equal_weight_evaluations=1,weights_per_seed_track=len(w),seed_tracks=9,original_weight_rules_reproduced=len(checks),max_original_metric_difference=max(c['max_absolute_error']for c in checks),scalar_and_vector_winners_verified=True,all_fold_predictions_cover_each_condition_and_query_once=True,selection='frozen inner matrices; fixed point-MAE then lexical-ID tie breaking',aggregation='Original source-balanced point weights and query-block-pair-source contrast/loss weights; median across 3 seeds',surface='piecewise-constant horizontal triangle at its directly evaluated centroid; no smoothing or interpolation of outcomes',boundary='shared grid edges whose neighbouring centroid selections differ in at least one of 15 broad-group seed-folds; discretized at 0.005 resolution',retention='fraction of 15 broad-group seed-fold choices matching their own equal-weight choice',outer_weight_optimization=False,ranges=ranges)
 (O/'protocol_and_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 (O/'input_sha256.json').write_text(json.dumps(input_hashes,indent=2),encoding='utf8')
 pd.DataFrame(checks).to_csv(O/'original_grid_reproduction.csv',index=False)
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':run()
