from __future__ import annotations
import pickle, sys, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core.fortaleza_poisson_backend import (build_training_frame,build_poisson_pipeline,FortalezaPoissonRuntime,FEATURE_COLS,normalize_name)
from scripts.validate_stgat_v5_walk_forward import RegionConfig,load_region,score_stgat_v5,norm_adj
OUT=Path(__file__).parent
torch.set_num_threads(4)
HORIZON=30
WEIGHTS=(0.,.2,.3,.4,.5,.7,1.)
TRAIN_END=pd.Timestamp('2025-12-31')
START=pd.Timestamp('2026-07-01')
END=pd.Timestamp('2026-09-21')
SEED=20261022
def load(file):
    with open(file,'rb') as f: return pickle.load(f)
def dataset_subset(base,idx):
    ix=np.array(idx,dtype=int)
    return dict(node_features=base['node_features'][ix], dates=base['dates'],
                nodes_gdf=base['nodes_gdf'].iloc[ix].reset_index(drop=True).copy(),
                adj_geo=base['adj_geo'][np.ix_(ix,ix)],
                adj_conflict=base['adj_conflict'][np.ix_(ix,ix)])
def to_runtime(data,ti):
    hist=dataset_subset(data,np.arange(data['node_features'].shape[0]))
    hist['node_features']=hist['node_features'][:,:ti,:]
    hist['dates']=pd.DatetimeIndex(hist['dates'][:ti])
    nf=hist['node_features']
    hist['nodes_gdf']['recent_cvli']=nf[:,max(0,ti-30):ti,0].sum(axis=1)
    hist['nodes_gdf']['total_cvli']=nf[:,:,0].sum(axis=1)
    return hist
def rows_report(order, y_all, ix):
    chosen=ix[order]
    total=float(y_all.sum())
    res={'pool_capture':float(y_all[ix].sum())/total if total else 0,
         'events':total}
    for k in (5,10,20):
        picks=chosen[:k]
        hit=int((y_all[picks]>0).sum())
        res['p'+str(k)]=hit/min(k,len(chosen))
        res['capt'+str(k)]=float(y_all[picks].sum())/total if total else 0
        res['fp'+str(k)]=min(k,len(chosen))-hit
    return res
def main():
    tstart=time.perf_counter()
    base=load(OUT/'exp_data/all_official/processed_fortaleza.pkl')
    active=load(ROOT/'data/processed/processed_fortaleza.pkl')
    rebuilt=load(OUT/'exp_data/top40_rebuild/processed_fortaleza.pkl')
    old_names=active['nodes_gdf']['name'].tolist()
    a_names=base['nodes_gdf']['name'].tolist()
    assert set(old_names)==set(rebuilt['nodes_gdf']['name'])
    assert len(set(a_names))==121
    index={n:i for i,n in enumerate(a_names)}
    fixed40=[index[n] for n in old_names]
    extra=base['nodes_gdf'].drop(index=fixed40).sort_values(['recent_cvli','name'],ascending=[False,True])['name'].tolist()
    fixed60=fixed40+[index[n] for n in extra[:20]]
    meshes={'fixed40':fixed40,'fixed60':fixed60,'all121':list(range(121))}
    for key,inds in meshes.items(): print('MESH',key,len(inds),flush=True)
    dates=pd.DatetimeIndex(pd.to_datetime(base['dates']))
    nf=base['node_features']
    # Train 30-day Poisson on cutoff-past data only; use every 14th date to constrain compute.
    train=dataset_subset(base,np.arange(121))
    ti_train=int(np.searchsorted(dates,TRAIN_END,side='right'))
    train_nf=nf[:,:ti_train,0]
    train['nodes_gdf']['total_cvli']=train_nf.sum(axis=1)
    train['nodes_gdf']['recent_cvli']=train_nf[:,max(0,ti_train-730):].sum(axis=1)
    print('TRAINFRAME 30D start',flush=True)
    train_df=build_training_frame(train,train_start='2022-01-01',train_end='2025-12-31',horizon_days=30,window=90)
    sample_days=pd.DatetimeIndex(pd.to_datetime(train_df['sample_date']))
    m=((sample_days - pd.Timestamp('2022-01-01')).days % 14)==0
    tr=train_df.loc[m]
    print('FIT rows',len(tr),'full rows',len(train_df),'events target',float(tr['target_count'].sum()),flush=True)
    reg=build_poisson_pipeline(alpha=1e-4,max_iter=250)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        reg.fit(tr[FEATURE_COLS],tr['target_count'].to_numpy(dtype=float))
        print('FIT done','sec',round(time.perf_counter()-tstart,1),'warnings',[str(q.message)[:110] for q in w][:3],flush=True)
    payload={'pipeline':reg,'region':'fortaleza','window':90,'feature_cols':FEATURE_COLS,
             'horizon_days':30,'train_range':{'start':'2022-01-01','end':'2025-12-31'}}
    # Keep models only in isolated experiment directory.
    (OUT/'exp_models').mkdir(exist_ok=True)
    with (OUT/'exp_models/poisson30_all121_stride14.pkl').open('wb') as f:pickle.dump(payload,f)
    _,trained_model,_,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    checkpoint=trained_model.state_dict()
    mdata={}
    mod={}
    adj={}
    for label,inds in meshes.items():
        d=dataset_subset(base,inds)
        mdata[label]=d
        model=type(trained_model)(num_nodes=len(inds),in_channels=cfg.channels,time_steps=cfg.window).cpu()
        model.load_state_dict(checkpoint,strict=True)
        model.eval()
        mod[label]=model
        adj[label]=norm_adj(d['adj_geo'],d['adj_conflict'])
    ti_list=[i for i,d in enumerate(dates) if START<=d<=END and i+HORIZON<=len(dates)][::7]
    all_metrics=[]; top_details=[]; pool_stats=[]
    print('BACKTEST',len(ti_list),'cutoffs',str(dates[ti_list[0]].date()),str(dates[ti_list[-1]].date()),flush=True)
    for ti in ti_list:
        cutoff=dates[ti]
        y_all=nf[:,ti:ti+HORIZON,0].sum(axis=1).astype(float)
        if y_all.sum()<=0:continue
        historical=nf[:,max(0,ti-730):ti,0].sum(axis=1)
        # Rolling K uses ONLY events prior to cutoff.
        roll_order=sorted(range(121),key=lambda x:(-historical[x],a_names[x]))
        for label,indices in {'rolling40':roll_order[:40],'rolling60':roll_order[:60],'fixed40':fixed40,'fixed60':fixed60,'all121':meshes['all121']}.items():
            ix=np.asarray(indices,dtype=int)
            pool_stats.append(dict(date=str(cutoff.date()),mesh=label,nodes=len(ix),
                   events=float(y_all.sum()),pool_events=float(y_all[ix].sum()),
                   pool_coverage=float(y_all[ix].sum()/y_all.sum()),
                   overlap_with_fixed40=len(set(ix).intersection(fixed40))))
        for label,idx in meshes.items():
            hist=to_runtime(mdata[label],ti)
            pmap,_=FortalezaPoissonRuntime(payload,hist).predict_scores()
            p=np.asarray([pmap[normalize_name(name)] for name in hist['nodes_gdf']['name']])
            s=score_stgat_v5(hist,mod[label],adj[label],cfg,ti)
            assert len(p)==len(s)==len(idx) and np.isfinite(p).all() and np.isfinite(s).all()
            rp=pd.Series(p).rank(pct=True)
            rs=pd.Series(s).rank(pct=True)
            for w in WEIGHTS:
                blend=(1-w)*rp+w*rs
                order=pd.DataFrame({'blend':blend,'p':p}).sort_values(['blend','p'],ascending=False,kind='mergesort').index.to_numpy(dtype=int)
                row=rows_report(order,y_all,np.asarray(idx))
                row.update(date=str(cutoff.date()),mesh=label,stgat_weight=w,period='sep' if cutoff.month==9 else 'jul_aug')
                all_metrics.append(row)
                top_details.append(dict(date=row['date'],mesh=label,weight=w,top10=' | '.join(a_names[idx[j]] for j in order[:10])))
        print('FINISHED',str(cutoff.date()),'events',int(y_all.sum()),'secs',round(time.perf_counter()-tstart,1),flush=True)
    metrics=pd.DataFrame(all_metrics)
    pools=pd.DataFrame(pool_stats)
    summ=metrics.groupby(['mesh','stgat_weight'],as_index=False).agg(
       n=('date','nunique'),p10=('p10','mean'),capt10=('capt10','mean'),
       capt20=('capt20','mean'),p20=('p20','mean'),pool_capture=('pool_capture','mean'),
       fp10=('fp10','mean'))
    sept=metrics[metrics.period=='sep'].groupby(['mesh','stgat_weight'],as_index=False).agg(
       n=('date','nunique'),p10=('p10','mean'),capt10=('capt10','mean'),
       capt20=('capt20','mean'),p20=('p20','mean'),pool_capture=('pool_capture','mean'))
    metrics.to_csv(OUT/'expand_align_30d_rows_20261022.csv',index=False,encoding='utf-8-sig')
    summ.round(6).to_csv(OUT/'expand_align_30d_summary_20261022.csv',index=False,encoding='utf-8-sig')
    sept.round(6).to_csv(OUT/'expand_align_30d_september_20261022.csv',index=False,encoding='utf-8-sig')
    pools.to_csv(OUT/'pool_coverage_dynamic_20261022.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(top_details).to_csv(OUT/'expand_align_30d_top10_20261022.csv',index=False,encoding='utf-8-sig')
    json.dump({'train':'2022-01-01..2025-12-31','poisson_horizon':30,'stgat_horizon':30,
       'training_stride_days':14,'poisson_train_examples':len(tr),'eval_horizon_days':30,
       'eval_start':str(dates[ti_list[0]].date()),'eval_end':str(dates[ti_list[-1]].date()),
       'stgat_transfer':'zero-shot: checkpoint 40 transferred to 60 and 121 with new graph adjacency; not retrained',
       'reference':'official Poisson14 and STGAT30 existing, remain untouched',
       'limitations':['ST-GAT training validation July-Aug 2026 means September preferred for testing',
       'only three-to-four September cutoffs; not statistically robust',
       'Static geographic faction/tension attributes may reflect post-cutoff intelligence',
       'graph attention uses changed node count and adjacency in zero-shot experiments',
       'Poisson stride14 is experimental retraining, not official identical 2022-2025 daily fit',
       'Rolling nodes selected past 730 days at each cutoff, with no held-out selection leak'],
       'duration_s':time.perf_counter()-tstart},(OUT/'expand_align_30d_metadata_20261022.json').open('w',encoding='utf-8'),indent=2,ensure_ascii=False)
    print('\nPOOL COVERAGE',flush=True)
    print(pools.groupby('mesh').agg(n=('date','count'),avg_coverage=('pool_coverage','mean'),total_pool_events=('pool_events','sum'),total_events=('events','sum'),overlap40=('overlap_with_fixed40','mean')).round(3).to_string(),flush=True)
    print('\nSUMMARY ALL',flush=True)
    print(summ.round(3).to_string(index=False),flush=True)
    print('\nSEPTEMBER',flush=True)
    print(sept.round(3).to_string(index=False),flush=True)
    print('WROTE',OUT,'sec',round(time.perf_counter()-tstart,1),flush=True)
if __name__=='__main__':main()
