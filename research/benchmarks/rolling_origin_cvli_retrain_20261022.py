from __future__ import annotations
"""Strict rolling-origin ST-GAT + Poisson historical CVLI backtest (2024-2026).
All training/evaluation split decisions are made before reading holdout labels.
Experimental only: no production checkpoints or live settings change.
"""
import argparse
import copy
import json
import math
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.core.architectures import get_model_class
from src.core.fortaleza_poisson_backend import (
    FEATURE_COLS, build_current_frame, build_poisson_pipeline, FortalezaPoissonRuntime)
from scripts.training.Active.train_all_specialists import NegBinomRankingLoss,get_active_nodes_bimestral
from scripts.validate_stgat_v5_walk_forward import build_input, norm_adj
from expand_align_30d_20261022 import dataset_subset,load
HORIZON=30
SEED=261022
WEIGHTS=(0.,.2,.4,.7,1.)
CHANNELS=41
WINDOW=120
CHANNEL_WHITELIST=(0,*range(3,23),33,34,35,36)
OUT=HERE/'historical_rolling_cvli_only_20261022'
torch.set_num_threads(3)
def geo_graph(gdf):
    lat=np.deg2rad(gdf['lat'].to_numpy(float))
    lon=np.deg2rad(gdf['long'].to_numpy(float))
    a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    dist=2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))
    geo=np.where((dist<=3)&(dist>0),1/(dist+.1),0.)
    np.fill_diagonal(geo,1.)
    return geo.astype(np.float32),np.eye(len(lat),dtype=np.float32)
def sample_input(nf,t):
    a=build_input(nf,t,WINDOW,CHANNELS)
    unused=[i for i in range(CHANNELS) if i not in CHANNEL_WHITELIST]
    a[:,:,unused]=0.
    assert np.isfinite(a).all()
    return torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0)
def build_hist(full,t):
    # At cutoff t, exclusively features/dates before target index t.
    assert t<=len(full['dates'])
    hist=dict(full)
    hist['node_features']=full['node_features'][:,:t,:]
    hist['dates']=pd.DatetimeIndex(full['dates'][:t])
    nodes=full['nodes_gdf'].copy()
    cvli=hist['node_features'][:,:,0]
    nodes['total_cvli']=cvli.sum(axis=1)
    nodes['recent_cvli']=cvli[:,max(0,t-730):t].sum(axis=1)
    nodes['tension_index']=0.
    hist['nodes_gdf']=nodes
    return hist
def rolling_indices(full,t,n):
    # Only CVLI from [t-730,t) is allowed to select candidates.
    nf=full['node_features'][:,:,0]
    look=nf[:,max(0,t-730):t].sum(axis=1)
    names=full['nodes_gdf']['name'].tolist()
    order=sorted(range(len(names)),key=lambda i:(-float(look[i]),names[i]))
    return order[:n]
def training_cutoffs(dates,train_end):
    indices=[i for i,x in enumerate(dates)
          if x>=pd.Timestamp('2022-07-01') and i>=WINDOW+60
          and i+HORIZON<=len(dates) and dates[i+HORIZON-1]<=train_end]
    return indices[::21]
def validation_cutoffs(year,index):
    # Exactly three mutually non-overlapping 30-day blocks Jan1..Mar30/31.
    start=pd.Timestamp(f'{year}-01-01')
    return [index[start+pd.Timedelta(days=30*k)] for k in range(3)]
def test_cutoffs(year,index,dates):
    targets=[]
    for month in range(4,13):
        stamp=pd.Timestamp(year=year,month=month,day=1)
        if stamp not in index: continue
        t=index[stamp]
        if t+HORIZON<=len(dates) and dates[t+HORIZON-1]<=pd.Timestamp('2026-09-30'):
            targets.append(t)
    return targets
def poisson_fit(full,train,dates):
    # Vectorized data date slicing; official feature generator but corrected
    # point-in-time static fields. Each target ends no later than Dec31.
    t0=time.monotonic()
    parts=[]
    for t in train:
        hist=build_hist(full,t)
        f=build_current_frame(hist,window=90)
        assert (f['sample_date']==str(dates[t].date())).all()
        y=full['node_features'][:,t:t+HORIZON,0].sum(axis=1).astype(float)
        f['target_count']=y
        parts.append(f)
    frame=pd.concat(parts,ignore_index=True)
    assert (frame['tension_index']==0).all()
    assert (frame[FEATURE_COLS].notna().all().all())
    pipeline=build_poisson_pipeline(alpha=1e-4,max_iter=300)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter('always')
        pipeline.fit(frame[FEATURE_COLS],frame['target_count'])
    print('POISSON_FIT rows',len(frame),'secs',round(time.monotonic()-t0,1),
          'warnings',[str(w.message)[:140] for w in warned],flush=True)
    return {'pipeline':pipeline,'window':90,'horizon_days':30,'region':'fortaleza',
            'feature_cols':FEATURE_COLS}
def cold_from_history(nf,t):
    obs=nf[:,max(0,t-30):t,0]
    streak=np.zeros(obs.shape[0],dtype=np.float32)
    for day in range(obs.shape[1]):
        streak=np.where(obs[:,day]>0,0,streak+1)
    return torch.from_numpy(-np.minimum(streak,30)).float()
def metrics(out,order,target,idx):
    chosen=np.asarray(idx,dtype=int)[order]
    total=target.sum()
    if total<=0:return {}
    return dict(
        p10=float(np.mean(target[chosen[:10]]>0)),
        p20=float(np.mean(target[chosen[:20]]>0)),
        captured10=int(target[chosen[:10]].sum()),
        captured20=int(target[chosen[:20]].sum()),
        capt10=float(target[chosen[:10]].sum()/total),
        capt20=float(target[chosen[:20]].sum()/total),
        false_positives10=int(np.sum(target[chosen[:10]]==0)),
        coverage=float(target[idx].sum()/total),
        city_events=int(total))
def stgat_raw(model,full,indices,t):
    df=dataset_subset(full,indices)
    a,c=geo_graph(df['nodes_gdf'])
    adj=norm_adj(a,c)
    model.eval()
    with torch.inference_mode():
        out=model(sample_input(df['node_features'],t),adj).flatten()
        neural=F.softplus(out).numpy()
    normalized=(neural-neural.min())/(neural.max()-neural.min()+1e-6)
    cvli=df['node_features'][:,max(0,t-30):t,0].sum(axis=1)
    support=np.clip(cvli/2,0,1)
    score=(.50*normalized+.25*support)*100.
    return neural,score
def val_metrics(model,full,indices,val):
    rows=[]
    for t in val:
        y=full['node_features'][:,t:t+HORIZON,0].sum(axis=1)
        _,s=stgat_raw(model,full,indices,t)
        order=np.argsort(-s,kind='stable')
        rows.append(metrics(None,order,y,indices))
    d=pd.DataFrame(rows)
    return {'p10':float(d.p10.mean()),'p20':float(d.p20.mean()),
       'capt20':float(d.capt20.mean()),'selection':float(.6*d.p10.mean()+.4*d.p20.mean())}
def train_stgat(full,indices,tr,val,year,max_epochs=6):
    # Model is initialized randomly for EACH origin, never inherits a checkpoint
    # that may have seen labels past that origin.
    torch.manual_seed(SEED+year)
    model=get_model_class('DeepSTGAT_v5')(num_nodes=60,in_channels=CHANNELS,time_steps=WINDOW)
    d=dataset_subset(full,indices)
    a,c=geo_graph(d['nodes_gdf'])
    adj=norm_adj(a,c)
    xs=[sample_input(d['node_features'],i) for i in tr]
    ys=[torch.from_numpy(d['node_features'][:,i:i+30,0].sum(axis=1)).float() for i in tr]
    masks=[torch.from_numpy(get_active_nodes_bimestral(d['node_features'],i,WINDOW)) for i in tr]
    cold=[cold_from_history(d['node_features'],i) for i in tr]
    assert len(xs)>=15
    crit=NegBinomRankingLoss(nb_r_init=2.,ranking_weight=2.,indecision_weight=.2,
                            nb_r_min=.75,nb_r_max=50.)
    opt=torch.optim.AdamW([
        {'params':model.parameters(),'lr':2.5e-4,'weight_decay':.02},
        {'params':crit.parameters(),'lr':5e-5,'weight_decay':0.}])
    best=None;bestscore=-1
    rows=[];patience=0
    for ep in range(1,max_epochs+1):
        model.train()
        perm=np.random.default_rng(SEED+year*10+ep).permutation(len(xs))
        opt.zero_grad(set_to_none=True)
        losses=[]
        for k,j in enumerate(perm):
            logits=model(xs[j],adj).flatten()
            loss=crit(logits,ys[j],active_mask=masks[j],cold_streak_signal=cold[j])
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite STGAT loss')
            (loss/4.).backward()
            losses.append(float(loss.item()))
            if k%4==3 or k==len(perm)-1:
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(crit.parameters()),1.)
                opt.step();crit.constrain_dispersion_();opt.zero_grad(set_to_none=True)
        v=val_metrics(model,full,indices,val)
        rows.append({'fold':year,'epoch':ep,'train_loss':float(np.mean(losses)),**v})
        if v['selection']>bestscore+.0025:
            bestscore=v['selection']
            best={k:vv.detach().cpu().clone() for k,vv in model.state_dict().items()}
            best_epoch=ep
            patience=0
        else:patience+=1
        print('FOLD_EPOCH',year,ep,'loss',round(rows[-1]['train_loss'],4),
              'valP10',round(v['p10'],3),'valP20',round(v['p20'],3),
              'selection',round(v['selection'],3),'best',best_epoch,flush=True)
        if patience>=3:break
    assert best is not None
    model.load_state_dict(best,strict=True)
    return model,rows,best_epoch
def simulate_fold(year,full):
    elapsed=time.monotonic()
    dates=pd.DatetimeIndex(pd.to_datetime(full['dates']))
    assert dates.min()<=pd.Timestamp('2022-01-01')
    assert dates[-1]>=pd.Timestamp('2026-09-30')
    index={d:i for i,d in enumerate(dates)}
    train_end=pd.Timestamp(f'{year-1}-12-31')
    first_val=pd.Timestamp(f'{year}-01-01')
    end_idx=index[first_val]
    train=training_cutoffs(dates,train_end)
    val=validation_cutoffs(year,index)
    test=test_cutoffs(year,index,dates)
    assert len(val)==3 and len(test)>=6
    assert dates[train[-1]+29]<=train_end
    assert dates[val[-1]+29]<dates[test[0]]
    assert all(dates[t+HORIZON-1]<=dates[-1] for t in test)
    # POISSON trains on all 121 neighborhoods, using prior labels only.
    pool_train=rolling_indices(full,end_idx,60)
    name_list=full['nodes_gdf']['name'].tolist()
    print('FOLD_BEGIN',year,'train_end',train_end.date(),
       'train_windows',len(train),'validation',[(str(dates[t].date())) for t in val],
       'test',[(str(dates[t].date())) for t in test],
       'pool',len(pool_train),flush=True)
    poisson=poisson_fit(full,train,dates)
    model,learning,best_epoch=train_stgat(full,pool_train,train,val,year)
    rows=[]
    for t in test:
        # Actual issue-time rolling mesh, recalculated ONLY using history before t.
        ids60=rolling_indices(full,t,60)
        ids40=ids60[:40]
        # Model learned inductively on 60 nodes; candidate identities need not be fixed.
        nn_raw,nn_oper=stgat_raw(model,full,ids60,t)
        neural_ranks=pd.Series(nn_raw).rank(pct=True)
        op_ranks=pd.Series(nn_oper).rank(pct=True)
        phist=build_hist(full,t)
        pmap,_=FortalezaPoissonRuntime(poisson,phist).predict_scores()
        p=np.asarray([pmap.get(str(n).strip(),float('nan')) for n in name_list],dtype=float)
        # Use backend normalization's exact normalization of names if needed.
        if not np.isfinite(p).all():
            from src.core.fortaleza_poisson_backend import normalize_name
            p=np.asarray([pmap[normalize_name(n)] for n in name_list],dtype=float)
        assert np.isfinite(p).all()
        y=full['node_features'][:,t:t+HORIZON,0].sum(axis=1).astype(float)
        for pool_size,ids in [(40,ids40),(60,ids60)]:
            ids=np.asarray(ids,dtype=int)
            include_local=np.arange(pool_size)
            pi=p[ids]
            ps=pd.Series(pi).rank(pct=True)
            so=op_ranks.iloc[include_local].reset_index(drop=True)
            sn=neural_ranks.iloc[include_local].reset_index(drop=True)
            for component,ss in [('operational',so),('neural_raw',sn)]:
                for weight in WEIGHTS:
                    blended=(1-weight)*ps+weight*ss
                    order=pd.DataFrame({'blend':blended,'poisson':pi}).sort_values(
                        ['blend','poisson'],ascending=False,kind='mergesort').index.to_numpy()
                    report=metrics(None,order,y,ids)
                    rows.append({'fold':year,'train_target_end':str(train_end.date()),
                         'date':str(dates[t].date()),'future_end':str(dates[t+HORIZON-1].date()),
                         'pool':f'rolling{pool_size}_stgat60_graph',
                         'component':component,
                         'stgat_weight':float(weight),
                         'poisson_weight':float(1-weight),
                         'stgat_best_epoch':best_epoch,
                         **report})
        print('FOLD_TEST',year,dates[t].date(),'cvli_city',int(y.sum()),
              'pool60_captured',int(y[ids60].sum()),flush=True)
    OUT.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f'fold_{year}_rows.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(learning).to_csv(OUT/f'fold_{year}_learning.csv',index=False,encoding='utf-8-sig')
    summary=df.groupby(['pool','component','stgat_weight'],as_index=False).agg(
        n=('date','count'),p10=('p10','mean'),capt20=('capt20','mean'),
        capt10=('capt10','mean'),coverage=('coverage','mean'),false_positives10=('false_positives10','mean'))
    summary.to_csv(OUT/f'fold_{year}_summary.csv',index=False,encoding='utf-8-sig')
    meta={'fold':year,'train_until':str(train_end.date()),'train_samples':len(train),
      'validation_dates':[str(dates[i].date()) for i in val],
      'test_dates':[str(dates[i].date()) for i in test],
      'model':'DeepSTGAT_v5 trained from RANDOM initialization at each year origin',
      'train_channels':CHANNEL_WHITELIST,'training_pool':['rolling 60 based on 730 previous days to fold origin'],
      'metric_denominator':'all 121 Fortaleza neighborhoods','source_range':[str(dates[0].date()),str(dates[-1].date())],
      'stgat_epochs':len(learning),'best_epoch':best_epoch,'graph':'distance-only plus identity, conflict adjacency identity',
      'poisson_fit':'Official Poisson feature generator, 30d target, static attributes rebuilt point-in-time, all121',
      'selection':'target previous 730 days, reselected before each test month',
      'leakage_controls':'all channels except CVLI, calendar and momentum masked, no original pretrained checkpoint',
      'nonsynchronized_31st':'30d windows commence on day1 and exclude day31 of 31-day months; do not double-count labels',
      'production_untouched':True,'secs':round(time.monotonic()-elapsed,1)}
    (OUT/f'fold_{year}_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FOLD_DONE',year,'windows',len(test),'best_epoch',best_epoch,
          'seconds',meta['secs'],flush=True)
    print(summary[(summary.pool=='rolling60_stgat60_graph')&(summary.component=='operational')]
          [['stgat_weight','p10','capt20','coverage']].round(3).to_string(index=False),flush=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,choices=[2024,2025,2026])
    args=ap.parse_args()
    if not args.fold:ap.error('--fold required')
    d=load(HERE/'exp_data/all_official/processed_fortaleza.pkl')
    assert len(d['nodes_gdf'])==121
    simulate_fold(args.fold,d)
if __name__=='__main__':
    main()
