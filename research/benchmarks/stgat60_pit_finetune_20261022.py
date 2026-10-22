from __future__ import annotations
import sys, json, math, time, copy, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from expand_align_30d_20261022 import dataset_subset, load
from src.core.fortaleza_poisson_backend import (build_training_frame,build_poisson_pipeline,FortalezaPoissonRuntime,FEATURE_COLS,normalize_name)
from scripts.validate_stgat_v5_walk_forward import (RegionConfig,load_region,build_input,norm_adj)
OUT=Path(__file__).parent
SEED=22026
TRAIN_END=pd.Timestamp('2026-06-30')
VAL_START=pd.Timestamp('2026-07-01')
VAL_END=pd.Timestamp('2026-07-31')
TEST_START=pd.Timestamp('2026-09-01')
TEST_END=pd.Timestamp('2026-09-21')
HORIZON=30
WEIGHTS=(0.,.2,.3,.4,.5,.7,1.)
KEEP_CHANNELS=set([0,24,28,29,30,33,34,35,36]+list(range(3,23)))
torch.set_num_threads(3)
torch.manual_seed(SEED); np.random.seed(SEED)
def safe_graph(nodes):
    lat=np.deg2rad(nodes['lat'].to_numpy(float))
    lon=np.deg2rad(nodes['long'].to_numpy(float))
    dlat=lat[:,None]-lat[None,:]
    dlon=lon[:,None]-lon[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    km=2*6371.0*np.arcsin(np.sqrt(np.clip(a,0,1)))
    geo=np.where((km<=3.0)&(km>0),1/(km+.1),0)
    np.fill_diagonal(geo,1)
    return geo.astype(np.float32),np.eye(len(nodes),dtype=np.float32)
def input_at(features,t,win,channels):
    # Crop is done inside official input builder; zero any source whose
    # availability cannot be proven at the historical cutoff.
    a=build_input(features,t,win,channels)
    x=a[0] if isinstance(a,tuple) else a
    for ch in range(x.shape[2]):
        if ch not in KEEP_CHANNELS:x[:,:,ch]=0.
    return torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0)
def score(model, features, ti, adj, cfg):
    model.eval()
    with torch.inference_mode():
        raw=model(input_at(features,ti,cfg.window,cfg.channels),adj).squeeze()
        mu=F.softplus(raw).numpy()
    neural=(mu-mu.min())/(mu.max()-mu.min()+1e-6)
    live=np.clip(features[:,max(0,ti-30):ti,0].sum(axis=1)/2.0,0,1)
    return np.clip((.5*neural+.25*live)*100,0,100)
def history(d,ti):
    out=dict(d)
    feat=d['node_features'][:,:ti,:]
    out['node_features']=feat
    out['dates']=pd.DatetimeIndex(d['dates'][:ti])
    nodes=d['nodes_gdf'].copy().reset_index(drop=True)
    nodes['tension_index']=0.0
    nodes['total_cvli']=feat[:,:,0].sum(axis=1)
    nodes['recent_cvli']=feat[:,max(0,ti-730):ti,0].sum(axis=1)
    out['nodes_gdf']=nodes
    return out
def rank_y(y,s):
    order=np.argsort(-np.asarray(s),kind='stable')
    r={}
    for k in (5,10,20):
        chosen=order[:k]
        r[f'p{k}']=float(np.count_nonzero(y[chosen]>0)/k)
        r[f'capt{k}']=float(y[chosen].sum()/max(1.,y.sum()))
    return r
def time_points(dates,start,end,step=7):
    return [i for i,x in enumerate(dates) if start<=x<=end and i+HORIZON<=len(dates)][::step]
def scores_at(model,df,ti,adj,cfg,poisson):
    st=score(model,df['node_features'],ti,adj,cfg)
    h=history(df,ti)
    pmap,_=FortalezaPoissonRuntime(poisson,h).predict_scores()
    p=np.asarray([pmap[normalize_name(n)] for n in df['nodes_gdf']['name']],dtype=float)
    return st,p
def evaluate(model,df,dates,ti_list,adj,cfg,poisson,tag):
    res=[]
    for ti in ti_list:
        y=df['node_features'][:,ti:ti+HORIZON,0].sum(axis=1).astype(float)
        s,p=scores_at(model,df,ti,adj,cfg,poisson)
        ps=pd.Series(p).rank(pct=True)
        ss=pd.Series(s).rank(pct=True)
        for w in WEIGHTS:
            blended=(1-w)*ps+w*ss
            order=pd.DataFrame({'b':blended,'p':p}).sort_values(['b','p'],ascending=False,kind='mergesort').index.to_numpy()
            rr=rank_y(y,np.arange(len(order))[np.argsort(order)]) if False else {}
            rr={'p10':np.mean(y[order[:10]]>0), 'p20':np.mean(y[order[:20]]>0),
                'capt10':float(y[order[:10]].sum()/max(y.sum(),1)),
                'capt20':float(y[order[:20]].sum()/max(y.sum(),1)),
                'fp10':int(np.count_nonzero(y[order[:10]]==0)),
                'date':str(dates[ti].date()),'model':tag,'weight':w,'events':float(y.sum())}
            res.append(rr)
    return pd.DataFrame(res)
def main():
    t0=time.monotonic()
    b=load(OUT/'exp_data/all_official/processed_fortaleza.pkl')
    a=load(ROOT/'data/processed/processed_fortaleza.pkl')
    names=a['nodes_gdf']['name'].tolist()
    extra=b['nodes_gdf'][~b['nodes_gdf']['name'].isin(names)].sort_values(['recent_cvli','name'],ascending=[False,True])['name'].tolist()[:20]
    idx={n:i for i,n in enumerate(b['nodes_gdf']['name'])}
    ix=[idx[n] for n in names+extra]
    d=dataset_subset(b,ix)
    assert len(ix)==60 and len(set(ix))==60
    orig_meta=d['nodes_gdf'][['name','recent_cvli']].copy()
    geo,conf=safe_graph(d['nodes_gdf'])
    d['adj_geo']=geo;d['adj_conflict']=conf
    d['nodes_gdf']['tension_index']=0
    dates=pd.DatetimeIndex(pd.to_datetime(d['dates']))
    print('DATA',d['node_features'].shape,'dates',dates[0].date(),dates[-1].date(),flush=True)
    print('SAFE graph edges',int(np.count_nonzero(geo)-len(ix)),'channels',sorted(KEEP_CHANNELS),flush=True)
    # Point-in-time Poisson training, including static counts at the date rather than 2026 snapshots.
    train_end_idx=int(np.searchsorted(dates,TRAIN_END,side='right'))
    datatrain=dict(d,node_features=d['node_features'][:,:train_end_idx],dates=dates[:train_end_idx])
    print('POISSON frame start',flush=True)
    frame=build_training_frame(datatrain,train_start='2022-01-01',train_end='2026-06-30',horizon_days=30,window=90)
    selected=pd.to_datetime(frame['sample_date'])
    pts=np.searchsorted(dates.values,selected.to_numpy(dtype='datetime64[ns]'))
    ni=frame['node_idx'].to_numpy(dtype=int)
    cov=d['node_features'][:,:,0]
    cum=np.pad(cov.cumsum(axis=1),((0,0),(1,0)))
    frame['total_cvli_static']=cum[ni,pts]
    frame['recent_cvli_static']=cum[ni,pts]-cum[ni,np.maximum(0,pts-730)]
    frame['tension_index']=0.
    take=(pts%14==0)
    sub=frame.loc[take]
    print('POISSON fit',len(sub),'rows','from',len(frame),flush=True)
    poisson=build_poisson_pipeline(alpha=1e-4,max_iter=350)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter('always')
        poisson.fit(sub[FEATURE_COLS],sub['target_count'])
    print('POISSON done',round(time.monotonic()-t0,1),'warn',[str(w.message)[:90] for w in warned],flush=True)
    payload={'pipeline':poisson,'region':'fortaleza','window':90,'feature_cols':FEATURE_COLS,
        'horizon_days':30,'train_range':{'start':'2022-01-01','end':'2026-06-30'},
        'point_in_time':True,'spatial_tension_masked':True}
    model_dir=OUT/'exp_models'
    model_dir.mkdir(exist_ok=True)
    import pickle
    with (model_dir/'poisson30_pit_60_20261022.pkl').open('wb') as f: pickle.dump(payload,f)
    _,m,_,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    a1=norm_adj(geo,conf)
    val=time_points(dates,VAL_START,VAL_END,step=14)
    test=time_points(dates,TEST_START,TEST_END,step=7)
    print('TIMELINE',len(val),'validation dates',[str(dates[i].date()) for i in val],'test',[str(dates[i].date()) for i in test],flush=True)
    assert max(dates[i+30-1] for i in val)<=pd.Timestamp('2026-08-29')
    assert min(dates[i] for i in test)>pd.Timestamp('2026-08-29')
    first=evaluate(m,d,dates,val,a1,cfg,payload,'before_finetune')
    print('VAL baseline P10 30%',first[first.weight==.3]['p10'].mean(),flush=True)
    train_i=[i for i,x in enumerate(dates) if pd.Timestamp('2023-01-01')<=x<=pd.Timestamp('2026-05-31') and i>=cfg.window+60 and dates[i+HORIZON-1]<=TRAIN_END]
    train_i=train_i[::21]
    inputs=[input_at(d['node_features'],i,cfg.window,cfg.channels) for i in train_i]
    targets=[torch.from_numpy(d['node_features'][:,i:i+HORIZON,0].sum(axis=1)).float() for i in train_i]
    print('TRAIN windows',len(inputs),'latest target ends',dates[train_i[-1]+29].date(),flush=True)
    optimizer=torch.optim.AdamW(m.parameters(),lr=1.8e-4,weight_decay=1e-4)
    best_sd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    best_metric=(first[first.weight==.3]['p10'].mean(), first[first.weight==.3]['capt20'].mean())
    records=[{'epoch':0,'train_loss':None,'val_p10_30':best_metric[0],'val_capt20_30':best_metric[1]}]
    for epoch in range(1,7):
        m.train()
        rng=np.random.default_rng(SEED+epoch)
        order=rng.permutation(len(inputs))
        train_loss=[]
        for j in order:
            x=inputs[j]; y=targets[j]
            optimizer.zero_grad(set_to_none=True)
            raw=m(x,a1).squeeze()
            mu=F.softplus(raw).clamp_min(1e-5)
            r=2.
            # Negative Binomial negative log-likelihood, matching checkpoint family.
            nb=(torch.lgamma(y+r)-torch.lgamma(y+1)-torch.lgamma(torch.tensor(r))
                +r*torch.log(torch.tensor(r)/(mu+r))+y*torch.log(mu/(mu+r)))
            w=1.+2.*torch.clamp(y,0,3)
            loss=-(nb*w).mean()
            pos=(y>0).nonzero(as_tuple=True)[0]
            neg=(y==0).nonzero(as_tuple=True)[0]
            if len(pos)>0 and len(neg)>0:
                n_neg=neg[torch.randperm(len(neg))[:min(12,len(neg))]]
                pair=F.relu(.15-raw[pos][:,None]+raw[n_neg][None,:]).mean()
                loss=loss+.12*pair
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss at '+str(j))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            optimizer.step()
            train_loss.append(float(loss.item()))
        ev=evaluate(m,d,dates,val,a1,cfg,payload,'validation_epoch'+str(epoch))
        row=ev[ev.weight==.3]
        now=(row['p10'].mean(),row['capt20'].mean())
        records.append({'epoch':epoch,'train_loss':float(np.mean(train_loss)),'val_p10_30':now[0],'val_capt20_30':now[1]})
        if now>best_metric:
            best_metric=now;best_sd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        print('EPOCH',epoch,'loss',round(np.mean(train_loss),4),'valP10',round(now[0],4),'valC20',round(now[1],4),'best',best_metric,'secs',round(time.monotonic()-t0,1),flush=True)
    (OUT/'stgat60_finetune_learning_20261022.csv').write_text(pd.DataFrame(records).to_csv(index=False),encoding='utf-8')
    # Evaluate on strict holdout ONLY AFTER training & model selection; preserve before model from original checkpoint.
    _,initial,_,_=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    pre=evaluate(initial,d,dates,test,a1,cfg,payload,'base_checkpoint40_transfer_to60')
    m.load_state_dict(best_sd)
    post=evaluate(m,d,dates,test,a1,cfg,payload,'stgat60_pit_finetuned')
    out=pd.concat([pre,post],ignore_index=True)
    out.to_csv(OUT/'stgat60_pit_holdout_rows_20261022.csv',index=False,encoding='utf-8-sig')
    sm=out.groupby(['model','weight'],as_index=False).agg(n=('date','nunique'),p10=('p10','mean'),capt10=('capt10','mean'),capt20=('capt20','mean'),p20=('p20','mean'),fp10=('fp10','mean'))
    sm.to_csv(OUT/'stgat60_pit_holdout_summary_20261022.csv',index=False,encoding='utf-8-sig')
    torch.save({'model_state_dict':best_sd,'model_class':'DeepSTGAT_v5','window':cfg.window,
       'in_channels':cfg.channels,'horizon_days':30,'nodes':list(d['nodes_gdf']['name']),
       'training_window':{'start':'2023-01-01','last_target':'2026-06-30'},
       'validation_end':'2026-08-29','feature_mask':sorted(KEEP_CHANNELS),
       'graph_kind':'pure_distance','val_best':best_metric,'epochs':6,
       'NOT_FOR_PRODUCTION':True},model_dir/'stgat_v5_60_30d_pit_finetuned_20261022.pth')
    meta={'generated_at':datetime.now().isoformat(),'training_count':len(train_i),'val_cutoffs':[str(dates[i].date()) for i in val],
      'holdout_cutoffs':[str(dates[i].date()) for i in test],'duration_sec':time.monotonic()-t0,
      'weights':[float(z) for z in WEIGHTS],'model_family':'STGAT v5 NegBinomial fine-tune warm start from 40-node official checkpoint',
      'leakage_protection':'cutoff-only CVLI/calendar features; geometry only, no faction/conflict or troop intelligence; point-in-time static Poisson counts',
      'poisson_window':'2022-01-01..2026-06-30','stgat_train':'2023-01-01..2026-05-31 with target <=2026-06-30',
      'known_limits':['original STGAT40 checkpoint trained on data through June 2026 but may have used historical snapshot-derived features',
        'fixed selection of nodes uses CVLI through end of 2025; fair for 2026 evaluation, selection for early training dates can be historical hindsight',
        'small September holdout: low statistical power','finetuning on CPU few sparse windows not a full hyperparameter study',
        'scikit-learn experimental environment 1.5.2'] }
    (OUT/'stgat60_pit_holdout_metadata_20261022.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('HOLDOUT SUMMARY\n'+sm.round(4).to_string(index=False),flush=True)
    print('DONE secs',round(time.monotonic()-t0,1),'saved',model_dir,flush=True)
if __name__=='__main__':main()
