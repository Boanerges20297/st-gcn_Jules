from __future__ import annotations
import json, pickle, sys, time, copy, math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.core.architectures import get_model_class
from src.core.fortaleza_poisson_backend import FortalezaPoissonRuntime,normalize_name
from scripts.training.Active.train_all_specialists import NegBinomRankingLoss, get_active_nodes_bimestral
from scripts.validate_stgat_v5_walk_forward import RegionConfig,load_region,build_input,norm_adj,score_stgat_v5
from expand_align_30d_20261022 import dataset_subset, load
SEED=221026
HORIZON=30
TEST_DATES=['2026-09-01','2026-09-08','2026-09-15']
VAL_DATES=['2026-07-01','2026-07-15','2026-07-22','2026-08-01']
WEIGHTS=(0.0,0.2,0.3,0.4,0.5,0.7,1.0)
MODEL_DIR=OUT/'exp_models'
torch.set_num_threads(3)
torch.manual_seed(SEED)
np.random.seed(SEED)
def features_input(nf,ti,cfg):
    x=build_input(nf,ti,cfg.window,cfg.channels)
    if not np.isfinite(x).all():
        raise ValueError(f'non-finite input {ti}')
    return torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0)
def eval_raw(model,d,adj,cfg,indexlist,dates,poisson=None):
    nf=d['node_features']
    if poisson is not None:
        pass
    out={}
    model.eval()
    with torch.inference_mode():
        for ti in indexlist:
            x=features_input(nf,ti,cfg)
            raw=model(x,adj).flatten().numpy()
            out[ti]=raw.copy()
    return out
def eval_candidates(model,d,adj,cfg,ti_list):
    model.eval()
    r=[]
    nf=d['node_features']
    for ti in ti_list:
        sc=score_stgat_v5(d,model,adj,cfg,ti)
        y=nf[:,ti:ti+HORIZON,0].sum(axis=1)
        idx=np.argsort(-sc,kind='stable')
        r.append({'p10':float(np.mean(y[idx[:10]]>0)),
                  'p20':float(np.mean(y[idx[:20]]>0)),
                  'capt10':float(y[idx[:10]].sum()/max(float(y.sum()),1.)),
                  'capt20':float(y[idx[:20]].sum()/max(float(y.sum()),1.))})
    return pd.DataFrame(r)
def train_model(tag,model,df,adj,cfg,tr,vr,epochs,lr,seed=SEED):
    # Keep 41 production-inference channels, with momentum recomputed at each historical cutoff.
    nf=df['node_features']
    xs=[features_input(nf,i,cfg) for i in tr]
    ys=[torch.from_numpy(nf[:,i:i+HORIZON,0].sum(axis=1)).float() for i in tr]
    masks=[torch.from_numpy(get_active_nodes_bimestral(nf,i,cfg.window)) for i in tr]
    # This channel is negative unnormalized cold-streak derived only from past CVLI.
    cold=[]
    for i in tr:
        look=nf[:,max(0,i-31):i,0]
        c=np.zeros(nf.shape[0],dtype=np.float32)
        for k in range(look.shape[1]):
            c=np.where(look[:,k]>0,0,c+1)
        cold.append(torch.from_numpy(-np.minimum(c,30)).float())
    criterion=NegBinomRankingLoss(nb_r_init=2.,ranking_weight=2.,indecision_weight=.2,
          nb_r_min=.75,nb_r_max=50.)
    optim=torch.optim.AdamW([{'params':model.parameters(),'lr':lr,'weight_decay':.02},
        {'params':criterion.parameters(),'lr':.00005,'weight_decay':0.}],lr=lr)
    best_metric=(-1.,-1.)
    best_epoch=None
    baseline=evaluate_val(model,df,adj,cfg,vr)
    if tag=='transfer':
        best_metric=(baseline['selection_score'],baseline['capt20'])
        best_epoch=0
        best_sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        best_sd=None
    rows=[dict(tag=tag,epoch=0,lr=lr,loss=float('nan'),**baseline)]
    print('INITIAL',tag,'val',baseline,'windows',len(xs),flush=True)
    persist_learning(rows)
    wait=0
    for epoch in range(1,epochs+1):
        model.train()
        rng=np.random.default_rng(seed+epoch)
        order=rng.permutation(len(xs))
        optim.zero_grad(set_to_none=True)
        loss_sum=0.
        acc=0
        for k,j in enumerate(order):
            prediction=model(xs[j],adj).flatten()
            loss=criterion(prediction,ys[j],active_mask=masks[j],cold_streak_signal=cold[j])
            if not bool(torch.isfinite(loss)):raise RuntimeError(f'nonfinite {tag} epoch{epoch} row{j}')
            (loss/4.0).backward()
            loss_sum+=float(loss.item())
            acc+=1
            if acc>=4 or k==len(order)-1:
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(criterion.parameters()),1.)
                optim.step()
                criterion.constrain_dispersion_()
                optim.zero_grad(set_to_none=True)
                acc=0
        val=evaluate_val(model,df,adj,cfg,vr)
        now=(val['selection_score'],val['capt20'])
        # Fixed criterion: 60% P@10 + 40% P@20. Never inspect September to select epoch.
        if now[0] > best_metric[0]+.0025 or (best_metric[0]<0):
            best_metric=now
            best_epoch=epoch
            best_sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_weights(tag,best_sd,cfg,df,best_epoch,best_metric,lr,criterion.dispersion().item())
            wait=0
        else:wait+=1
        rec=dict(tag=tag,epoch=epoch,lr=lr,loss=loss_sum/max(1,len(xs)),dispersion=float(criterion.dispersion().item()),**val)
        rows.append(rec)
        persist_learning(rows)
        print('EPOCH',tag,epoch,'loss',round(rec['loss'],4),
              'valP10',round(val['p10'],3),'valP20',round(val['p20'],3),
              'score',round(val['selection_score'],4),'best_epoch',best_epoch,
              'elapsed_s',round(time.monotonic()-START,1),flush=True)
        if wait>=4:
            print('EARLY_STOP',tag,'epoch',epoch,flush=True)
            break
    if best_sd is None: raise RuntimeError('no trained checkpoint selected for '+tag)
    model.load_state_dict(best_sd,strict=True)
    save_weights(tag,best_sd,cfg,df,best_epoch,best_metric,lr,criterion.dispersion().item())
    print('SELECTED',tag,'best_epoch',best_epoch,'val_metric',best_metric,flush=True)
    return model,rows,best_epoch
def evaluate_val(model,df,adj,cfg,vr):
    r=eval_candidates(model,df,adj,cfg,vr)
    av=r.mean(numeric_only=True)
    return dict(p10=float(av['p10']),p20=float(av['p20']),
       capt10=float(av['capt10']),capt20=float(av['capt20']),
       selection_score=float(.6*av['p10']+.4*av['p20']))
def persist_learning(rows):
    # Per-branch files, saved after every epoch in research only
    if not rows:return
    tag=rows[0]['tag']
    pd.DataFrame(rows).to_csv(OUT/f'train60_threeway_{tag}_learning_20261022.csv',index=False,encoding='utf-8')
def save_weights(tag,state,cfg,data,epoch,metric,lr,dispersion):
    ckpt={'model_state_dict':state,'model_class':'DeepSTGAT_v5',
        'window':cfg.window,'in_channels':cfg.channels,'predict_horizon':HORIZON,
        'num_nodes':60,'nodes':data['nodes_gdf']['name'].tolist(),
        'train_target_end':'2026-06-30','validation_cutoffs':VAL_DATES,
        'best_epoch':int(epoch),'best_validation':metric,
        'learning_rate':lr,'dispersion_r':float(dispersion),
        'experimental_only':True,'do_not_promote':True}
    torch.save(ckpt,MODEL_DIR/f'stgat_v5_60_30d_{tag}_original_features_20261022.pth')
def score_hybrid(model,df,adj,cfg,ti,poisson):
    sc=score_stgat_v5(df,model,adj,cfg,ti)
    hist=dict(df)
    hist['node_features']=df['node_features'][:,:ti,:]
    hist['dates']=pd.DatetimeIndex(df['dates'][:ti])
    nodes=df['nodes_gdf'].copy()
    nodes['recent_cvli']=hist['node_features'][:,max(0,ti-730):ti,0].sum(axis=1)
    nodes['total_cvli']=hist['node_features'][:,:,0].sum(axis=1)
    nodes['tension_index']=0.  # matching point-in-time Poisson training (no snapshot)
    hist['nodes_gdf']=nodes
    pmap,_=FortalezaPoissonRuntime(poisson,hist).predict_scores()
    p=np.asarray([pmap[normalize_name(n)] for n in df['nodes_gdf']['name']],dtype=float)
    return sc,p
def summary_scores(name,model,df,adj,cfg,cutoffs,poisson,yall,idx,dates):
    rows=[]
    for ti in cutoffs:
        neural,p=score_hybrid(model,df,adj,cfg,ti,poisson)
        pr=pd.Series(p).rank(pct=True)
        nr=pd.Series(neural).rank(pct=True)
        y=yall[:,ti:ti+HORIZON].sum(axis=1)
        picked_y=y[np.asarray(idx,dtype=int)]
        assert len(picked_y)==len(neural)
        tot=y.sum()
        assert tot>0
        for w in WEIGHTS:
            b=(1-w)*pr+w*nr
            pos=pd.DataFrame({'blend':b,'p':p}).sort_values(['blend','p'],ascending=False,kind='mergesort').index.to_numpy()
            top10=picked_y[pos[:10]]
            top20=picked_y[pos[:20]]
            rows.append({'variant':name,'date':str(dates[ti].date()),'weight':w,'n_candidates':len(idx),
              'events_city':float(tot),'pool_capture':float(picked_y.sum()/tot),
              'p10':float((top10>0).mean()),'p20':float((top20>0).mean()),
              'capt10_city':float(top10.sum()/tot),'capt20_city':float(top20.sum()/tot),
              'false_positive10':int((top10==0).sum())})
    return rows
def main():
    global START
    START=time.monotonic()
    MODEL_DIR.mkdir(exist_ok=True)
    full=load(OUT/'exp_data/all_official/processed_fortaleza.pkl')
    orig=load(ROOT/'data/processed/processed_fortaleza.pkl')
    dates=pd.DatetimeIndex(pd.to_datetime(full['dates']))
    names=full['nodes_gdf']['name'].tolist()
    idx={n:i for i,n in enumerate(names)}
    fixed40=[idx[n] for n in orig['nodes_gdf']['name'].tolist()]
    extra=full['nodes_gdf'].drop(index=fixed40).sort_values(['recent_cvli','name'],ascending=[False,True])['name'].tolist()[:20]
    fixed60=fixed40+[idx[n] for n in extra]
    assert len(set(fixed60))==60
    d40=dataset_subset(full,fixed40)
    d60=dataset_subset(full,fixed60)
    assert d40['nodes_gdf']['name'].tolist()==orig['nodes_gdf']['name'].tolist(), 'Original40 node identity mismatch'
    _,base,_,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    # Check graph shapes and enforce strict state-transfer with no skipped parameters.
    kind=get_model_class('DeepSTGAT_v5')
    warm=kind(num_nodes=60,in_channels=cfg.channels,time_steps=cfg.window)
    warm.load_state_dict(base.state_dict(),strict=True)
    warm_start={k:v.detach().clone() for k,v in warm.state_dict().items()}
    torch.manual_seed(SEED)
    scratch=kind(num_nodes=60,in_channels=cfg.channels,time_steps=cfg.window)
    assert any(not torch.equal(v,scratch.state_dict()[k]) for k,v in warm_start.items())
    ad40=norm_adj(d40['adj_geo'],d40['adj_conflict'])
    ad60=norm_adj(d60['adj_geo'],d60['adj_conflict'])
    timap={str(d.date()):i for i,d in enumerate(dates)}
    val=[timap[d] for d in VAL_DATES]
    test=[timap[d] for d in TEST_DATES]
    assert all(dates[t+HORIZON-1]<=pd.Timestamp('2026-08-30') for t in val)
    assert all(dates[t]>=pd.Timestamp('2026-09-01') for t in test)
    tr=[i for i,t in enumerate(dates) if pd.Timestamp('2022-09-01')<=t and dates[i+HORIZON-1]<=pd.Timestamp('2026-06-30') and i>=cfg.window+60] if False else [
        i for i,t in enumerate(dates) if pd.Timestamp('2022-09-01')<=t<=pd.Timestamp('2026-06-01')
        and i>=cfg.window+60 and i+HORIZON<=len(dates) and dates[i+HORIZON-1]<=pd.Timestamp('2026-06-30')]
    tr=tr[::14]
    print('START',dates[0].date(),'..',dates[-1].date(),
          '60 candidates 40+20,',len(tr),'train cutoffs; val',VAL_DATES,'test',TEST_DATES,
          'train_last',str(dates[tr[-1]+HORIZON-1].date()),'threads',torch.get_num_threads(),flush=True)
    print('FULL_VARIABLES graph geo+conflict; 37 raw channels + 4 zero padded; production normalization per cutoff',flush=True)
    print('WARNING: static tension/faction/graph snapshots are not point-in-time historical snapshots',flush=True)
    with (OUT/'exp_models/poisson30_pit_60_20261022.pkl').open('rb') as f:
        poisson=pickle.load(f)
    # A: archived original 40. A60: same state transferred zero-shot 60.
    zero=kind(num_nodes=60,in_channels=cfg.channels,time_steps=cfg.window)
    zero.load_state_dict(warm_start,strict=True)
    print('A BASE VAL 40',evaluate_val(base,d40,ad40,cfg,val),flush=True)
    print('A BASE VAL 60',evaluate_val(zero,d60,ad60,cfg,val),flush=True)
    warm,log_b,eb=train_model('transfer',warm,d60,ad60,cfg,tr,val,epochs=9,lr=6e-5)
    scratch,log_c,ec=train_model('scratch',scratch,d60,ad60,cfg,tr,val,epochs=15,lr=2.5e-4)
    models=[('A_original40',base,d40,ad40,fixed40),
            ('A_original_transferred60',zero,d60,ad60,fixed60),
            ('B_transfer60',warm,d60,ad60,fixed60),
            ('C_scratch60',scratch,d60,ad60,fixed60)]
    rows=[]
    for name,m,df,ad,ix in models:
        rows+=summary_scores(name,m,df,ad,cfg,val,poisson,full['node_features'][:,:,0],ix,dates)
        rows+=summary_scores(name,m,df,ad,cfg,test,poisson,full['node_features'][:,:,0],ix,dates)
        print('EVALUATED',name,flush=True)
    res=pd.DataFrame(rows)
    res['period']=np.where(res['date'].isin(TEST_DATES),'holdout_sep','validation_jul_aug')
    res.to_csv(OUT/'threeway_stgat60_30d_rows_20261022.csv',index=False,encoding='utf-8-sig')
    sm=res.groupby(['period','variant','weight'],as_index=False).agg(n=('date','count'),p10=('p10','mean'),
         p20=('p20','mean'),capt10=('capt10_city','mean'),capt20=('capt20_city','mean'),
         pool_cov=('pool_capture','mean'),fp10=('false_positive10','mean'))
    sm.to_csv(OUT/'threeway_stgat60_30d_summary_20261022.csv',index=False,encoding='utf-8-sig')
    print('===HOLDOUT SEP===')
    print(sm[(sm.period=='holdout_sep') & (sm.weight.isin([0,.2,.3,.4,.7,1]))][['variant','weight','p10','capt10','capt20','pool_cov','fp10']].round(3).to_string(index=False),flush=True)
    meta={'date':datetime.now().isoformat(),'seed':SEED,'full_features':True,
      'graph':'original tactical geo/conflict 2026 snapshot','train_samples':len(tr),
      'train_start':str(dates[tr[0]].date()),'train_last_target_end':str(dates[tr[-1]+HORIZON-1].date()),
      'validation_cutoffs':VAL_DATES,'test_cutoffs':TEST_DATES,
      'source_checkpoint_train_end':'2026-06-30','transfer_best_epoch':eb,'scratch_best_epoch':ec,
      'cpu_threads':torch.get_num_threads(),
      'limitations':['spatial metadata and graphs include contemporary snapshot, historical as-of validity not fully proven',
        'top60 selected using 2025 historical totals while training dates start 2022',
        'pretrained initialization already saw data through June 2026; no independent additional training sample',
        'tiny and overlapping July/Aug validation and September holdout; not enough evidence for operational promotion',
        'original STGAT training normalization inconsistent with historical trainer, comparison uses production inference transform consistently',
        'different learning rates selected in advance to accommodate warm start vs random initialization',
        'ranking scores are not calibrated future probabilities'],
      'elapsed_seconds':round(time.monotonic()-START,1),'production_untouched':True}
    (OUT/'threeway_stgat60_30d_metadata_20261022.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('DONE_SECONDS',meta['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
