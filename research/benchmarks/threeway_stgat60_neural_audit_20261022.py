from pathlib import Path
import json,pickle,sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.core.architectures import get_model_class
from scripts.validate_stgat_v5_walk_forward import RegionConfig,load_region,norm_adj,score_stgat_v5
from expand_align_30d_20261022 import dataset_subset,load
from threeway_stgat60_30d_20261022 import features_input, TEST_DATES, VAL_DATES
torch.set_num_threads(3)
def metrics_at(y,ids,local_scores):
    order=np.argsort(-np.asarray(local_scores),kind='stable')
    tot=y.sum()
    return dict(p10=float(np.mean(y[np.asarray(ids)[order[:10]]]>0)),
                capture10=float(y[np.asarray(ids)[order[:10]]].sum()/tot),
                capture20=float(y[np.asarray(ids)[order[:20]]].sum()/tot))
def main():
    full=load(OUT/'exp_data/all_official/processed_fortaleza.pkl')
    active=load(ROOT/'data/processed/processed_fortaleza.pkl')
    names=full['nodes_gdf']['name'].tolist();dic={n:i for i,n in enumerate(names)}
    id40=[dic[n] for n in active['nodes_gdf']['name']]
    extra=full['nodes_gdf'].drop(index=id40).sort_values(['recent_cvli','name'],ascending=[False,True])['name'].tolist()[:20]
    id60=id40+[dic[n] for n in extra]
    df40=dataset_subset(full,id40)
    df60=dataset_subset(full,id60)
    dates=pd.DatetimeIndex(pd.to_datetime(full['dates']))
    date_index={str(x.date()):i for i,x in enumerate(dates)}
    _,base,_,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    models={'A_original40':(base,df40,id40)}
    kind=get_model_class('DeepSTGAT_v5')
    zero=kind(num_nodes=60,in_channels=cfg.channels,time_steps=cfg.window)
    zero.load_state_dict(base.state_dict(),strict=True)
    models['A_original_transferred60']=(zero,df60,id60)
    for tag,name in [('transfer','B_transfer60'),('scratch','C_scratch60')]:
        ckpt=torch.load(OUT/'exp_models'/f'stgat_v5_60_30d_{tag}_original_features_20261022.pth',map_location='cpu',weights_only=False)
        assert ckpt['experimental_only'] is True
        model=kind(num_nodes=60,in_channels=cfg.channels,time_steps=cfg.window)
        model.load_state_dict(ckpt['model_state_dict'],strict=True)
        models[name]=(model,df60,id60)
        print('CHECKPOINT',tag,'epoch',ckpt['best_epoch'],'validation',ckpt['best_validation'],flush=True)
    nf=full['node_features'][:,:,0]
    rows=[]
    for period,days in [('validation',VAL_DATES),('holdout',TEST_DATES)]:
        for dt in days:
            ti=date_index[dt]
            y=nf[:,ti:ti+30].sum(axis=1)
            coverage40=y[id40].sum()/y.sum()
            coverage60=y[id60].sum()/y.sum()
            print('WINDOW',dt,'total CVLI city',y.sum(),'Top40',y[id40].sum(),'Top60',y[id60].sum(),flush=True)
            for tag,(m,df,ix) in models.items():
                adj=norm_adj(df['adj_geo'],df['adj_conflict'])
                x=features_input(df['node_features'],ti,cfg)
                m.eval()
                with torch.no_grad():
                    raw=m(x,adj).flatten()
                    neural=F.softplus(raw).cpu().numpy()
                op=score_stgat_v5(df,m,adj,cfg,ti)
                for score_type,score in [('neural_raw',neural),('operational_stgat',op)]:
                    z=metrics_at(y,ix,score)
                    rows.append(dict(period=period,date=dt,variant=tag,score_type=score_type,
                          pool_cov=float(y[ix].sum()/y.sum()),city_events=float(y.sum()),**z))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'threeway_stgat60_neural_audit_rows_20261022.csv',index=False,encoding='utf-8-sig')
    sm=out.groupby(['period','variant','score_type'],as_index=False).agg(n=('date','nunique'),
             p10=('p10','mean'),capt10=('capture10','mean'),capt20=('capture20','mean'),pool_cov=('pool_cov','mean'))
    sm.to_csv(OUT/'threeway_stgat60_neural_audit_summary_20261022.csv',index=False,encoding='utf-8-sig')
    print('===HOLDOUT NEURAL vs OPERATIONAL===')
    print(sm[sm.period=='holdout'][['variant','score_type','p10','capt20','pool_cov']].round(3).to_string(index=False),flush=True)
    # Fixed 60 scenario is defined by 2025 records; no re-selection with Sept labels.
    assert all(names[i] in set(active['nodes_gdf']['name']) for i in id40)
    assert all((out['pool_cov']<=1+1e-9)&(out['capture20']<=out['pool_cov']+1e-9))
    f=OUT/'threeway_stgat60_30d_summary_20261022.csv'
    hybrid=pd.read_csv(f)
    print('===HOLDOUT BY THREE CUTOFFS, 70% STGAT===')
    z=pd.read_csv(OUT/'threeway_stgat60_30d_rows_20261022.csv')
    print(z[(z['date'].isin(TEST_DATES)) & (z['weight']==.7)][['variant','date','p10','capt20_city','pool_capture','events_city']].to_string(index=False),flush=True)
    val=hybrid[(hybrid.period=='validation_jul_aug') & (hybrid.weight.isin([.2,.4,.7]))]
    print('===VALIDATION HYBRID===')
    print(val[['variant','weight','p10','capt20']].round(3).to_string(index=False),flush=True)
    print('AUDIT_PASS',len(out),'rows')
if __name__=='__main__':main()
