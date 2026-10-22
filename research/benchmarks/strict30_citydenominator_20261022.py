from __future__ import annotations
import sys,json,hashlib,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from stgat60_pit_finetune_20261022 import (safe_graph,score,history,load)
from src.core.fortaleza_poisson_backend import (FortalezaPoissonRuntime,normalize_name)
from scripts.validate_stgat_v5_walk_forward import (RegionConfig,load_region,norm_adj)
from expand_align_30d_20261022 import dataset_subset
OUT=Path(__file__).parent
torch.set_num_threads(3)
WEIGHTS=[0,.2,.3,.4,.5,1.]
def main():
    base=load(OUT/'exp_data/all_official/processed_fortaleza.pkl')
    orig=load(ROOT/'data/processed/processed_fortaleza.pkl')
    pp=load(OUT/'exp_models/poisson30_pit_60_20261022.pkl')
    _,m,_,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    saved=torch.load(OUT/'exp_models/stgat_v5_60_30d_pit_finetuned_20261022.pth',map_location='cpu',weights_only=False)
    same=all(torch.equal(v,saved['model_state_dict'][k]) for k,v in m.state_dict().items())
    print('BEST_CKPT_IDENTICAL_TO_ORIGINAL',same,flush=True)
    assert same, 'unexpected: chosen epoch model differs; audit retraining'
    names=base['nodes_gdf']['name'].tolist()
    ids={n:i for i,n in enumerate(names)}
    fixed40=[ids[n] for n in orig['nodes_gdf']['name']]
    fixed60=fixed40+[ids[n] for n in base['nodes_gdf'][~base['nodes_gdf']['name'].isin(orig['nodes_gdf']['name'])].sort_values(['recent_cvli','name'],ascending=[False,True])['name'].head(20)]
    dates=pd.DatetimeIndex(pd.to_datetime(base['dates']))
    nf=base['node_features']
    cutoffs=[i for i,x in enumerate(dates) if pd.Timestamp('2026-07-01')<=x<=pd.Timestamp('2026-09-21') and i+30<=len(dates)][::7]
    allrows=[]
    for ti in cutoffs:
        y=nf[:,ti:ti+30,0].sum(axis=1).astype(float)
        past=nf[:,max(0,ti-730):ti,0].sum(axis=1)
        rolling=sorted(range(121),key=lambda i:(-past[i],names[i]))
        scenarios={'fixed40':fixed40,'rolling40':rolling[:40],'fixed60':fixed60,'rolling60':rolling[:60]}
        for scenario,ix in scenarios.items():
            d=dataset_subset(base,ix)
            gg,cc=safe_graph(d['nodes_gdf'])
            d['adj_geo']=gg;d['adj_conflict']=cc
            d['nodes_gdf']['tension_index']=0
            adj=norm_adj(gg,cc)
            h=history(d,ti)
            pmap,_=FortalezaPoissonRuntime(pp,h).predict_scores()
            p=np.asarray([pmap[normalize_name(n)] for n in h['nodes_gdf']['name']],dtype=float)
            neural=score(m,d['node_features'],ti,adj,cfg)
            pr=pd.Series(p).rank(pct=True)
            nr=pd.Series(neural).rank(pct=True)
            for w in WEIGHTS:
                blend=(1-w)*pr+w*nr
                order=pd.DataFrame({'b':blend,'p':p}).sort_values(['b','p'],ascending=False,kind='mergesort').index.to_numpy(dtype=int)
                yy=y[np.asarray(ix)]
                tot=y.sum()
                rr={'date':str(dates[ti].date()),'set':scenario,'weight':w,'n':len(ix),'pool_cov':float(yy.sum()/tot),
                    'p10':float(np.mean(yy[order[:10]]>0)),'p20':float(np.mean(yy[order[:20]]>0)),
                    'capt10_city':float(yy[order[:10]].sum()/tot),'capt20_city':float(yy[order[:20]].sum()/tot),
                    'fp10':int(np.count_nonzero(yy[order[:10]]==0)),
                    'events_city':float(tot)}
                allrows.append(rr)
        print('SCORED',dates[ti].date(),'CVLI',int(y.sum()),flush=True)
    df=pd.DataFrame(allrows)
    df['period']=np.where(df.date<'2026-09-01','validation_jul_aug','holdout_sep')
    df.to_csv(OUT/'strict30_citydenominator_rows_20261022.csv',index=False,encoding='utf-8-sig')
    sm=df.groupby(['period','set','weight'],as_index=False).agg(k=('date','nunique'),p10=('p10','mean'),p20=('p20','mean'),
       capt10=('capt10_city','mean'),capt20=('capt20_city','mean'),pool_cov=('pool_cov','mean'),fp10=('fp10','mean'))
    sm.to_csv(OUT/'strict30_citydenominator_summary_20261022.csv',index=False,encoding='utf-8-sig')
    for period in ('validation_jul_aug','holdout_sep'):
        print('\n'+period)
        s=sm[(sm.period==period)&(sm.weight.isin([.2,.3,.4]))]
        print(s[['set','weight','k','p10','capt10','capt20','pool_cov','fp10']].round(3).to_string(index=False),flush=True)
    assert len(df)==len(cutoffs)*4*len(WEIGHTS)
    assert ((df['capt20_city']<=df['pool_cov']+1e-9)).all()
    assert df['date'].max()<='2026-09-21'
    print('INVARIANTS PASSED rows',len(df),'checkpoint unchanged',same,flush=True)
if __name__=='__main__':main()
