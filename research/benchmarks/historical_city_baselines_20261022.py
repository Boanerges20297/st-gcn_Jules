from __future__ import annotations
"""Fold-matched baselines: full-city Poisson and CVLI persistence."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from expand_align_30d_20261022 import load
from rolling_origin_cvli_retrain_20261022 import (
    HORIZON, poisson_fit, build_hist, rolling_indices, training_cutoffs, test_cutoffs)
from src.core.fortaleza_poisson_backend import FortalezaPoissonRuntime, normalize_name
def score_persist(x,t):
    score=np.zeros(x.shape[0],dtype=float)
    for d,w in [(7,.4),(14,.3),(30,.2),(60,.1)]:
        score+=w*x[:,t-d:t].sum(axis=1)
    return score
def main():
    full=load(HERE/'exp_data/all_official/processed_fortaleza.pkl')
    dates=pd.DatetimeIndex(pd.to_datetime(full['dates']))
    index={d:i for i,d in enumerate(dates)}
    names=full['nodes_gdf']['name'].tolist()
    cvli=full['node_features'][:,:,0]
    records=[]
    for year in (2024,2025,2026):
        train_end=pd.Timestamp(f'{year-1}-12-31')
        train=training_cutoffs(dates,train_end)
        poisson=poisson_fit(full,train,dates)
        for t in test_cutoffs(year,index,dates):
            hist=build_hist(full,t)
            pmap,_=FortalezaPoissonRuntime(poisson,hist).predict_scores()
            p=np.asarray([pmap[normalize_name(n)] for n in names],dtype=float)
            ewma=score_persist(cvli,t)
            look=cvli[:,max(0,t-30):t].sum(axis=1)
            y=cvli[:,t:t+HORIZON].sum(axis=1)
            tot=y.sum()
            assert tot>0
            for pool,idx in [('rolling40',rolling_indices(full,t,40)),
                             ('rolling60',rolling_indices(full,t,60)),('all121',list(range(121)))]:
                for label,s in [('poisson',p),('EWMA',ewma),('recent30',look)]:
                    order=np.argsort(-s[idx],kind='stable')
                    picked=np.asarray(idx)[order]
                    records.append(dict(
                        fold=year,date=str(dates[t].date()),pool=pool,baseline=label,
                        p10=float(np.mean(y[picked[:10]]>0)),
                        capt20=float(y[picked[:20]].sum()/tot),
                        coverage=float(y[idx].sum()/tot),
                        events_city=int(tot),captured20=int(y[picked[:20]].sum())))
        print('BASELINE_FOLD',year,'months',len(test_cutoffs(year,index,dates)),flush=True)
    df=pd.DataFrame(records)
    out=HERE/'historical_rolling_cvli_only_20261022'
    df.to_csv(out/'full_city_baselines_rows.csv',index=False,encoding='utf-8-sig')
    df['period']=np.where(df.fold<2026,'2024_2025','2026_sensitivity')
    sm=df.groupby(['period','pool','baseline'],as_index=False).agg(n=('date','count'),
          p10=('p10','mean'),capt20=('capt20','mean'),coverage=('coverage','mean'))
    sm.to_csv(HERE/'rolling_origin_cvli_only_baselines_20261022.csv',index=False,encoding='utf-8-sig')
    print('===STABLE YEARS BASELINES===')
    print(sm[sm.period=='2024_2025'][['pool','baseline','p10','capt20','coverage']].round(4).to_string(index=False),flush=True)
    old=pd.read_csv(out/'all_folds_rows.csv')
    old=old[(old.fold<2026)&(old.pool=='rolling60_stgat60_graph')&
            (old.component=='operational')&(old.stgat_weight==0)]
    same=df[(df.fold<2026)&(df.pool=='rolling60')&(df.baseline=='poisson')]
    chk=old.merge(same,on=['fold','date'],suffixes=['_old','_new'])
    assert len(chk)==18
    assert np.allclose(chk.p10_old,chk.p10_new)
    assert np.allclose(chk.capt20_old,chk.capt20_new)
    print('BASELINE_PARITY_PASS',len(chk),flush=True)
if __name__=='__main__':main()
