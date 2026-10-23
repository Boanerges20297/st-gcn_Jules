from __future__ import annotations
"""Study interannual CVLI tendency + short-term 60/90/120d:
2022/23 -> 2024; 2023/24 -> 2025; 2024/25 -> 2026.
No use of future labels at any test cutoff.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from expand_align_30d_20261022 import load
from rolling_origin_cvli_retrain_20261022 import rolling_indices, test_cutoffs
from aggregate_rolling_origin_cvli_only_20261022 import paired_bootstrap
OUT=HERE/'interannual_context_cvli_20261022'
PERIODS=(60,90,120)
POOLS=(40,60,121)
def pct(v):
    return pd.Series(np.asarray(v,float)).rank(pct=True,method='average').to_numpy()
def safe_share(v,pseudo=.5):
    v=np.maximum(np.asarray(v,float),0)
    return (v+pseudo)/float(v.sum()+pseudo*len(v))
def trend_features(x,dates,year):
    previous=dates.year==year-1
    before=dates.year==year-2
    assert previous.sum() in (365,366) and before.sum() in (365,366)
    a=x[:,before].sum(axis=1)
    b=x[:,previous].sum(axis=1)
    s_a=safe_share(a)
    s_b=safe_share(b)
    # B has different citywide intensity each year; compare shares, not raw totals.
    v_half=np.maximum(s_b+0.5*(s_b-s_a),1e-8)
    v_full=np.maximum(s_b+(s_b-s_a),1e-8)
    local_log_growth=np.log((b+2)/(a+2))
    local_trend=safe_share(b*np.exp(.5*np.clip(local_log_growth,-1,1)))
    return {'last_year':s_b,'annual_share_trend_half':v_half,'annual_share_trend_full':v_full,
            'annual_log_growth_half':local_trend},dict(sum_prev2=int(a.sum()),sum_prev1=int(b.sum()),
                 year_prev2=year-2,year_prev1=year-1)
def score_eval(v,ids,y):
    idx=np.asarray(ids)
    order=np.argsort(-np.asarray(v)[idx],kind='stable')
    ranked=idx[order]
    total=y.sum()
    assert total>0
    return dict(p10=float(np.mean(y[ranked[:10]]>0)),
        capt10=float(y[ranked[:10]].sum()/total),
        capt20=float(y[ranked[:20]].sum()/total),
        fp10=int((y[ranked[:10]]==0).sum()),
        coverage=float(y[idx].sum()/total),
        city_events=int(total),captured20=int(y[ranked[:20]].sum()))
def spearman(a,b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))
def test():
    full=load(HERE/'exp_data/all_official/processed_fortaleza.pkl')
    dates=pd.DatetimeIndex(pd.to_datetime(full['dates']))
    x=full['node_features'][:,:,0]
    lookup={d:i for i,d in enumerate(dates)}
    test_counts={2024:9,2025:9,2026:6}
    rows=[]
    contexts=[]
    for year in (2024,2025,2026):
        annual,detail=trend_features(x,dates,year)
        test_i=test_cutoffs(year,lookup,dates)
        assert len(test_i)==test_counts[year]
        for t in test_i:
            target=x[:,t:t+30].sum(axis=1)
            before={L:x[:,t-L:t].sum(axis=1) for L in PERIODS}
            recently=x[:,t-30:t].sum(axis=1)
            scores=dict(annual)
            scores['recent30']=recently
            for L,r in before.items():
                scores[f'recent{L}']=r
                for weight in (.25,.5,.75):
                    scores[f'blend_annualhalf_recent{L}_annual{int(weight*100):02d}'] = (
                        weight*pct(annual['annual_share_trend_half'])+(1-weight)*pct(r))
            # Choose context window from agreement (observable signal only),
            # NOT which would have predicted the next 30 days.
            agreement={L:spearman(annual['annual_share_trend_half'],safe_share(before[L])) for L in PERIODS}
            chosen=max(PERIODS,key=lambda L:(agreement[L],-abs(L-90)))
            # Annual structural trend gets more weight when short-term and
            # interannual rankings support each other.
            align=max(0.,min(1.,(agreement[chosen]+.2)/.9))
            structural_weight=.25+.5*align
            scores['adaptive_agreement']= (
                structural_weight*pct(annual['annual_share_trend_half'])
                +(1-structural_weight)*pct(before[chosen]))
            # Alternative: recent 30d support should be a strong comparator.
            scores['blend_annualhalf_recent30_annual50']= (
                .5*pct(annual['annual_share_trend_half'])+.5*pct(recently))
            info=dict(year=year,date=str(dates[t].date()),
                      available_data_before=str(dates[t-1].date()),
                      chosen_window_days=chosen,
                      chosen_annual_weight=float(structural_weight),
                      agreement60=agreement[60],agreement90=agreement[90],
                      agreement120=agreement[120],**detail)
            contexts.append(info)
            pools={'rolling40':rolling_indices(full,t,40),
                'rolling60':rolling_indices(full,t,60),'all121':list(range(121))}
            for name,v in scores.items():
                for pool,idx in pools.items():
                    rows.append(dict(year=year,date=str(dates[t].date()),
                        future_end=str(dates[t+29].date()),pool=pool,
                        strategy=name,**score_eval(v,idx,target)))
        print('ANNUAL_CUTOFF',year,detail,'n_tests',len(test_i),flush=True)
    OUT.mkdir(exist_ok=True)
    raw=pd.DataFrame(rows);c=pd.DataFrame(contexts)
    assert len(c)==24 and raw.groupby(['year','pool','strategy']).size().max()==9
    raw.to_csv(OUT/'interannual_window_rows.csv',index=False,encoding='utf-8-sig')
    c.to_csv(OUT/'selected_contexts.csv',index=False,encoding='utf-8-sig')
    raw['period']=np.where(raw.year<2026,'2024_25','2026')
    sm=raw.groupby(['period','pool','strategy'],as_index=False).agg(
        n=('date','count'),p10=('p10','mean'),capt10=('capt10','mean'),
        capt20=('capt20','mean'),fp10=('fp10','mean'),coverage=('coverage','mean'),
        events=('city_events','sum'))
    sm.to_csv(OUT/'interannual_window_summary.csv',index=False,encoding='utf-8-sig')
    stable=sm[(sm.period=='2024_25')&(sm.pool=='rolling60')].copy()
    stable['joint_utility']=.4*stable.p10+.6*stable.capt20
    print('===2024_25 TOP60 RANKED BY JOINT UTILITY===')
    print(stable.sort_values('joint_utility',ascending=False)[
        ['strategy','n','p10','capt20','coverage','joint_utility']].head(30).round(4).to_string(index=False),flush=True)
    print('===YEARLY TOP60 SELECTED STRATEGIES===')
    keep=('last_year','annual_share_trend_half','annual_share_trend_full',
          'recent30','recent60','recent90','recent120',
          'adaptive_agreement','blend_annualhalf_recent30_annual50',
          'blend_annualhalf_recent60_annual50',
          'blend_annualhalf_recent90_annual50',
          'blend_annualhalf_recent120_annual50')
    yy=raw[(raw.pool=='rolling60')&raw.strategy.isin(keep)].groupby(['year','strategy'],as_index=False).agg(n=('date','count'),p10=('p10','mean'),capt20=('capt20','mean'))
    yy.to_csv(OUT/'interannual_yearly_comparison.csv',index=False,encoding='utf-8-sig')
    print(yy[['year','strategy','p10','capt20']].round(3).to_string(index=False),flush=True)
    dev=sm[(sm.period=='2024_25')&(sm.pool=='rolling60')]
    # Discard naive global winner as evidence; predeclare selection on 2024
    # followed by 2025 holdout for out-of-sample confirmation.
    for base_year,target_year in [(2024,2025),(2025,2026)]:
        candidates=raw[(raw.year==base_year)&(raw.pool=='rolling60')]
        sums=candidates.groupby('strategy').agg(p10=('p10','mean'),capt20=('capt20','mean'))
        sums['utility']=.4*sums.p10+.6*sums.capt20
        selected=sums.sort_values(['utility','capt20'],ascending=False).index[0]
        x=yy[(yy.year==target_year)&(yy.strategy==selected)]
        if len(x):print('PREVIOUS_YEAR_SELECTED',base_year,selected,'TEST',target_year,
                      'p10',round(float(x.p10.iloc[0]),3),'capt20',round(float(x.capt20.iloc[0]),3),flush=True)
    pooled=raw[(raw.year<2026)&(raw.pool=='rolling60')]
    for strategy in ('annual_share_trend_half','adaptive_agreement',
                     'blend_annualhalf_recent90_annual50',
                     'blend_annualhalf_recent120_annual50'):
        a=pooled[pooled.strategy==strategy]
        b=pooled[pooled.strategy=='recent30']
        m=a.merge(b,on=['year','date'],suffixes=['_a','_b'])
        assert len(m)==18
        print('PAIRED_VS_RECENT30',strategy,
              {metric:paired_bootstrap(m[metric+'_a'],m[metric+'_b']) for metric in ('p10','capt20')},flush=True)
    print('===CONTEXT CHOICES===')
    print(c.groupby(['year','chosen_window_days']).agg(n=('date','count'),
       avg_interannual_alignment=('agreement90','mean')).round(3).to_string(),flush=True)
    (OUT/'interannual_metadata.json').write_text(json.dumps(
        {'source_2022_2026':True,'year_pair_prediction':[
        '2022->2023 for 2024','2023->2024 for 2025','2024->2025 for 2026'],
         'selection_for_2025':'best of 2024, not best 2025',
         'history':list(PERIODS),'primary_pool':'rolling60','time_frame':'Apr-Dec 2024-25; Apr-Sep2026',
         'future_labels_used_for_selection':False,'production_untouched':True,
         'key_caveat':'2026 CVLI source anomalously sparse',
         'score_basis':'rank and shares only; no calibrated probabilities'},
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',len(raw),'rows',flush=True)
if __name__=='__main__':test()
