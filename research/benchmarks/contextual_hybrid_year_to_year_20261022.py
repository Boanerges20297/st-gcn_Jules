from __future__ import annotations
"""Pre-year contextual neural/Poisson weight selection.
Fit policy on 2024; verify on 2025; 2026 sensitivity only.
No CVLI labels from evaluation year to choose policy.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from aggregate_rolling_origin_cvli_only_20261022 import paired_bootstrap
from interannual_signal_context_20261022 import trend_features,safe_share,spearman
from expand_align_30d_20261022 import load
HERE=Path(__file__).resolve().parent
OUT=HERE/'interannual_context_cvli_20261022'
MONTHLY=HERE/'historical_rolling_cvli_only_20261022/all_folds_rows.csv'
WINDOWS=(60,90,120)
THRESHOLDS=(-.15,0,.15,.30,.45)
RULES=('positive_to_high','positive_to_low')
WEIGHTS=(.2,.4,.7)
def contexts(full,dates):
    nf=full['node_features'][:,:,0]
    original=pd.read_csv(OUT/'selected_contexts.csv')
    lookup={str(d.date()):i for i,d in enumerate(dates)}
    records=[]
    for year in (2024,2025,2026):
        annual,_=trend_features(nf,dates,year)
        a=annual['annual_share_trend_half']
        # Recover the SIGNED change in neighborhood share between years.
        b=(dates.year==year-1)
        before=(dates.year==year-2)
        velocity=safe_share(nf[:,b].sum(axis=1))-safe_share(nf[:,before].sum(axis=1))
        for date in sorted(original[original.year==year].date):
            t=lookup[date]
            for L in WINDOWS:
                current=safe_share(nf[:,t-L:t].sum(axis=1))
                prior=safe_share(nf[:,t-2*L:t-L].sum(axis=1))
                direction=current-prior
                rows={'date':date,'fold':year,'window':L,
                  'rank_concordance_level':spearman(a,current),
                  'rank_concordance_trend':spearman(velocity,direction)}
                records.append(rows)
    return pd.DataFrame(records)
def eval_policy(df,frame,window,threshold,direction):
    c=frame[frame.window==window]
    joined=c.merge(df,on=['fold','date'])
    assert len(joined)==df[['fold','date']].drop_duplicates().shape[0]*3
    choices=[]
    for (year,date), group in joined.groupby(['fold','date']):
        context=float(group.rank_concordance_trend.iloc[0])
        if direction=='positive_to_high':
            weight=.7 if context>=threshold else .2
        else:
            weight=.2 if context>=threshold else .7
        metrics=group[group.stgat_weight==weight].iloc[0]
        choices.append({'fold':year,'date':date,'window':window,'threshold':threshold,
                        'direction':direction,'annual_signal_corr':context,'weight':weight,
                        'p10':metrics.p10,'capt20':metrics.capt20,'coverage':metrics.coverage})
    return pd.DataFrame(choices)
def summary(frame):
    return {'p10':float(frame.p10.mean()),
            'capt20':float(frame.capt20.mean()),
            'utility':float(.4*frame.p10.mean()+.6*frame.capt20.mean())}
def main():
    full=load(HERE/'exp_data/all_official/processed_fortaleza.pkl')
    dates=pd.DatetimeIndex(pd.to_datetime(full['dates']))
    frame=contexts(full,dates)
    frame.to_csv(OUT/'yearly_context_slope_alignment.csv',index=False,encoding='utf-8-sig')
    base=pd.read_csv(MONTHLY)
    base=base[(base.pool=='rolling60_stgat60_graph')&
              (base.component=='operational')&
              (base.stgat_weight.isin(WEIGHTS))]
    seen=base[base.fold==2024]
    trials=[]
    for L in WINDOWS:
        for tau in THRESHOLDS:
            for direction in RULES:
                trial=eval_policy(seen,frame,L,tau,direction)
                trials.append({'window':L,'threshold':tau,'direction':direction,
                               **summary(trial),
                               'fraction70':float(np.mean(trial.weight==.7))})
    tuning=pd.DataFrame(trials).sort_values(['utility','capt20','p10'],
             ascending=[False,False,False])
    tuning.to_csv(OUT/'context_weight_policy_candidates_2024.csv',index=False,encoding='utf-8-sig')
    print('TOP_2024_CONTEXT_POLICIES')
    print(tuning.head(12).round(4).to_string(index=False),flush=True)
    best=tuning.iloc[0]
    L=int(best.window);tau=float(best.threshold);direction=best.direction
    outputs=[]
    for year in (2024,2025,2026):
        inputdata=base[base.fold==year]
        strategy=eval_policy(inputdata,frame,L,tau,direction)
        fixed20=inputdata[inputdata.stgat_weight==.2][['fold','date','p10','capt20']].copy()
        fixed70=inputdata[inputdata.stgat_weight==.7][['fold','date','p10','capt20']].copy()
        for name,z in [('adaptive_prior_year',strategy),
                       ('fixed_20',fixed20),('fixed_70',fixed70)]:
            outputs.append({'year':year,'strategy':name,'n':len(z),**summary(z),
                            'freq70':float(np.mean(z.weight==.7)) if 'weight' in z else
                                (1. if name=='fixed_70' else 0.)})
        print('CONTEXT_SELECTION',year,strategy[['date','annual_signal_corr','weight']].round(3).to_string(index=False),flush=True)
    rep=pd.DataFrame(outputs)
    rep.to_csv(OUT/'context_policy_yearly_test_summary.csv',index=False,encoding='utf-8-sig')
    print('RESULT_YEARLY')
    print(rep.round(4).to_string(index=False),flush=True)
    hold=base[base.fold==2025]
    a=eval_policy(hold,frame,L,tau,direction)
    for benchmark in (.2,.7):
        b=hold[hold.stgat_weight==benchmark]
        q=a.merge(b,on=['fold','date'],suffixes=['_adaptive','_fixed'])
        assert len(q)==9
        print('PAIRED_2025_ADAPT_VS',benchmark,
          {k:paired_bootstrap(q[f'{k}_adaptive'],q[f'{k}_fixed'])
           for k in ('p10','capt20')},flush=True)
    meta={'train_policy_on':2024,'evaluation_on':2025,'sensitivity_on':2026,
        'selected_window':L,'selected_threshold':tau,'direction':direction,
        'training_2024_utility':float(best.utility),
        'potential_overfit':'30 policy candidates tried on 9 development months',
        'future_current_year_labels_used_to_choose_weight':False,
        'production_untouched':True,
        'reason':'Use relative neighborhood CVLI SHARE between preceding two complete years and sign of its update over past window',
        'not_a_calibrated_probability':True}
    (OUT/'context_policy_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':main()
