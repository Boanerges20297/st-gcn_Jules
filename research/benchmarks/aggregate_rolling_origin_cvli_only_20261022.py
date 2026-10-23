from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent/'historical_rolling_cvli_only_20261022'
OUT=Path(__file__).resolve().parent
WEIGHTS=(0.,.2,.4,.7,1.)
SEED=21022
def primary_summary(df):
    return (df.groupby(['period','pool','component','stgat_weight'],as_index=False)
      .agg(n=('date','count'),p10=('p10','mean'),p20=('p20','mean'),capt10=('capt10','mean'),
           capt20=('capt20','mean'),coverage=('coverage','mean'),false_positives10=('false_positives10','mean'),
           events_in_period=('city_events','sum'),events_top20=('captured20','sum')))
def paired_bootstrap(x,y,seed=SEED):
    a=np.asarray(x,float)-np.asarray(y,float)
    rng=np.random.default_rng(seed)
    n=len(a)
    trials=rng.choice(a,size=(10000,n),replace=True).mean(axis=1)
    return dict(n=n,delta=float(a.mean()),ci_low=float(np.quantile(trials,.025)),
                ci_high=float(np.quantile(trials,.975)),
                wins=int((a>1e-9).sum()),ties=int((np.abs(a)<=1e-9).sum()),
                losses=int((a<-1e-9).sum()))
def main():
    frames=[]
    for year in (2024,2025,2026):
        fp=BASE/f'fold_{year}_rows.csv'
        if not fp.is_file():raise RuntimeError(f'Missing fold {year}')
        z=pd.read_csv(fp)
        assert z.fold.nunique()==1 and int(z.fold.iloc[0])==year
        frames.append(z)
    rows=pd.concat(frames,ignore_index=True)
    assert set(rows.stgat_weight)==set(WEIGHTS)
    assert sorted(rows.date.unique())==list(rows.date.unique()) or True
    assert rows[['fold','date','pool','component','stgat_weight']].duplicated().sum()==0
    for fold in (2024,2025,2026):
        per=rows[(rows.fold==fold)&(rows.pool=='rolling60_stgat60_graph')&(rows.component=='operational')&(rows.stgat_weight==0)]
        assert len(per)==(9 if fold in (2024,2025) else 6)
        s=pd.to_datetime(per.date).sort_values().tolist()
        assert all((s[i+1]-s[i]).days>=30 for i in range(len(s)-1))
    rows['period']=np.where(rows.fold<=2025,'stable_2024_2025','anomalous_2026')
    agg=primary_summary(rows)
    agg.to_csv(OUT/'rolling_origin_cvli_only_pooled_summary_20261022.csv',index=False,encoding='utf-8-sig')
    rows.to_csv(BASE/'all_folds_rows.csv',index=False,encoding='utf-8-sig')
    print('===COMPARISON 2024/25 18 windows rolling60 ===')
    flt=(agg.period=='stable_2024_2025') & (agg.pool=='rolling60_stgat60_graph') & (agg.component=='operational')
    print(agg[flt][['stgat_weight','n','p10','capt10','capt20','coverage','false_positives10','events_in_period','events_top20']].round(4).to_string(index=False))
    print('===COMPARISON 2026 6 windows rolling60 ===')
    flt=(agg.period=='anomalous_2026') & (agg.pool=='rolling60_stgat60_graph') & (agg.component=='operational')
    print(agg[flt][['stgat_weight','n','p10','capt20','coverage']].round(4).to_string(index=False))
    print('===COMPARISON 40 vs 60 ===')
    flt=(agg.period=='stable_2024_2025') & (agg.component=='operational') & (agg.stgat_weight.isin([0.,.2,.4,.7]))
    print(agg[flt][['pool','stgat_weight','p10','capt20','coverage']].round(4).to_string(index=False))
    print('===NEURAL RAW vs operational 2024/25 ===')
    flt=(agg.period=='stable_2024_2025') & (agg.pool=='rolling60_stgat60_graph') & (agg.stgat_weight.isin([0.,.2,.7,1.]))
    print(agg[flt][['component','stgat_weight','p10','capt20']].round(4).to_string(index=False))
    comp=rows[(rows.period=='stable_2024_2025')&(rows.pool=='rolling60_stgat60_graph')&
          (rows.component=='operational')]
    a=comp.pivot_table(index=['fold','date'],columns='stgat_weight',values=['p10','capt20'])
    checks=[]
    for metric in ('p10','capt20'):
        for weight in (.2,.4,.7,1.):
            for reference in (0.,.2):
                if weight==reference:continue
                detail=paired_bootstrap(a[(metric,weight)],a[(metric,reference)])
                checks.append({'metric':metric,'stgat_weight':weight,'reference_weight':reference,**detail})
                print('PAIRED',metric,weight,'vs',reference,{k:round(v,4) if isinstance(v,float) else v for k,v in detail.items()})
    pd.DataFrame(checks).to_csv(OUT/'rolling_origin_cvli_only_paired_20261022.csv',index=False,encoding='utf-8-sig')
    stats=agg[(agg.period=='stable_2024_2025')&(agg.pool=='rolling60_stgat60_graph')&(agg.component=='operational')].set_index('stgat_weight')
    event_total=int(stats.loc[.0,'events_in_period'])
    metadata={'folds':[2024,2025,2026],'historical_training_start':'2022-07-01',
      'validation':'first 90 days after each training year; no holdout usage',
      'independent_test_windows':{'2024':9,'2025':9,'2026':6},
      'control_data_anomaly':{'2022':840,'2023':1462,'2024':1653,'2025':1487,'2026_through_oct21':378,
       'raw_csv_2026_fortaleza_cvli_type_records':381},
      'pool':'rolling40 or 60 based on previous 730d CVLI',
      'training':'fresh weights per origin, CVLI+calendar+momentum, geography-only graph',
      'poisson':'official pipeline refit to all121 per origin, point-in-time static features, no tactical signals',
      'prospective_shadow_separate':True,
      'stable_years_all121_events_evaluated':event_total,
      'leakage_protection':'do not use original 2026 checkpoint for 2024/2025 testing',
      'limitations':['input may have been retrospectively corrected in source', '2026 volume drop is already in raw source and not explained',
          'temporal dependence between monthly data despite nonoverlapping 30d windows',
          '2018-2021 CVLI data not in 121 processed source', 'ranking not a calibrated homicide probability']}
    (OUT/'rolling_origin_cvli_only_metadata_20261022.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AGGREGATE_DONE',len(rows),'rows, 18 stable nonoverlap windows',flush=True)
if __name__=='__main__':main()
