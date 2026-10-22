from pathlib import Path
import pandas as pd
import pickle
import numpy as np
root = Path(__file__).resolve().parent
a = pd.read_csv(root / 'expand_align_30d_rows_20261022.csv')
b = pd.read_csv(root / 'rolling_risk_rows_20261022.csv').rename(columns={'weight':'stgat_weight'})
df = pd.concat([a,b], ignore_index=True)
periods = {
    'nonoverlap_A': ['2026-07-01','2026-08-05','2026-09-09'],
    'nonoverlap_B': ['2026-07-08','2026-08-12','2026-09-16']
}
blocks = []
for panel, days in periods.items():
    x = df[df.date.isin(days)].copy()
    z = x.groupby(['mesh','stgat_weight'],as_index=False).agg(
        n=('date','count'),p10=('p10','mean'),capt20=('capt20','mean'),
        capt10=('capt10','mean'),pool_capture=('pool_capture','mean'))
    z['panel']=panel
    blocks.append(z)
    report = z[
        ((z.mesh=='fixed40') & (z.stgat_weight==.2)) |
        ((z.mesh=='fixed60') & (z.stgat_weight.isin([.2,.4]))) |
        ((z.mesh=='rolling40') & (z.stgat_weight==.3)) |
        ((z.mesh=='all121') & (z.stgat_weight==.4))
    ]
    print('\n'+panel+'\n'+report[['mesh','stgat_weight','p10','capt20','pool_capture']].round(3).to_string(index=False))
pd.concat(blocks).to_csv(root/'nonoverlap_30d_20261022.csv',index=False,encoding='utf-8-sig')
with open(root/'exp_data/all_official/processed_fortaleza.pkl','rb') as f:
    d = pickle.load(f)
with open(root/'exp_data/top40_rebuild/processed_fortaleza.pkl','rb') as f:
    old = pickle.load(f)
names = d['nodes_gdf']['name'].tolist()
index = {name:i for i,name in enumerate(names)}
old40=[index[n] for n in old['nodes_gdf']['name']]
extra=d['nodes_gdf'].drop(index=old40).sort_values(['recent_cvli','name'],ascending=[False,True])['name'].tolist()
old60=old40+[index[n] for n in extra[:20]]
dates=pd.to_datetime(d['dates'])
series=d['node_features'][:,:,0]
records=[]
for month in sorted(set(dates.to_period('M'))):
    if str(month)<'2026-01':
        continue
    mask=np.asarray(dates.to_period('M')==month)
    y=series[:,mask].sum(axis=1)
    records.append(dict(month=str(month),all121=int(y.sum()),fixed40=int(y[old40].sum()),fixed60=int(y[old60].sum()),added20=int(y[old60].sum()-y[old40].sum())))
print('\nMONTHLY INDEPENDENT EVENTS\n',pd.DataFrame(records).to_string(index=False))
pd.DataFrame(records).to_csv(root/'monthly_coverage_20261022.csv',index=False,encoding='utf-8-sig')
