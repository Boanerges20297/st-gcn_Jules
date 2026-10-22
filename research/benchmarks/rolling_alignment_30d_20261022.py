from __future__ import annotations
import sys, pickle, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from expand_align_30d_20261022 import dataset_subset,to_runtime,rows_report,load
from src.core.fortaleza_poisson_backend import FortalezaPoissonRuntime,normalize_name
from scripts.validate_stgat_v5_walk_forward import RegionConfig,load_region,score_stgat_v5,norm_adj
OUT=Path(__file__).parent
torch.set_num_threads(4)
def main():
 t0=time.perf_counter()
 base=load(OUT/'exp_data/all_official/processed_fortaleza.pkl')
 payload=load(OUT/'exp_models/poisson30_all121_stride14.pkl')
 _,old_model,_,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
 ckpt=old_model.state_dict()
 dates=pd.DatetimeIndex(pd.to_datetime(base['dates']))
 nf=base['node_features']
 names=base['nodes_gdf']['name'].tolist()
 timesteps=[i for i,d in enumerate(dates) if pd.Timestamp('2026-07-01')<=d<=pd.Timestamp('2026-09-21') and i+30<=len(dates)][::7]
 rows=[]; names_by_period={}
 for ti in timesteps:
  lookback=nf[:,max(0,ti-730):ti,0].sum(axis=1)
  candidates=sorted(range(121),key=lambda x:(-lookback[x],names[x]))
  y=nf[:,ti:ti+30,0].sum(axis=1).astype(float)
  for kind,ix in [('rolling40',candidates[:40]),('rolling60',candidates[:60])]:
   data=dataset_subset(base,ix)
   h=to_runtime(data,ti)
   pmap,_=FortalezaPoissonRuntime(payload,h).predict_scores()
   p=np.asarray([pmap[normalize_name(name)] for name in h['nodes_gdf']['name']])
   model=type(old_model)(num_nodes=len(ix),in_channels=cfg.channels,time_steps=cfg.window).cpu()
   model.load_state_dict(ckpt)
   model.eval()
   adj=norm_adj(data['adj_geo'],data['adj_conflict'])
   s=score_stgat_v5(h,model,adj,cfg,ti)
   rp=pd.Series(p).rank(pct=True)
   rs=pd.Series(s).rank(pct=True)
   for w in (0.,.2,.3,.4,.5,.7,1.):
    b=(1-w)*rp+w*rs
    order=pd.DataFrame(dict(b=b,p=p)).sort_values(['b','p'],ascending=False,kind='mergesort').index.to_numpy(dtype=int)
    row=rows_report(order,y,np.array(ix))
    row.update(date=str(dates[ti].date()),mesh=kind,weight=w,period='sep' if dates[ti].month==9 else 'jul_aug')
    rows.append(row)
   names_by_period[f'{dates[ti].date()}_{kind}']=[names[z] for z in ix]
  print('DONE',dates[ti].date(),'sec',round(time.perf_counter()-t0,1),flush=True)
 df=pd.DataFrame(rows)
 for period,subset in [('all',df),('sep',df[df.period=='sep'])]:
  summ=subset.groupby(['mesh','weight'],as_index=False).agg(n=('date','nunique'),p10=('p10','mean'),capt10=('capt10','mean'),capt20=('capt20','mean'),pool_capture=('pool_capture','mean'))
  summ.to_csv(OUT/f'rolling_risk_{period}_20261022.csv',index=False,encoding='utf-8-sig')
  print(period, '\n'+summ.round(3).to_string(index=False),flush=True)
 df.to_csv(OUT/'rolling_risk_rows_20261022.csv',index=False,encoding='utf-8-sig')
 (OUT/'rolling_selected_bairros_20261022.json').write_text(json.dumps(names_by_period,indent=2,ensure_ascii=False),encoding='utf-8')
 print('OK',round(time.perf_counter()-t0,1),flush=True)
if __name__=='__main__':main()
