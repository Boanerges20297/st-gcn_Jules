from __future__ import annotations
import json, warnings, pickle, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core.fortaleza_poisson_backend import load_payload, FortalezaPoissonRuntime
from scripts.validate_stgat_v5_walk_forward import RegionConfig, load_region, score_stgat_v5, score_ewma
OUT=Path(__file__).parent
OUT.mkdir(parents=True,exist_ok=True)
WEIGHTS=[0,.10,.20,.30,.40,.50,.60,.70,.80,.90,1.]
HORIZON=14
START=pd.Timestamp('2026-07-01')
END=pd.Timestamp('2026-10-08')
K=(5,10,20)
def measure(order,y):
    pos=int(np.count_nonzero(y>0)); events=float(np.sum(y))
    res={}
    for k in K:
        chosen=order[:k]
        hits=int(np.count_nonzero(y[chosen]>0))
        res['p'+str(k)]=hits/k
        res['r'+str(k)]=hits/pos if pos else 0
        res['capture'+str(k)]=float(np.sum(y[chosen]))/events if events else 0
        res['false'+str(k)]=k-hits
    return res
def main():
    warnings.filterwarnings('ignore',category=UserWarning,module='sklearn')
    data,model,adj,cfg=load_region(RegionConfig('fortaleza','fortaleza_model_active.pth','processed_fortaleza.pkl',120))
    payload=load_payload(ROOT/'models/active/production/poisson/fortaleza_poisson_regressor.pkl')
    dates=pd.DatetimeIndex(pd.to_datetime(data['dates']))
    nf=np.asarray(data['node_features'])
    names=data['nodes_gdf']['name'].tolist()
    assert len(names)==len(set(names))
    idxs=[i for i,d in enumerate(dates) if START<=d<=END and i>=cfg.window+60 and i+HORIZON<=len(dates)]
    idxs=idxs[::14]
    print('RUN',len(idxs),'cutoffs',dates[idxs[0]].date(),dates[idxs[-1]].date(),'nodes',len(names),'ckpt horizon 30d; Poisson 14d',flush=True)
    rows=[]
    detail=[]
    for ti in idxs:
        cutoff=dates[ti]
        y=nf[:,ti:ti+HORIZON,0].sum(axis=1).astype(float)
        if y.sum()<=0: continue
        hist=dict(data)
        hist['node_features']=nf[:,:ti,:]
        hist['dates']=dates[:ti]
        nodes=data['nodes_gdf'].copy().reset_index(drop=True)
        nodes['total_cvli']=nf[:,:ti,0].sum(axis=1)
        nodes['recent_cvli']=nf[:,max(0,ti-30):ti,0].sum(axis=1)
        hist['nodes_gdf']=nodes
        pmap,_=FortalezaPoissonRuntime(payload,hist).predict_scores()
        # Same native ST-GAT inference implementation and score postprocessing as official orchestrator
        stgat=score_stgat_v5(hist,model,adj,cfg,ti)
        from src.core.fortaleza_poisson_backend import normalize_name
        names_norm=[normalize_name(name) for name in names]
        if len(set(names_norm))!=len(names_norm): raise RuntimeError('Nomes territorialmente ambiguos')
        p=np.asarray([pmap[n] for n in names_norm],dtype=float)
        if not (np.isfinite(p).all() and np.isfinite(stgat).all()): raise RuntimeError('Score nao finito')
        p_pct=pd.Series(p).rank(pct=True)
        s_pct=pd.Series(stgat).rank(pct=True)
        baseline=sorted(range(len(y)),key=lambda j:(-float(p[j]),j))
        ewma=score_ewma(hist,ti)
        extra={'Poisson':np.array(baseline,dtype=int),
               'STGAT':np.lexsort(( -p,-s_pct.to_numpy())).astype(int),
               'EWMA':np.argsort(-ewma,kind='stable')}
        for w in WEIGHTS:
            blend=(1-w)*p_pct+w*s_pct
            order=pd.DataFrame({'b':blend,'p':p}).sort_values(['b','p'],ascending=False,kind='mergesort').index.to_numpy()
            extra[f'ST{int(w*100):03}']=order
        ref=set(extra['ST020'][:10])
        period='sep_oct' if cutoff>=pd.Timestamp('2026-09-01') else 'jul_aug'
        for label,order in extra.items():
            m=measure(order,y)
            m.update(model=label,date=cutoff.strftime('%Y-%m-%d'),period=period,
                     events=float(y.sum()),positive_locations=int(np.count_nonzero(y>0)),
                     top10_overlap_ref=len(ref.intersection(order[:10]))/10.0)
            rows.append(m)
            detail.append(dict(date=m['date'],model=label,top10=' | '.join(names[j] for j in order[:10])))
        print('DONE',cutoff.date(),'events',int(y.sum()),'positives',int(np.count_nonzero(y)),flush=True)
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('Sem janelas')
    grp=df.groupby(['period','model'],as_index=False).agg(
      windows=('date','nunique'), p5=('p5','mean'),p10=('p10','mean'),p20=('p20','mean'),
      r10=('r10','mean'),r20=('r20','mean'),capture10=('capture10','mean'),
      capture20=('capture20','mean'),false10=('false10','mean'),
      overlap=('top10_overlap_ref','mean'))
    grp2=df.assign(period='all').groupby(['period','model'],as_index=False).agg(
      windows=('date','nunique'),p5=('p5','mean'),p10=('p10','mean'),p20=('p20','mean'),
      r10=('r10','mean'),r20=('r20','mean'),capture10=('capture10','mean'),
      capture20=('capture20','mean'),false10=('false10','mean'),
      overlap=('top10_overlap_ref','mean'))
    sm=pd.concat([grp2,grp],ignore_index=True)
    stamp='20261022_stride14'
    (OUT/f'window_metrics_{stamp}.csv').write_text(df.to_csv(index=False),encoding='utf-8')
    (OUT/f'summary_{stamp}.csv').write_text(sm.round(6).to_csv(index=False),encoding='utf-8')
    (OUT/f'top10_{stamp}.csv').write_text(pd.DataFrame(detail).to_csv(index=False),encoding='utf-8')
    (OUT/f'metadata_{stamp}.json').write_text(json.dumps({'generated_at':datetime.now().isoformat(),'dates':[str(dates[i].date()) for i in idxs], 'weights':WEIGHTS,
    'poisson_train':payload.get('train_range'),'poisson_horizon':payload.get('horizon_days'),'stgat_horizon':30,
    'stgat_training_end':'2026-06-30','stgat_validation_end':'2026-08-29','horizon_eval':HORIZON,
    'candidate_nodes':len(names),'evaluation_universe':'40 bairros do processed_fortaleza.pkl',
    'limitations':['Tension index e grafo do snapshot de 21/10 podem conter sinais posteriores aos cutoffs',
       'Checkpoints treinados com horizontes diferentes (Poisson 14, STGAT 30)',
       'STGAT usa dados jul-ago em selecao/validacao; sep-out mais confiavel',
       'Poisson foi salvo com sklearn 1.8 mas ambiente atual 1.5.2; warning de compatibilidade',
       'janelas de 14 dias a cada 7 dias sao sobrepostas'],
    'sources':['src/core/orchestrator.py','src/core/fortaleza_poisson_backend.py',
               'scripts/validate_stgat_v5_walk_forward.py']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nSUMMARY',flush=True)
    for period in ('all','sep_oct','jul_aug'):
        d=sm[sm.period==period].sort_values('model')
        print('\n'+period+' '+str(len(d))+' candidates',flush=True)
        print(d[['model','windows','p10','r10','capture10','p20','capture20','overlap']].round(3).to_string(index=False),flush=True)
    print('\nOUTPUT',OUT,flush=True)
if __name__=='__main__': main()
