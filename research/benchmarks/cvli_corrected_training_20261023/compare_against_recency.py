"""Compare corrected scratch models against simple CVLI recency controls.
Every test month uses only data before its issue date for candidate selection
and baseline scoring; labels are exactly the following 30 days.
"""
from __future__ import annotations
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/"research/benchmarks"
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BENCH))
from expand_align_30d_20261022 import load
from rolling_origin_cvli_retrain_20261022 import rolling_indices
SRC=ROOT/"data/runtime/research/cvli_rebuilt_20261022/all_official/processed_fortaleza.pkl"
H=30
def metrics(y, chosen):
    total=float(y.sum())
    return {"p10":float(np.mean(y[chosen[:10]]>0)),
            "capt20":float(y[chosen[:20]].sum()/total) if total>0 else float("nan"),
            "city_deaths":int(total),
            "captured20":int(y[chosen[:20]].sum()),
            "fp10":int(np.sum(y[chosen[:10]]==0))}
def main():
    full=load(SRC)
    names=full["nodes_gdf"]["name"].tolist()
    assert len(names)==121
    nf=full["node_features"]
    dates=pd.DatetimeIndex(pd.to_datetime(full["dates"]))
    idx={stamp:i for i,stamp in enumerate(dates)}
    rows=[]
    model_sets={}
    for fold in (2024,2025,2026):
        report_path=OUT/f"fold_{fold}_rows.csv"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        fr=pd.read_csv(report_path)
        model_sets[fold]=fr
        test_dates=sorted(fr["date"].unique().tolist())
        for date in test_dates:
            t=idx[pd.Timestamp(date)]
            assert t+H<=len(dates)
            pool=np.array(rolling_indices(full,t,60),dtype=int)
            target=nf[:,t:t+H,0].sum(axis=1).astype(float)
            assert target.sum()>0
            for horizon in (30,60,90,120):
                prev=nf[:,max(0,t-horizon):t,0].sum(axis=1)
                sorted_ids=pool[np.argsort(-prev[pool],kind="stable")]
                row={"fold":fold,"date":date,"component":"recency",
                     "method":f"CVLI_{horizon}d","pool":"rolling60_stgat60_graph"}
                row.update(metrics(target,sorted_ids))
                rows.append(row)
            for comp,weight,method in [
                ("operational",0.0,"Poisson"),
                ("operational",0.7,"STGAT_70_Poisson_30"),
                ("operational",1.0,"STGAT_operational"),
                ("neural_raw",1.0,"STGAT_neural_raw")]:
                sel=fr[(fr["date"]==date)
                    &(fr["pool"]=="rolling60_stgat60_graph")
                    &(fr["component"]==comp)
                    &(np.isclose(fr["stgat_weight"],weight))]
                if len(sel)!=1:
                    raise ValueError((fold,date,comp,weight,len(sel)))
                r=sel.iloc[0]
                rows.append({"fold":fold,"date":date,"component":comp,
                  "method":method,"pool":r["pool"],"p10":float(r["p10"]),
                  "capt20":float(r["capt20"]), "city_deaths":int(r["city_events"]),
                  "captured20":int(r["captured20"]), "fp10":int(r["false_positives10"])})
    allrows=pd.DataFrame(rows)
    allrows.to_csv(OUT/"comparison_corrected_cvli_rows.csv",index=False)
    summary=allrows.groupby(["fold","method"],as_index=False).agg(
        n=("date","nunique"),p10=("p10","mean"),capt20=("capt20","mean"),
        city_deaths=("city_deaths","sum"),captured20=("captured20","sum"),
        fp10=("fp10","mean"))
    summary["observed_death_capture_ratio"]=summary["captured20"]/summary["city_deaths"]
    summary.to_csv(OUT/"comparison_corrected_cvli_by_fold.csv",index=False)
    pooled=allrows.groupby("method",as_index=False).agg(
        n=("date","nunique"),p10=("p10","mean"),capt20=("capt20","mean"),
        city_deaths=("city_deaths","sum"),captured20=("captured20","sum"),
        fp10=("fp10","mean"))
    pooled["observed_death_capture_ratio"]=pooled["captured20"]/pooled["city_deaths"]
    pooled.to_csv(OUT/"comparison_corrected_cvli_pooled.csv",index=False)
    print("===BY_FOLD===",flush=True)
    print(summary[summary["method"].isin(("Poisson","STGAT_operational","STGAT_neural_raw","STGAT_70_Poisson_30","CVLI_120d"))][["fold","method","n","p10","capt20"]].round(4).to_string(index=False),flush=True)
    print("===POOLED===",flush=True)
    print(pooled[["method","n","p10","capt20","observed_death_capture_ratio"]].round(4).to_string(index=False),flush=True)
    print("DONE",flush=True)
if __name__=="__main__":
    main()
