from __future__ import annotations
"""Verify and evaluate sealed prospective CVLI rankings when 30d labels mature."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN_DIR=OUT/"shadow_20261022"
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()
def checked_manifest():
    m=json.loads((RUN_DIR/"manifest.json").read_text(encoding="utf-8"))
    ranked=RUN_DIR/m["rankings_file"]
    actual=sha(ranked)
    assert actual==m["rankings_sha256"],"SEALED RANKING MODIFIED -- STOP"
    data=pd.read_csv(ranked,encoding="utf-8-sig")
    assert len(data)==m["ranking_rows"]
    assert data["target_start"].nunique()==data["target_end"].nunique()==1
    assert data["target_start"].iloc[0]==m["target_start"]
    assert data["target_end"].iloc[0]==m["target_end"]
    assert set(data["weight_stgat"].unique())==set([0.0,0.2,0.4,0.7,1.0])
    for (variant,w), part in data.groupby(["variant","weight_stgat"]):
        assert sorted(part["rank"].tolist())==list(range(1,len(part)+1))
        assert part["bairro"].is_unique
        assert part["candidate_pool_size"].nunique()==1
        assert len(part)==part["candidate_pool_size"].iloc[0]
    assert data["variant"].nunique()==4
    # Verify source archive persisted without mutation; this is optional for future
    # evaluation (scores themselves are fully sealed in the CSV).
    archive=OUT/m["archive_files_relative_to_benchmarks"]
    verified=0
    for fn,digest in m["archive_sha256"].items():
        path=archive/fn
        if path.exists():
            assert sha(path)==digest,f"ARCHIVED MODEL/DATA CHANGED: {fn}"
            verified+=1
    print("INTEGRITY_OK", "rankings",len(data),"archive_files_verified",verified,
          "ranking_sha256",actual,flush=True)
    return m,data
def evaluate(full_path:Path,manifest,rankings):
    import pickle
    start=pd.Timestamp(manifest["target_start"])
    end=pd.Timestamp(manifest["target_end"])
    ready=pd.Timestamp(manifest["labels_mature_after"])
    now=pd.Timestamp.now().normalize()
    if now<ready:
        print("NOT_READY_CLOCK",str(now.date()),"earliest",str(ready.date()),
              "-- no outcome assessment performed",flush=True)
        return
    if not full_path.is_file():
        print("NOT_READY_SOURCE_MISSING",str(full_path),flush=True)
        return
    with full_path.open("rb") as f:
        d=pickle.load(f)
    names=d["nodes_gdf"]["name"].tolist()
    assert len(names)==len(set(names))==121,"Expected complete 121 Fortaleza official neighborhoods"
    dates=pd.DatetimeIndex(pd.to_datetime(d["dates"]))
    assert dates.is_unique and dates.is_monotonic_increasing
    if dates[-1]<ready:
        print("NOT_READY_SERIES",str(dates[-1].date()),"expected_on_or_after",
              str(ready.date()),flush=True)
        return
    expected=pd.date_range(start,end,freq="D")
    present=dates[(dates>=start)&(dates<=end)]
    assert len(present)==30 and present.equals(expected),"Future label dates incomplete or noncontiguous"
    nf=d["node_features"]
    assert nf.shape[0]==121 and nf.shape[1]==len(dates)
    counts=nf[:,(dates>=start)&(dates<=end),0]
    assert np.isfinite(counts).all() and (counts>=0).all()
    city_events=counts.sum(axis=1)
    if city_events.sum()<=0:
        raise RuntimeError("Zero CVLI in whole city; verify ingestion/completeness before judging")
    city_total=float(city_events.sum())
    byname=dict(zip(names,city_events))
    city_mean=float(city_total/121)
    results=[]
    for (variant,weight),group in rankings.groupby(["variant","weight_stgat"],sort=True):
        ordered=group.sort_values("rank")
        names_this=ordered["bairro"].tolist()
        assert set(names_this)<=set(names)
        y=np.asarray([byname[n] for n in names_this])
        y10=y[:10]
        y20=y[:20]
        r=dict(
            variant=variant,
            weight_stgat=float(weight),
            weight_poisson=1-float(weight),
            selected_bairros=len(names_this),
            bairro_positives_top10=int((y10>0).sum()),
            p10=float((y10>0).mean()),
            captured_events_top10=float(y10.sum()),
            captured_events_top20=float(y20.sum()),
            capture10_city=float(y10.sum()/city_total),
            capture20_city=float(y20.sum()/city_total),
            pool_coverage=float(y.sum()/city_total),
            false_positives_top10=int((y10==0).sum()),
            all121_events=city_total,
            labels_data_end=str(dates[-1].date()),
            target_start=str(start.date()),
            target_end=str(end.date()),
            source_file_sha256=sha(full_path),
        )
        assert 0<=r["capture20_city"]<=r["pool_coverage"]+1e-9
        results.append(r)
    result=pd.DataFrame(results)
    source_digest=sha(full_path)
    assessment=RUN_DIR/f"assessment_after_{ready.date()}_{source_digest[:12]}.csv"
    with assessment.open("x",encoding="utf-8-sig",newline="") as f:
        result.to_csv(f,index=False)
    diag=(result[result.weight_stgat.isin(manifest["main_weights_stgat"])]
          .sort_values(["variant","weight_stgat"]))
    print("PROSPECTIVE_RESULTS",str(assessment),"CVLI_city",city_total,flush=True)
    print(diag[["variant","weight_stgat","p10","capture20_city","pool_coverage","false_positives_top10"]]
          .round(3).to_string(index=False),flush=True)
    print("NO_MODEL_PROMOTION_OR_DEPLOY",flush=True)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrity-only",action="store_true")
    ap.add_argument("--all121-processed",type=Path,default=OUT/"exp_data/all_official/processed_fortaleza.pkl")
    args=ap.parse_args()
    manifest,ranked=checked_manifest()
    if args.integrity_only:
        return
    evaluate(args.all121_processed,manifest,ranked)
if __name__=="__main__":
    main()
