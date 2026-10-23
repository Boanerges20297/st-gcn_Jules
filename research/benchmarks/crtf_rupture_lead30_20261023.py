"""CRTF lead-30 diagnostic on CVLI-corrected, out-of-time ST-GAT checkpoints.
This is a retrospective ranking evaluation, NOT a fitted change-point model.
Forecast is issued at t. Future confirmation windows are t+30..59 and t+60..89;
there is a 30-day lead buffer t..29. Main label requires >=2 CVLI in EACH
future 30-day window, each city's smoothed local share >=1.35x historic share
(t-120..t-1). 'Delayed' additionally requires no already-elevated share in the
first 30-day buffer.
The 60 candidate nodes are selected ONLY from data before t.
Results are aggregate only; no personal records or neighborhood rankings exported.
"""
from __future__ import annotations
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "research/benchmarks"
HERE = BENCH / "crtf_rupture_lead30_20261023"
TR = BENCH / "cvli_corrected_training_20261023"
SRC = ROOT / "data/runtime/research/cvli_rebuilt_20261022/all_official/processed_fortaleza.pkl"
HASH = "aa32f5fe829ab8345d68e67e2a19523f3886d8740c2f2b392943d29324807d24"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT))
import rolling_origin_cvli_retrain_20261022 as engine
torch.set_num_threads(3)
rng = np.random.default_rng(261023)
def checked_load(year, source_hash):
    path = TR / "exp_models" / f"stgat_v5_scratch_cvli_corrected_fold_{year}.pth"
    payload = torch.load(path, map_location="cpu")
    assert payload["source_tensor_sha256"] == source_hash
    assert payload["train_end"] == f"{year-1}-12-31"
    assert payload["weights_not_from_production"]
    cls = engine.get_model_class(payload["model"])
    model = cls(num_nodes=60, in_channels=payload["in_channels"],
                time_steps=payload["time_steps"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with (TR / "exp_models" / f"poisson_cvli_corrected_fold_{year}.pkl").open("rb") as handle:
        poisson = pickle.load(handle)
    assert poisson["source_tensor_sha256"] == source_hash
    return model, poisson
def share(counts, k):
    return (counts + .5) / (float(counts.sum()) + .5 * k)
def targets(nf, t):
    K = nf.shape[0]
    hist = nf[:, t-120:t].sum(axis=1)
    buffer = nf[:, t:t+30].sum(axis=1)
    f1 = nf[:, t+30:t+60].sum(axis=1)
    f2 = nf[:, t+60:t+90].sum(axis=1)
    sh = share(hist, K)
    pbuf = share(buffer, K)
    future1 = share(f1, K)
    future2 = share(f2, K)
    def sustained(factor):
        return (f1 >= 2) & (f2 >= 2) & (future1 >= factor*sh) & (future2 >= factor*sh)
    mainstream = sustained(1.35)
    return {
      "sustained135": mainstream,
      "delayed135": mainstream & (pbuf < 1.35*sh),
      "dormant135": mainstream & (hist <= 1),
      "sustained120": sustained(1.20),
      "sustained150": sustained(1.50),
    }
def score_methods(full, t, pool, model, poisson):
    nf = full["node_features"][:,:,0]
    K = nf.shape[0]
    h120 = nf[:,t-120:t].sum(axis=1)
    h90 = nf[:,t-90:t].sum(axis=1)
    h60 = nf[:,t-60:t].sum(axis=1)
    h30 = nf[:,t-30:t].sum(axis=1)
    hprev90 = nf[:,t-120:t-30].sum(axis=1)
    neural, operational = engine.stgat_raw(model, full, pool, t)
    hist = engine.build_hist(full, t)
    scored, _details = engine.FortalezaPoissonRuntime(poisson, hist).predict_scores()
    names = full["nodes_gdf"]["name"].tolist()
    from src.core.fortaleza_poisson_backend import normalize_name
    p_all = np.asarray([scored.get(str(n).strip(), scored.get(normalize_name(n), float("nan"))) for n in names])
    assert np.isfinite(p_all).all(), "Poisson mapping incomplete"
    p_pool = p_all[pool]
    growth = np.log(share(h30,K) / share(hprev90,K))
    neural_rank = pd.Series(neural).rank(pct=True).to_numpy()
    history_rank = pd.Series(h120[pool]).rank(pct=True).to_numpy()
    poisson_rank = pd.Series(p_pool).rank(pct=True).to_numpy()
    recent30_rank = pd.Series(h30[pool]).rank(pct=True).to_numpy()
    recent60_rank = pd.Series(h60[pool]).rank(pct=True).to_numpy()
    oper_rank = pd.Series(operational).rank(pct=True).to_numpy()
    return {
       "recency30": h30[pool],
       "recency60": h60[pool],
       "recency90": h90[pool],
       "recency120": h120[pool],
       "share_momentum_30_vs_90": growth[pool],
       "poisson": p_pool,
       "stgat_raw": neural,
       "stgat_operational": operational,
       "stgat_surprise_vs_120": neural_rank - history_rank,
       # Counterfactual controls: novelty can be a mathematical property of
       # negating old risk, rather than a learned emerging-violence signal.
       "inverse_recency120": -h120[pool],
       "inverse_recency30": -h30[pool],
       "poisson_surprise_vs_120": poisson_rank - history_rank,
       "recency30_surprise_vs_120": recent30_rank - history_rank,
       "recency60_surprise_vs_120": recent60_rank - history_rank,
       "operational_surprise_vs_120": oper_rank - history_rank,
    }
def rank(array):
    return np.argsort(-np.asarray(array, dtype=float), kind="stable")
def main():
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == HASH, "Tensor provenance mismatch"
    full = engine.load(SRC)
    K = len(full["nodes_gdf"])
    assert K == 121
    dates = pd.DatetimeIndex(pd.to_datetime(full["dates"]))
    dates_ix = {str(v.date()): i for i,v in enumerate(dates)}
    nf = full["node_features"][:,:,0].astype(float)
    rows=[]
    eval_count=0
    checked_30d = 0
    for year in (2023,2024,2025,2026):
        model, poisson = checked_load(year, HASH)
        previous = pd.read_csv(TR / f"fold_{year}_rows.csv")
        cuts = previous["date"].drop_duplicates().tolist()
        for day in cuts:
            t = dates_ix[day]
            if t < 120 or t+90>len(dates):
                continue
            if year in (2023,2024,2025) and dates[t+89].year != year:
                continue
            if year == 2026 and dates[t+89] >= pd.Timestamp("2026-09-01"):
                continue
            pool = np.asarray(engine.rolling_indices(full,t,60),dtype=int)
            assert len(set(pool)) == 60
            scores = score_methods(full,t,pool,model,poisson)
            # Confirm that the replayed ranking uses the exact saved holdout model.
            immediate = nf[:,t:t+30].sum(axis=1)
            for method,comp in (("poisson",0.0),("stgat_operational",1.0)):
                saved = previous.loc[
                    (previous["date"] == day) & (previous["component"] == "operational") &
                    (previous["pool"] == "rolling60_stgat60_graph") &
                    np.isclose(previous["stgat_weight"],comp)]
                assert len(saved) == 1
                got=float(np.mean(immediate[pool[rank(scores[method])[:10]]]>0))
                assert abs(got-float(saved.iloc[0]["p10"])) <= 1e-8, (
                    "Replay mismatch", year,day,method,got,float(saved.iloc[0]["p10"]))
                checked_30d += 1
            labels = targets(nf,t)
            for key,label in labels.items():
                positives_city=int(label.sum())
                positives_pool=int(label[pool].sum())
                for method,values in scores.items():
                    order=pool[rank(values)]
                    hit10=int(label[order[:10]].sum())
                    hit20=int(label[order[:20]].sum())
                    ap=(float(average_precision_score(label[pool],values))
                        if positives_pool else float("nan"))
                    rows.append({
                        "year":year,"issue_date":day,"target":key,
                        "method":method,"positives_city":positives_city,
                        "positives_pool":positives_pool,
                        "hits10":hit10,"hits20":hit20,
                        "precision10":hit10/10,
                        "recall10_city":hit10/positives_city if positives_city else float("nan"),
                        "recall20_city":hit20/positives_city if positives_city else float("nan"),
                        "average_precision_pool":ap,
                        "random_expected_hits10":positives_pool/6,
                        "valid_date":True
                    })
            eval_count += 1
            print("CUT",year,day,"future_sustained",int(labels["sustained135"].sum()),
                  "delayed",int(labels["delayed135"].sum()),
                  "in_pool",int(labels["sustained135"][pool].sum()),flush=True)
    HERE.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    assert eval_count == 24 and checked_30d == 2 * eval_count, (eval_count,checked_30d)
    assert df.groupby(["year","issue_date","target"]).size().eq(15).all()
    df.to_csv(HERE / "lead30_method_by_cutoff_aggregates.csv",index=False)
    summary=df.groupby(["target","method","year"],as_index=False).agg(
       n=("issue_date","nunique"),positive_city=("positives_city","sum"),
       positive_pool=("positives_pool","sum"),
       hits10=("hits10","sum"),hits20=("hits20","sum"),
       mean_ap=("average_precision_pool","mean"),
       mean_p10=("precision10","mean"))
    summary["recall10_city"]=summary["hits10"]/summary["positive_city"].replace(0,np.nan)
    summary["recall20_city"]=summary["hits20"]/summary["positive_city"].replace(0,np.nan)
    summary.to_csv(HERE / "lead30_method_by_year.csv",index=False)
    primary=df[(df["year"] != 2026)&(df["target"].isin(("sustained135","delayed135")))]
    pooled=primary.groupby(["target","method"],as_index=False).agg(
       n=("issue_date","nunique"),positive_city=("positives_city","sum"),
       positive_pool=("positives_pool","sum"),hits10=("hits10","sum"),
       hits20=("hits20","sum"),mean_ap=("average_precision_pool","mean"))
    pooled["recall10_city"]=pooled["hits10"]/pooled["positive_city"]
    pooled["recall20_city"]=pooled["hits20"]/pooled["positive_city"]
    pooled.to_csv(HERE/"lead30_primary_pooled_2023_2025.csv",index=False)
    meta={
       "tensor_sha256":HASH,"model":"DeepSTGAT_v5 scratch with frozen fold checkpoints",
       "folds":[2023,2024,2025,2026],"main_analysis_years":[2023,2024,2025],
       "2026_caveat":"Sep 2026 incomplete vs SSPDS; only windows ending by Aug 2026 included",
       "city_bairros":121,"historical_window_days":120,"lead_buffer_days":30,
       "confirmation_period_1_days":"31-60","confirmation_period_2_days":"61-90",
       "min_deaths_each_confirmation_window":2,
       "share_growth_ratio":1.35,"beta_smoothing_per_territory":0.5,
       "delayed_label_requires_buffer_below_threshold":True,
       "replay_30day_p10_checks":checked_30d,
       "no_future_data_in_scores":True,"no_person_data_or_territory_rankings_exported":True,
       "post_hoc_proxy_not_original_training_target":True,
       "controls_added":"inverse history, recency-rank innovation, Poisson innovation, operational innovation",
       "overlapping_future_windows_across_monthly_cutoffs":True,
       "geometry":"121 bairros; model ranks only the top60 CVLI-730d history pool",
       "2023_training_protocol":"2022-only 7-day stride (23 training windows), other years 21-day stride",
       "caution":"Proxy episodes are not independently certified real causal regime change"
    }
    (HERE/"protocol.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf8")
    print("===PRIMARY POOLED 2023-25===",flush=True)
    print(pooled[["target","method","n","positive_city","positive_pool","hits10","hits20","recall10_city","recall20_city","mean_ap"]].round(3).to_string(index=False),flush=True)
    print("===PER YEAR CORE===",flush=True)
    print(summary[(summary.target=="sustained135") &
      (summary.method.isin(["recency120","poisson","stgat_raw","stgat_operational",
                          "share_momentum_30_vs_90","stgat_surprise_vs_120"]))][
       ["year","method","n","positive_city","hits10","hits20","mean_ap"]].round(3).to_string(index=False),flush=True)
    print("COMPLETE",eval_count,"CUTOFFS",checked_30d,"REPLAY_CHECKS",flush=True)
if __name__ == "__main__":
    main()
