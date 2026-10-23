"""Episode-level independence check for the CRTF lead-30 proxy experiment."""
from collections import defaultdict
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "research/benchmarks"
OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(BENCH))
import crtf_rupture_lead30_20261023 as exp
def candidate_events():
    full = exp.engine.load(exp.SRC)
    times = pd.DatetimeIndex(pd.to_datetime(full["dates"]))
    index = {str(x.date()):i for i,x in enumerate(times)}
    nf = full["node_features"][:,:,0].astype(float)
    groups = defaultdict(list)
    for year in (2023,2024,2025):
        model,poisson = exp.checked_load(year,exp.HASH)
        saved = pd.read_csv(exp.TR/f"fold_{year}_rows.csv")
        for day in saved.date.drop_duplicates():
            t = index[day]
            if t+90>len(times) or times[t+89].year!=year:
                continue
            pool=np.asarray(exp.engine.rolling_indices(full,t,60),dtype=int)
            labels=exp.targets(nf,t)
            scores=exp.score_methods(full,t,pool,model,poisson)
            priority={method:set(pool[exp.rank(values)[:10]]) for method,values in scores.items()}
            for target in ("sustained135","delayed135"):
                for node in np.where(labels[target])[0]:
                    record={"year":year,"target":target,"day":day,"node":int(node),
                            "in_pool":bool(node in pool)}
                    record.update({m:bool(node in places) for m,places in priority.items()})
                    groups[(year,target,int(node))].append(record)
    return groups
def summarize_episode_groups(grouped):
    episodes=[]
    methods=("stgat_surprise_vs_120","operational_surprise_vs_120",
        "share_momentum_30_vs_90","poisson_surprise_vs_120",
        "recency30","recency120","stgat_operational","stgat_raw")
    for (year,target,node), records in grouped.items():
        records=sorted(records,key=lambda r:r["day"])
        one=[]
        for record in records:
            if one and (pd.Timestamp(record["day"])-pd.Timestamp(one[-1]["day"])).days>40:
                episodes.append(one)
                one=[]
            one.append(record)
        if one: episodes.append(one)
    rows=[]
    for target in ("sustained135","delayed135"):
        ep=[group for group in episodes if group[0]["target"]==target]
        for method in methods:
            rows.append({
                "target":target,"method":method,
                "episodes":len(ep),
                "first_issue_top10_hits":sum(group[0][method] for group in ep),
                "any_issue_top10_hits":sum(any(r[method] for r in group) for group in ep),
                "episode_first_issue_pool_coverage":sum(group[0]["in_pool"] for group in ep),
                "unique_episode_territory_year_groups":len(set((g[0]["year"],g[0]["node"]) for g in ep)),
            })
    return pd.DataFrame(rows)
def paired_control_diagnostics():
    df=pd.read_csv(OUT/"lead30_method_by_cutoff_aggregates.csv")
    focus=df[(df.year.isin((2023,2024,2025)))&(df.target=="sustained135")]
    base=focus[focus.method=="stgat_surprise_vs_120"].sort_values(["year","issue_date"])
    pmf=np.asarray([1.])
    for k in base.positives_pool:
        x=hypergeom.pmf(np.arange(min(int(k),10)+1),60,int(k),10)
        pmf=np.convolve(pmf,x)
    obs=int(base.hits10.sum())
    chance={"target":"sustained135","city_events":int(base.positives_city.sum()),
        "pool_events":int(base.positives_pool.sum()),"monthly_cutoffs":len(base),
        "stgat_surprise_hits10":obs,
        "random_top10_expected_hits":float(sum(base.positives_pool)/6),
        "single_method_one_sided_conditional_null_p":float(pmf[obs:].sum()),
        "warning":"Unadjusted for selecting 15 methods and multiple event labels post hoc; monthly target windows overlap"}
    out=[]
    for alternative in ("share_momentum_30_vs_90","inverse_recency30","poisson_surprise_vs_120",
                        "operational_surprise_vs_120","recency30"):
        other=focus[focus.method==alternative].sort_values(["year","issue_date"])
        assert other.issue_date.tolist()==base.issue_date.tolist()
        deltas=base.hits10.to_numpy()-other.hits10.to_numpy()
        rng=np.random.default_rng(261023)
        sampled=rng.integers(0,len(deltas),size=(20000,len(deltas)))
        totals=deltas[sampled].sum(axis=1)
        out.append({
            "control":alternative,"stgat_surprise_hits":int(base.hits10.sum()),
            "control_hits":int(other.hits10.sum()),"difference":int(deltas.sum()),
            "unadjusted_date_resample_ci_low":float(np.quantile(totals,.025)),
            "unadjusted_date_resample_ci_high":float(np.quantile(totals,.975)),
            "per_year_diff":{str(y):int(sum(deltas[base.year.to_numpy()==y])) for y in (2023,2024,2025)}
        })
    (OUT/"lead30_statistical_caveats.json").write_text(
        json.dumps({"random_null":chance,"paired_controls":out},indent=2,ensure_ascii=False),encoding="utf-8")
    print("STATISTICAL_NULL",chance,flush=True)
    for row in out:print("PAIRED_CONTROL",row,flush=True)
def main():
    episodes=summarize_episode_groups(candidate_events())
    episodes.to_csv(OUT/"lead30_independent_episode_summary.csv",index=False)
    print("EPISODE_SUMMARY",flush=True)
    print(episodes.to_string(index=False),flush=True)
    paired_control_diagnostics()
if __name__=="__main__":main()
