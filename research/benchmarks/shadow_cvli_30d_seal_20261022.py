from __future__ import annotations
"""Seal prospective CVLI neighborhood rankings. Does not score future outcomes."""
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from expand_align_30d_20261022 import dataset_subset, load
from src.core.architectures import get_model_class
from src.core.fortaleza_poisson_backend import FortalezaPoissonRuntime, normalize_name
from scripts.validate_stgat_v5_walk_forward import RegionConfig, load_region, norm_adj, score_stgat_v5
from threeway_stgat60_30d_20261022 import features_input
RUN_ID = "shadow_20261022"
AS_OF = pd.Timestamp("2026-10-21")
HORIZON = 30
PRIMARY_WEIGHTS = (0.20, 0.40, 0.70)
DIAGNOSTIC_WEIGHTS = (0.0, 1.0)
WEIGHTS = (0.0, 0.20, 0.40, 0.70, 1.0)
LABEL_GRACE_DAYS = 7
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        f.write(content)
def main() -> None:
    torch.set_num_threads(3)
    model_dir = OUT / "exp_models"
    base_pickle = OUT / "exp_data/all_official/processed_fortaleza.pkl"
    active40_pickle = ROOT / "data/processed/processed_fortaleza.pkl"
    source_ckpt = ROOT / "models/active/legacy_torch/fortaleza_model_active.pth"
    transfer_ckpt = model_dir / "stgat_v5_60_30d_transfer_original_features_20261022.pth"
    scratch_ckpt = model_dir / "stgat_v5_60_30d_scratch_original_features_20261022.pth"
    poisson_ckpt = model_dir / "poisson30_pit_60_20261022.pkl"
    sources = {
        "input_all_121.pkl": base_pickle,
        "input_active40.pkl": active40_pickle,
        "stgat_active40.pth": source_ckpt,
        "stgat_transfer60.pth": transfer_ckpt,
        "stgat_scratch60.pth": scratch_ckpt,
        "poisson30.pkl": poisson_ckpt,
    }
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    shadow_dir = OUT / RUN_ID
    archive_dir = OUT / "exp_data" / RUN_ID
    if shadow_dir.exists() or archive_dir.exists():
        raise FileExistsError(f"Shadow snapshot exists. Must not overwrite: {shadow_dir} / {archive_dir}")
    full = load(base_pickle)
    active40 = load(active40_pickle)
    dates = pd.DatetimeIndex(pd.to_datetime(full["dates"]))
    assert dates.is_unique and dates.is_monotonic_increasing
    assert dates[-1] == AS_OF, f"unexpected last date {dates[-1]}"
    assert pd.DatetimeIndex(pd.to_datetime(active40["dates"]))[-1] == AS_OF
    issue_date = AS_OF + pd.Timedelta(days=1)
    target_end = issue_date + pd.Timedelta(days=HORIZON-1)
    assert (target_end - issue_date).days == 29
    assert len(full["nodes_gdf"]) == 121
    allnames = full["nodes_gdf"]["name"].tolist()
    assert len(set(allnames)) == 121
    lookup = {name:i for i,name in enumerate(allnames)}
    old = active40["nodes_gdf"]["name"].tolist()
    assert len(old) == len(set(old)) == 40
    fixed40 = [lookup[name] for name in old]
    # Only 2024/2025 CVLI controls who enters the fixed Top-60.
    historic = (dates >= pd.Timestamp("2024-01-01")) & (dates <= pd.Timestamp("2025-12-31"))
    cvli = full["node_features"][:,historic,0].sum(axis=1)
    remaining = [name for name in allnames if name not in set(old)]
    extra = sorted(remaining, key=lambda name: (-float(cvli[lookup[name]]), name))[:20]
    fixed60 = fixed40 + [lookup[name] for name in extra]
    assert len(set(fixed60)) == 60
    assert np.array_equal(full["node_features"][fixed40,:,0], active40["node_features"][:,:,0]), "CVLI count series diverged"
    ckpt_b = torch.load(transfer_ckpt, map_location="cpu", weights_only=False)
    ckpt_c = torch.load(scratch_ckpt, map_location="cpu", weights_only=False)
    assert ckpt_b["nodes"] == ckpt_c["nodes"] == old+extra
    assert ckpt_b["experimental_only"] and ckpt_c["experimental_only"]
    assert ckpt_b["best_epoch"] == 7 and ckpt_c["best_epoch"] == 1
    _, baseline, _, cfg = load_region(RegionConfig("fortaleza", "fortaleza_model_active.pth", "processed_fortaleza.pkl", 120))
    kind = get_model_class("DeepSTGAT_v5")
    models = {
        "A_original40": (baseline, fixed40),
    }
    untrained60 = kind(num_nodes=60, in_channels=cfg.channels, time_steps=cfg.window)
    untrained60.load_state_dict(baseline.state_dict(), strict=True)
    models["A_zero_shot60"] = (untrained60, fixed60)
    for checkpoint, name in [(ckpt_b, "B_transfer60"), (ckpt_c, "C_scratch60")]:
        m = kind(num_nodes=60, in_channels=cfg.channels, time_steps=cfg.window)
        m.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[name] = (m, fixed60)
    with poisson_ckpt.open("rb") as f:
        poisson = pickle.load(f)
    assert poisson["horizon_days"] == HORIZON
    assert poisson["train_range"]["end"] == "2026-06-30"
    assert cfg.window == 120 and cfg.channels == 41
    ti = len(dates)
    rankings = []
    raw_features = full["node_features"]
    for label, (model, ix) in models.items():
        data = dataset_subset(full, ix)
        adj = norm_adj(data["adj_geo"], data["adj_conflict"])
        model.eval()
        with torch.inference_mode():
            raw = model(features_input(data["node_features"], ti, cfg), adj).flatten()
            neural = F.softplus(raw).numpy()
        operational = score_stgat_v5(data, model, adj, cfg, ti)
        # Poisson's static CVLI covariates must be known by cutoff.
        hist = dict(data)
        hist["node_features"] = data["node_features"][:,:ti,:]
        hist["dates"] = dates[:ti]
        nodes = data["nodes_gdf"].copy()
        nodes["total_cvli"] = hist["node_features"][:,:,0].sum(axis=1)
        nodes["recent_cvli"] = hist["node_features"][:,max(0,ti-730):ti,0].sum(axis=1)
        nodes["tension_index"] = 0.0  # Poisson fitted without tactical tension.
        hist["nodes_gdf"] = nodes
        pmap, _ = FortalezaPoissonRuntime(poisson, hist).predict_scores()
        poisson_scores = np.asarray([pmap[normalize_name(name)] for name in data["nodes_gdf"]["name"]], dtype=float)
        ps = pd.Series(poisson_scores).rank(pct=True)
        ss = pd.Series(operational).rank(pct=True)
        names = data["nodes_gdf"]["name"].tolist()
        for weight in WEIGHTS:
            combined = (1.0 - weight) * ps + weight * ss
            order = (pd.DataFrame({"blend":combined,"poisson":poisson_scores})
                     .sort_values(["blend","poisson"], ascending=False, kind="mergesort").index.to_numpy())
            assert sorted(order.tolist()) == list(range(len(ix)))
            for rank, i in enumerate(order, 1):
                rankings.append({
                    "issue_date": str(issue_date.date()),
                    "cutoff_end": str(AS_OF.date()),
                    "target_start": str(issue_date.date()),
                    "target_end": str(target_end.date()),
                    "variant":label,
                    "weight_stgat":float(weight),
                    "weight_poisson":float(1-weight),
                    "rank":rank,
                    "bairro":names[i],
                    "hybrid_percentile":float(combined.iloc[i]),
                    "poisson_score":float(poisson_scores[i]),
                    "stgat_operational_score":float(operational[i]),
                    "neural_mean_uncalibrated":float(neural[i]),
                    "candidate_pool_size":len(ix),
                })
        print("SNAPSHOT",label,"candidate_pool",len(ix),
              "top10_70/30",[
                  row["bairro"] for row in rankings if row["variant"]==label
                  and row["weight_stgat"]==.70 and row["rank"]<=10
              ],flush=True)
    ranks = pd.DataFrame(rankings)
    assert len(ranks)==(40+60+60+60)*len(WEIGHTS)
    assert len(ranks.groupby(["variant","weight_stgat"]))==20
    assert not ranks.select_dtypes(include="number").isna().any().any()
    source_hashes={key:sha256(p) for key,p in sources.items()}
    archive_dir.mkdir(parents=True)
    frozen_hashes={}
    for name, src in sources.items():
        dst=archive_dir / name
        with src.open("rb") as f, dst.open("xb") as w:
            shutil.copyfileobj(f,w,length=1024*1024)
        frozen_hashes[name]=sha256(dst)
        assert frozen_hashes[name]==source_hashes[name]
        print("FROZEN",name,dst.stat().st_size,"bytes",flush=True)
    shadow_dir.mkdir()
    ranks_path=shadow_dir/"sealed_rankings.csv"
    with ranks_path.open("x",encoding="utf-8-sig",newline="") as fp:
        ranks.to_csv(fp,index=False)
    csv_sha=sha256(ranks_path)
    manifest={
        "id":RUN_ID,
        "purpose":"prospective locked neighborhood CVLI ranking, not live allocations",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "cutoff_end":str(AS_OF.date()),
        "target_start":str(issue_date.date()),
        "target_end":str(target_end.date()),
        "labels_mature_after":str((target_end+pd.Timedelta(days=LABEL_GRACE_DAYS)).date()),
        "horizon_days":HORIZON,
        "universe_neighborhoods":121,
        "source_base_last_date":str(dates[-1].date()),
        "origin_snapshot_cvli_active40_identical":True,
        "selected_top40":old,
        "selected_added20":extra,
        "selection_window":"2024-01-01 through 2025-12-31 only",
        "models":[{"name":lab,"candidate_count":len(ix)} for lab,(_,ix) in models.items()],
        "main_weights_stgat":PRIMARY_WEIGHTS,
        "diagnostic_weights_stgat":DIAGNOSTIC_WEIGHTS,
        "ranking_rows":len(ranks),
        "rankings_file":ranks_path.name,
        "rankings_sha256":csv_sha,
        "archive_files_relative_to_benchmarks":str(archive_dir.relative_to(OUT)).replace("\\","/"),
        "archive_sha256":frozen_hashes,
        "post_hoc_weight_change_forbidden":True,
        "future_cvli_labels_not_used":True,
        "caveats":[
            "Tactical metadata are contemporary as of October 2026, not time-travel historical snapshots",
            "ST-GAT scores and hybrid percentiles are not calibrated probabilities",
            "Top-60 fixed from historical CVLI; changing its membership invalidates this sealed run",
            "One 30-day prospective window has limited statistical power",
            "CVLI reports may be corrected after the window; evaluation requires a seven-day minimum lag",
            "No operational deployment, allocation decisions or automatic alerts",
        ],
    }
    write_exclusive(shadow_dir/"manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    short=ranks[(ranks["weight_stgat"].isin(PRIMARY_WEIGHTS)) & (ranks["rank"]<=10)]
    write_exclusive(shadow_dir/"TOP10_PRIORIDADES.md",
       "# Ranking prospectivo selado (uso exclusivamente experimental)\n\n"
       +f"Corte: **{AS_OF.date()}**; alvo: **{issue_date.date()} a {target_end.date()}**.\n"
       +"Pesos predefinidos de ST-GAT: **20%, 40%, 70%**. N├úo h├í resultados futuros nesta captura.\n\n"
       +"\n".join(
        f"## {v} ÔÇö {int(w*100)}% ST-GAT\n"
        +", ".join(group.sort_values("rank")["bairro"])
        for (v,w),group in short.groupby(["variant","weight_stgat"],sort=True)
       )+"\n")
    print("SEALED_SNAPSHOT",str(shadow_dir),"rows",len(ranks),
          "ranking_sha256",csv_sha,
          "target",issue_date.date(),target_end.date(),
          "label_maturity",manifest["labels_mature_after"],flush=True)
if __name__=="__main__":
    main()
