import io
import json
import os
import sys
import importlib.util
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FREEZE_SCRIPT = os.path.join(os.path.dirname(__file__), "freeze_total_v3.py")
OUT_DIR = os.path.join(BASE_PATH, "outputs")


def load_freeze_module():
    spec = importlib.util.spec_from_file_location("freeze_total_v3_local", FREEZE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", closefd=False)
    module.sys.stdout = sys.stdout
    module.sys.stderr = sys.stderr
    return module


def normalize(values):
    arr = np.asarray(values, dtype=float)
    low, high = arr.min(), arr.max()
    if high - low < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def train_cvli_reranker(feats, cvli_raw, horizon):
    feature_names = [
        "hist_pct",
        "target_enc",
        "intel_ewma_14d",
        "intel_ewma_7d",
        "inter_intel_cvli",
        "nbr_cvli_30d",
        "inter_chuva_hist",
        "inter_feriado_hist",
    ]
    rows = []
    num_nodes, total_days = cvli_raw.shape
    for time_index in range(90, total_days - horizon, 2):
        targets = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if targets.sum() == 0:
            continue
        for node_index in range(num_nodes):
            row = {name: float(feats[name][node_index, time_index]) for name in feature_names}
            row["ti"] = time_index
            row["label"] = min(int(targets[node_index]), 5) + (1 if targets[node_index] > 0 else 0)
            rows.append(row)

    train_df = pd.DataFrame(rows).sort_values("ti")
    max_time_index = train_df["ti"].max()
    train_df["sample_weight"] = np.exp(-(max_time_index - train_df["ti"]) / 450.0)
    groups = train_df.groupby("ti").size().values

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 5, 10],
        n_estimators=1000,
        num_leaves=63,
        learning_rate=0.01,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=3.0,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        extra_trees=True,
    )
    ranker.fit(
        train_df[feature_names],
        train_df["label"].astype("int32"),
        group=groups,
        sample_weight=train_df["sample_weight"],
    )
    return ranker, feature_names


def tactical_cvli_score(feats, time_index):
    return (
        0.50 * normalize(feats["cvli_ewma_7d"][:, time_index])
        + 0.35 * normalize(feats["cvli_ewma_14d"][:, time_index])
        + 0.15 * normalize(feats["cvli_ewma_30d"][:, time_index])
    )


def structural_intel_score(feats, time_index, ranker, ranker_features, num_nodes):
    rank_df = pd.DataFrame(
        [[float(feats[name][node_index, time_index]) for name in ranker_features] for node_index in range(num_nodes)],
        columns=ranker_features,
    )
    score_ranker = normalize(ranker.predict(rank_df))
    score_structural = 0.60 * normalize(feats["hist_pct"][:, time_index]) + 0.40 * normalize(feats["target_enc"][:, time_index])
    score_intel = normalize(feats["inter_intel_cvli"][:, time_index])
    return 0.55 * score_structural + 0.35 * score_ranker + 0.10 * score_intel


def shortlist_mix(shortlist_source, baseline_source, shortlist_size=20, boost=0.25):
    mask = np.zeros_like(shortlist_source, dtype=float)
    shortlist = np.argsort(shortlist_source)[::-1][:shortlist_size]
    mask[shortlist] = 1.0
    return normalize(baseline_source) + boost * mask


def build_rank_table(name, bairros, scores, feats, time_index):
    order = np.argsort(scores)[::-1]
    rows = []
    for rank, idx in enumerate(order, 1):
        rows.append(
            {
                "modelo": name,
                "rank": rank,
                "bairro": bairros[idx],
                "score": round(float(scores[idx]), 6),
                "cvli_ewma_7d": round(float(feats["cvli_ewma_7d"][idx, time_index]), 6),
                "cvli_ewma_14d": round(float(feats["cvli_ewma_14d"][idx, time_index]), 6),
                "cvli_ewma_30d": round(float(feats["cvli_ewma_30d"][idx, time_index]), 6),
                "hist_pct": round(float(feats["hist_pct"][idx, time_index]), 6),
                "target_enc": round(float(feats["target_enc"][idx, time_index]), 6),
                "intel_ewma_14d": round(float(feats["intel_ewma_14d"][idx, time_index]), 6),
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    freeze = load_freeze_module()
    feats, _, dates, cvli_raw, top_bairros = freeze.build_all()
    horizon = freeze.HORIZON
    num_nodes = len(top_bairros)
    last_index = len(dates) - 1
    reference_date = dates[last_index].date()
    prediction_end = (dates[last_index] + timedelta(days=horizon)).date()

    ranker, ranker_features = train_cvli_reranker(feats, cvli_raw, horizon)

    tactical = tactical_cvli_score(feats, last_index)
    rerank = structural_intel_score(feats, last_index, ranker, ranker_features, num_nodes)
    short20 = shortlist_mix(tactical, 0.65 * tactical + 0.35 * rerank, shortlist_size=20, boost=0.25)

    tactical_df = build_rank_table("CVLI_TACTICAL_ONLY", top_bairros, tactical, feats, last_index)
    short20_df = build_rank_table("SHORT20_MIX", top_bairros, short20, feats, last_index)
    ranking_df = pd.concat([tactical_df, short20_df], ignore_index=True)

    csv_path = os.path.join(OUT_DIR, "cvli_first_candidate_rankings.csv")
    json_path = os.path.join(OUT_DIR, "cvli_first_candidate_rankings.json")
    meta_path = os.path.join(OUT_DIR, "cvli_first_candidate_metadata.json")

    ranking_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(ranking_df.to_dict("records"), handle, ensure_ascii=False, indent=2)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "reference_date": str(reference_date),
        "prediction_start": str(reference_date),
        "prediction_end": str(prediction_end),
        "models": [
            {
                "name": "CVLI_TACTICAL_ONLY",
                "description": "Score baseado apenas em momentum recente de CVLI (7d/14d/30d).",
                "focus": "maximizar P@10",
            },
            {
                "name": "SHORT20_MIX",
                "description": "Shortlist tática por CVLI recente com reranqueamento leve estrutural + intel.",
                "focus": "equilibrar topo e cobertura",
            },
        ],
        "top10": {
            "CVLI_TACTICAL_ONLY": tactical_df.head(10)["bairro"].tolist(),
            "SHORT20_MIX": short20_df.head(10)["bairro"].tolist(),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\n=== Candidatos CVLI-first gerados ===")
    print(f"Referência: {reference_date} | Horizonte: {reference_date} -> {prediction_end}")
    print("\nTop-10 CVLI_TACTICAL_ONLY:")
    print(tactical_df.head(10)[["rank", "bairro", "score"]].to_string(index=False))
    print("\nTop-10 SHORT20_MIX:")
    print(short20_df.head(10)[["rank", "bairro", "score"]].to_string(index=False))
    print(f"\n[OK] CSV: {csv_path}")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] META: {meta_path}")


if __name__ == "__main__":
    main()
