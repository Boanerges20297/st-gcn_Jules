import io
import json
import os
import sys
import importlib.util

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


def precision_at_k(scores, targets, k):
    top = np.argsort(scores)[::-1][:k]
    return float(np.sum(targets[top] > 0) / k)


def recall_at_k(scores, targets, k):
    top = np.argsort(scores)[::-1][:k]
    denom = max(int((targets > 0).sum()), 1)
    return float(np.sum(targets[top] > 0) / denom)


def train_cvli_reranker(feats, cvli_raw, cutoff_i, horizon):
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
    num_nodes = cvli_raw.shape[0]
    for time_index in range(90, cutoff_i - horizon, 2):
        targets = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if targets.sum() == 0 or time_index + horizon > cutoff_i:
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


def structural_intel_score(feats, time_index, ranker, ranker_features):
    rank_df = pd.DataFrame(
        [[float(feats[name][node_index, time_index]) for name in ranker_features] for node_index in range(feats["hist_pct"].shape[0])],
        columns=ranker_features,
    )
    score_ranker = normalize(ranker.predict(rank_df))
    score_structural = 0.60 * normalize(feats["hist_pct"][:, time_index]) + 0.40 * normalize(feats["target_enc"][:, time_index])
    score_intel = normalize(feats["inter_intel_cvli"][:, time_index])
    return 0.55 * score_structural + 0.35 * score_ranker + 0.10 * score_intel


def apply_shortlist_rerank(shortlist_source, rerank_source, shortlist_size, shortlist_boost):
    mask = np.zeros_like(shortlist_source, dtype=float)
    shortlist = np.argsort(shortlist_source)[::-1][:shortlist_size]
    mask[shortlist] = 1.0
    return normalize(rerank_source) + shortlist_boost * mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    freeze = load_freeze_module()
    feats, _, dates, cvli_raw, top_bairros = freeze.build_all()
    horizon = freeze.HORIZON
    cutoff = pd.Timestamp("2026-02-28")
    date_map = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    cutoff_i = date_map[cutoff]

    ranker, ranker_features = train_cvli_reranker(feats, cvli_raw, cutoff_i, horizon)

    start_eval = cutoff + pd.Timedelta(days=1)
    last_pred = dates[-1] - pd.Timedelta(days=horizon)
    eval_dates = pd.date_range(start_eval, last_pred, freq="D")

    all_rows = []
    shortlist_configs = [
        ("SHORT12", 12, 0.35),
        ("SHORT15", 15, 0.30),
        ("SHORT20", 20, 0.25),
        ("SHORT25", 25, 0.20),
    ]

    for pred_date in eval_dates:
        time_index = date_map[pd.Timestamp(pred_date)]
        targets = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if int((targets > 0).sum()) == 0:
            continue

        score_shortlist = tactical_cvli_score(feats, time_index)
        score_rerank = structural_intel_score(feats, time_index, ranker, ranker_features)
        baseline = 0.65 * score_shortlist + 0.35 * score_rerank

        candidates = {"BASELINE_MIX": baseline}
        for label, shortlist_size, boost in shortlist_configs:
            candidates[f"{label}_RERANK"] = apply_shortlist_rerank(score_shortlist, score_rerank, shortlist_size, boost)
            candidates[f"{label}_MIX"] = apply_shortlist_rerank(score_shortlist, baseline, shortlist_size, boost)

        for family, scores in candidates.items():
            all_rows.append(
                {
                    "pred_date": str(pred_date.date()),
                    "family": family,
                    "p10": precision_at_k(scores, targets, 10),
                    "p20": precision_at_k(scores, targets, 20),
                    "r10": recall_at_k(scores, targets, 10),
                    "r20": recall_at_k(scores, targets, 20),
                }
            )

    results = pd.DataFrame(all_rows)
    summary = (
        results.groupby("family")[["p10", "p20", "r10", "r20"]]
        .mean()
        .sort_values(["p10", "p20", "r20"], ascending=False)
        .reset_index()
    )

    results.to_csv(os.path.join(OUT_DIR, "cvli_shortlist_rerank_daily.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(OUT_DIR, "cvli_shortlist_rerank_summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "cvli_shortlist_rerank_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict("records"), handle, ensure_ascii=False, indent=2)

    print("\n=== CVLI shortlist + rerank summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
