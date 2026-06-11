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


def reciprocal_rank_fusion(*score_vectors):
    rank_arrays = [np.argsort(np.argsort(-normalize(scores))) + 1 for scores in score_vectors]
    fused = np.zeros_like(rank_arrays[0], dtype=float)
    for ranks in rank_arrays:
        fused += 1.0 / (60.0 + ranks)
    return fused


def apply_shortlist(mask_source, rerank_source, shortlist_size, boost=0.25):
    base_mask = np.zeros_like(mask_source, dtype=float)
    shortlist = np.argsort(mask_source)[::-1][:shortlist_size]
    base_mask[shortlist] = 1.0
    return normalize(rerank_source) + boost * base_mask


def train_ranker(feats, feat_names, cvli_raw, cutoff_i, horizon):
    rows = []
    for time_index in range(90, cutoff_i - horizon, 2):
        targets = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if targets.sum() == 0 or time_index + horizon > cutoff_i:
            continue
        for node_index in range(cvli_raw.shape[0]):
            row = {name: float(feats[name][node_index, time_index]) for name in feat_names}
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
        n_estimators=1500,
        num_leaves=127,
        learning_rate=0.008,
        min_child_samples=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        extra_trees=True,
    )
    ranker.fit(
        train_df[feat_names],
        train_df["label"].astype("int32"),
        group=groups,
        sample_weight=train_df["sample_weight"],
    )
    return ranker


def summarize(name, daily_rows):
    frame = pd.DataFrame(daily_rows)
    return {
        "family": name,
        "p10": float(frame["p10"].mean()),
        "p20": float(frame["p20"].mean()),
        "r10": float(frame["r10"].mean()),
        "r20": float(frame["r20"].mean()),
        "windows": int(len(frame)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    freeze = load_freeze_module()
    feats, feat_names, dates, cvli_raw, top_bairros = freeze.build_all()
    horizon = freeze.HORIZON
    cutoff = pd.Timestamp("2026-02-28")
    date_map = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    cutoff_i = date_map[cutoff]

    ranker = train_ranker(feats, feat_names, cvli_raw, cutoff_i, horizon)

    hist_score = feats["hist_pct"][:, cutoff_i]
    target_score = feats["target_enc"][:, cutoff_i]

    start_eval = cutoff + pd.Timedelta(days=1)
    last_pred = dates[-1] - pd.Timedelta(days=horizon)
    eval_dates = pd.date_range(start_eval, last_pred, freq="D")

    family_rows = {}

    for pred_date in eval_dates:
        time_index = date_map[pd.Timestamp(pred_date)]
        targets = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if int((targets > 0).sum()) == 0:
            continue

        features_df = pd.DataFrame(
            [[float(feats[name][node_index, time_index]) for name in feat_names] for node_index in range(len(top_bairros))],
            columns=feat_names,
        )
        score_lgbm = ranker.predict(features_df)
        score_ewma = np.zeros(len(top_bairros), np.float32)
        for name, weight in freeze.EWMA_WEIGHTS.items():
            if name in feats:
                score_ewma += weight * feats[name][:, time_index]
        score_hist = feats["hist_pct"][:, time_index]
        score_target = feats["target_enc"][:, time_index]
        score_neighbor = feats["nbr_cvli_30d"][:, time_index]
        score_intel = feats["intel_ewma_14d"][:, time_index]

        candidate_scores = {
            "EWMA": score_ewma,
            "LGBM": score_lgbm,
            "EWMA_60_LGBM_20_HIST_20": 0.6 * normalize(score_ewma) + 0.2 * normalize(score_lgbm) + 0.2 * normalize(score_hist),
            "EWMA_50_LGBM_20_HIST_20_INTEL_10": 0.5 * normalize(score_ewma) + 0.2 * normalize(score_lgbm) + 0.2 * normalize(score_hist) + 0.1 * normalize(score_intel),
            "RRF_EWMA_LGBM": reciprocal_rank_fusion(score_ewma, score_lgbm),
            "RRF_EWMA_LGBM_HIST": reciprocal_rank_fusion(score_ewma, score_lgbm, score_hist),
            "SHORTLIST20_HIST_RERANK_EWMA": apply_shortlist(score_hist, score_ewma, shortlist_size=20, boost=0.25),
            "SHORTLIST20_HIST_RERANK_BLEND": apply_shortlist(score_hist, 0.7 * normalize(score_ewma) + 0.3 * normalize(score_lgbm), shortlist_size=20, boost=0.25),
            "SHORTLIST15_HIST_RERANK_BLEND": apply_shortlist(score_hist, 0.7 * normalize(score_ewma) + 0.3 * normalize(score_lgbm), shortlist_size=15, boost=0.25),
            "SHORTLIST20_TARGET_RERANK_EWMA": apply_shortlist(score_target, score_ewma, shortlist_size=20, boost=0.25),
            "SHORTLIST20_HIST_RERANK_NEIGHBOR": apply_shortlist(score_hist, 0.6 * normalize(score_ewma) + 0.4 * normalize(score_neighbor), shortlist_size=20, boost=0.25),
            "GATED_HIST_NEIGHBOR": 0.5 * normalize(score_ewma) + 0.2 * normalize(score_lgbm) + 0.2 * normalize(score_hist) + 0.1 * normalize(score_neighbor),
        }

        for family, scores in candidate_scores.items():
            family_rows.setdefault(family, []).append(
                {
                    "pred_date": str(pred_date.date()),
                    "p10": precision_at_k(scores, targets, 10),
                    "p20": precision_at_k(scores, targets, 20),
                    "r10": recall_at_k(scores, targets, 10),
                    "r20": recall_at_k(scores, targets, 20),
                }
            )

    summary = pd.DataFrame([summarize(name, rows) for name, rows in family_rows.items()])
    summary = summary.sort_values(["p10", "p20", "r20"], ascending=False).reset_index(drop=True)

    summary.to_csv(os.path.join(OUT_DIR, "hybrid_family_summary.csv"), index=False, encoding="utf-8-sig")

    details = []
    for family, rows in family_rows.items():
        frame = pd.DataFrame(rows)
        frame["family"] = family
        details.append(frame)
    pd.concat(details, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "hybrid_family_daily.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(OUT_DIR, "hybrid_family_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict("records"), handle, ensure_ascii=False, indent=2)

    print("\n=== Hybrid family summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
