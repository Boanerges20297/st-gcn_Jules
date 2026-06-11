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
OUT_DIR = os.path.join(BASE_PATH, "outputs")
FREEZE_SCRIPT = os.path.join(os.path.dirname(__file__), "freeze_total_v3.py")


def load_freeze_module():
    spec = importlib.util.spec_from_file_location("freeze_total_v3_local", FREEZE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", closefd=False)
    module.sys.stdout = sys.stdout
    module.sys.stderr = sys.stderr
    return module


def normalize_scores(values):
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


def train_ranker(feats, feat_names, cvli_raw, cutoff_i, horizon):
    rows = []
    for time_index in range(90, cutoff_i - horizon, 2):
        future_targets = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if future_targets.sum() == 0 or time_index + horizon > cutoff_i:
            continue
        for node_index in range(cvli_raw.shape[0]):
            row = {name: float(feats[name][node_index, time_index]) for name in feat_names}
            row["ti"] = time_index
            row["label"] = min(int(future_targets[node_index]), 5) + (1 if future_targets[node_index] > 0 else 0)
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    freeze = load_freeze_module()
    feats, feat_names, dates, cvli_raw, top_bairros = freeze.build_all()
    horizon = freeze.HORIZON
    date_map = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    cutoff = pd.Timestamp("2026-02-28")
    cutoff_i = date_map[cutoff]

    print("\n=== Historical concentration ===")
    hist_totals = cvli_raw.sum(axis=1)
    hist_order = np.argsort(hist_totals)[::-1]
    cumulative = np.cumsum(hist_totals[hist_order]) / hist_totals.sum()
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    concentration = {}
    for threshold in thresholds:
        count = int(np.searchsorted(cumulative, threshold) + 1)
        concentration[f"{int(threshold * 100)}%"] = count
        print(f"{int(threshold * 100)}% dos CVLI históricos estão em {count} bairros")

    ranker = train_ranker(feats, feat_names, cvli_raw, cutoff_i, horizon)

    start_eval = cutoff + pd.Timedelta(days=1)
    last_pred = dates[-1] - pd.Timedelta(days=horizon)
    eval_dates = pd.date_range(start_eval, last_pred, freq="D")

    records = []
    for pred_date in eval_dates:
        time_index = date_map[pd.Timestamp(pred_date)]
        ground_truth = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        if int((ground_truth > 0).sum()) == 0:
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
        records.append(
            (
                str(pred_date.date()),
                ground_truth,
                score_ewma.astype(float),
                score_lgbm.astype(float),
                score_hist.astype(float),
                score_target.astype(float),
            )
        )

    grid_rows = []
    best = None
    steps = [value / 10 for value in range(11)]
    for w_ewma in steps:
        for w_lgbm in steps:
            for w_hist in steps:
                w_target = round(1 - w_ewma - w_lgbm - w_hist, 10)
                if w_target < 0 or w_target > 1:
                    continue
                p10_list, p20_list, r10_list, r20_list = [], [], [], []
                for _, ground_truth, score_ewma, score_lgbm, score_hist, score_target in records:
                    score = (
                        w_ewma * normalize_scores(score_ewma)
                        + w_lgbm * normalize_scores(score_lgbm)
                        + w_hist * normalize_scores(score_hist)
                        + w_target * normalize_scores(score_target)
                    )
                    p10_list.append(precision_at_k(score, ground_truth, 10))
                    p20_list.append(precision_at_k(score, ground_truth, 20))
                    r10_list.append(recall_at_k(score, ground_truth, 10))
                    r20_list.append(recall_at_k(score, ground_truth, 20))
                row = {
                    "w_ewma": w_ewma,
                    "w_lgbm": w_lgbm,
                    "w_hist": w_hist,
                    "w_target": w_target,
                    "p10": float(np.mean(p10_list)),
                    "p20": float(np.mean(p20_list)),
                    "r10": float(np.mean(r10_list)),
                    "r20": float(np.mean(r20_list)),
                }
                grid_rows.append(row)
                if best is None or (row["p10"], row["p20"], row["r20"]) > (best["p10"], best["p20"], best["r20"]):
                    best = row

    feasible = [row for row in grid_rows if row["r20"] >= 0.65]
    feasible.sort(key=lambda row: (row["p10"], row["p20"], row["r20"]), reverse=True)
    chosen = feasible[0] if feasible else best

    print("\n=== Best blend ===")
    print(chosen)

    hist_top10 = hist_order[:10]
    hist_top32 = set(hist_order[:32])
    fixed_rows = []
    inside_top32 = []
    for pred_date in eval_dates:
        time_index = date_map[pd.Timestamp(pred_date)]
        ground_truth = cvli_raw[:, time_index + 1 : time_index + horizon + 1].sum(axis=1)
        positive = set(np.where(ground_truth > 0)[0])
        if not positive:
            continue
        fixed_rows.append(
            {
                "pred_date": str(pred_date.date()),
                "hist_top10_p10": precision_at_k(np.isin(np.arange(len(top_bairros)), hist_top10).astype(float), ground_truth, 10),
                "hist_top10_r10": recall_at_k(np.isin(np.arange(len(top_bairros)), hist_top10).astype(float), ground_truth, 10),
            }
        )
        inside_top32.append(len(positive & hist_top32) / len(positive))

    ranking_stats = {}
    chosen_weights = (chosen["w_ewma"], chosen["w_lgbm"], chosen["w_hist"], chosen["w_target"])
    hist_top10_names = [top_bairros[index] for index in hist_order[:10]]
    for bairro in hist_top10_names:
        ranking_stats[bairro] = []

    for _, _, score_ewma, score_lgbm, score_hist, score_target in records:
        score = (
            chosen_weights[0] * normalize_scores(score_ewma)
            + chosen_weights[1] * normalize_scores(score_lgbm)
            + chosen_weights[2] * normalize_scores(score_hist)
            + chosen_weights[3] * normalize_scores(score_target)
        )
        order = np.argsort(score)[::-1]
        for bairro in hist_top10_names:
            index = top_bairros.index(bairro)
            ranking_stats[bairro].append(int(np.where(order == index)[0][0] + 1))

    ranking_rows = []
    for bairro, values in ranking_stats.items():
        arr = np.asarray(values)
        ranking_rows.append(
            {
                "bairro": bairro,
                "avg_rank": float(arr.mean()),
                "min_rank": int(arr.min()),
                "max_rank": int(arr.max()),
                "std_rank": float(arr.std()),
            }
        )

    pd.DataFrame(grid_rows).sort_values(["p10", "p20", "r20"], ascending=False).to_csv(
        os.path.join(OUT_DIR, "blend_gridsearch_p10.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fixed_rows).to_csv(
        os.path.join(OUT_DIR, "historical_fixed_set_validation.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(ranking_rows).to_csv(
        os.path.join(OUT_DIR, "shortlist_rerank_position_volatility.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "cutoff": str(cutoff.date()),
        "historical_concentration": concentration,
        "best_blend": chosen,
        "inside_top32_mean": float(np.mean(inside_top32)),
        "inside_top32_median": float(np.median(inside_top32)),
        "inside_top32_all_windows": int(np.sum(np.asarray(inside_top32) == 1.0)),
        "inside_top32_windows": int(len(inside_top32)),
        "historical_top10_names": hist_top10_names,
        "position_volatility": ranking_rows,
    }
    with open(os.path.join(OUT_DIR, "blend_gridsearch_p10_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
