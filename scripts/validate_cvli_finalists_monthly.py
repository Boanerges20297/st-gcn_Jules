"""
Playoff mensal dos finalistas para o alvo:
prever o proximo bairro com CVLI.

- Reusa a mesma montagem de dataset do benchmark principal.
- Foca em modelos classicos finalistas.
- Reporta metricas mes a mes em 2025 para medir estabilidade.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_cvli_stochastic_suite import (
    DEFAULT_OUT_DIR,
    REGION_CONFIGS,
    build_classical_models,
    build_region_datasets,
    dataframe_to_markdown,
    evaluate_daily_predictions,
    format_seconds,
    load_processed_region,
    progress_bar,
    set_seed,
)


DEFAULT_FINALISTS = [
    "poisson_regressor",
    "histgb_classifier",
    "logit_classifier",
    "hurdle_logit_poisson",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="fortaleza", choices=sorted(REGION_CONFIGS.keys()))
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-start", default="2025-01-01")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--horizon-days", type=int, default=14)
    parser.add_argument("--models", nargs="+", default=DEFAULT_FINALISTS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def monthly_eval(
    val_df: pd.DataFrame,
    predictions: np.ndarray,
    month_targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    pred_df = val_df[["sample_date", "node_idx"]].copy()
    pred_df["pred"] = predictions
    pred_df["month"] = pred_df["sample_date"].str.slice(0, 7)

    rows = []
    for month, group in pred_df.groupby("month"):
        predictions_by_day = {}
        month_days = sorted(group["sample_date"].unique().tolist())
        for day in month_days:
            day_group = group[group["sample_date"] == day]
            predictions_by_day[day] = day_group.sort_values("node_idx")["pred"].to_numpy(dtype=float)
        targets_by_day = {day: month_targets[day] for day in month_days if day in month_targets}
        if not targets_by_day:
            continue
        metrics = evaluate_daily_predictions(predictions_by_day, targets_by_day, model_type="classic")
        metrics["month"] = month
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("month")


def build_predictions_for_model(
    model_name: str,
    model,
    target_kind: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    X_train = train_df[feature_cols]
    y_train_count = train_df["target_count"].to_numpy(dtype=float)
    y_train_binary = train_df["target_binary"].to_numpy(dtype=int)
    X_val = val_df[feature_cols]

    if target_kind == "binary":
        model.fit(X_train, y_train_binary)
        if hasattr(model, "predict_proba"):
            pred = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
        else:
            pred = np.asarray(model.predict(X_val), dtype=float)
    elif target_kind == "count":
        model.fit(X_train, y_train_count)
        pred = np.clip(np.asarray(model.predict(X_val), dtype=float), 0.0, None)
    else:
        model.fit(X_train, y_train_binary)
        pred = np.clip(np.asarray(model.predict(X_val), dtype=float), 0.0, None)
    return pred


def summarize_stability(monthly_df: pd.DataFrame) -> dict:
    return {
        "months": int(len(monthly_df)),
        "hit1_mean": float(monthly_df["hit1_event"].mean()),
        "hit1_std": float(monthly_df["hit1_event"].std(ddof=0)),
        "p10_mean": float(monthly_df["p10_event"].mean()),
        "p10_std": float(monthly_df["p10_event"].std(ddof=0)),
        "mrr_mean": float(monthly_df["mrr_event"].mean()),
        "mrr_std": float(monthly_df["mrr_event"].std(ddof=0)),
        "worst_hit1": float(monthly_df["hit1_event"].min()),
        "worst_p10": float(monthly_df["p10_event"].min()),
        "best_hit1": float(monthly_df["hit1_event"].max()),
        "best_p10": float(monthly_df["p10_event"].max()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    started_at = datetime.now()

    cfg = REGION_CONFIGS[args.region]
    data = load_processed_region(cfg)
    datasets = build_region_datasets(
        cfg=cfg,
        data=data,
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        val_start=pd.Timestamp(args.val_start),
        val_end=pd.Timestamp(args.val_end),
        horizon_days=args.horizon_days,
    )

    train_df = datasets["classical_train"]
    val_df = datasets["classical_val"]
    targets_by_day = datasets["daily_val_targets"]

    feature_cols = [
        "lag_1",
        "sum_7",
        "sum_14",
        "sum_30",
        "sum_60",
        "mean_7",
        "mean_14",
        "mean_30",
        "mean_60",
        "hist_mean",
        "hist_sum",
        "momentum_7",
        "momentum_14",
        "momentum_30",
        "cold_streak_inv",
        "tension_index",
        "recent_cvli_static",
        "total_cvli_static",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "is_weekend",
    ]

    available = {name: (model, kind) for name, model, kind in build_classical_models(set(args.models))}
    monthly_results = {}
    summary_rows = []

    for idx, model_name in enumerate(args.models, start=1):
        if model_name not in available:
            raise ValueError(f"Modelo finalista desconhecido: {model_name}")
        model, kind = available[model_name]
        print(
            f"[finalists] {progress_bar(idx - 1, len(args.models))} "
            f"{idx - 1}/{len(args.models)} concluídos | próximo={model_name}"
        )
        pred = build_predictions_for_model(model_name, model, kind, train_df, val_df, feature_cols)
        monthly_df = monthly_eval(val_df, pred, targets_by_day)
        monthly_results[model_name] = monthly_df
        stability = summarize_stability(monthly_df)
        stability["model"] = model_name
        summary_rows.append(stability)
        print(
            f"[finalists] {progress_bar(idx, len(args.models))} "
            f"{idx}/{len(args.models)} {model_name} | "
            f"hit1_mean={stability['hit1_mean']:.3f} p10_mean={stability['p10_mean']:.3f} "
            f"mrr_mean={stability['mrr_mean']:.3f} "
            f"| hit1_std={stability['hit1_std']:.3f} p10_std={stability['p10_std']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["hit1_mean", "p10_mean", "mrr_mean", "hit1_std", "p10_std"],
        ascending=[False, False, False, True, True],
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = DEFAULT_OUT_DIR / f"cvli_finalists_monthly_{args.region}_{stamp}.json"
    out_csv = DEFAULT_OUT_DIR / f"cvli_finalists_monthly_{args.region}_{stamp}.csv"
    out_md = DEFAULT_OUT_DIR / f"cvli_finalists_monthly_{args.region}_{stamp}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "region": args.region,
        "args": vars(args),
        "summary": summary_df.to_dict(orient="records"),
        "monthly": {
            model_name: df.to_dict(orient="records")
            for model_name, df in monthly_results.items()
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# Finalists Monthly Validation - {args.region}\n\n")
        f.write(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Train: {args.train_start} -> {args.train_end}\n")
        f.write(f"- Validation: {args.val_start} -> {args.val_end}\n")
        f.write(f"- Horizon: {args.horizon_days} days\n")
        f.write(f"- Models: {', '.join(args.models)}\n\n")
        f.write("## Stability Summary\n\n")
        f.write(dataframe_to_markdown(summary_df))
        for model_name, df in monthly_results.items():
            f.write(f"\n\n## {model_name}\n\n")
            f.write(dataframe_to_markdown(df))
    elapsed = datetime.now() - started_at
    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"\nElapsed: {format_seconds(elapsed.total_seconds())}")
    print(f"\nFiles:\n  {out_json}\n  {out_csv}\n  {out_md}")


if __name__ == "__main__":
    main()
