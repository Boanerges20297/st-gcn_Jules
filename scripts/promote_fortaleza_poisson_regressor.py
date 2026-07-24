from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cvli_stochastic_suite import (  # noqa: E402
    REGION_CONFIGS,
    build_region_datasets,
    build_classical_models,
    evaluate_daily_predictions,
    load_processed_region,
)
from src.core.fortaleza_poisson_backend import (  # noqa: E402
    FEATURE_COLS,
    save_payload,
    train_poisson_payload,
)


OUTPUT_PKL = ROOT / "models" / "active" / "production" / "poisson" / "fortaleza_poisson_regressor.pkl"
OUTPUT_JSON = ROOT / "models" / "active" / "production" / "poisson" / "fortaleza_poisson_regressor.json"


def main() -> None:
    cfg = REGION_CONFIGS["fortaleza"]
    data = load_processed_region(cfg)
    datasets = build_region_datasets(
        cfg=cfg,
        data=data,
        train_start=pd.Timestamp("2022-01-01"),
        train_end=pd.Timestamp("2024-12-31"),
        val_start=pd.Timestamp("2025-01-01"),
        val_end=pd.Timestamp("2025-12-31"),
        horizon_days=30,
    )

    train_df = datasets["classical_train"]
    val_df = datasets["classical_val"]
    targets_by_day = datasets["daily_val_targets"]

    models = {name: (model, kind) for name, model, kind in build_classical_models({"poisson_regressor"})}
    model, kind = models["poisson_regressor"]
    model.fit(train_df[FEATURE_COLS], train_df["target_count"].to_numpy(dtype=float))
    pred = model.predict(val_df[FEATURE_COLS])

    pred_df = val_df[["sample_date", "node_idx"]].copy()
    pred_df["pred"] = pred
    predictions_by_day = {
        day: group.sort_values("node_idx")["pred"].to_numpy(dtype=float)
        for day, group in pred_df.groupby("sample_date")
    }
    metrics = evaluate_daily_predictions(predictions_by_day, targets_by_day, model_type="classic")

    benchmark_metrics = {
        "validation_split": {
            "train_start": "2022-01-01",
            "train_end": "2024-12-31",
            "val_start": "2025-01-01",
            "val_end": "2025-12-31",
            "horizon_days": 14,
        },
        "fortaleza_2025": metrics,
        "selection_rationale": "Promovido como champion operacional por melhor equilibrio entre hit1, p10 e mrr.",
    }

    payload = train_poisson_payload(
        data=data,
        train_start="2022-01-01",
        train_end="2024-12-31",
        horizon_days=30,
        window=90,
        benchmark_metrics=benchmark_metrics,
    )
    save_payload(payload, OUTPUT_PKL)

    sidecar = {
        "generated_at": datetime.now().isoformat(),
        "artifact_path": str(OUTPUT_PKL),
        "backend_type": payload["backend_type"],
        "window": payload["window"],
        "horizon_days": payload["horizon_days"],
        "feature_cols": payload["feature_cols"],
        "train_rows": payload["train_rows"],
        "benchmark_metrics": benchmark_metrics,
    }
    OUTPUT_JSON.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(sidecar, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
