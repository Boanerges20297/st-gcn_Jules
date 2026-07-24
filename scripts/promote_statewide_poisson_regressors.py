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
    build_classical_models,
    build_region_datasets,
    evaluate_daily_predictions,
    load_processed_region,
)
from src.core.fortaleza_poisson_backend import (  # noqa: E402
    FEATURE_COLS,
    save_payload,
    train_poisson_payload,
)


REGION_WINDOWS = {
    "fortaleza": 90,
    "rmf": 14,
    "interior": 14,
}


def evaluate_poisson_region(region: str) -> dict:
    cfg = REGION_CONFIGS[region]
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
    model, _ = models["poisson_regressor"]
    model.fit(train_df[FEATURE_COLS], train_df["target_count"].to_numpy(dtype=float))
    pred = model.predict(val_df[FEATURE_COLS])
    pred_df = val_df[["sample_date", "node_idx"]].copy()
    pred_df["pred"] = pred
    predictions_by_day = {
        day: group.sort_values("node_idx")["pred"].to_numpy(dtype=float)
        for day, group in pred_df.groupby("sample_date")
    }
    metrics = evaluate_daily_predictions(predictions_by_day, targets_by_day, model_type="classic")

    payload = train_poisson_payload(
        data=data,
        region=region,
        train_start="2022-01-01",
        train_end="2024-12-31",
        horizon_days=30,
        window=REGION_WINDOWS[region],
        benchmark_metrics={
            "validation_split": {
                "train_start": "2022-01-01",
                "train_end": "2024-12-31",
                "val_start": "2025-01-01",
                "val_end": "2025-12-31",
                "horizon_days": 14,
            },
            f"{region}_2025": metrics,
            "selection_rationale": "Padronização estadual do backend Poisson para reduzir custo operacional do app.",
        },
    )

    model_path = ROOT / "models" / "active" / "production" / "poisson" / f"{region}_poisson_regressor.pkl"
    meta_path = ROOT / "models" / "active" / "production" / "poisson" / f"{region}_poisson_regressor.json"
    save_payload(payload, model_path)

    sidecar = {
        "generated_at": datetime.now().isoformat(),
        "artifact_path": str(model_path),
        "backend_type": payload["backend_type"],
        "region": region,
        "window": payload["window"],
        "horizon_days": payload["horizon_days"],
        "feature_cols": payload["feature_cols"],
        "train_rows": payload["train_rows"],
        "benchmark_metrics": payload["benchmark_metrics"],
    }
    meta_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def main() -> None:
    results = {region: evaluate_poisson_region(region) for region in ("fortaleza", "rmf", "interior")}
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
