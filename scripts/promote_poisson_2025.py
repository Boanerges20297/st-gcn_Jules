"""
Promove o Poisson retreinado (treino 2022-2025) para production/.

Diferença em relação ao promote_statewide_poisson_regressors.py:
  - train_end = 2025-12-31  (inclui dados de 2025)
  - val_start = 2026-01-01 / val_end = 2026-05-31  (dados reais de 2026)

Uso:
    python scripts/promote_poisson_2025.py
    python scripts/promote_poisson_2025.py --region fortaleza
"""

from __future__ import annotations

import argparse
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


TRAIN_START = "2022-01-01"
TRAIN_END = "2025-12-31"
VAL_START = "2026-01-01"
VAL_END = "2026-05-31"
HORIZON_DAYS = 14

REGION_WINDOWS = {
    "fortaleza": 90,
    "rmf": 14,
    "interior": 14,
}


def promote_region(region: str) -> dict:
    print(f"[promote] {region.upper()} | treino {TRAIN_START} -> {TRAIN_END} | val {VAL_START} -> {VAL_END}")

    cfg = REGION_CONFIGS[region]
    data = load_processed_region(cfg)

    datasets = build_region_datasets(
        cfg=cfg,
        data=data,
        train_start=pd.Timestamp(TRAIN_START),
        train_end=pd.Timestamp(TRAIN_END),
        val_start=pd.Timestamp(VAL_START),
        val_end=pd.Timestamp(VAL_END),
        horizon_days=HORIZON_DAYS,
    )

    train_df = datasets["classical_train"]
    val_df = datasets["classical_val"]
    targets_by_day = datasets["daily_val_targets"]

    print(f"  Treino: {len(train_df):,} linhas | Val: {len(targets_by_day)} dias")

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
    print(f"  hit@1={metrics['hit1_event']:.3f}  p@10={metrics['p10_event']:.3f}  mrr={metrics['mrr_event']:.3f}")

    benchmark_metrics = {
        "validation_split": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "val_start": VAL_START,
            "val_end": VAL_END,
            "horizon_days": HORIZON_DAYS,
        },
        f"{region}_2026": metrics,
        "selection_rationale": (
            "Poisson retreinado com dados ate 2025-12-31. "
            "Venceu LGBMRanker em p@10 e mrr para todas as regioes no benchmark jan-mai/2026."
        ),
    }

    payload = train_poisson_payload(
        data=data,
        region=region,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        horizon_days=HORIZON_DAYS,
        window=REGION_WINDOWS[region],
        benchmark_metrics=benchmark_metrics,
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
        "benchmark_metrics": benchmark_metrics,
    }
    meta_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Salvo: {model_path.name}")

    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Promove Poisson retreinado 2022-2025 para production/")
    parser.add_argument("--region", choices=["fortaleza", "rmf", "interior"], default=None,
                        help="Região específica (padrão: todas)")
    args = parser.parse_args()

    regions = [args.region] if args.region else ["fortaleza", "rmf", "interior"]
    results = {}
    for region in regions:
        results[region] = promote_region(region)

    print("\n[promote] Resumo final:")
    for region, sidecar in results.items():
        m = sidecar["benchmark_metrics"].get(f"{region}_2026", {})
        print(
            f"  {region.upper():12} | "
            f"hit@1={m.get('hit1_event', 0):.3f} | "
            f"p@10={m.get('p10_event', 0):.3f} | "
            f"mrr={m.get('mrr_event', 0):.3f} | "
            f"treino={sidecar['train_rows']:,} linhas"
        )


if __name__ == "__main__":
    main()
