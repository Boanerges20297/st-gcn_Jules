"""
Benchmark: Poisson Retreinado (2022-2025) vs LGBMRanker (LambdaMART)

Compara o Poisson com dados estendidos até dez/2025 contra um LGBMRanker
treinado com objective='lambdarank' otimizando diretamente P@10.

Uso:
    python scripts/benchmark_lgbm_ranker.py
    python scripts/benchmark_lgbm_ranker.py --region fortaleza --val-end 2026-03-31

Saídas:
    outputs/benchmarks/lgbm_ranker_benchmark_<timestamp>.md
    outputs/benchmarks/lgbm_ranker_benchmark_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cvli_stochastic_suite import (  # noqa: E402
    REGION_CONFIGS,
    build_region_datasets,
    evaluate_daily_predictions,
    load_processed_region,
    precision_at_k,
    recall_at_k,
    rank_overlap_at_k,
)
from src.core.fortaleza_poisson_backend import FEATURE_COLS  # noqa: E402

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ---------------------------------------------------------------------------
# LGBMRanker wrapper compatível com a interface fit/predict do benchmark
# ---------------------------------------------------------------------------

class LGBMRankerModel:
    """
    LambdaMART treinado por dia: cada dia = 1 query, cada bairro = 1 doc.

    Usa relevância binária (0/1) — adequado para dados de crime esparsos
    onde 80%+ dos registros são zero e contagens altas são raras.
    eval_at=[10] alinha o sinal de treino diretamente com P@10.
    """

    name = "lgbm_ranker"

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        if not HAS_LGBM:
            raise ImportError("lightgbm nao instalado. Execute: pip install lightgbm")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model: lgb.LGBMRanker | None = None

    def _build_groups(self, df: pd.DataFrame) -> np.ndarray:
        """Retorna array com o tamanho de cada grupo (query = dia)."""
        return df.groupby("sample_date", sort=False).size().to_numpy(dtype=np.int32)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LGBMRankerModel":
        # Garante ordenação por dia para que os grupos fiquem contíguos
        df = df.copy()
        df["_y"] = y
        df = df.sort_values("sample_date").reset_index(drop=True)
        y_sorted = df["_y"].to_numpy(dtype=float)
        X = df[FEATURE_COLS]
        groups = self._build_groups(df)
        # Relevância binária: qualquer crime = 1, zero = 0
        relevance = (y_sorted > 0).astype(np.int32)

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=[0, 1],
            eval_at=[10],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X, relevance, group=groups)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[FEATURE_COLS])


# ---------------------------------------------------------------------------
# Poisson retreinado — mesmo pipeline, datas estendidas
# ---------------------------------------------------------------------------

def build_poisson_retrained(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> object:
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", PoissonRegressor(alpha=1e-4, max_iter=400)),
    ])
    pipeline.fit(train_df[feature_cols], train_df["target_count"].to_numpy(dtype=float))
    return pipeline


# ---------------------------------------------------------------------------
# Avaliação diária — retorna predictions_by_day no formato esperado
# ---------------------------------------------------------------------------

def predictions_from_classical(
    model,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    pred = model.predict(val_df[feature_cols])
    pred_df = val_df[["sample_date", "node_idx"]].copy()
    pred_df["pred"] = pred
    return {
        day: group.sort_values("node_idx")["pred"].to_numpy(dtype=float)
        for day, group in pred_df.groupby("sample_date")
    }


def predictions_from_ranker(
    model: LGBMRankerModel,
    val_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    scores = model.model.predict(val_df[FEATURE_COLS])
    pred_df = val_df[["sample_date", "node_idx"]].copy()
    pred_df["pred"] = scores
    return {
        day: group.sort_values("node_idx")["pred"].to_numpy(dtype=float)
        for day, group in pred_df.groupby("sample_date")
    }


# ---------------------------------------------------------------------------
# Relatório de texto
# ---------------------------------------------------------------------------

def format_report(results: list[dict], region: str, train_end: str, val_start: str, val_end: str) -> str:
    lines = [
        f"# Benchmark: Poisson Retreinado vs LGBMRanker — {region.upper()}",
        "",
        f"- Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Treino: 2022-01-01 → {train_end}",
        f"- Validação: {val_start} → {val_end}",
        f"- Horizonte: 14 dias",
        "",
        "## Resultados",
        "",
        f"| {'Modelo':<25} | {'hit@1':>6} | {'p@5':>6} | {'p@10':>6} | {'p@20':>6} | {'r@10':>6} | {'mrr':>6} | {'dias':>5} |",
        f"|{'-'*27}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*7}|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['model']:<25} "
            f"| {m['hit1_event']:>6.3f} "
            f"| {m['p5_event']:>6.3f} "
            f"| {m['p10_event']:>6.3f} "
            f"| {m['p20_event']:>6.3f} "
            f"| {m['recall10_event']:>6.3f} "
            f"| {m['mrr_event']:>6.3f} "
            f"| {m['days_scored']:>5} |"
        )

    # Destaca vencedor por p@10
    best = max(results, key=lambda r: r["metrics"]["p10_event"])
    lines += [
        "",
        f"**Vencedor (p@10):** `{best['model']}` com p@10 = {best['metrics']['p10_event']:.3f}",
        "",
        "## Interpretação",
        "",
        "- **p@10**: dos 10 bairros mais alertados, quantos tiveram CVLI real (primária de promoção)",
        "- **hit@1**: o bairro #1 acertou? (operacionalmente crítico)",
        "- **mrr**: posição média do primeiro acerto no ranking",
    ]

    # Delta vs Poisson
    poisson_r = next((r for r in results if "poisson" in r["model"]), None)
    ranker_r = next((r for r in results if "lgbm" in r["model"]), None)
    if poisson_r and ranker_r:
        delta_p10 = ranker_r["metrics"]["p10_event"] - poisson_r["metrics"]["p10_event"]
        delta_mrr = ranker_r["metrics"]["mrr_event"] - poisson_r["metrics"]["mrr_event"]
        sign_p10 = "+" if delta_p10 >= 0 else ""
        sign_mrr = "+" if delta_mrr >= 0 else ""
        lines += [
            "",
            "## Delta LGBMRanker vs Poisson Retreinado",
            "",
            f"| Métrica | Delta |",
            f"|---------|-------|",
            f"| p@10    | {sign_p10}{delta_p10:.3f} ({sign_p10}{delta_p10*100:.1f}pp) |",
            f"| mrr     | {sign_mrr}{delta_mrr:.3f} |",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    region: str,
    train_end: str,
    val_start: str,
    val_end: str,
) -> list[dict]:
    print(f"[benchmark] Regiao: {region.upper()} | treino ate {train_end} | val {val_start} -> {val_end}")

    cfg = REGION_CONFIGS[region]
    data = load_processed_region(cfg)

    datasets = build_region_datasets(
        cfg=cfg,
        data=data,
        train_start=pd.Timestamp("2022-01-01"),
        train_end=pd.Timestamp(train_end),
        val_start=pd.Timestamp(val_start),
        val_end=pd.Timestamp(val_end),
        horizon_days=30,
    )

    train_df = datasets["classical_train"]
    val_df = datasets["classical_val"]
    targets_by_day = datasets["daily_val_targets"]

    n_val_days = len(targets_by_day)
    if n_val_days == 0:
        print(f"[benchmark] AVISO: nenhum dia de validação encontrado para {region}. Verifique val_end.")
        return []

    print(f"[benchmark] Treino: {len(train_df):,} linhas | Val: {n_val_days} dias")

    results = []

    # --- Poisson retreinado ---
    print("[benchmark] Treinando Poisson retreinado...")
    poisson = build_poisson_retrained(train_df, FEATURE_COLS)
    preds_poisson = predictions_from_classical(poisson, val_df, FEATURE_COLS)
    metrics_poisson = evaluate_daily_predictions(preds_poisson, targets_by_day, model_type="classic")
    results.append({"model": f"poisson_retreinado ({train_end[:4]})", "metrics": metrics_poisson})
    print(f"  p@10={metrics_poisson['p10_event']:.3f}  mrr={metrics_poisson['mrr_event']:.3f}  hit@1={metrics_poisson['hit1_event']:.3f}")

    # --- LGBMRanker ---
    if HAS_LGBM:
        print("[benchmark] Treinando LGBMRanker (LambdaMART)...")
        ranker = LGBMRankerModel(n_estimators=300, learning_rate=0.05, max_depth=5)
        # Passa o df completo (com sample_date) para montar os grupos
        train_df_sorted = train_df.sort_values("sample_date").reset_index(drop=True)
        ranker.fit(train_df_sorted, train_df_sorted["target_count"].to_numpy(dtype=float))
        preds_ranker = predictions_from_ranker(ranker, val_df)
        metrics_ranker = evaluate_daily_predictions(preds_ranker, targets_by_day, model_type="classic")
        results.append({"model": "lgbm_ranker (lambdamart)", "metrics": metrics_ranker})
        print(f"  p@10={metrics_ranker['p10_event']:.3f}  mrr={metrics_ranker['mrr_event']:.3f}  hit@1={metrics_ranker['hit1_event']:.3f}")
    else:
        print("[benchmark] lightgbm não encontrado — pulando LGBMRanker.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Poisson Retreinado vs LGBMRanker")
    parser.add_argument("--region", default="fortaleza", choices=["fortaleza", "rmf", "interior"])
    parser.add_argument("--train-end", default="2025-12-31", help="Fim do período de treino (YYYY-MM-DD)")
    parser.add_argument("--val-start", default="2026-01-01", help="Início da validação (YYYY-MM-DD)")
    parser.add_argument("--val-end", default="2026-05-31", help="Fim da validação (YYYY-MM-DD)")
    parser.add_argument("--all-regions", action="store_true", help="Roda para todas as regiões")
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: dict[str, list[dict]] = {}
    regions = ["fortaleza", "rmf", "interior"] if args.all_regions else [args.region]

    for region in regions:
        results = run_benchmark(
            region=region,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
        )
        all_results[region] = results

        if results:
            report = format_report(results, region, args.train_end, args.val_start, args.val_end)
            md_path = out_dir / f"lgbm_ranker_benchmark_{region}_{ts}.md"
            md_path.write_text(report, encoding="utf-8")
            print(f"\n[benchmark] Relatório salvo em: {md_path.name}")
            print(report)

    json_path = out_dir / f"lgbm_ranker_benchmark_{ts}.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "train_end": args.train_end,
                "val_start": args.val_start,
                "val_end": args.val_end,
                "results": all_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[benchmark] JSON salvo em: {json_path.name}")


if __name__ == "__main__":
    main()
