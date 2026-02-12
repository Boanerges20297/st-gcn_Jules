"""
Validação do pipeline ATUAL do app (compute_predictions de app.py)
usando dados recentes mapeados para nós.

Objetivo:
- Medir se o ranking de risco (final_risk) está realmente concentrando
  os CVLIs nos nós de maior score (P@10, P@20, cobertura etc.).

Uso:
    python scripts/validate_current_app.py
"""

import os
import sys
from datetime import datetime

import numpy as np

# Garantir que o diretório raiz esteja no PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import load_data_and_models, compute_predictions  # noqa: E402
from scripts.validate_recent_data import (  # noqa: E402
    load_recent_data,
    load_graph_structure,
    map_events_to_nodes,
)


def evaluate_rank_quality(risk_scores: np.ndarray, y_true_total: np.ndarray, k_values=(10, 20)):
    """
    Avalia a qualidade do ranking de risco.

    - risk_scores: vetor (N,) com score final do app (0–100).
    - y_true_total: vetor (N,) com total de eventos reais no período.

    Retorna dicionário com P@k e cobertura.
    """
    metrics = {}

    # Ordena nós por risco (maior primeiro)
    order = np.argsort(-risk_scores)

    total_events = float(y_true_total.sum())

    for k in k_values:
        top_k_idx = order[:k]
        hits = (y_true_total[top_k_idx] > 0).sum()
        events_in_top_k = y_true_total[top_k_idx].sum()

        p_at_k = hits / max(k, 1)
        coverage = events_in_top_k / total_events if total_events > 0 else 0.0

        metrics[f"P@{k}"] = float(p_at_k)
        metrics[f"coverage@{k}"] = float(coverage)
        metrics[f"events_in_top{k}"] = float(events_in_top_k)

    return metrics


def main():
    print("\n" + "#" * 80)
    print("VALIDAÇÃO DO PIPELINE ATUAL DO APP (compute_predictions)")
    print("Data de validação:", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("#" * 80)

    # 1) Carregar dados recentes e estrutura do grafo (ground truth)
    df_recent = load_recent_data()
    graph_data, nodes_gdf = load_graph_structure()
    event_counts, cvli_df = map_events_to_nodes(df_recent, nodes_gdf)

    all_dates = sorted(event_counts.keys())
    num_nodes = graph_data["node_features"].shape[0]

    # Montar Y_true total por nó no período
    y_true_total = np.zeros(num_nodes, dtype=float)
    for date in all_dates:
        for node_idx, count in event_counts[date].items():
            if 0 <= node_idx < num_nodes:
                y_true_total[node_idx] += float(count)

    print(f"\nPeríodo de validação: {all_dates[0]} a {all_dates[-1]}")
    print(f"Total de eventos CVLI no período: {int(y_true_total.sum())}")
    print(f"Nós com pelo menos 1 evento: {(y_true_total > 0).sum()}")

    # 2) Carregar modelos e dados do app (usa o mesmo processed_graph_data.pkl)
    print("\nCarregando dados e modelos do app.py...")
    load_data_and_models()

    # 3) Rodar compute_predictions uma vez (como o endpoint /api/risk)
    print("Executando compute_predictions() ...")
    meta, results, final_risk, stgcn_score, hist_sum = compute_predictions()

    risk_scores = np.array([r["risk_score"] for r in results], dtype=float)

    if risk_scores.shape[0] != num_nodes:
        print(
            f"\n[AVISO] Número de nós em risk_scores ({risk_scores.shape[0]}) "
            f"difere de num_nodes em graph_data ({num_nodes}). "
            "Os índices podem não estar perfeitamente alinhados."
        )

    # 4) Avaliar qualidade do ranking
    print("\n" + "=" * 80)
    print("MÉTRICAS DE RANKING (usar para meta de 'previsão > 60%')")
    print("=" * 80)

    metrics = evaluate_rank_quality(risk_scores, y_true_total, k_values=(10, 20))

    for k in (10, 20):
        print(f"\nTop-{k}:")
        print(f"  P@{k}:         {metrics[f'P@{k}']:.2%}")
        print(
            f"  Cobertura@{k}: {metrics[f'coverage@{k}']:.2%} "
            f"({metrics[f'events_in_top{k}']:.0f}/{int(y_true_total.sum())} eventos)"
        )

    # 5) Resumo e salvamento opcional
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_path = os.path.join(
        reports_dir, f"validate_current_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    import json

    payload = {
        "timestamp": datetime.now().isoformat(),
        "period": {"start": str(all_dates[0]), "end": str(all_dates[-1])},
        "total_events": int(y_true_total.sum()),
        "nodes_with_events": int((y_true_total > 0).sum()),
        "metrics": metrics,
        "meta_counts": meta.get("counts", {}),
        "meta_stats": {
            "top5_mean": meta.get("stats_top5_mean"),
            "top10_mean": meta.get("stats_top10_mean"),
            "overall_mean": meta.get("stats_overall_mean"),
            "overall_std": meta.get("stats_overall_std"),
        },
        "ranking_status": meta.get("ranking_info", {}),
        "ranking_source": meta.get("ranking_source"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"✓ Resultados de validação salvos em: {out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

