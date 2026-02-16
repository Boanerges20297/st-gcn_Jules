"""
Validação do pipeline atual do app (`compute_predictions` de app.py)
usando diretamente o arquivo oficial bruto:

    data/raw/dados_status_ocorrencias_gerais.json

Objetivo:
- Medir P@10, P@20 e cobertura em cima dos CVLIs reais dos
  últimos N dias (por padrão 7), sem depender de recortes auxiliares.

Uso:
    # Usando 7 dias de janela (default)
    python scripts/validate_current_app_from_raw.py

    # Usando outra janela (ex.: 14 dias)
    set VALIDATION_DAYS_BACK=14  # Windows / PowerShell
    python scripts/validate_current_app_from_raw.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Garantir que o diretório raiz esteja no PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import load_data_and_models, compute_predictions  # noqa: E402
from scripts.validate_recent_data import (  # noqa: E402
    load_graph_structure,
    map_events_to_nodes,
)


RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "dados_status_ocorrencias_gerais.json")

# Janela de dias para validação (usando dados oficiais)
VALIDATION_DAYS_BACK = int(os.getenv("VALIDATION_DAYS_BACK", "7"))


def load_official_recent_cvli() -> pd.DataFrame:
    """
    Carrega o arquivo bruto oficial e retorna apenas CVLIs recentes.

    Estrutura esperada de cada registro:
        {
            "tipo": "cvli" ou "cvp",
            "data": "YYYY-MM-DD",
            "latitude": "...",
            "longitude": "...",
            ...
        }
    """
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {RAW_FILE}")

    print("\n" + "=" * 80)
    print("CARREGANDO DADOS OFICIAIS (dados_status_ocorrencias_gerais.json)")
    print("=" * 80)

    # Leitura via pandas (arquivo é um array JSON grande)
    with open(RAW_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    df = pd.DataFrame(data)

    # Normalizar colunas mínimas
    if "tipo" not in df.columns:
        raise ValueError("Coluna 'tipo' não encontrada no arquivo bruto.")
    if "data" not in df.columns:
        raise ValueError("Coluna 'data' não encontrada no arquivo bruto.")

    # Alguns registros podem trazer 'data' como lista/objeto; normalizar para string simples
    def _normalize_date(val):
        if isinstance(val, list) and val:
            return str(val[0])
        if isinstance(val, dict):
            # tenta chaves comuns
            for k in ("data", "date", "timestamp"):
                if k in val:
                    return str(val[k])
            return str(val)
        return str(val)

    df["data"] = df["data"].apply(_normalize_date)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    # Remover linhas sem data válida
    df = df.dropna(subset=["data"])

    # Filtrar apenas CVLI consolidados
    df = df[df["tipo"].str.lower() == "cvli"].copy()
    if "consolidado" in df.columns:
        df = df[df["consolidado"] == "Sim"].copy()

    # Remover linhas sem coordenadas
    df = df.dropna(subset=["latitude", "longitude"])

    # Converter coordenadas para float
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    # Selecionar últimos N dias em relação à data máxima disponível
    max_date = df["data"].max()
    start_date = max_date - timedelta(days=VALIDATION_DAYS_BACK - 1)

    df_recent = df[(df["data"] >= start_date) & (df["data"] <= max_date)].copy()

    print(f"✓ Período considerado: {start_date.date()} até {max_date.date()} "
          f"({VALIDATION_DAYS_BACK} dias)")
    print(f"✓ Total de registros CVLI no período: {len(df_recent)}")

    # Renomear/garantir colunas usadas pelo map_events_to_nodes
    # (ele espera 'data', 'tipo', 'latitude', 'longitude')
    if "tipo_evento" in df_recent.columns:
        df_recent["tipo_evento"] = df_recent["tipo_evento"].fillna("")

    return df_recent


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
    print("USANDO DADOS OFICIAIS: dados_status_ocorrencias_gerais.json")
    print("Data de validação:", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("#" * 80)

    # 1) Carregar CVLIs oficiais recentes
    df_recent = load_official_recent_cvli()

    # 2) Carregar estrutura do grafo (mesma de produção)
    from scripts.validate_recent_data import load_graph_structure as _lg

    graph_data, nodes_gdf = _lg()

    # 3) Mapear eventos oficiais para nós
    event_counts, cvli_df = map_events_to_nodes(df_recent, nodes_gdf)

    all_dates = sorted(event_counts.keys())
    num_nodes = graph_data["node_features"].shape[0]

    # Montar Y_true total por nó no período
    y_true_total = np.zeros(num_nodes, dtype=float)
    for date in all_dates:
        for node_idx, count in event_counts[date].items():
            if 0 <= node_idx < num_nodes:
                y_true_total[node_idx] += float(count)

    print(f"\nPeríodo de validação efetivo: {all_dates[0]} a {all_dates[-1]}")
    print(f"Total de eventos CVLI no período: {int(y_true_total.sum())}")
    print(f"Nós com pelo menos 1 evento: {(y_true_total > 0).sum()}")

    # 4) Carregar modelos e dados do app (usa o mesmo processed_graph_data.pkl)
    print("\nCarregando dados e modelos do app.py...")
    load_data_and_models()

    # 5) Rodar compute_predictions uma vez (como o endpoint /api/risk)
    print("Executando compute_predictions() ...")
    meta, results, final_risk, stgcn_score, hist_sum = compute_predictions()

    risk_scores = np.array([r["risk_score"] for r in results], dtype=float)

    if risk_scores.shape[0] != num_nodes:
        print(
            f"\n[AVISO] Número de nós em risk_scores ({risk_scores.shape[0]}) "
            f"difere de num_nodes em graph_data ({num_nodes}). "
            "Os índices podem não estar perfeitamente alinhados."
        )

    # 6) Avaliar qualidade do ranking
    print("\n" + "=" * 80)
    print("MÉTRICAS DE RANKING (dados oficiais)")
    print("=" * 80)

    metrics = evaluate_rank_quality(risk_scores, y_true_total, k_values=(10, 20))

    for k in (10, 20):
        print(f"\nTop-{k}:")
        print(f"  P@{k}:         {metrics[f'P@{k}']:.2%}")
        print(
            f"  Cobertura@{k}: {metrics[f'coverage@{k}']:.2%} "
            f"({metrics[f'events_in_top{k}']:.0f}/{int(y_true_total.sum())} eventos)"
        )

    # 7) Salvar resumo em reports
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_path = os.path.join(
        reports_dir,
        f"validate_current_app_from_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "period": {"start": str(all_dates[0]), "end": str(all_dates[-1])},
        "validation_days_back": VALIDATION_DAYS_BACK,
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
    print(f"✓ Resultados de validação (dados oficiais) salvos em: {out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

