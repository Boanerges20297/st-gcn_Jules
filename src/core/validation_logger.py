from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from .orchestrator import StateOrchestrator, normalize_name


def _status_for_p10(p10: float) -> str:
    if p10 >= 0.4:
        return "✅"
    if p10 >= 0.2:
        return "⚠️"
    return "🚨"


def _load_scores(project_root: str, orchestrator: StateOrchestrator | None):
    if orchestrator is not None:
        scores_map = orchestrator.get_combined_risk()
        return orchestrator, scores_map
    temp_orchestrator = StateOrchestrator(project_root)
    scores_map = temp_orchestrator.get_combined_risk()
    return temp_orchestrator, scores_map


def append_validation_log(
    df_eval: pd.DataFrame,
    project_root: str,
    window_days: int = 14,
    source_label: str = "startup",
    orchestrator: StateOrchestrator | None = None,
    model_label: str | None = None,
) -> bool:
    """
    Registra uma sessão regional no VALIDATION_LOG.md.
    Retorna True quando uma nova sessão é gravada e False quando é ignorada.
    """
    if df_eval is None or df_eval.empty:
        print("  - Validação ignorada: dataframe vazio.")
        return False

    print(f"\n📊 Iniciando Validação Regional Detalhada ({source_label} - últimos {window_days} dias)...")
    try:
        orchestrator_ref, scores_map = _load_scores(project_root, orchestrator)
        if not scores_map:
            print("  - Não foi possível obter scores do Orquestrador para validação.")
            return False
    except Exception as exc:
        print(f"  - Erro ao carregar StateOrchestrator: {exc}")
        return False

    node_to_region = {}
    for reg, spec in orchestrator_ref.specialists.items():
        for _, row in spec["data"]["nodes_gdf"].iterrows():
            node_to_region[normalize_name(str(row["name"]))] = reg

    df_eval = df_eval.copy()
    df_eval["data"] = pd.to_datetime(df_eval["data"], errors="coerce")
    max_date = df_eval["data"].max()
    if pd.isna(max_date):
        print("  - Erro: nenhuma data válida encontrada na base.")
        return False

    cutoff_date = max_date - pd.Timedelta(days=window_days)
    mask_time = df_eval["data"] >= cutoff_date
    mask_cvli = df_eval["tipo"].astype(str).str.lower() == "cvli"
    cvlis = df_eval[mask_time & mask_cvli].copy()
    if cvlis.empty:
        print(f"  - Nenhum CVLI nos últimos {window_days} dias para validar.")
        return False

    cvlis["node_norm"] = cvlis["bairro"].apply(normalize_name)
    cvlis["region"] = cvlis["node_norm"].map(node_to_region)

    start_d = cvlis["data"].min().strftime("%Y-%m-%d")
    end_d = cvlis["data"].max().strftime("%Y-%m-%d")
    model_label = model_label or getattr(orchestrator_ref, "HYBRID_MODEL_LABEL", "arquitetura_ativa")
    session_key = f"{source_label}|{start_d}|{end_d}|{model_label}"
    log_path = os.path.join(project_root, "VALIDATION_LOG.md")

    if os.path.exists(log_path):
        existing = open(log_path, "r", encoding="utf-8").read()
        if f"<!-- validation-session: {session_key} -->" in existing:
            print(f"  - Validação já registrada para {session_key}.")
            return False

    regions = ["fortaleza", "rmf", "interior"]
    results = []
    for reg in regions:
        reg_cvlis = cvlis[cvlis["region"] == reg]
        total_bruto = len(reg_cvlis)
        reg_nodes = [n for n, r in node_to_region.items() if r == reg]
        reg_scores = {n: scores_map.get(n, 0.0) for n in reg_nodes}
        top_pred = sorted(reg_scores.keys(), key=lambda x: reg_scores[x], reverse=True)
        gt_bairros = set(reg_cvlis["node_norm"].unique())
        hits10 = len(gt_bairros.intersection(set(top_pred[:10])))
        hits20 = len(gt_bairros.intersection(set(top_pred[:20])))
        p10 = hits10 / 10.0
        p20 = hits20 / 20.0
        r10 = hits10 / total_bruto if total_bruto > 0 else 0.0
        r20 = hits20 / total_bruto if total_bruto > 0 else 0.0
        results.append(
            {
                "region": reg.upper(),
                "total": total_bruto,
                "hits": hits10,
                "p10": f"{p10*100:.1f}%",
                "p20": f"{p20*100:.1f}%",
                "r10": f"{r10*100:.1f}%",
                "r20": f"{r20*100:.1f}%",
                "status": _status_for_p10(p10),
            }
        )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n<!-- validation-session: {session_key} -->\n")
        f.write(f"\n### 🔄 Sessão de Validação: {now_str}\n")
        f.write(f"**Período Gabarito:** {start_d} a {end_d}\n\n")
        f.write(f"**Origem:** {source_label}\n\n")
        f.write(f"**Arquitetura:** {model_label}\n\n")
        f.write("| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |\n")
        f.write("|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|\n")
        for res in results:
            region_padded = res["region"].ljust(9)
            f.write(
                f"| {region_padded} | {res['total']:^12} | {res['hits']:^10} | {res['p10']:^5} | "
                f"{res['p20']:^5} | {res['r10']:^6} | {res['r20']:^6} | {res['status']:^6} |\n"
            )
        f.write("\n---\n")

    print(f"  ✅ Validação regional concluída e registrada em {log_path}")
    return True
