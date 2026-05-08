"""
====================================================================
SENTINELA V3 — TREINO COMPLETO + VALIDAÇÃO SOMBRA + EXPLICABILIDADE
====================================================================
Tentativa 57 — 2026-04-14

Pipeline:
  1. Treino completo: Jan/2024 → 31/Mar/2026 (820d)
  2. Validação Sombra: prediz 14d → compara CVLI real 01/Abr→14/Abr/2026
  3. Ranking atual (hoje) com contribuição de cada feature por bairro
  4. Recomendação de promoção (P@10>=45% OU P@20>=60%)
  5. Salva modelo em tests/Sentinela/ (NÃO promovido ainda)

Rodando:
  .\.venv\Scripts\python.exe tests/Sentinela/train_validate_v3.py
====================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, time, warnings, unicodedata, pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist
from lightgbm import LGBMRanker

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURACOES
# ─────────────────────────────────────────────────────────────────
BASE_PATH   = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
DATA_RAW    = os.path.join(BASE_PATH, "data", "raw")
DATA_PROC   = os.path.join(BASE_PATH, "data", "processed")
OUT_PATH    = os.path.join(BASE_PATH, "tests", "Sentinela")

CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
LATLON_FILE = os.path.join(DATA_RAW, "bairros_centros_latlong.json")

HORIZON     = 14
MIN_CVLI_MES = 0.4
TOP_N       = 40

# Corte para validação sombra: treina até aqui, valida nos 14d seguintes
SHADOW_CUTOFF = pd.Timestamp("2026-03-31")

# Pesos por natureza de ocorrência
PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO':  15.0,
    'PORTE ILEGAL ART 14':        12.0,
    'TRAFICO DE DROGAS':           8.0,
    'APREENSAO DE DROGAS':         6.0,
    'APREENSAO DE ENTORPECENTES':  6.0,
    'MANDADO DE PRISAO':           4.0,
    'MANDADO EM ABERTO':           3.5,
    'MANDADO DE PRISAO EM ABERTO': 3.5,
    'VEICULO ROUBADO RECUPERADO':  2.5,
    'VEICULO ROUBADO LOCALIZADO':  2.0,
    'ABANDONO DE MATERIAL':        1.5,
    'NAO INFORMADA':               0.5,
}

# ─────────────────────────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────────────────────────
def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def section(title, w=68):
    print(f"\n{'='*w}\n  {title}\n{'='*w}")

def pk_from_ranking(scores_N, cvli_horizon_N, k):
    top_pred = np.argsort(scores_N)[::-1][:k]
    top_real = np.argsort(cvli_horizon_N)[::-1][:k]
    return len(set(top_pred) & set(top_real)) / k

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9: return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

# ─────────────────────────────────────────────────────────────────
#  1. CARREGAR E PRÉ-PROCESSAR DADOS
# ─────────────────────────────────────────────────────────────────
def load_data():
    """Carrega CSVs, filtra Fortaleza, normaliza bairros."""
    print("[1/5] Carregando dados brutos...")

    df = pd.read_csv(CSV_ENRICH, low_memory=False)
    df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
    df["bairro"]  = df["bairro"].apply(norm)
    df["data"]    = pd.to_datetime(df["data"], errors="coerce")
    df            = df.dropna(subset=["data", "bairro"])
    df["is_cvli"] = (df["tipo"] == "cvli").astype("int8")
    df["is_cvp"]  = (df["tipo"] == "cvp").astype("int8")
    df["eh_feriado"] = df["eh_feriado"].fillna(False)

    df_t = pd.read_csv(CSV_TROPA, low_memory=False, encoding="utf-8-sig")
    df_t["bairro"] = df_t["bairro"].apply(norm)
    df_t["data"]   = pd.to_datetime(df_t["data"], errors="coerce")
    df_t           = df_t.dropna(subset=["data", "bairro"])
    df_t["peso_nat"] = df_t["natureza"].str.upper().str.strip().map(
        lambda x: PESO_NATUREZA.get(x, 1.0)
    )
    df_t["score_intel"] = (
        df_t["qtd_armas"] * 15.0
        + np.log1p(df_t["qtd_drogas"].fillna(0)) * 4.0
        + df_t["qtd_drogas_itens"] * 2.0
        + df_t["qtd_veiculos_apreendidos"] * 3.0
        + df_t["peso_nat"]
    ).astype("float32")

    # Top-40 bairros por CVLI/mês nos últimos 2 anos
    cutoff      = df["data"].max() - pd.Timedelta(days=730)
    cvli_rank   = df[df["data"] >= cutoff].groupby("bairro")["is_cvli"].sum().sort_values(ascending=False)
    cvli_pm     = cvli_rank / 24.0
    top_bairros = cvli_pm[cvli_pm > MIN_CVLI_MES].index[:TOP_N].tolist()

    print(f"    {len(top_bairros)} bairros | dados: {df['data'].min().date()} → {df['data'].max().date()}")
    print(f"    CVLI Fortaleza total: {df['is_cvli'].sum():,} | TROPA: {len(df_t):,} ocorrencias")

    return df, df_t, top_bairros

# ─────────────────────────────────────────────────────────────────
#  2. CONSTRUIR MATRIZ DE FEATURES
# ─────────────────────────────────────────────────────────────────
def build_feature_matrix(df, df_t, top_bairros, start_d=pd.Timestamp("2024-01-01")):
    """Constrói matriz (N, T) para cada feature lean."""
    print("[2/5] Construindo features lean (top-10 do V2)...")

    end_d    = df["data"].max()
    dates    = pd.date_range(start_d, end_d, freq="D")
    N, T     = len(top_bairros), len(dates)
    node_map = {b: i for i, b in enumerate(top_bairros)}
    date_map = {d: i for i, d in enumerate(dates)}

    # Matrizes brutas
    cvli_raw = np.zeros((N, T), dtype=np.float32)
    cvp_raw  = np.zeros((N, T), dtype=np.float32)
    intel_raw= np.zeros((N, T), dtype=np.float32)
    feriado  = np.zeros((N, T), dtype=np.float32)

    df_p = df[df["data"] >= start_d].copy()
    df_tp= df_t[df_t["data"] >= start_d].copy()

    for _, row in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: cvli_raw[ni, ti] = row["v"]

    for _, row in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: cvp_raw[ni, ti] = row["v"]

    for _, row in df_tp.groupby(["data","bairro"])["score_intel"].sum().reset_index().iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: intel_raw[ni, ti] = float(row["score_intel"])

    feriados_set = set(df_p[df_p["eh_feriado"]==True]["data"].dt.normalize().unique())
    for di, dt in enumerate(dates):
        if pd.Timestamp(dt.date()) in feriados_set:
            feriado[:, di] = 1.0

    # Vizinhança geográfica
    with open(LATLON_FILE, encoding="utf-8") as f:
        raw_ll = json.load(f)
    ll_norm  = {norm(k): v for k, v in raw_ll.items()}
    coords   = np.array([[ll_norm.get(b, {"lat":0,"long":0})["lat"],
                          ll_norm.get(b, {"lat":0,"long":0})["long"]]
                         for b in top_bairros], dtype=np.float32)
    dist_mat = cdist(coords, coords, "euclidean").astype(np.float32)
    adj_mask = (dist_mat < 0.05).astype(np.float32)
    np.fill_diagonal(adj_mask, 0)

    print("    Calculando features derivadas...", flush=True)

    feats = {}

    # ── F1: cvp_cvli_ratio calibrado por sqrt(hist_pct) ──
    # Evita falsos positivos em bairros com muito CVP mas pouco CVLI histórico
    # (ex: zonas comerciais com alta taxa de roubos mas sem homicídios)
    cvli_cum = np.zeros((N, T), np.float32)
    cvp_cum  = np.zeros((N, T), np.float32)
    for ni in range(N):
        cvli_cum[ni] = pd.Series(cvli_raw[ni]).expanding().sum().values
        cvp_cum[ni]  = pd.Series(cvp_raw[ni]).expanding().sum().values
    hist_total_cvli = cvli_cum[:, -1]
    hist_pct_base   = np.argsort(np.argsort(hist_total_cvli)) / max(N - 1, 1)
    sqrt_hist_pct   = np.sqrt(np.clip(hist_pct_base, 0, 1))[:, None]
    feats["cvp_cvli_ratio"] = (cvp_cum / (cvli_cum + 1)) * sqrt_hist_pct

    # ── F2: target_enc (#2 importância) ──
    target_enc = np.zeros((N, T), np.float32)
    for ni in range(N):
        target_enc[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
    feats["target_enc"] = target_enc

    # ── F3-5: cvp_ewma 7/14/30d (#3, #7, #8) ──
    for hl in [7, 14, 30]:
        arr = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(cvp_raw[ni]).ewm(halflife=hl, min_periods=1).mean().values
        feats[f"cvp_ewma_{hl}d"] = arr

    # ── F6-7: intel_ewma 7/14d (#4) ──
    for hl in [7, 14]:
        arr = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(intel_raw[ni]).ewm(halflife=hl, min_periods=1).mean().values
        feats[f"intel_ewma_{hl}d"] = arr

    # ── F8: nbr_cvli_30d (#5) ──
    roll30 = np.zeros((N, T), np.float32)
    for ni in range(N):
        roll30[ni] = pd.Series(cvli_raw[ni]).rolling(30, min_periods=1).sum().values
    nbr30 = np.zeros((N, T), np.float32)
    for ti in range(T):
        nbr30[:, ti] = adj_mask @ roll30[:, ti]
    feats["nbr_cvli_30d"] = nbr30

    # ── F9: hist_pct (#8) ──
    hist_total = cvli_raw.sum(axis=1)
    hist_pct   = np.argsort(np.argsort(hist_total)) / (N - 1)
    feats["hist_pct"] = np.broadcast_to(hist_pct[:, None], (N, T)).astype(np.float32).copy()

    # ── F10: inter_intel_cvli (#10) ──
    cvli_ewma14 = np.zeros((N, T), np.float32)
    for ni in range(N):
        cvli_ewma14[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=14, min_periods=1).mean().values
    feats["inter_intel_cvli"] = feats["intel_ewma_7d"] * cvli_ewma14

    # ── EWMA-Multi para ranking atual ──
    for hl in [3, 7, 14, 30, 60, 90]:
        arr = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl, min_periods=1).mean().values
        feats[f"cvli_ewma_{hl}d"] = arr

    feat_names_lgbm = [k for k in sorted(feats.keys()) if not k.startswith("cvli_ewma_")]
    print(f"    {len(feat_names_lgbm)} features LGBM | {len(feats)} features totais")

    return feats, feat_names_lgbm, dates, cvli_raw, node_map, date_map

# ─────────────────────────────────────────────────────────────────
#  3. TREINO LGBM + EWMA-MULTI
# ─────────────────────────────────────────────────────────────────
def train_models(feats, feat_names_lgbm, dates, cvli_raw, node_map, shadow_cutoff):
    """
    Treina LGBM Lean com dados até shadow_cutoff.
    Retorna modelo treinado + ranker.
    """
    print(f"[3/5] Treinando modelos (corte: {shadow_cutoff.date()})...")

    N, T = cvli_raw.shape
    date_map = {d: i for i, d in enumerate(dates)}
    cutoff_i = date_map.get(shadow_cutoff, T - HORIZON - 1)

    # Garante que cutoff_i está dentro dos limites
    cutoff_i = min(cutoff_i, T - HORIZON - 2)
    print(f"    Dias de treino: 0 → {cutoff_i} ({dates[0].date()} → {dates[cutoff_i].date()})")

    # Construir dataset de treino
    rows = []
    train_indices = list(range(90, cutoff_i, 2))  # step=2 para acelerar
    for ti in train_indices:
        if ti + HORIZON >= cutoff_i: continue
        targets_h = cvli_raw[:, ti+1:ti+HORIZON+1].sum(axis=1)
        if targets_h.sum() == 0: continue
        for ni in range(N):
            row = {f: float(feats[f][ni, ti]) for f in feat_names_lgbm}
            row["ni"]     = ni
            row["ti"]     = ti
            row["cvli_h"] = float(targets_h[ni])
            row["label"]  = min(int(targets_h[ni]), 5) + (1 if targets_h[ni] > 0 else 0)
            rows.append(row)

    df_tr = pd.DataFrame(rows).sort_values("ti")
    groups_tr = df_tr.groupby("ti").size().values
    print(f"    {len(df_tr):,} amostras de treino | {len(feat_names_lgbm)} features")

    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg",
        ndcg_eval_at=[5, 10],
        n_estimators=300, num_leaves=31,
        learning_rate=0.05, min_child_samples=10,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    ranker.fit(
        df_tr[feat_names_lgbm],
        df_tr["label"].astype("int32"),
        group=groups_tr
    )
    print(f"    Treino concluido em {time.time()-t0:.1f}s")

    return ranker, cutoff_i

# ─────────────────────────────────────────────────────────────────
#  4. VALIDAÇÃO SOMBRA (14 dias após cutoff)
# ─────────────────────────────────────────────────────────────────
def shadow_validation(ranker, feats, feat_names_lgbm, dates, cvli_raw, cutoff_i, top_bairros):
    """
    Prediz risco para os 14 dias após cutoff_i.
    Compara ranking predito vs CVLI real.
    """
    section("VALIDAÇÃO SOMBRA (Out-of-Sample)")
    N, T = cvli_raw.shape

    # Ponto de predição = último dia de treino
    pred_ti    = cutoff_i
    shadow_end = min(pred_ti + HORIZON + 1, T)
    shadow_days = shadow_end - pred_ti - 1

    print(f"\n  Predição em: {dates[pred_ti].date()}")
    print(f"  Janela sombra: {dates[pred_ti+1].date()} → {dates[shadow_end-1].date()} ({shadow_days} dias)")

    # Ground truth: CVLI real na janela sombra
    gt_cvli = cvli_raw[:, pred_ti+1:shadow_end].sum(axis=1)
    bairros_com_cvli = (gt_cvli > 0).sum()
    print(f"  CVLI real na sombra: {int(gt_cvli.sum())} eventos em {bairros_com_cvli} bairros")

    # ── Score LGBM no ponto de predição ──
    xi = np.array([[float(feats[f][ni, pred_ti]) for f in feat_names_lgbm] for ni in range(N)])
    xi_df = pd.DataFrame(xi, columns=feat_names_lgbm)
    scores_lgbm = ranker.predict(xi_df)

    # ── Score EWMA-Multi ──
    weights = {"cvli_ewma_7d":0.40, "cvli_ewma_14d":0.35,
               "cvli_ewma_30d":0.15, "cvli_ewma_90d":0.10}
    scores_ewma = np.zeros(N, np.float32)
    for fname, w in weights.items():
        scores_ewma += w * feats[fname][:, pred_ti]

    # ── Ensemble (50/50) ──
    scores_ens = 0.5 * normalize_scores(scores_ewma) + 0.5 * normalize_scores(scores_lgbm)

    # ── Métricas ──
    print("\n  ┌─────────────────┬────────┬────────┐")
    print("  │ Modelo          │  P@10  │  P@20  │")
    print("  ├─────────────────┼────────┼────────┤")
    results = {}
    for nome, sc in [("LGBM-Lean", scores_lgbm),
                     ("EWMA-Multi", scores_ewma),
                     ("Ensemble-V3", scores_ens)]:
        p10 = pk_from_ranking(sc, gt_cvli, 10) * 100
        p20 = pk_from_ranking(sc, gt_cvli, 20) * 100
        results[nome] = {"P@10": p10, "P@20": p20}
        print(f"  │ {nome:<15}  │ {p10:>5.1f}% │ {p20:>5.1f}% │")
    print("  ├─────────────────┼────────┼────────┤")
    print("  │ Chance (random) │  25.0% │  50.0% │")
    print("  └─────────────────┴────────┴────────┘")

    # ── Ranking predito dos Top-20 ──
    print("\n  Ranking Predito (Ensemble) vs Real:")
    print(f"  {'#':<3}  {'Bairro':<30}  {'Score':>6}  {'CVLI Real':>9}  {'Acerto?':>7}")
    print("  " + "-"*65)
    top20_pred = np.argsort(scores_ens)[::-1][:20]
    top10_real = set(np.argsort(gt_cvli)[::-1][:10])
    top20_real = set(np.argsort(gt_cvli)[::-1][:20])
    for rank_i, ni in enumerate(top20_pred, 1):
        acerto = "✅ P@10" if ni in top10_real else ("✅ P@20" if ni in top20_real else "—")
        print(f"  {rank_i:<3}  {top_bairros[ni]:<30}  {scores_ens[ni]:>6.3f}  {int(gt_cvli[ni]):>9}  {acerto}")

    # ── Bairros que tiveram CVLI mas não foram previstos (falsos negativos) ──
    missed = [ni for ni in np.argsort(gt_cvli)[::-1][:10] if ni not in set(top20_pred[:10])]
    if missed:
        print(f"\n  Bairros com CVLI real NÃO capturados no top-10:")
        for ni in missed:
            print(f"    - {top_bairros[ni]}: {int(gt_cvli[ni])} CVLI (rank predito: {list(np.argsort(scores_ens)[::-1]).index(ni)+1})")

    # ── Decisão de promoção ──
    best_p10 = max(r["P@10"] for r in results.values())
    best_p20 = max(r["P@20"] for r in results.values())
    promover = best_p10 >= 45 or best_p20 >= 60
    section("DECISÃO DE PROMOÇÃO")
    print(f"\n  P@10 validação sombra: {best_p10:.1f}%  (threshold: 45%)")
    print(f"  P@20 validação sombra: {best_p20:.1f}%  (threshold: 60%)")
    if promover:
        print("\n  🟢 RECOMENDAÇÃO: PROMOVER")
        print("     Critério atingido. Modelo pronto para review de promoção.")
    else:
        print("\n  🔴 RECOMENDAÇÃO: NÃO PROMOVER")
        print("     Abaixo dos thresholds. Manter em tests/ e iterar.")

    return scores_lgbm, scores_ewma, scores_ens, gt_cvli, results, promover

# ─────────────────────────────────────────────────────────────────
#  5. RANKING ATUAL + EXPLICABILIDADE
# ─────────────────────────────────────────────────────────────────
def generate_explainability(ranker, feats, feat_names_lgbm, dates, top_bairros):
    """
    Gera ranking de hoje com contribuição de cada feature por bairro.
    """
    section("RANKING ATUAL + EXPLICABILIDADE")

    N    = len(top_bairros)
    ti   = len(dates) - 1  # último dia disponível

    print(f"\n  Data de referência: {dates[ti].date()} (último dado disponível)")

    # Scores no último dia
    xi     = np.array([[float(feats[f][ni, ti]) for f in feat_names_lgbm] for ni in range(N)])
    xi_df  = pd.DataFrame(xi, columns=feat_names_lgbm)
    scores_lgbm = ranker.predict(xi_df)

    weights = {"cvli_ewma_7d":0.40, "cvli_ewma_14d":0.35,
               "cvli_ewma_30d":0.15, "cvli_ewma_90d":0.10}
    scores_ewma = np.zeros(N, np.float32)
    for fname, w in weights.items():
        scores_ewma += w * feats[fname][:, ti]

    scores_ens = 0.5 * normalize_scores(scores_ewma) + 0.5 * normalize_scores(scores_lgbm)

    # Feature importance do ranker
    fi = ranker.feature_importances_
    fi_total = fi.sum()

    # Tabela de ranking com contribuições
    ranking = []
    for rank_i, ni in enumerate(np.argsort(scores_ens)[::-1], 1):
        row = {
            "Rank":   rank_i,
            "Bairro": top_bairros[ni],
            "Score_Ensemble": round(float(scores_ens[ni]), 4),
            "Score_LGBM":     round(float(normalize_scores(scores_lgbm)[ni]), 4),
            "Score_EWMA":     round(float(normalize_scores(scores_ewma)[ni]), 4),
        }
        for j, fname in enumerate(feat_names_lgbm):
            row[f"feat_{fname}"] = round(float(feats[fname][ni, ti]), 4)
        ranking.append(row)

    df_rank = pd.DataFrame(ranking)

    # Print top-20
    print(f"\n  {'Rank':<5}  {'Bairro':<30}  {'Score':>7}  {'cvp_ratio':>10}  {'intel_14d':>10}  {'nbr30d':>7}")
    print("  " + "-"*78)
    for _, row in df_rank.head(20).iterrows():
        print(f"  {int(row['Rank']):<5}  {row['Bairro']:<30}  {row['Score_Ensemble']:>7.4f}"
              f"  {row['feat_cvp_cvli_ratio']:>10.3f}"
              f"  {row['feat_intel_ewma_14d']:>10.2f}"
              f"  {row['feat_nbr_cvli_30d']:>7.2f}")

    # Feature importance global
    print(f"\n  Feature Importance Global (LGBM):")
    print(f"  {'Feature':<25}  {'Importância':>12}  {'Peso%':>7}")
    print("  " + "-"*48)
    for j, fname in enumerate(feat_names_lgbm):
        pct = fi[j] / fi_total * 100
        bar = "█" * int(pct / 2)
        print(f"  {fname:<25}  {fi[j]:>12,.0f}  {pct:>6.1f}%  {bar}")

    return df_rank, scores_ens

# ─────────────────────────────────────────────────────────────────
#  6. SALVAR ARTEFATOS
# ─────────────────────────────────────────────────────────────────
def save_artifacts(ranker, feats, feat_names_lgbm, df_rank, results, promover, top_bairros, dates):
    section("SALVANDO ARTEFATOS")

    modelo_path = os.path.join(OUT_PATH, "lgbm_lean_v3.pkl")
    pipeline_path = os.path.join(OUT_PATH, "feat_pipeline_v3.pkl")
    ranking_csv   = os.path.join(OUT_PATH, "ranking_atual_v3.csv")
    report_txt    = os.path.join(OUT_PATH, "train_validate_v3_report.txt")

    # Modelo
    with open(modelo_path, "wb") as f:
        pickle.dump({"ranker": ranker, "feat_names": feat_names_lgbm,
                     "top_bairros": top_bairros, "trained_at": str(datetime.now()),
                     "shadow_cutoff": str(SHADOW_CUTOFF.date()),
                     "results_shadow": results, "promover": promover}, f)
    print(f"\n  [OK] Modelo: {modelo_path}")

    # Pipeline de features
    with open(pipeline_path, "wb") as f:
        pickle.dump({"feat_names_lgbm": feat_names_lgbm,
                     "top_bairros": top_bairros,
                     "dates_range": (str(dates[0].date()), str(dates[-1].date())),
                     "ewma_weights": {"cvli_ewma_7d":0.40,"cvli_ewma_14d":0.35,
                                      "cvli_ewma_30d":0.15,"cvli_ewma_90d":0.10}}, f)
    print(f"  [OK] Pipeline: {pipeline_path}")

    # Ranking atual
    df_rank.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    print(f"  [OK] Ranking: {ranking_csv}")

    # Relatório
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write("SENTINELA V3 — TREINO COMPLETO + VALIDAÇÃO SOMBRA\n")
        f.write(f"Tentativa 57 | {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Shadow cutoff: {SHADOW_CUTOFF.date()} → validação: {14} dias seguintes\n")
        f.write("=" * 68 + "\n\n")
        f.write("RESULTADOS VALIDAÇÃO SOMBRA:\n")
        for nome, r in results.items():
            f.write(f"  {nome:<15}: P@10={r['P@10']:.1f}%  P@20={r['P@20']:.1f}%\n")
        f.write(f"\nRECOMENDAÇÃO: {'PROMOVER' if promover else 'NÃO PROMOVER'}\n")
        f.write("\nRANKING ATUAL (top-20):\n")
        f.write(df_rank[["Rank","Bairro","Score_Ensemble","feat_cvp_cvli_ratio","feat_intel_ewma_14d","feat_nbr_cvli_30d"]].head(20).to_string(index=False))
        f.write(f"\n\nNOTA: Modelo salvo em tests/ aguardando revisão.\n")
        f.write("Para promover: copiar lgbm_lean_v3.pkl para models/active/\n")
    print(f"  [OK] Relatório: {report_txt}")

    if promover:
        print("\n  ⚠️  Para promover após revisão manual:")
        print(f"      copy \"{modelo_path}\" \"{os.path.join(BASE_PATH, 'models', 'active', 'lgbm_lean_v3.pkl')}\"")
    else:
        print("\n  ℹ️  Modelo mantido em tests/ — thresholds não atingidos.")

# ─────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def run():
    section("SENTINELA V3 — TREINO COMPLETO + VALIDAÇÃO SOMBRA (Tentativa 57)")
    print(f"\n  Inicio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Shadow cutoff: {SHADOW_CUTOFF.date()}")
    print(f"  Threshold promoção: P@10≥45% OU P@20≥60%")

    # Step 1: Carregar dados
    df, df_t, top_bairros = load_data()

    # Step 2: Construir features
    feats, feat_names_lgbm, dates, cvli_raw, node_map, date_map = \
        build_feature_matrix(df, df_t, top_bairros)

    # Step 3: Treinar
    ranker, cutoff_i = train_models(feats, feat_names_lgbm, dates, cvli_raw, node_map, SHADOW_CUTOFF)

    # Step 4: Validação sombra
    scores_lgbm, scores_ewma, scores_ens, gt_cvli, results, promover = \
        shadow_validation(ranker, feats, feat_names_lgbm, dates, cvli_raw, cutoff_i, top_bairros)

    # Step 5: Ranking atual + explicabilidade
    df_rank, _ = generate_explainability(ranker, feats, feat_names_lgbm, dates, top_bairros)

    # Step 6: Salvar artefatos
    save_artifacts(ranker, feats, feat_names_lgbm, df_rank, results, promover, top_bairros, dates)

    section("RESUMO FINAL")
    print(f"\n  Validação sombra ({SHADOW_CUTOFF.date()} → +{HORIZON}d):")
    for nome, r in results.items():
        m10 = "✅" if r["P@10"] >= 45 else "❌"
        m20 = "✅" if r["P@20"] >= 60 else "❌"
        print(f"    {nome:<15}: P@10={r['P@10']:.1f}% {m10}  P@20={r['P@20']:.1f}% {m20}")
    print(f"\n  Decisão: {'🟢 PROMOVER' if promover else '🔴 MANTER EM tests/'}")
    print(f"\n  Concluido: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":
    run()
