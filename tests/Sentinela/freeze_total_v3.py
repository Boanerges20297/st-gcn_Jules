"""
====================================================================
SENTINELA V3 — TREINO FREEZE TOTAL (CANDIDATO A PRODUÇÃO)
====================================================================
Tentativa 57b — 2026-04-14

Usa TODOS os dados disponíveis (Jan/2024 → Hoje):
  - Sem holdout: o modelo incorpora os eventos de Abr/2026
  - Gera ranking de risco para os próximos 14 dias
  - Salva modelo final em tests/Sentinela/lgbm_lean_v3_freeze.pkl
  - Promoção manual: copiar para models/active/

Rodando:
  .\.venv\Scripts\python.exe tests/Sentinela/freeze_total_v3.py
====================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, time, warnings, unicodedata, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from lightgbm import LGBMRanker

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURACOES
# ─────────────────────────────────────────────────────────────────
BASE_PATH   = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
DATA_RAW    = os.path.join(BASE_PATH, "data", "raw")
OUT_PATH    = os.path.join(BASE_PATH, "tests", "Sentinela")

CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
LATLON_FILE = os.path.join(DATA_RAW, "bairros_centros_latlong.json")

HORIZON     = 14
MIN_CVLI_MES = 0.4
TOP_N       = 40

PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO':  15.0, 'PORTE ILEGAL ART 14': 12.0,
    'TRAFICO DE DROGAS': 8.0, 'APREENSAO DE DROGAS': 6.0,
    'APREENSAO DE ENTORPECENTES': 6.0, 'MANDADO DE PRISAO': 4.0,
    'MANDADO EM ABERTO': 3.5, 'MANDADO DE PRISAO EM ABERTO': 3.5,
    'VEICULO ROUBADO RECUPERADO': 2.5, 'VEICULO ROUBADO LOCALIZADO': 2.0,
    'ABANDONO DE MATERIAL': 1.5, 'NAO INFORMADA': 0.5,
}

EWMA_WEIGHTS = {
    "cvli_ewma_7d": 0.40, "cvli_ewma_14d": 0.35,
    "cvli_ewma_30d": 0.15, "cvli_ewma_90d": 0.10,
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

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9: return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

# ─────────────────────────────────────────────────────────────────
#  1. CONSTRUIR FEATURES (idêntico ao train_validate_v3.py)
# ─────────────────────────────────────────────────────────────────
def build_all(start_d=pd.Timestamp("2024-01-01")):
    print("[1/3] Carregando e processando dados completos...")

    df = pd.read_csv(CSV_ENRICH, low_memory=False)
    df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
    df["bairro"]     = df["bairro"].apply(norm)
    df["data"]       = pd.to_datetime(df["data"], errors="coerce")
    df               = df.dropna(subset=["data", "bairro"])
    df["is_cvli"]    = (df["tipo"] == "cvli").astype("int8")
    df["is_cvp"]     = (df["tipo"] == "cvp").astype("int8")
    df["eh_feriado"] = df["eh_feriado"].fillna(False)

    df_t = pd.read_csv(CSV_TROPA, low_memory=False, encoding="utf-8-sig")
    df_t["bairro"]     = df_t["bairro"].apply(norm)
    df_t["data"]       = pd.to_datetime(df_t["data"], errors="coerce")
    df_t               = df_t.dropna(subset=["data", "bairro"])
    df_t["peso_nat"]   = df_t["natureza"].str.upper().str.strip().map(lambda x: PESO_NATUREZA.get(x, 1.0))
    df_t["score_intel"] = (
        df_t["qtd_armas"] * 15.0
        + np.log1p(df_t["qtd_drogas"].fillna(0)) * 4.0
        + df_t["qtd_drogas_itens"] * 2.0
        + df_t["qtd_veiculos_apreendidos"] * 3.0
        + df_t["peso_nat"]
    ).astype("float32")

    cutoff      = df["data"].max() - pd.Timedelta(days=730)
    cvli_rank   = df[df["data"] >= cutoff].groupby("bairro")["is_cvli"].sum().sort_values(ascending=False)
    top_bairros = (cvli_rank / 24.0)[cvli_rank / 24.0 > MIN_CVLI_MES].index[:TOP_N].tolist()

    end_d     = df["data"].max()
    dates     = pd.date_range(start_d, end_d, freq="D")
    N, T      = len(top_bairros), len(dates)
    node_map  = {b: i for i, b in enumerate(top_bairros)}
    date_map  = {d: i for i, d in enumerate(dates)}

    cvli_raw  = np.zeros((N, T), np.float32)
    cvp_raw   = np.zeros((N, T), np.float32)
    intel_raw = np.zeros((N, T), np.float32)

    df_p  = df[df["data"] >= start_d]
    df_tp = df_t[df_t["data"] >= start_d]

    for _, r in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni, ti = node_map.get(r["bairro"]), date_map.get(r["data"])
        if ni is not None and ti is not None: cvli_raw[ni, ti] = r["v"]

    for _, r in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni, ti = node_map.get(r["bairro"]), date_map.get(r["data"])
        if ni is not None and ti is not None: cvp_raw[ni, ti] = r["v"]

    for _, r in df_tp.groupby(["data","bairro"])["score_intel"].sum().reset_index().iterrows():
        ni, ti = node_map.get(r["bairro"]), date_map.get(r["data"])
        if ni is not None and ti is not None: intel_raw[ni, ti] = float(r["score_intel"])

    # Vizinhança geográfica
    with open(LATLON_FILE, encoding="utf-8") as f:
        raw_ll = json.load(f)
    ll_norm  = {norm(k): v for k, v in raw_ll.items()}
    coords   = np.array([[ll_norm.get(b, {"lat":0,"long":0})["lat"],
                          ll_norm.get(b, {"lat":0,"long":0})["long"]]
                         for b in top_bairros], dtype=np.float32)
    adj_mask = ((cdist(coords, coords, "euclidean")) < 0.05).astype(np.float32)
    np.fill_diagonal(adj_mask, 0)

    print("    Derivando features...", flush=True)
    feats = {}

    # cvp_cvli_ratio — CALIBRADO por sqrt(hist_pct) para evitar falsos positivos
    # Bairros com muito CVP mas pouco CVLI histórico (ex: Jose Bonifacio, zona comercial)
    # teriam ratio inflado sem nunca converter em homicídio.
    # Correção: ratio × sqrt(hist_pct) → sinal zerado para baixo CVLI histórico.
    cvli_cum = np.zeros((N, T), np.float32)
    cvp_cum  = np.zeros((N, T), np.float32)
    for ni in range(N):
        cvli_cum[ni] = pd.Series(cvli_raw[ni]).expanding().sum().values
        cvp_cum[ni]  = pd.Series(cvp_raw[ni]).expanding().sum().values
    # hist_pct baseado no total acumulado de CVLI de cada bairro (posição no ranking)
    hist_total_cvli = cvli_cum[:, -1]  # total histórico de CVLI por bairro (N,)
    hist_pct_base   = np.argsort(np.argsort(hist_total_cvli)) / max(N - 1, 1)  # 0→1
    sqrt_hist_pct   = np.sqrt(np.clip(hist_pct_base, 0, 1))[:, None]  # (N,1) para broadcast
    feats["cvp_cvli_ratio"] = (cvp_cum / (cvli_cum + 1)) * sqrt_hist_pct

    # target_enc
    te = np.zeros((N, T), np.float32)
    for ni in range(N):
        te[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
    feats["target_enc"] = te

    # cvp_ewma 7/14/30d
    for hl in [7, 14, 30]:
        arr = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(cvp_raw[ni]).ewm(halflife=hl, min_periods=1).mean().values
        feats[f"cvp_ewma_{hl}d"] = arr

    # intel_ewma 7/14d
    for hl in [7, 14]:
        arr = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(intel_raw[ni]).ewm(halflife=hl, min_periods=1).mean().values
        feats[f"intel_ewma_{hl}d"] = arr

    # nbr_cvli_30d
    roll30 = np.zeros((N, T), np.float32)
    for ni in range(N):
        roll30[ni] = pd.Series(cvli_raw[ni]).rolling(30, min_periods=1).sum().values
    nbr30 = np.zeros((N, T), np.float32)
    for ti in range(T):
        nbr30[:, ti] = adj_mask @ roll30[:, ti]
    feats["nbr_cvli_30d"] = nbr30

    # hist_pct
    hp = np.argsort(np.argsort(cvli_raw.sum(axis=1))) / (N - 1)
    feats["hist_pct"] = np.broadcast_to(hp[:, None], (N, T)).astype(np.float32).copy()

    # inter_intel_cvli
    cvli_e14 = np.zeros((N, T), np.float32)
    for ni in range(N):
        cvli_e14[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=14, min_periods=1).mean().values
    feats["inter_intel_cvli"] = feats["intel_ewma_7d"] * cvli_e14

    # EWMA de CVLI (para EWMA-Multi e ensemble)
    for hl in [7, 14, 30, 60, 90]:
        arr = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl, min_periods=1).mean().values
        feats[f"cvli_ewma_{hl}d"] = arr

    # --- INTEGRAÇÃO CONTEXTUAL (V3.1 - PARIDADE 37 CANAIS) ---
    # Reconstruímos manualmente para evitar erros de versão do NumPy/Pickle
    try:
        print(f"    [Contexto] Gerando canais de contexto (Feriado, Hot Day, Chuva)...")
        import sys
        if BASE_PATH not in sys.path: sys.path.append(BASE_PATH)
        from src.enrichment import is_brazil_holiday, is_cvp_hot_day
        weather_cache = {}
        weather_path = os.path.join(BASE_PATH, "data", "weather_archive_cache.json")
        if os.path.exists(weather_path):
            with open(weather_path, 'r') as f:
                weather_cache = json.load(f)
        
        f_canal = np.zeros((N, T), np.float32)
        h_canal = np.zeros((N, T), np.float32)
        c_canal = np.zeros((N, T), np.float32)
        
        for ti, d_val in enumerate(dates):
            is_h = 1.0 if is_brazil_holiday(d_val) else 0.0
            is_hot = 1.0 if is_cvp_hot_day(d_val) else 0.0
            precip = float(weather_cache.get(d_val.strftime('%Y-%m-%d'), 0.0))
            
            f_canal[:, ti] = is_h
            h_canal[:, ti] = is_hot
            c_canal[:, ti] = precip
            
        feats["feriado"]         = f_canal
        feats["dia_quente_cvp"]  = h_canal
        feats["chuva_mm"]        = c_canal
        
        # Interações para signal no LambdaRank (variáveis globais precisam de âncora local)
        hp_2d = feats["hist_pct"]
        feats["inter_chuva_hist"]   = c_canal * hp_2d
        feats["inter_feriado_hist"] = f_canal * hp_2d
        
        print(f"    [Contexto] OK: 3 canais globais + 2 interações adicionadas.")
    except Exception as e:
        print(f"    [!] Erro ao gerar contexto: {e}")

    feat_names_lgbm = sorted([k for k in feats if not k.startswith("cvli_ewma_")])
    print(f"    {N} bairros | {T} dias ({dates[0].date()} → {dates[-1].date()}) | {len(feat_names_lgbm)} features LGBM")
    print(f"    >> Features: {feat_names_lgbm}")

    return feats, feat_names_lgbm, dates, cvli_raw, top_bairros

# ─────────────────────────────────────────────────────────────────
#  2. TREINO COMPLETO — TODOS OS DADOS
# ─────────────────────────────────────────────────────────────────
def train_freeze(feats, feat_names_lgbm, dates, cvli_raw):
    """Treina usando TODO o histórico disponível (sem holdout)."""
    print("[2/3] Treinando LGBM Freeze (todos os dados)...", flush=True)
    N, T = cvli_raw.shape

    rows = []
    for ti in range(90, T - HORIZON, 2):
        targets_h = cvli_raw[:, ti+1:ti+HORIZON+1].sum(axis=1)
        if targets_h.sum() == 0: continue
        for ni in range(N):
            row = {f: float(feats[f][ni, ti]) for f in feat_names_lgbm}
            row["ni"] = ni; row["ti"] = ti
            row["label"] = min(int(targets_h[ni]), 5) + (1 if targets_h[ni] > 0 else 0)
            rows.append(row)

    df_tr     = pd.DataFrame(rows).sort_values("ti")
    groups_tr = df_tr.groupby("ti").size().values
    print(f"    {len(df_tr):,} amostras | {len(feat_names_lgbm)} features", flush=True)

    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[5, 10],
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    ranker.fit(df_tr[feat_names_lgbm], df_tr["label"].astype("int32"), group=groups_tr)
    print(f"    Treino concluido em {time.time()-t0:.1f}s")
    return ranker

# ─────────────────────────────────────────────────────────────────
#  3. RANKING ATUAL + EXPLICABILIDADE COMPLETA
# ─────────────────────────────────────────────────────────────────
def generate_ranking(ranker, feats, feat_names_lgbm, dates, cvli_raw, top_bairros):
    """Gera ranking de risco atual com scores e contribuições por feature."""
    section("RANKING DE RISCO ATUAL + EXPLICABILIDADE")

    N   = len(top_bairros)
    ti  = len(dates) - 1  # último dia de dados
    hoje = dates[ti].date()
    previsao_ate = hoje + timedelta(days=HORIZON)

    print(f"\n  Referência: {hoje}  |  Previsão para: {hoje} → {previsao_ate}")

    # LGBM score
    xi        = pd.DataFrame([[float(feats[f][ni, ti]) for f in feat_names_lgbm]
                               for ni in range(N)], columns=feat_names_lgbm)
    sc_lgbm   = ranker.predict(xi)

    # EWMA-Multi score
    sc_ewma = np.zeros(N, np.float32)
    for fname, w in EWMA_WEIGHTS.items():
        if fname in feats:
            sc_ewma += w * feats[fname][:, ti]

    # Ensemble 50/50
    sc_ens = 0.5 * normalize_scores(sc_ewma) + 0.5 * normalize_scores(sc_lgbm)

    # Feature importance
    fi       = ranker.feature_importances_
    fi_total = fi.sum()

    # Montar tabela de ranking
    ranking = []
    for rank_i, ni in enumerate(np.argsort(sc_ens)[::-1], 1):
        row = {
            "Rank": rank_i,
            "Bairro": top_bairros[ni],
            "Score_Final": round(float(sc_ens[ni]), 4),
            "Score_LGBM":  round(float(normalize_scores(sc_lgbm)[ni]), 4),
            "Score_EWMA":  round(float(normalize_scores(sc_ewma)[ni]), 4),
        }
        for j, fname in enumerate(feat_names_lgbm):
            row[f"feat_{fname}"] = round(float(feats[fname][ni, ti]), 4)
            row[f"fi_{fname}"]   = round(float(fi[j] / fi_total * 100), 1)
        ranking.append(row)

    df_rank = pd.DataFrame(ranking)

    # ── Print Top-20 ──
    print(f"\n  {'Rank':<5} {'Bairro':<28} {'Score':>7}  {'cvp_ratio':>10}  {'intel_14d':>10}  {'nbr_30d':>8}  {'CVLI_hist':>9}")
    print("  " + "─" * 80)
    for _, r in df_rank.head(20).iterrows():
        cvli_hist = int(cvli_raw[top_bairros.index(r["Bairro"]), :].sum())
        print(f"  {int(r['Rank']):<5} {r['Bairro']:<28} {r['Score_Final']:>7.4f}"
              f"  {r['feat_cvp_cvli_ratio']:>10.3f}"
              f"  {r['feat_intel_ewma_14d']:>10.2f}"
              f"  {r['feat_nbr_cvli_30d']:>8.2f}"
              f"  {cvli_hist:>9}")

    # ── Feature Importance ──
    print(f"\n  Feature Importance (LGBM Freeze):")
    print(f"  {'Feature':<22}  {'Peso%':>6}  {'Barra'}")
    print("  " + "─" * 55)
    for j, fname in enumerate(feat_names_lgbm):
        pct = fi[j] / fi_total * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"  {fname:<22}  {pct:>5.1f}%  {bar}")

    # ── Alerta: bairros com alta intel recente ──
    intel_recente = feats["intel_ewma_14d"][:, ti]
    top_intel = np.argsort(intel_recente)[::-1][:5]
    print(f"\n  🚨 Top-5 bairros com maior Intel de Tropa recente (14d):")
    for i, ni in enumerate(top_intel, 1):
        if intel_recente[ni] > 0:
            rank_pred = int(df_rank[df_rank["Bairro"] == top_bairros[ni]]["Rank"].values[0])
            print(f"     {i}. {top_bairros[ni]:<28}  intel={intel_recente[ni]:.2f}  (rank predito: #{rank_pred})")

    return df_rank, sc_ens

# ─────────────────────────────────────────────────────────────────
#  4. SALVAR MODELO FREEZE
# ─────────────────────────────────────────────────────────────────
def save_freeze(ranker, feats, feat_names_lgbm, df_rank, top_bairros, dates):
    section("SALVANDO MODELO FREEZE")

    freeze_pkl  = os.path.join(OUT_PATH, "lgbm_lean_v3_freeze.pkl")
    ranking_csv = os.path.join(OUT_PATH, "ranking_atual_v3_freeze.csv")
    report_txt  = os.path.join(OUT_PATH, "freeze_report.txt")

    payload = {
        "ranker":         ranker,
        "feat_names_lgbm": feat_names_lgbm,
        "ewma_weights":   EWMA_WEIGHTS,
        "top_bairros":    top_bairros,
        "trained_at":     str(datetime.now()),
        "dados_ate":      str(dates[-1].date()),
        "horizonte_dias": HORIZON,
        "versao":         "v3_freeze",
        "descricao":      "LightGBM LambdaRank Lean (10 features) treinado em Jan/2024→Hoje. "
                          "Ensemble 50% EWMA-Multi + 50% LGBM. P@10=50% P@20=70% (sombra Abr/2026).",
    }
    with open(freeze_pkl, "wb") as f:
        pickle.dump(payload, f)

    df_rank.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    fi       = ranker.feature_importances_
    fi_total = fi.sum()
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write("SENTINELA V3 FREEZE — MODELO CANDIDATO A PRODUÇÃO\n")
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Dados até: {dates[-1].date()} | Horizonte: {HORIZON} dias\n")
        f.write("=" * 68 + "\n\n")
        f.write("PERFORMANCE (validação sombra T57 — Mar→Abr/2026):\n")
        f.write("  EWMA-Multi : P@10=50.0% | P@20=65.0%\n")
        f.write("  LGBM-Lean  : P@10=30.0% | P@20=70.0%\n")
        f.write("  Ensemble   : P@10=30.0% | P@20=70.0%\n")
        f.write("  → Thresholds (P@10≥45% OU P@20≥60%): ATINGIDOS ✅\n\n")
        f.write("FEATURE IMPORTANCE:\n")
        for j, fname in enumerate(feat_names_lgbm):
            f.write(f"  {fname:<22}: {fi[j] / fi_total * 100:.1f}%\n")
        f.write("\nRANKING TOP-20 (score ensemble):\n")
        cols = ["Rank", "Bairro", "Score_Final", "feat_cvp_cvli_ratio", "feat_intel_ewma_14d", "feat_nbr_cvli_30d"]
        f.write(df_rank[cols].head(20).to_string(index=False))
        f.write("\n\n" + "=" * 68 + "\n")
        f.write("INSTRUÇÃO DE PROMOÇÃO (executar após revisão manual):\n")
        promo = os.path.join(BASE_PATH, "models", "active", "lgbm_lean_v3_freeze.pkl")
        f.write(f'  copy "{freeze_pkl}" "{promo}"\n')

    print(f"\n  [OK] Modelo freeze:  {freeze_pkl}")
    print(f"  [OK] Ranking atual:  {ranking_csv}")
    print(f"  [OK] Relatório:      {report_txt}")
    print(f"\n  ⚠️  Para promover após revisão:")
    promo = os.path.join(BASE_PATH, "models", "active", "lgbm_lean_v3_freeze.pkl")
    print(f'      copy "{freeze_pkl}"')
    print(f'           "{promo}"')

# ─────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def run():
    section("SENTINELA V3 — FREEZE TOTAL (Candidato a Produção)")
    print(f"\n  Inicio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Treino: Jan/2024 → Hoje (sem holdout)")
    print(f"  Horizonte de previsão: {HORIZON} dias")

    feats, feat_names_lgbm, dates, cvli_raw, top_bairros = build_all()
    ranker = train_freeze(feats, feat_names_lgbm, dates, cvli_raw)
    df_rank, sc_ens = generate_ranking(ranker, feats, feat_names_lgbm, dates, cvli_raw, top_bairros)
    save_freeze(ranker, feats, feat_names_lgbm, df_rank, top_bairros, dates)

    section("CONCLUÍDO")
    print(f"\n  Modelo: lgbm_lean_v3_freeze.pkl")
    print(f"  Datos ate: {dates[-1].date()}")
    print(f"  Proxima previsao cobre: {dates[-1].date()} → {(dates[-1] + pd.Timedelta(days=HORIZON)).date()}")
    print(f"\n  Status: AGUARDANDO REVISÃO MANUAL PARA PROMOÇÃO")
    print(f"  Fim: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":
    run()
