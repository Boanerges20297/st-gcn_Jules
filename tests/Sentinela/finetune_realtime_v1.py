"""
====================================================================
SENTINELA — FINE-TUNING EM TEMPO REAL (Fase 6)
====================================================================
Ajusta o modelo base (freeze) com dados dos últimos N dias,
capturando padrões emergentes sem perder a estabilidade histórica.

Arquitetura:
  Score_final = w_base × Score_base + w_ft × Score_fine_tuner

Regras de ativação do fine-tuner:
  1. Mínimo de MIN_CVLI_FT eventos nos últimos JANELA_FT dias
  2. P@10 do fine-tuner > P@10 do modelo base (últimos 14d)
  → Se não atingir: ensemble retorna 100% modelo base (fallback)

Uso:
  .\.venv\Scripts\python.exe tests/Sentinela/finetune_realtime_v1.py
  .\.venv\Scripts\python.exe tests/Sentinela/finetune_realtime_v1.py --janela 45 --peso-ft 0.35
====================================================================
"""

import sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, warnings, unicodedata, pickle, time
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from scipy.spatial.distance import cdist
from lightgbm import LGBMRanker

warnings.filterwarnings("ignore")

BASE_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW     = os.path.join(BASE_PATH, "data", "raw")
MODEL_DIR    = os.path.join(BASE_PATH, "tests", "Sentinela")
BASE_MODEL   = os.path.join(MODEL_DIR, "lgbm_lean_v3_freeze.pkl")
FT_MODEL_OUT = os.path.join(MODEL_DIR, "lgbm_finetune_current.pkl")
OUT_JSON     = os.path.join(MODEL_DIR, "ranking_realtime.json")
OUT_CSV      = os.path.join(MODEL_DIR, "ranking_realtime.csv")

CSV_ENRICH   = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA    = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
LATLON_FILE  = os.path.join(DATA_RAW, "bairros_centros_latlong.json")

HORIZON      = 14
MIN_CVLI_FT  = 3   # mínimo de eventos CVLI na janela para ativar fine-tuner

PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO': 15.0, 'PORTE ILEGAL ART 14': 12.0,
    'TRAFICO DE DROGAS': 8.0, 'APREENSAO DE DROGAS': 6.0,
    'APREENSAO DE ENTORPECENTES': 6.0, 'MANDADO DE PRISAO': 4.0,
    'MANDADO EM ABERTO': 3.5, 'MANDADO DE PRISAO EM ABERTO': 3.5,
    'VEICULO ROUBADO RECUPERADO': 2.5, 'VEICULO ROUBADO LOCALIZADO': 2.0,
    'ABANDONO DE MATERIAL': 1.5, 'NAO INFORMADA': 0.5,
}


def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9: return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

def pk(scores, targets, k):
    return len(set(np.argsort(scores)[::-1][:k]) & set(np.argsort(targets)[::-1][:k])) / k


# ─────────────────────────────────────────────────────────────────
#  1. CARREGAR DADOS E CONSTRUIR FEATURES
# ─────────────────────────────────────────────────────────────────
def build_features(top_bairros, start_d=pd.Timestamp("2024-01-01")):
    """Constrói matrix de features (N×T) para todos os bairros."""
    df = pd.read_csv(CSV_ENRICH, low_memory=False)

    # --- REPARO DE BAIRROS VIA GEOLOCALIZACAO (Sincronizado com v3_freeze) ---
    from scipy.spatial import KDTree
    with open(LATLON_FILE, encoding="utf-8") as f:
        raw_ll = json.load(f)
    fort_ll = {norm(k): v for k, v in raw_ll.items() if v.get('regiao') == 'fortaleza'}
    names_ll = list(fort_ll.keys())
    coords_ll = np.array([[fort_ll[n]['lat'], fort_ll[n]['long']] for n in names_ll])
    tree = KDTree(coords_ll)
    
    mask_null = df['bairro'].isna() | (df['bairro'].apply(norm) == 'DESCONHECIDO')
    mask_gps  = df['latitude'].notna() & df['longitude'].notna()
    mask_rep  = mask_null & mask_gps
    
    if mask_rep.sum() > 0:
        points = df.loc[mask_rep, ['latitude', 'longitude']].values
        dist, idx = tree.query(points)
        THRESHOLD = 0.045 # Aprox 5km
        df_update = df.loc[mask_rep].copy()
        for i, (d, ix) in enumerate(zip(dist, idx)):
            if d < THRESHOLD:
                df_update.iloc[i, df_update.columns.get_loc('bairro')] = names_ll[ix]
                df_update.iloc[i, df_update.columns.get_loc('cidade')] = 'FORTALEZA'
            else:
                cid = str(df_update.iloc[i, df_update.columns.get_loc('cidade')]).upper()
                if cid != 'FORTALEZA' and cid != 'NAN' and cid != 'DESCONHECIDO':
                    df_update.iloc[i, df_update.columns.get_loc('bairro')] = cid
        df.update(df_update)

    df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
    df["bairro"] = df["bairro"].apply(norm)
    df["data"]   = pd.to_datetime(df["data"], errors="coerce")
    df           = df.dropna(subset=["data","bairro"])
    df["is_cvli"]= (df["tipo"] == "cvli").astype("int8")
    df["is_cvp"] = (df["tipo"] == "cvp").astype("int8")

    df_t = pd.read_csv(CSV_TROPA, low_memory=False, encoding="utf-8-sig")
    df_t["bairro"]     = df_t["bairro"].apply(norm)
    df_t["data"]       = pd.to_datetime(df_t["data"], errors="coerce")
    df_t               = df_t.dropna(subset=["data","bairro"])
    df_t["peso_nat"]   = df_t["natureza"].str.upper().str.strip().map(lambda x: PESO_NATUREZA.get(x,1.0))
    df_t["score_intel"]= (
        df_t["qtd_armas"]*15 + np.log1p(df_t["qtd_drogas"].fillna(0))*4
        + df_t["qtd_drogas_itens"]*2 + df_t["qtd_veiculos_apreendidos"]*3
        + df_t["peso_nat"]
    ).astype("float32")

    dates = pd.date_range(start_d, df["data"].max(), freq="D")
    N, T  = len(top_bairros), len(dates)
    nm    = {b: i for i,b in enumerate(top_bairros)}
    dm    = {d: i for i,d in enumerate(dates)}

    cvli_raw  = np.zeros((N,T), np.float32)
    cvp_raw   = np.zeros((N,T), np.float32)
    intel_raw = np.zeros((N,T), np.float32)

    df_p  = df[df["data"] >= start_d]
    df_tp = df_t[df_t["data"] >= start_d]

    for _,r in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
        if ni is not None and ti is not None: cvli_raw[ni,ti] = r["v"]
    for _,r in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
        if ni is not None and ti is not None: cvp_raw[ni,ti] = r["v"]
    for _,r in df_tp.groupby(["data","bairro"])["score_intel"].sum().reset_index().iterrows():
        ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
        if ni is not None and ti is not None: intel_raw[ni,ti] = float(r["score_intel"])

    with open(LATLON_FILE, encoding="utf-8") as f:
        raw_ll = json.load(f)
    ll_n     = {norm(k): v for k,v in raw_ll.items()}
    coords   = np.array([[ll_n.get(b,{"lat":0,"long":0})["lat"],
                          ll_n.get(b,{"lat":0,"long":0})["long"]]
                         for b in top_bairros], np.float32)
    adj_mask = (cdist(coords,coords,"euclidean") < 0.05).astype(np.float32)
    np.fill_diagonal(adj_mask, 0)

    feats = {}

    cvli_cum = np.zeros((N,T), np.float32)
    cvp_cum  = np.zeros((N,T), np.float32)
    for ni in range(N):
        cvli_cum[ni] = pd.Series(cvli_raw[ni]).expanding().sum().values
        cvp_cum[ni]  = pd.Series(cvp_raw[ni]).expanding().sum().values
    hist_pct_base = np.argsort(np.argsort(cvli_cum[:,-1])) / max(N-1,1)
    sqrt_hp = np.sqrt(np.clip(hist_pct_base,0,1))[:,None]
    feats["cvp_cvli_ratio"] = (cvp_cum / (cvli_cum + 1)) * sqrt_hp

    te = np.zeros((N,T), np.float32)
    for ni in range(N):
        te[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
    feats["target_enc"] = te

    for hl in [7,14,30]:
        arr = np.zeros((N,T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(cvp_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
        feats[f"cvp_ewma_{hl}d"] = arr

    for hl in [7,14]:
        arr = np.zeros((N,T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(intel_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
        feats[f"intel_ewma_{hl}d"] = arr

    r30 = np.zeros((N,T), np.float32)
    for ni in range(N):
        r30[ni] = pd.Series(cvli_raw[ni]).rolling(30,min_periods=1).sum().values
    nb = np.zeros((N,T), np.float32)
    for ti in range(T): nb[:,ti] = adj_mask @ r30[:,ti]
    feats["nbr_cvli_30d"] = nb

    hp = np.argsort(np.argsort(cvli_raw.sum(axis=1))) / max(N-1,1)
    feats["hist_pct"] = np.broadcast_to(hp[:,None],(N,T)).astype(np.float32).copy()

    e14 = np.zeros((N,T), np.float32)
    for ni in range(N):
        e14[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=14,min_periods=1).mean().values
    feats["inter_intel_cvli"] = feats["intel_ewma_7d"] * e14

    for hl in [7,14,30,60,90]:
        arr = np.zeros((N,T), np.float32)
        for ni in range(N):
            arr[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
        feats[f"cvli_ewma_{hl}d"] = arr

    # --- INTEGRAÇÃO CONTEXTUAL (Sincronizado com v3_freeze) ---
    try:
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
        
        hp_2d = feats["hist_pct"]
        feats["inter_chuva_hist"]   = c_canal * hp_2d
        feats["inter_feriado_hist"] = f_canal * hp_2d
    except Exception as e:
        print(f"    [!] Erro ao gerar contexto no fine-tuner: {e}")

    return feats, dates, cvli_raw


# ─────────────────────────────────────────────────────────────────
#  2. FINE-TUNER: LGBM NA JANELA RECENTE
# ─────────────────────────────────────────────────────────────────
def train_finetuner(feats, feat_names, dates, cvli_raw, janela_dias, cutoff_ti):
    """
    Treina um LGBM apenas na janela recente (últimos janela_dias).
    Retorna (ranker, n_cvli_na_janela).
    """
    N, T   = cvli_raw.shape
    start  = max(0, cutoff_ti - janela_dias)
    rows   = []

    for ti in range(start, cutoff_ti - HORIZON):
        targets_h = cvli_raw[:, ti+1:ti+HORIZON+1].sum(axis=1)
        if targets_h.sum() == 0: continue
        for ni in range(N):
            row = {f: float(feats[f][ni, ti]) for f in feat_names}
            row["ni"]    = ni
            row["ti"]    = ti
            row["label"] = min(int(targets_h[ni]), 5) + (1 if targets_h[ni] > 0 else 0)
            rows.append(row)

    if not rows:
        return None, 0

    df_ft     = pd.DataFrame(rows).sort_values("ti")
    groups_ft = df_ft.groupby("ti").size().values
    n_cvli    = int(cvli_raw[:, start:cutoff_ti].sum())

    ranker_ft = LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[5,10],
        n_estimators=150,    # menos árvores — dados limitados
        num_leaves=15,       # muito conservador para janela curta
        learning_rate=0.05,
        min_child_samples=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,      # regularização forte
        random_state=42, n_jobs=-1, verbose=-1,
    )
    ranker_ft.fit(df_ft[feat_names], df_ft["label"].astype("int32"), group=groups_ft)
    return ranker_ft, n_cvli


# ─────────────────────────────────────────────────────────────────
#  3. VALIDAÇÃO INTERNA DO FINE-TUNER
# ─────────────────────────────────────────────────────────────────
def validate_finetuner(ranker_base, ranker_ft, feats, feat_names,
                       dates, cvli_raw, ewma_weights, cutoff_ti):
    """
    Compara P@10 do base vs fine-tuner nos últimos HORIZON dias antes do cutoff.
    Retorna (p10_base, p10_ft, ft_e_melhor).
    """
    val_end   = cutoff_ti
    val_start = max(0, val_end - HORIZON)
    N = cvli_raw.shape[0]

    p10_base_list, p10_ft_list = [], []

    for ti in range(val_start, val_end):
        if ti + HORIZON >= cvli_raw.shape[1]: continue
        targets_h = cvli_raw[:, ti+1:ti+HORIZON+1].sum(axis=1)
        if targets_h.sum() == 0: continue

        xi = pd.DataFrame([[float(feats[f][ni,ti]) for f in feat_names]
                            for ni in range(N)], columns=feat_names)

        # Base
        sc_b_lgbm = ranker_base.predict(xi)
        sc_b_ewma = np.zeros(N, np.float32)
        for fn, w in ewma_weights.items():
            if fn in feats: sc_b_ewma += w * feats[fn][:, ti]
        sc_base = 0.5*normalize_scores(sc_b_ewma) + 0.5*normalize_scores(sc_b_lgbm)

        # Fine-tuner
        sc_ft = normalize_scores(ranker_ft.predict(xi))

        p10_base_list.append(pk(sc_base, targets_h, 10))
        p10_ft_list.append(pk(sc_ft, targets_h, 10))

    if not p10_base_list:
        return 0.0, 0.0, False

    p10_base = float(np.mean(p10_base_list)) * 100
    p10_ft   = float(np.mean(p10_ft_list))   * 100
    return p10_base, p10_ft, p10_ft > p10_base


# ─────────────────────────────────────────────────────────────────
#  4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def run(janela_dias: int = 30, peso_ft: float = 0.30):
    print("\n" + "="*68)
    print("  SENTINELA — FINE-TUNING EM TEMPO REAL (Fase 6)")
    print("="*68)
    print(f"\n  Janela fine-tuner: últimos {janela_dias} dias")
    print(f"  Pesos: {1-peso_ft:.0%} base + {peso_ft:.0%} fine-tuner")
    print(f"  Critério ativação: ≥{MIN_CVLI_FT} CVLI na janela E P@10 FT > P@10 base")

    # Carregar modelo base
    print("\n[1/5] Carregando modelo base (freeze)...")
    with open(BASE_MODEL, "rb") as f:
        payload = pickle.load(f)
    ranker_base  = payload["ranker"]
    feat_names   = payload["feat_names_lgbm"]
    ewma_weights = payload["ewma_weights"]
    top_bairros  = payload["top_bairros"]
    N = len(top_bairros)
    print(f"       Treinado em: {payload.get('dados_ate','?')} | {N} bairros | {len(feat_names)} features")

    # Construir features
    print("[2/5] Construindo features...")
    t0 = time.time()
    feats, dates, cvli_raw = build_features(top_bairros)
    T  = len(dates)
    ti_now = T - 1  # cutoff = hoje
    print(f"       {T} dias ({dates[0].date()} → {dates[-1].date()}) em {time.time()-t0:.1f}s")

    # Treinar fine-tuner
    print(f"[3/5] Treinando fine-tuner (janela {janela_dias}d)...")
    t0 = time.time()
    ranker_ft, n_cvli = train_finetuner(feats, feat_names, dates, cvli_raw,
                                         janela_dias, ti_now)
    print(f"       {n_cvli} CVLI na janela | treino em {time.time()-t0:.1f}s")

    # Critério 1: sinal suficiente
    if ranker_ft is None or n_cvli < MIN_CVLI_FT:
        print(f"\n  ⚠️  FINE-TUNER DESATIVADO — sinal insuficiente ({n_cvli} < {MIN_CVLI_FT} CVLI)")
        print("       Usando 100% modelo base (fallback seguro).\n")
        ft_ativo = False
        modo_str = "base_only"
    else:
        # Critério 2: fine-tuner supera base
        print("[4/5] Validando fine-tuner vs base...")
        p10_base, p10_ft, ft_melhor = validate_finetuner(
            ranker_base, ranker_ft, feats, feat_names,
            dates, cvli_raw, ewma_weights, ti_now
        )
        print(f"       P@10 Base: {p10_base:.1f}%  |  P@10 Fine-tuner: {p10_ft:.1f}%")

        if ft_melhor:
            ft_ativo = True
            modo_str = f"ensemble_{int((1-peso_ft)*100)}_base_{int(peso_ft*100)}_ft"
            print(f"       ✅ Fine-tuner ATIVADO (+{p10_ft-p10_base:.1f}pp vs base)")
        else:
            ft_ativo = False
            modo_str = "base_only"
            print(f"       ❌ Fine-tuner NÃO ativado (base é melhor). Usando fallback.")

    # Gerar ranking final
    print("[5/5] Gerando ranking final...")
    ti = ti_now

    xi = pd.DataFrame([[float(feats[f][ni,ti]) for f in feat_names]
                        for ni in range(N)], columns=feat_names)

    sc_lgbm_base = ranker_base.predict(xi)
    sc_ewma = np.zeros(N, np.float32)
    for fn, w in ewma_weights.items():
        if fn in feats: sc_ewma += w * feats[fn][:, ti]
    sc_base_ens = 0.5*normalize_scores(sc_ewma) + 0.5*normalize_scores(sc_lgbm_base)

    if ft_ativo and ranker_ft is not None:
        sc_ft    = normalize_scores(ranker_ft.predict(xi))
        sc_final = (1 - peso_ft) * normalize_scores(sc_base_ens) + peso_ft * sc_ft
        # Calcular delta de posições
        rank_base = {ni: r for r, ni in enumerate(np.argsort(sc_base_ens)[::-1])}
        rank_final= {ni: r for r, ni in enumerate(np.argsort(sc_final)[::-1])}
    else:
        sc_final  = sc_base_ens
        rank_base = {ni: r for r, ni in enumerate(np.argsort(sc_base_ens)[::-1])}
        rank_final= rank_base

    previsao_inicio = dates[ti].date()
    previsao_fim    = previsao_inicio + timedelta(days=HORIZON)

    # Print top-20
    print(f"\n  {'Rank':<5} {'Bairro':<28} {'Score':>7}  {'Δ':>5}  {'Status'}")
    print("  " + "─"*65)
    ranking_out = []
    for rank_i, ni in enumerate(np.argsort(sc_final)[::-1], 1):
        delta = rank_base[ni] - rank_final[ni]  # positivo = subiu
        delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "=")
        status = "↑ SUBIU" if delta > 2 else ("↓ CAIU" if delta < -2 else "")
        print(f"  {rank_i:<5} {top_bairros[ni]:<28} {normalize_scores(sc_final)[ni]:>7.4f}"
              f"  {delta_str:>5}  {status}")
        ranking_out.append({
            "rank":            rank_i,
            "bairro":          top_bairros[ni],
            "score_final":     round(float(normalize_scores(sc_final)[ni]), 4),
            "score_base":      round(float(normalize_scores(sc_base_ens)[ni]), 4),
            "delta_posicoes":  delta,
            "ft_ativo":        ft_ativo,
        })

    # Salvar fine-tuner
    if ft_ativo and ranker_ft is not None:
        with open(FT_MODEL_OUT, "wb") as f:
            pickle.dump({
                "ranker_ft":    ranker_ft,
                "feat_names":   feat_names,
                "janela_dias":  janela_dias,
                "peso_ft":      peso_ft,
                "n_cvli":       n_cvli,
                "gerado_em":    str(datetime.now()),
                "modo":         modo_str,
            }, f)
        print(f"\n  [OK] Fine-tuner salvo: {FT_MODEL_OUT}")

    # Salvar outputs
    import json as jlib
    out_data = {
        "gerado_em":        str(datetime.now()),
        "modo":             modo_str,
        "ft_ativo":         ft_ativo,
        "janela_dias":      janela_dias,
        "n_cvli_janela":    n_cvli,
        "previsao_inicio":  str(previsao_inicio),
        "previsao_fim":     str(previsao_fim),
        "ranking":          ranking_out,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        jlib.dump(out_data, f, ensure_ascii=False, indent=2)

    pd.DataFrame(ranking_out).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"  [OK] JSON: {OUT_JSON}")
    print(f"  [OK] CSV:  {OUT_CSV}")
    print(f"\n  Modo ativo: {modo_str}")
    print(f"  Previsão:   {previsao_inicio} → {previsao_fim}")
    print("="*68)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinela Fine-Tuner em Tempo Real")
    parser.add_argument("--janela",   type=int,   default=30,   help="Janela do fine-tuner em dias (padrão: 30)")
    parser.add_argument("--peso-ft",  type=float, default=0.30, help="Peso do fine-tuner no ensemble (padrão: 0.30)")
    args = parser.parse_args()
    run(janela_dias=args.janela, peso_ft=args.peso_ft)
