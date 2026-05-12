"""
====================================================================
FINALIZAÇÃO SOLO FLIGHT — REGIME ULTRA-REATIVO (30D -> 7D)
====================================================================
Configuração Vencedora: Config E (Ultra-Fast 0.1 LR)
Objetivo: Produzir o modelo final para promoção em produção.
====================================================================
"""

import os, json, time, warnings, unicodedata, pickle, shutil
import numpy as np
import pandas as pd
from datetime import datetime
from lightgbm import LGBMRanker

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURACOES
# ─────────────────────────────────────────────────────────────────
BASE_PATH   = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
DATA_RAW    = os.path.join(BASE_PATH, "data", "raw")
OUT_PATH    = os.path.join(BASE_PATH, "tests", "Sentinela")
ACTIVE_PATH = os.path.join(BASE_PATH, "models", "active")

CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")

HORIZON      = 7
LOOKBACK     = 30 
TOP_N        = 60

# Config E (Vencedora)
PARAMS = {
    "objective": "rank_xendcg",
    "n_estimators": 200,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO':  15.0,
    'PORTE ILEGAL ART 14':        12.0,
    'TRAFICO DE DROGAS':           8.0,
    'APREENSAO DE DRGAS':          6.0,
    'APREENSAO DE ENTORPECENTES':  6.0,
    'MANDADO DE PRISAO':           4.0,
}

def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def load_data():
    print("[1/3] Carregando dados para produção...")
    df = pd.read_csv(CSV_ENRICH, low_memory=False)
    df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
    df["bairro"]  = df["bairro"].apply(norm)
    df["data"]    = pd.to_datetime(df["data"], errors="coerce")
    df            = df.dropna(subset=["data", "bairro"])
    df["is_cvli"] = (df["tipo"] == "cvli").astype("int8")
    df["is_cvp"]  = (df["tipo"] == "cvp").astype("int8")

    df_t = pd.read_csv(CSV_TROPA, low_memory=False)
    df_t["bairro"] = df_t["bairro"].apply(norm)
    df_t["data"]   = pd.to_datetime(df_t["data"], errors="coerce")
    df_t           = df_t.dropna(subset=["data", "bairro"])
    df_t["score_intel"] = (df_t["qtd_armas"] * 15.0).astype("float32")

    cutoff      = df["data"].max() - pd.Timedelta(days=730)
    top_bairros = df[df["data"] >= cutoff].groupby("bairro")["is_cvli"].sum().sort_values(ascending=False).index[:TOP_N].tolist()
    return df, df_t, top_bairros

def build_final_features(df, df_t, top_bairros):
    print("[2/3] Construindo matriz de features (30d window)...")
    start_d = df["data"].max() - pd.Timedelta(days=120) # Buffer para EWMAs
    end_d   = df["data"].max()
    dates   = pd.date_range(start_d, end_d, freq="D")
    N, T    = len(top_bairros), len(dates)
    node_map = {b: i for i, b in enumerate(top_bairros)}
    date_map = {d: i for i, d in enumerate(dates)}

    cvli_raw = np.zeros((N, T), dtype=np.float32)
    cvp_raw  = np.zeros((N, T), dtype=np.float32)

    df_p = df[df["data"] >= start_d].copy()
    for _, row in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: cvli_raw[ni, ti] = row["v"]
    for _, row in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: cvp_raw[ni, ti] = row["v"]

    feats = {}
    for hl in [3, 7, 14, 30]:
        a_cvli = np.zeros((N, T), np.float32); a_cvp = np.zeros((N, T), np.float32)
        for ni in range(N):
            a_cvli[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl).mean().values
            a_cvp[ni]  = pd.Series(cvp_raw[ni]).ewm(halflife=hl).mean().values
        feats[f"cvli_ewma_{hl}d"] = a_cvli
        feats[f"cvp_ewma_{hl}d"]  = a_cvp

    days_since = np.zeros((N, T), np.float32)
    for ni in range(N):
        last = -100
        for ti in range(T):
            if cvli_raw[ni, ti] > 0: last = ti
            days_since[ni, ti] = min(ti - last, 60)
    feats["recency"] = days_since
    
    return feats, sorted(feats.keys()), dates, cvli_raw

def train_and_promote(feats, feat_names, dates, cvli_raw, top_bairros):
    print("[3/3] Treinando modelo definitivo e promovendo...")
    N, T = cvli_raw.shape
    # Treino nos últimos 30 dias disponíveis
    cutoff_i = T - 1
    start_i  = max(0, cutoff_i - LOOKBACK)
    
    rows = []
    for ti in range(start_i, cutoff_i - HORIZON):
        targets_h = cvli_raw[:, ti+1:ti+HORIZON+1].sum(axis=1)
        if targets_h.sum() == 0: continue
        for ni in range(N):
            row = {fn: float(feats[fn][ni, ti]) for fn in feat_names}
            row["label"] = min(int(targets_h[ni]), 5) + (1 if targets_h[ni] > 0 else 0)
            row["ti"] = ti
            rows.append(row)
    
    df_tr = pd.DataFrame(rows)
    groups = df_tr.groupby("ti").size().values
    
    ranker = LGBMRanker(**PARAMS)
    ranker.fit(df_tr[feat_names], df_tr["label"], group=groups)
    
    # Salvar
    model_data = {
        "ranker": ranker,
        "feat_names": feat_names,
        "top_bairros": top_bairros,
        "params": PARAMS,
        "trained_at": str(datetime.now()),
        "regime": "30d_7d_solo"
    }
    
    filename = "lgbm_solo_challenger.pkl"
    local_path = os.path.join(OUT_PATH, filename)
    active_dest = os.path.join(ACTIVE_PATH, filename)
    
    with open(local_path, "wb") as f:
        pickle.dump(model_data, f)
    
    shutil.copy(local_path, active_dest)
    
    print(f"\n✅ MODELO PROMOVIDO!")
    print(f"   Local: {local_path}")
    print(f"   Ativo: {active_dest}")
    
    return active_dest

if __name__ == "__main__":
    df, df_t, top_bairros = load_data()
    feats, feat_names, dates, cvli_raw = build_features = build_final_features(df, df_t, top_bairros)
    train_and_promote(feats, feat_names, dates, cvli_raw, top_bairros)
