"""
====================================================================
SOLO CHALLENGER — POTÊNCIA TOTAL (800D -> 7D)
====================================================================
Objetivo: Treinar a arquitetura Solo com o histórico completo de 800 dias.
Comparação direta e justa com o Sentinela V3 original.
====================================================================
"""

import os, json, time, warnings, unicodedata, pickle
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

CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")

HORIZON      = 14
LOOKBACK     = 800 # Memória profunda (Simetria com V3)
TOP_N        = 60
SHADOW_CUTOFF = pd.Timestamp("2026-03-31")

# Parâmetros Elite do Solo
PARAMS = {
    "objective": "rank_xendcg",
    "n_estimators": 300,       # Aumentado levemente para o volume de 800 dias
    "num_leaves": 31,          # Aumentado para 31 (equilíbrio entre 15 e 63)
    "learning_rate": 0.05,     # Reduzido de 0.1 para 0.05 para estabilidade no longo prazo
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO':  15.0,
    'PORTE ILEGAL ART 14':        12.0,
    'TRAFICO DE DROGAS':           8.0,
    'APREENSAO DE DROGAS':         6.0,
}

def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def pk_from_ranking(scores_N, cvli_horizon_N, k):
    top_pred = np.argsort(scores_N)[::-1][:k]
    top_real = np.argsort(cvli_horizon_N)[::-1][:k]
    if len(top_real) == 0: return 0
    return len(set(top_pred) & set(top_real)) / k

def load_data():
    print("[1/3] Carregando dataset completo (2024-2026)...")
    df = pd.read_csv(CSV_ENRICH, low_memory=False)
    df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
    df["bairro"]  = df["bairro"].apply(norm)
    df["data"]    = pd.to_datetime(df["data"], errors="coerce")
    df            = df.dropna(subset=["data", "bairro"])
    df["is_cvli"] = (df["tipo"] == "cvli").astype("int8")
    df["is_cvp"]  = (df["tipo"] == "cvp").astype("int8")

    cutoff      = df["data"].max() - pd.Timedelta(days=730)
    top_bairros = df[df["data"] >= cutoff].groupby("bairro")["is_cvli"].sum().sort_values(ascending=False).index[:TOP_N].tolist()
    return df, top_bairros

def build_deep_features(df, top_bairros):
    print("[2/3] Construindo features de memória profunda...")
    start_d = pd.Timestamp("2024-01-01")
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
    for hl in [3, 7, 14, 30, 90]: # Incluído 90d para memória longa
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
            days_since[ni, ti] = min(ti - last, 180)
    feats["recency"] = days_since
    
    # Target encoding (histórico real)
    te = np.zeros((N, T), np.float32)
    for ni in range(N): te[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
    feats["target_enc"] = te

    return feats, sorted(feats.keys()), dates, cvli_raw

def train_and_eval_800d(feats, feat_names, dates, cvli_raw, top_bairros):
    print("[3/3] Treinando Solo Challenger 800d...")
    N, T = cvli_raw.shape
    date_map = {d: i for i, d in enumerate(dates)}
    cutoff_i = date_map.get(SHADOW_CUTOFF, T - HORIZON - 1)
    
    # Treino: TODO O HISTÓRICO até o cutoff
    rows = []
    train_indices = list(range(120, cutoff_i - HORIZON, 1)) # Denso
    for ti in train_indices:
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
    
    # Validação Shadow (os 7 dias após o cutoff)
    pred_ti = cutoff_i
    gt_cvli = cvli_raw[:, pred_ti+1:pred_ti+HORIZON+1].sum(axis=1)
    xi = np.array([[float(feats[fn][ni, pred_ti]) for fn in feat_names] for ni in range(N)])
    scores = ranker.predict(pd.DataFrame(xi, columns=feat_names))
    
    p10 = pk_from_ranking(scores, gt_cvli, 10) * 100
    p20 = pk_from_ranking(scores, gt_cvli, 20) * 100
    
    print(f"   P@10: {p10:.1f}%")
    print(f"   P@20: {p20:.1f}%")
    
    # Salvar para integração oficial
    model_data = {
        "ranker": ranker,
        "feat_names_lgbm": feat_names,
        "top_bairros": top_bairros,
        "ewma_weights": {}, # Desativado, Solo é 100% LGBM
        "p10": p10,
        "p20": p20,
        "trained_at": str(datetime.now())
    }
    
    output_file = os.path.join(OUT_PATH, "lgbm_solo_challenger_800d.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(model_data, f)
    print(f"   Modelo INTEGRÁVEL salvo em: {output_file}")

if __name__ == "__main__":
    df, top_bairros = load_data()
    feats, feat_names, dates, cvli_raw = build_deep_features(df, top_bairros)
    train_and_eval_800d(feats, feat_names, dates, cvli_raw, top_bairros)
