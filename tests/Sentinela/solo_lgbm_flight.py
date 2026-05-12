"""
====================================================================
SENTINELA SOLO FLIGHT — REGIME ULTRA-REATIVO (30D -> 7D)
====================================================================
Objetivo: Treinar com apenas 30 dias de histórico para prever 7 dias.
Foco em capturar a dinâmica de curtíssimo prazo.
====================================================================
"""

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
OUT_PATH    = os.path.join(BASE_PATH, "tests", "Sentinela")

CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
LATLON_FILE = os.path.join(DATA_RAW, "bairros_centros_latlong.json")

HORIZON      = 7
LOOKBACK     = 14 
MIN_CVLI_MES = 0.4
TOP_N        = 60
SHADOW_CUTOFF = pd.Timestamp("2026-03-31")

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

def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def section(title, w=68):
    print(f"\n{'='*w}\n  {title}\n{'='*w}")

def pk_from_ranking(scores_N, cvli_horizon_N, k):
    top_pred = np.argsort(scores_N)[::-1][:k]
    top_real = np.argsort(cvli_horizon_N)[::-1][:k]
    if len(top_real) == 0: return 0
    return len(set(top_pred) & set(top_real)) / k

def load_data():
    print("[1/4] Carregando dados...")
    df = pd.read_csv(CSV_ENRICH, low_memory=False)
    df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
    df["bairro"]  = df["bairro"].apply(norm)
    df["data"]    = pd.to_datetime(df["data"], errors="coerce")
    df            = df.dropna(subset=["data", "bairro"])
    df["is_cvli"] = (df["tipo"] == "cvli").astype("int8")
    df["is_cvp"]  = (df["tipo"] == "cvp").astype("int8")

    df_t = pd.read_csv(CSV_TROPA, low_memory=False, encoding="utf-8-sig")
    df_t["bairro"] = df_t["bairro"].apply(norm)
    df_t["data"]   = pd.to_datetime(df_t["data"], errors="coerce")
    df_t           = df_t.dropna(subset=["data", "bairro"])
    df_t["score_intel"] = (df_t["qtd_armas"] * 15.0 + df_t["natureza"].str.upper().map(lambda x: PESO_NATUREZA.get(x, 1.0))).astype("float32")

    cutoff      = df["data"].max() - pd.Timedelta(days=730)
    cvli_rank   = df[df["data"] >= cutoff].groupby("bairro")["is_cvli"].sum().sort_values(ascending=False)
    top_bairros = cvli_rank.index[:TOP_N].tolist()

    return df, df_t, top_bairros

def build_features(df, df_t, top_bairros, start_d=pd.Timestamp("2024-01-01")):
    print("[2/4] Construindo features ultra-reativas...")
    end_d    = df["data"].max()
    dates    = pd.date_range(start_d, end_d, freq="D")
    N, T     = len(top_bairros), len(dates)
    node_map = {b: i for i, b in enumerate(top_bairros)}
    date_map = {d: i for i, d in enumerate(dates)}

    cvli_raw = np.zeros((N, T), dtype=np.float32)
    cvp_raw  = np.zeros((N, T), dtype=np.float32)
    intel_raw= np.zeros((N, T), dtype=np.float32)

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

    feats = {}
    for hl in [3, 7, 14, 30]:
        arr_cvli = np.zeros((N, T), np.float32)
        arr_cvp  = np.zeros((N, T), np.float32)
        for ni in range(N):
            arr_cvli[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl).mean().values
            arr_cvp[ni]  = pd.Series(cvp_raw[ni]).ewm(halflife=hl).mean().values
        feats[f"cvli_ewma_{hl}d"] = arr_cvli
        feats[f"cvp_ewma_{hl}d"]  = arr_cvp

    days_since = np.zeros((N, T), np.float32)
    for ni in range(N):
        last = -100
        for ti in range(T):
            if cvli_raw[ni, ti] > 0: last = ti
            days_since[ni, ti] = min(ti - last, 60)
    feats["recency"] = days_since

    feat_names = sorted(feats.keys())
    return feats, feat_names, dates, cvli_raw, top_bairros

def walk_forward_eval(feats, feat_names, dates, cvli_raw, params, n_folds=4):
    N, T = cvli_raw.shape
    fold_results = []
    current_end_i = T - HORIZON - 1
    
    for f in range(n_folds):
        cutoff_i = current_end_i - (f * HORIZON)
        start_train_i = max(0, cutoff_i - LOOKBACK)
        train_indices = list(range(start_train_i, cutoff_i - HORIZON))
        
        rows = []
        for ti in train_indices:
            targets_h = cvli_raw[:, ti+1:ti+HORIZON+1].sum(axis=1)
            if targets_h.sum() == 0: continue
            for ni in range(N):
                row = {fn: float(feats[fn][ni, ti]) for fn in feat_names}
                row["label"] = min(int(targets_h[ni]), 5) + (1 if targets_h[ni] > 0 else 0)
                row["ti"] = ti
                rows.append(row)
        
        if not rows: continue
        df_tr = pd.DataFrame(rows)
        groups = df_tr.groupby("ti").size().values
        ranker = LGBMRanker(**params)
        ranker.fit(df_tr[feat_names], df_tr["label"], group=groups)
        
        pred_ti = cutoff_i
        gt_cvli = cvli_raw[:, pred_ti+1:pred_ti+HORIZON+1].sum(axis=1)
        xi = np.array([[float(feats[fn][ni, pred_ti]) for fn in feat_names] for ni in range(N)])
        scores = ranker.predict(pd.DataFrame(xi, columns=feat_names))
        
        fold_results.append((pk_from_ranking(scores, gt_cvli, 10)*100, pk_from_ranking(scores, gt_cvli, 20)*100))
    
    return np.mean([r[0] for r in fold_results]), np.mean([r[1] for r in fold_results])

def run():
    section(f"SENTINELA SOLO FLIGHT — {LOOKBACK}D -> {HORIZON}D")
    df, df_t, top_bairros = load_data()
    feats, feat_names, dates, cvli_raw, _ = build_features(df, df_t, top_bairros)
    
    configs = [
        {"name": "Config A (Baseline)", "params": {"objective": "lambdarank", "n_estimators": 600, "num_leaves": 31, "learning_rate": 0.05, "verbose": -1}},
        {"name": "Config C (rank_xendcg)", "params": {"objective": "rank_xendcg", "n_estimators": 600, "num_leaves": 31, "learning_rate": 0.05, "verbose": -1}},
        {"name": "Config E (Ultra-Fast 0.1 LR)", "params": {"objective": "rank_xendcg", "n_estimators": 200, "num_leaves": 15, "learning_rate": 0.1, "verbose": -1}}
    ]

    results = []
    for cfg in configs:
        print(f"[EXEC] Torneio: {cfg['name']}...")
        p10, p20 = walk_forward_eval(feats, feat_names, dates, cvli_raw, cfg['params'])
        print(f"     Média: P@10={p10:.1f}% | P@20={p20:.1f}%")
        results.append({"Config": cfg['name'], "P@10": p10, "P@20": p20})

    section("RESULTADO FINAL")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    run()
