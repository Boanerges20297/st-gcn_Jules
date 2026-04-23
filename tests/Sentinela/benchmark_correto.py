"""
====================================================================
SENTINELA - BENCHMARK COMPARATIVO P@20 (Protocolo Correto)
====================================================================
Protocolo identico ao ST-GAT:
  - Target: bairros com >= 1 CVLI nos proximos 14 dias (horizonte)
  - Avaliacao: P@K sobre ranking de risco (nao binario por dia)
  - Dados: 2024-01-01 a 2026-04-14 (periodo de treino/val do ST-GAT)
  - Filtro: apenas bairros com >0.4 CVLI/mes
  - Walk-forward: testar Out/Nov/Dez/2025 + Jan/2026

Metricas reportadas:
  P@10  -- igual ao training log (para calibrar vs ~43%)
  P@20  -- extensao natural para teste operacional

Rodando:
  .\.venv\Scripts\python.exe tests/Sentinela/benchmark_correto.py
====================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, pickle, json, time, warnings, unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRanker

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURACOES
# ─────────────────────────────────────────────────────────────────
BASE_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW    = os.path.join(BASE_PATH, "data", "raw")
DATA_PROC   = os.path.join(BASE_PATH, "data", "processed")
MODELS_PATH = os.path.join(BASE_PATH, "models", "active")
OUT_PATH    = os.path.join(BASE_PATH, "tests", "Sentinela")

CKPT_PATH   = os.path.join(MODELS_PATH, "fortaleza_model_active.pth")
CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
LATLON_FILE = os.path.join(DATA_RAW, "bairros_centros_latlong.json")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZON     = 14    # dias: identico ao predict_horizon_days do ST-GAT
K_LIST      = [10, 20]  # avaliar P@10 e P@20
MIN_CVLI_MES= 0.4   # filtro de ruido
WINDOW_STGAT= 120   # janela do ST-GAT
WINDOW_NN   = 60    # janela LSTM/TCN (menor = mais rapido)
N_FOLDS     = 4     # Out/Nov/Dez 2025 + Jan 2026

# ─────────────────────────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────────────────────────
def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def section(title, w=68):
    return f"\n{'='*w}\n  {title}\n{'='*w}"

def pk_from_ranking(scores_N, cvli_horizon_N, k):
    """
    scores_N: (N,) score de risco por bairro (maior = mais risco)
    cvli_horizon_N: (N,) CVLI real acumulado no horizonte de 14 dias
    k: quantos bairros pegar no topo
    Retorna: P@k (0.0 a 1.0)
    """
    top_pred = np.argsort(scores_N)[::-1][:k]
    top_real = np.argsort(cvli_horizon_N)[::-1][:k]
    hits = len(set(top_pred) & set(top_real))
    return hits / k

# ─────────────────────────────────────────────────────────────────
#  ARQUITETURA ST-GAT (espelho exato de architectures.py)
# ─────────────────────────────────────────────────────────────────
class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, channels, heads=2):
        super().__init__()
        self.mha  = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        B, C, N, T = x.shape
        xt = x.mean(dim=2).permute(0, 2, 1)
        ao, _ = self.mha(xt, xt, xt)
        xt = self.norm(xt + ao)
        return x * torch.sigmoid(xt.permute(0, 2, 1).unsqueeze(2))

class FastRelationalGCN(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.4):
        super().__init__()
        self.W_self = nn.Linear(in_f, out_f)
        self.W_geo  = nn.Linear(in_f, out_f)
        self.W_conf = nn.Linear(in_f, out_f)
        self.drop   = nn.Dropout(dropout)
        self.bn     = nn.BatchNorm1d(out_f)
        self.prelu  = nn.PReLU()
    def forward(self, x, adj_list):
        ag, ac = adj_list
        out = self.W_self(x) + torch.matmul(ag, self.W_geo(x)) + torch.matmul(ac, self.W_conf(x))
        BT, N, C = out.shape
        return self.drop(self.prelu(self.bn(out.view(-1, C)).view(BT, N, C)))

class GlobalSpatialAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.mha  = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        ao, _ = self.mha(x, x, x)
        return self.norm(x + ao)

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, T, dropout=0.4):
        super().__init__()
        self.time_conv           = nn.Conv2d(in_c, out_c, (1, 3), padding=(0, 1))
        self.prelu               = nn.PReLU()
        self.spatial_transformer = GlobalSpatialAttention(out_c)
        self.gcn                 = FastRelationalGCN(out_c, out_c, dropout)
        self.temp_attn           = MultiHeadTemporalAttention(out_c)
        self.residual            = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, adj):
        res = self.residual(x)
        x   = self.prelu(self.time_conv(x))
        B, C, N, T = x.shape
        xf = x.permute(0, 3, 2, 1).reshape(B*T, N, C)
        xf = self.gcn(self.spatial_transformer(xf), adj)
        x  = xf.reshape(B, T, N, C).permute(0, 3, 2, 1)
        return self.temp_attn(x) + res

class DeepSTGAT_64(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.4):
        super().__init__()
        self.layer1      = STGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2      = STGCNBlock(32, 64, time_steps, dropout)
        self.layer3      = STGCNBlock(64, 64, time_steps, dropout)
        self.final_conv  = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.prelu_final = nn.PReLU()
        self.fc          = nn.Sequential(nn.Linear(64, 32), nn.PReLU(), nn.Linear(32, 1))
    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.layer3(x, adj)
        x = self.prelu_final(self.final_conv(x)).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)

# ─────────────────────────────────────────────────────────────────
#  MODELOS NEURAIS (LSTM e TCN)
# ─────────────────────────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, padding=pad, dilation=dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.pad   = pad
    def _chomp(self, x): return x[:, :, :-self.pad] if self.pad > 0 else x
    def forward(self, x):
        res = self.res(x)
        out = self.drop(self.relu(self._chomp(self.conv1(x))))
        out = self.drop(self.relu(self._chomp(self.conv2(out))))
        return self.relu(out + res)

class TCNClassifier(nn.Module):
    def __init__(self, in_dim, channels=64, levels=3):
        super().__init__()
        blocks, ch = [], in_dim
        for i in range(levels):
            blocks.append(TCNBlock(ch, channels, kernel=3, dilation=2**i))
            ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.fc  = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out = self.tcn(x.permute(0, 2, 1))
        return self.fc(out[:, :, -1]).squeeze(-1)

# ─────────────────────────────────────────────────────────────────
#  1. CONSTRUCAO DOS DADOS (37 canais, identico ao ST-GAT)
# ─────────────────────────────────────────────────────────────────
def build_data():
    pkl_path = os.path.join(DATA_PROC, "processed_fortaleza_bench37.pkl")
    if os.path.exists(pkl_path):
        print("[*] Carregando cache bench37...")
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        print(f"    {len(d['top_bairros'])} bairros | {len(d['dates'])} dias | {d['in_channels']}ch")
        return d

    print("[*] Construindo features 37ch dos CSVs raw...")

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
    df_t["score_intent"] = (
        df_t["qtd_armas"] * 10.0
        + np.log1p(df_t["qtd_drogas"]) * 5.0
        + df_t["qtd_veiculos_apreendidos"] * 3.0
    ).astype("float32")

    # Selecionar top-40 bairros por CVLI nos últimos 2 anos
    cutoff = df["data"].max() - pd.Timedelta(days=730)
    cvli_rank = df[df["data"] >= cutoff].groupby("bairro")["is_cvli"].sum().sort_values(ascending=False)
    cvli_pm   = cvli_rank / 24.0
    top_bairros = cvli_pm[cvli_pm > MIN_CVLI_MES].index[:40].tolist()
    print(f"    {len(top_bairros)} bairros selecionados (>{MIN_CVLI_MES} CVLI/mes)")

    start_d = pd.Timestamp("2024-01-01")
    end_d   = df["data"].max()
    dates   = pd.date_range(start_d, end_d, freq="D")
    N, T, C = len(top_bairros), len(dates), 37

    features = np.zeros((N, T, C), dtype=np.float32)
    node_map = {b: i for i, b in enumerate(top_bairros)}
    date_map = {d: i for i, d in enumerate(dates)}

    df_p   = df[df["data"] >= start_d].copy()
    df_t_p = df_t[df_t["data"] >= start_d].copy()

    # Canal 0: CVLI
    for _, row in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: features[ni, ti, 0] = row["v"]

    # Canal 1: CVP * 2.5
    for _, row in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: features[ni, ti, 1] = row["v"] * 2.5

    # Canal 2: tension (0.5 default)
    features[:, :, 2] = 0.5

    # Canal 27: intel
    for _, row in df_t_p.groupby(["data","bairro"])["score_intent"].sum().reset_index().iterrows():
        ni = node_map.get(row["bairro"]); ti = date_map.get(row["data"])
        if ni is not None and ti is not None: features[ni, ti, 27] = float(row["score_intent"]) * 2.0

    # Canais temporais
    for di, dt in enumerate(dates):
        dow = dt.dayofweek
        features[:, di, 3 + dow] = 1.0
        if dow == 4: features[:, di, 7] = 1.5
        features[:, di, 10 + dt.month - 1] = 1.0
        if dow >= 5: features[:, di, 22] = 1.0
        if dt.day <= 10 or dt.day >= 30: features[:, di, 30] = 1.0

    # Canal 24: rolling 7d
    for ni in range(N):
        features[ni, :, 24] = pd.Series(features[ni, :, 0]).rolling(7, min_periods=1).sum().values

    # Canal 28: pressao global
    features[:, :, 28] = features[:, :, 0].sum(axis=0, keepdims=True)

    # Adjacência geográfica
    with open(LATLON_FILE, encoding="utf-8") as f:
        raw_ll = json.load(f)
    ll_norm = {norm(k): v for k, v in raw_ll.items()}
    coords  = np.array([[ll_norm.get(b, {"lat":0,"long":0})["lat"],
                         ll_norm.get(b, {"lat":0,"long":0})["long"]]
                        for b in top_bairros], dtype=np.float32)

    dist_mat     = cdist(coords, coords, "euclidean")
    adj_geo      = (dist_mat < 0.05).astype(np.float32)
    adj_conflict = np.eye(N, dtype=np.float32)

    def norm_adj(A):
        D = A.sum(axis=1, keepdims=True)
        D[D==0] = 1
        return A / D

    adj_geo      = norm_adj(adj_geo)
    adj_conflict = norm_adj(adj_conflict)

    d = {"node_features": features, "adj_geo": adj_geo,
         "adj_conflict": adj_conflict, "dates": dates,
         "top_bairros": top_bairros, "in_channels": C}

    with open(pkl_path, "wb") as f:
        pickle.dump(d, f)
    print(f"    Salvo: {pkl_path}")
    return d

# ─────────────────────────────────────────────────────────────────
#  2. TARGETS CORRETOS: CVLI ACUMULADO NO HORIZONTE DE 14 DIAS
# ─────────────────────────────────────────────────────────────────
def get_horizon_targets(feat, ti, horizon=HORIZON):
    """
    Retorna (N,) com o total de CVLI de cada bairro nos proximos
    `horizon` dias a partir de ti+1.
    Este e o target real que o ST-GAT aprende a rankear.
    """
    N = feat.shape[0]
    T = feat.shape[1]
    end = min(ti + horizon + 1, T)
    return feat[:, ti+1:end, 0].sum(axis=1)  # (N,)

# ─────────────────────────────────────────────────────────────────
#  3. AVALIACAO P@K (protocolo identico ao ST-GAT)
# ─────────────────────────────────────────────────────────────────
def evaluate_model(scores_matrix, test_range, feat, k_list=K_LIST):
    """
    scores_matrix: (N, T_test) ou funcao que retorna (N,) para cada ti
    Retorna dict {k: p@k_medio}
    """
    results = {k: [] for k in k_list}
    test_list = list(test_range)
    N = feat.shape[0]

    for col_i, ti in enumerate(test_list):
        if ti + HORIZON >= feat.shape[1]:
            continue
        targets = get_horizon_targets(feat, ti)
        if targets.sum() == 0:
            continue  # sem CVLI no horizonte = skip

        if hasattr(scores_matrix, "__call__"):
            scores = scores_matrix(ti)
        else:
            scores = scores_matrix[:, col_i]

        for k in k_list:
            results[k].append(pk_from_ranking(scores, targets, k))

    return {k: float(np.mean(v)) * 100 if v else 0.0 for k, v in results.items()}

# ─────────────────────────────────────────────────────────────────
#  4. MODELOS
# ─────────────────────────────────────────────────────────────────

# ── ST-GAT ──
def run_stgat(data, test_range):
    ckpt    = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg     = ckpt.get("config", {})
    W       = cfg.get("window", WINDOW_STGAT)
    C_saved = cfg.get("in_channels", 37)
    N       = len(data["top_bairros"])
    feat    = data["node_features"]

    adj_geo = torch.FloatTensor(data["adj_geo"]).to(DEVICE)
    adj_cf  = torch.FloatTensor(data["adj_conflict"]).to(DEVICE)
    adj     = [adj_geo, adj_cf]

    model = DeepSTGAT_64(num_nodes=N, in_channels=C_saved, time_steps=W)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE).eval()

    T_test  = len(list(test_range))
    scores  = np.zeros((N, T_test), dtype=np.float32)

    for col_i, ti in enumerate(test_range):
        if ti < W:
            continue
        window = feat[:, ti-W:ti, :]               # (N, W, C_bench)
        C_bench = window.shape[2]
        if C_bench < C_saved:
            pad    = np.zeros((N, W, C_saved - C_bench), dtype=np.float32)
            window = np.concatenate([window, pad], axis=-1)
        elif C_bench > C_saved:
            window = window[:, :, :C_saved]

        x = torch.FloatTensor(window).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x, adj)
            sc  = torch.sigmoid(out).squeeze().cpu().numpy()
        if sc.ndim == 0: sc = np.array([float(sc)])
        scores[:, col_i] = sc

    return scores

# ── Naive EWMA ──
def run_ewma(data, train_range, test_range, halflife=14):
    feat     = data["node_features"]
    N        = feat.shape[0]
    T_test   = len(list(test_range))
    train_end= list(train_range)[-1]

    # Score acumulado por horizonte (EWMA sobre CVLI historico)
    scores = np.zeros((N, T_test), dtype=np.float32)
    for col_i, ti in enumerate(test_range):
        for ni in range(N):
            s = pd.Series(feat[ni, :ti+1, 0]).ewm(halflife=halflife).mean()
            scores[ni, col_i] = float(s.iloc[-1])
    return scores

# ── LightGBM LambdaRank (com target = CVLI horizonte acumulado) ──
def run_lgbm_rank(data, train_range, test_range):
    feat      = data["node_features"]
    bairros   = data["top_bairros"]
    dates     = data["dates"]
    N, T, C   = feat.shape

    def build_flat(idx_range, is_train=True):
        rows = []
        for ti in idx_range:
            if ti < 60: continue
            if ti + HORIZON >= T: continue
            targets_h = get_horizon_targets(feat, ti)
            if targets_h.sum() == 0 and is_train: continue  # skip dias sem crime no treino

            for ni in range(N):
                row_feat = []
                for hl in [3, 7, 14, 30, 60]:
                    s = pd.Series(feat[ni, max(0,ti-60):ti, 0])
                    row_feat.append(float(s.ewm(halflife=hl).mean().iloc[-1]) if len(s) > 0 else 0.0)
                for hl in [7, 14, 30]:
                    s = pd.Series(feat[ni, max(0,ti-30):ti, 1])
                    row_feat.append(float(s.ewm(halflife=hl).mean().iloc[-1]) if len(s) > 0 else 0.0)
                for hl in [7, 14]:
                    s = pd.Series(feat[ni, max(0,ti-30):ti, 27])
                    row_feat.append(float(s.ewm(halflife=hl).mean().iloc[-1]) if len(s) > 0 else 0.0)
                # Rolling sums
                row_feat += [
                    float(feat[ni, max(0,ti-7):ti, 0].sum()),
                    float(feat[ni, max(0,ti-14):ti, 0].sum()),
                    float(feat[ni, max(0,ti-30):ti, 0].sum()),
                ]
                # Canais temporais
                row_feat += list(feat[ni, ti, 3:23])
                row_feat.append(float(ni))  # bairro encoding

                label = int(targets_h[ni] > 0)
                rows.append({"ti": ti, "ni": ni, "label": label,
                             "cvli_h": float(targets_h[ni]),
                             **{f"f{j}": v for j, v in enumerate(row_feat)}})

        return pd.DataFrame(rows)

    print("      [LGBM] Construindo features...", flush=True)
    tr_list   = list(train_range)
    # Usar step=3 no treino para acelerar
    tr_sample = tr_list[::3]
    df_tr = build_flat(tr_sample)
    df_te = build_flat(list(test_range), is_train=False)

    feat_cols = [c for c in df_tr.columns if c.startswith("f")]

    if df_tr.empty or df_te.empty:
        T_test = len(list(test_range))
        return np.zeros((N, T_test))

    df_tr = df_tr.sort_values("ti")
    df_te = df_te.sort_values("ti")
    groups_tr = df_tr.groupby("ti").size().values

    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[10, 20],
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        min_child_samples=3, random_state=42, n_jobs=-1, verbose=-1,
    )
    ranker.fit(
        df_tr[feat_cols],
        df_tr["cvli_h"].clip(0, 10).astype("int32"),
        group=groups_tr
    )
    df_te["score"] = ranker.predict(df_te[feat_cols])

    T_test    = len(list(test_range))
    scores    = np.zeros((N, T_test), dtype=np.float32)
    ti_to_col = {ti: col for col, ti in enumerate(list(test_range))}
    for _, row in df_te.iterrows():
        col = ti_to_col.get(int(row["ti"]), -1)
        if col >= 0: scores[int(row["ni"]), col] = float(row["score"])

    return scores

# ── LSTM (target = CVLI horizonte, regressao) ──
def run_lstm(data, train_range, test_range):
    feat  = data["node_features"]
    N, T, C = feat.shape
    W = WINDOW_NN

    def make_seqs(idx_range, step=1):
        Xs, ys, ni_list, ti_list = [], [], [], []
        for ti in list(idx_range)[::step]:
            if ti < W or ti + HORIZON >= T: continue
            targets_h = get_horizon_targets(feat, ti)
            for ni in range(N):
                Xs.append(feat[ni, ti-W:ti, :].copy())
                ys.append(float(np.log1p(targets_h[ni])))
                ni_list.append(ni); ti_list.append(ti)
        return np.array(Xs), np.array(ys, dtype=np.float32), ni_list, ti_list

    X_tr, y_tr, _, _         = make_seqs(train_range, step=3)
    X_te, y_te, ni_te, ti_te = make_seqs(test_range)

    if X_tr.shape[0] < 10:
        return np.zeros((N, len(list(test_range))))

    sc_  = StandardScaler()
    N_tr, Ww, Cc = X_tr.shape
    X_tr_s = sc_.fit_transform(X_tr.reshape(-1, Cc)).reshape(N_tr, Ww, Cc)
    N_te   = X_te.shape[0]
    X_te_s = sc_.transform(X_te.reshape(-1, Cc)).reshape(N_te, Ww, Cc)

    model = SimpleLSTM(in_dim=Cc).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    crit  = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        list(zip(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr))),
        batch_size=128, shuffle=True
    )
    model.train()
    for ep in range(8):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad(); crit(model(bx), by).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()

    T_test    = len(list(test_range))
    scores    = np.zeros((N, T_test), dtype=np.float32)
    ti_to_col = {ti: col for col, ti in enumerate(list(test_range))}
    for idx, (ni, ti, sc) in enumerate(zip(ni_te, ti_te, out)):
        col = ti_to_col.get(ti, -1)
        if col >= 0: scores[ni, col] = float(sc)
    return scores

# ── TCN (idem) ──
def run_tcn(data, train_range, test_range):
    feat  = data["node_features"]
    N, T, C = feat.shape
    W = WINDOW_NN

    def make_seqs(idx_range, step=1):
        Xs, ys, ni_list, ti_list = [], [], [], []
        for ti in list(idx_range)[::step]:
            if ti < W or ti + HORIZON >= T: continue
            targets_h = get_horizon_targets(feat, ti)
            for ni in range(N):
                Xs.append(feat[ni, ti-W:ti, :].copy())
                ys.append(float(np.log1p(targets_h[ni])))
                ni_list.append(ni); ti_list.append(ti)
        return np.array(Xs), np.array(ys, dtype=np.float32), ni_list, ti_list

    X_tr, y_tr, _, _         = make_seqs(train_range, step=3)
    X_te, y_te, ni_te, ti_te = make_seqs(test_range)

    if X_tr.shape[0] < 10:
        return np.zeros((N, len(list(test_range))))

    sc_  = StandardScaler()
    N_tr, Ww, Cc = X_tr.shape
    X_tr_s = sc_.fit_transform(X_tr.reshape(-1, Cc)).reshape(N_tr, Ww, Cc)
    N_te   = X_te.shape[0]
    X_te_s = sc_.transform(X_te.reshape(-1, Cc)).reshape(N_te, Ww, Cc)

    model = TCNClassifier(in_dim=Cc).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    crit  = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        list(zip(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr))),
        batch_size=128, shuffle=True
    )
    model.train()
    for ep in range(8):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad(); crit(model(bx), by).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()

    T_test    = len(list(test_range))
    scores    = np.zeros((N, T_test), dtype=np.float32)
    ti_to_col = {ti: col for col, ti in enumerate(list(test_range))}
    for idx, (ni, ti, sc) in enumerate(zip(ni_te, ti_te, out)):
        col = ti_to_col.get(ti, -1)
        if col >= 0: scores[ni, col] = float(sc)
    return scores

# ─────────────────────────────────────────────────────────────────
#  5. SPLITS (Out/Nov/Dez 2025 + Jan 2026)
# ─────────────────────────────────────────────────────────────────
def make_splits(dates, n_folds=N_FOLDS):
    """Folds mensais comecando em Out/2025 (dia 639 do grid 2024-)"""
    start_test = 639   # 2025-10-01
    splits = []
    for i in range(n_folds):
        t_start = start_test + i * 30
        t_end   = t_start + 30
        if t_end > len(dates): break
        label = f"Fold_{i+1}_{dates[t_start].strftime('%Y-%m')}"
        splits.append((range(0, t_start), range(t_start, t_end), label))
    return splits

# ─────────────────────────────────────────────────────────────────
#  6. BENCHMARK PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def run():
    print(section("SENTINELA - BENCHMARK P@10 e P@20 (Protocolo Correto)"))
    print(f"  Inicio:    {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Device:    {DEVICE}")
    print(f"  Horizonte: {HORIZON} dias (identico ao ST-GAT)")
    print(f"  Referencia ST-GAT: P@10 = 42.9% (checkpoint salvo)")

    data    = build_data()
    feat    = data["node_features"]
    dates   = data["dates"]
    bairros = data["top_bairros"]
    N       = len(bairros)
    splits  = make_splits(dates)

    print(f"\n  Grid: {N} bairros | {len(dates)} dias | {feat.shape[2]}ch")
    print(f"  Splits: {len(splits)} folds (Out/2025 -> Jan/2026)")

    MODELOS = [
        ("ST-GAT_Active", lambda d, tr, te: run_stgat(d, te)),
        ("Naive_EWMA",    run_ewma),
        ("LightGBM_Rank", run_lgbm_rank),
        ("LSTM",          run_lstm),
        ("TCN",           run_tcn),
    ]

    all_rows = []

    for tr_range, te_range, fold_name in splits:
        te_start = dates[list(te_range)[0]].strftime("%d/%m/%Y")
        te_end   = dates[list(te_range)[-1]].strftime("%d/%m/%Y")
        print(section(f"Fold: {fold_name}  [{te_start} -> {te_end}]  treino={len(list(tr_range))}d"))

        for m_name, m_fn in MODELOS:
            print(f"  [{m_name}]", end="  ", flush=True)
            t0 = time.time()
            try:
                scores = m_fn(data, tr_range, te_range)
                res    = evaluate_model(scores, te_range, feat)
                p10    = res.get(10, 0.0)
                p20    = res.get(20, 0.0)
                status = "OK"
            except Exception as e:
                p10, p20, status = 0.0, 0.0, f"ERRO: {e}"
                scores = None
            elapsed = round(time.time() - t0, 1)
            print(f"P@10={p10:.1f}%  P@20={p20:.1f}%  ({elapsed}s)  {status}")
            all_rows.append({
                "Modelo": m_name, "Fold": fold_name,
                "P@10": round(p10, 2), "P@20": round(p20, 2),
                "Tempo_s": elapsed
            })

    # ─────────────────────────────────────────────────────────────
    #  RANKING FINAL
    # ─────────────────────────────────────────────────────────────
    df_res  = pd.DataFrame(all_rows)
    summary = (
        df_res.groupby("Modelo")[["P@10","P@20","Tempo_s"]]
              .mean().round(2)
              .sort_values("P@10", ascending=False)
              .reset_index()
    )

    # Delta vs ST-GAT
    stgat_p10 = summary.loc[summary["Modelo"]=="ST-GAT_Active","P@10"].values
    stgat_p20 = summary.loc[summary["Modelo"]=="ST-GAT_Active","P@20"].values
    if len(stgat_p10) > 0:
        summary["Delta_P10"] = (summary["P@10"] - stgat_p10[0]).apply(lambda x: f"{x:+.1f}%")
        summary["Delta_P20"] = (summary["P@20"] - stgat_p20[0]).apply(lambda x: f"{x:+.1f}%")

    print(section("RANKING FINAL (media sobre todos os folds)"))
    print(summary.to_string(index=False))

    print("\n  Referencia do checkpoint salvo (training loop interno):")
    print(f"    ST-GAT P@10 = 42.9%  (superior a qualquer modelo flat esperado)")
    print()
    print("  INTERPRETACAO:")
    print("    - P@10/P@20 aqui = ranking de bairros por risco nos proximos 14d")
    print("    - Referencia aleatória P@10 com 40 bairros = 25%")
    print("    - Referencia aleatória P@20 com 40 bairros = 50%")

    # Salvar
    res_csv = os.path.join(OUT_PATH, "benchmark_correto_results.csv")
    sum_csv = os.path.join(OUT_PATH, "benchmark_correto_summary.csv")
    rep_txt = os.path.join(OUT_PATH, "benchmark_correto_report.txt")

    df_res.to_csv(res_csv,  index=False, encoding="utf-8-sig")
    summary.to_csv(sum_csv, index=False, encoding="utf-8-sig")

    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write("SENTINELA - BENCHMARK P@10/P@20 (Protocolo Correto)\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Horizonte: {HORIZON} dias | Min CVLI/mes: >{MIN_CVLI_MES}\n")
        f.write(f"N bairros: {N} | Folds: {len(splits)}\n")
        f.write("=" * 68 + "\n\n")
        f.write("Bairros:\n")
        for b in bairros: f.write(f"  - {b}\n")
        f.write("\nResultados por Fold:\n")
        f.write(df_res.to_string(index=False))
        f.write("\n\nRanking Final:\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print(f"\n[OK] Relatorio: {rep_txt}")
    print(section("FIM"))


if __name__ == "__main__":
    run()
