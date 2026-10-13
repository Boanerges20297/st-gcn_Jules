"""
Experimento de Hibridismo / Ensemble: ST-GAT DeepSTGAT_v5 + Poisson Regressor
Foco: Fortaleza (121 nós) nas janelas de validação oficial.

Compara:
- ST-GAT puro (alpha = 1.0)
- Poisson puro (alpha = 0.0)
- Blends intermediários (alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
- Combinações lineares e por ranking percentil.
"""

import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src", "core"))
sys.path.append(os.path.join(ROOT_DIR, "scripts", "training", "Active"))

from train_all_specialists import (
    REGION_CONFIGS,
    build_temporal_split_config,
    get_active_nodes_bimestral,
    inject_momentum_channels,
    normalize_adj,
    PREDICT_HORIZON,
)
from src.core.architectures import DeepSTGAT_v5
from src.core.fortaleza_poisson_backend import FEATURE_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_p_at_k(pred_scores, y_true, k, n_events):
    if n_events < 1:
        return 0.0
    k_adj = min(k, n_events, len(pred_scores))
    if k_adj <= 0:
        return 0.0
    top_pred_idx = np.argsort(pred_scores)[::-1][:k]
    top_true_idx = np.argsort(y_true)[::-1][:k_adj]
    hits = len(set(top_pred_idx) & set(top_true_idx))
    return hits / k_adj


def build_poisson_val_features(nf, enriched, nodes_gdf, dates, t, horizon_days=30):
    n_nodes = nf.shape[0]
    target_start = dates[t]
    day_of_week = target_start.dayofweek
    month = target_start.month
    sin_dow = math.sin(2.0 * math.pi * day_of_week / 7.0)
    cos_dow = math.cos(2.0 * math.pi * day_of_week / 7.0)
    sin_month = math.sin(2.0 * math.pi * month / 12.0)
    cos_month = math.cos(2.0 * math.pi * month / 12.0)
    hist_slice = nf[:, :t, 0]
    sum_7 = nf[:, max(0, t - 7):t, 0].sum(axis=1)
    sum_14 = nf[:, max(0, t - 14):t, 0].sum(axis=1)
    sum_30 = nf[:, max(0, t - 30):t, 0].sum(axis=1)
    sum_60 = nf[:, max(0, t - 60):t, 0].sum(axis=1)
    prev_7 = nf[:, max(0, t - 14):max(0, t - 7), 0].sum(axis=1)

    tension_index = nodes_gdf["tension_index"].fillna(0.0).to_numpy(dtype=float) if "tension_index" in nodes_gdf.columns else np.zeros(n_nodes)
    recent_cvli = nodes_gdf.get("recent_cvli", pd.Series(np.zeros(n_nodes))).fillna(0.0).to_numpy(dtype=float)
    total_cvli = nodes_gdf.get("total_cvli", pd.Series(np.zeros(n_nodes))).fillna(0.0).to_numpy(dtype=float)

    rows = []
    for node_idx in range(n_nodes):
        rows.append({
            "lag_1": float(nf[node_idx, t - 1, 0]),
            "sum_7": float(sum_7[node_idx]),
            "sum_14": float(sum_14[node_idx]),
            "sum_30": float(sum_30[node_idx]),
            "sum_60": float(sum_60[node_idx]),
            "mean_7": float(sum_7[node_idx] / 7.0),
            "mean_14": float(sum_14[node_idx] / 14.0),
            "mean_30": float(sum_30[node_idx] / 30.0),
            "mean_60": float(sum_60[node_idx] / 60.0),
            "hist_mean": float(hist_slice[node_idx].mean()) if hist_slice.shape[1] > 0 else 0.0,
            "hist_sum": float(hist_slice[node_idx].sum()),
            "momentum_7": float(sum_7[node_idx] - prev_7[node_idx]),
            "momentum_14": float(enriched[node_idx, t - 1, 34]) if enriched.shape[2] > 34 else 0.0,
            "momentum_30": float(enriched[node_idx, t - 1, 35]) if enriched.shape[2] > 35 else 0.0,
            "cold_streak_inv": float(enriched[node_idx, t - 1, 36]) if enriched.shape[2] > 36 else 0.0,
            "tension_index": float(tension_index[node_idx]),
            "recent_cvli_static": float(recent_cvli[node_idx]),
            "total_cvli_static": float(total_cvli[node_idx]),
            "sin_dow": sin_dow,
            "cos_dow": cos_dow,
            "sin_month": sin_month,
            "cos_month": cos_month,
            "is_weekend": float(day_of_week >= 5),
        })
    return pd.DataFrame(rows)


def run_experiment():
    print("=" * 80)
    print("EXPERIMENTO DE HIBRIDISMO / ENSEMBLE: ST-GAT (DeepSTGAT_v5) + POISSON REGRESSOR")
    print("Região: Fortaleza (121 nós) | Horizonte: 30 dias | Janela ST-GAT: 120 dias")
    print("=" * 80)

    # 1. Carregar dados processados
    data_path = os.path.join(ROOT_DIR, "data", "processed", "processed_fortaleza.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    nf = data["node_features"]
    dates = pd.to_datetime(data["dates"])
    adj_geo_np = data["adj_geo"]
    adj_conf_np = data["adj_conflict"]
    nodes_gdf = data["nodes_gdf"].reset_index(drop=True)
    n_nodes, total_steps, _ = nf.shape

    adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(adj_conf_np), dtype=torch.float32).to(DEVICE)
    features = inject_momentum_channels(nf)

    # 2. Split temporal oficial
    cfg = REGION_CONFIGS["fortaleza"]
    window = cfg["window"]  # 120
    split_cfg = build_temporal_split_config(dates)
    print(f"Treino: {split_cfg['train_start'].date()} -> {split_cfg['train_end'].date()}")
    print(f"Validação: {split_cfg['val_start'].date()} -> {split_cfg['val_end'].date()}")

    # 3. Montar janelas de validação
    val_windows = []
    for t in range(window, total_steps - PREDICT_HORIZON):
        target_start = dates[t]
        target_end = dates[t + PREDICT_HORIZON - 1]
        if target_start >= split_cfg["val_start"] and target_end <= split_cfg["val_end"]:
            x_win = features[:, t - window:t, :].copy()
            x = torch.tensor(x_win, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            if x.shape[1] < 41:
                padding = torch.zeros((1, 41 - x.shape[1], n_nodes, window))
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] > 41:
                x = x[:, :41, :, :]
            y = nf[:, t:t + PREDICT_HORIZON, 0].sum(axis=1).astype(np.float32)
            val_windows.append({
                "t": t,
                "x": x,
                "y": y,
                "date": target_start.strftime("%Y-%m-%d"),
            })

    print(f"Total de janelas de validação: {len(val_windows)}")

    # 4. Carregar ST-GAT
    stgat_model = DeepSTGAT_v5(
        num_nodes=n_nodes,
        in_channels=41,
        time_steps=window,
        num_graphs=2,
        dropout=cfg["dropout"],
    ).to(DEVICE)

    stgat_ckpt_path = os.path.join(ROOT_DIR, "models", "active", "legacy_torch", "fortaleza_model_active.pth")
    stgat_ckpt = torch.load(stgat_ckpt_path, map_location=DEVICE, weights_only=False)
    stgat_model.load_state_dict(stgat_ckpt["model_state_dict"], strict=False)
    stgat_model.eval()
    print(f"ST-GAT carregado de: {stgat_ckpt_path}")
    if "validation_metrics" in stgat_ckpt:
        print(f"Métricas no checkpoint ST-GAT: {stgat_ckpt['validation_metrics']}")

    # 5. Carregar Poisson
    poisson_path = os.path.join(ROOT_DIR, "models", "active", "production", "poisson", "fortaleza_poisson_regressor.pkl")
    with open(poisson_path, "rb") as f:
        poisson_payload = pickle.load(f)
    poisson_pipeline = poisson_payload["pipeline"]
    print(f"Poisson carregado de: {poisson_path}")

    # 6. Gerar predições de cada modelo nas janelas de validação
    predictions = []
    with torch.no_grad():
        for item in val_windows:
            t = item["t"]
            vx = item["x"].to(DEVICE)
            vy = item["y"]
            n_events = int((vy > 0).sum())

            # ST-GAT: saída bruta e softplus (escala de contagem mu)
            raw_stgat = stgat_model(vx, [adj_geo, adj_conf]).squeeze().cpu().numpy()
            mu_stgat = F.softplus(torch.from_numpy(raw_stgat)).numpy()

            # Poisson: predição count
            poisson_df = build_poisson_val_features(nf, features, nodes_gdf, dates, t)
            poisson_pred = np.clip(poisson_pipeline.predict(poisson_df[FEATURE_COLS]), 0.0, None)

            # Normalizações MinMax para cada janela [0, 1]
            def min_max(arr):
                rng = arr.max() - arr.min()
                return (arr - arr.min()) / (rng + 1e-9)

            norm_stgat = min_max(mu_stgat)
            norm_poisson = min_max(poisson_pred)

            # Ranking Percentil (0 a 1) para cada janela
            # Valores maiores recebem percentil mais alto
            rank_stgat = pd.Series(mu_stgat).rank(pct=True).to_numpy()
            rank_poisson = pd.Series(poisson_pred).rank(pct=True).to_numpy()

            predictions.append({
                "date": item["date"],
                "y": vy,
                "n_events": n_events,
                "stgat_mu": mu_stgat,
                "poisson_mu": poisson_pred,
                "norm_stgat": norm_stgat,
                "norm_poisson": norm_poisson,
                "rank_stgat": rank_stgat,
                "rank_poisson": rank_poisson,
            })

    # Filtrar janelas válidas com n_events >= k_eval (10) como no treinamento oficial
    k_eval = cfg.get("k_eval", 10)
    valid_preds = [p for p in predictions if p["n_events"] >= k_eval]
    print(f"Janelas com n_events >= {k_eval}: {len(valid_preds)} de {len(predictions)}")

    # 7. Testar diferentes pesos alpha
    # Score Composto: 0.60 * P@10 + 0.40 * P@20
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def evaluate_blend(blend_type="minmax"):
        results_list = []
        for a in alphas:
            p10_list, p20_list, mae_list = [], [], []
            for p in valid_preds:
                y = p["y"]
                n_events = p["n_events"]

                if blend_type == "minmax":
                    s = a * p["norm_stgat"] + (1.0 - a) * p["norm_poisson"]
                elif blend_type == "percentile":
                    s = a * p["rank_stgat"] + (1.0 - a) * p["rank_poisson"]
                elif blend_type == "direct_count":
                    s = a * p["stgat_mu"] + (1.0 - a) * p["poisson_mu"]
                elif blend_type == "geometric":
                    # Média geométrica ponderada
                    s = (p["norm_stgat"] + 1e-4) ** a * (p["norm_poisson"] + 1e-4) ** (1.0 - a)

                p10 = compute_p_at_k(s, y, 10, n_events)
                p20 = compute_p_at_k(s, y, 20, n_events)
                mae = np.mean(np.abs(s - y)) if blend_type == "direct_count" else np.nan

                p10_list.append(p10)
                p20_list.append(p20)
                if blend_type == "direct_count":
                    mae_list.append(mae)

            avg_p10 = np.mean(p10_list)
            avg_p20 = np.mean(p20_list)
            comp_score = 0.60 * avg_p10 + 0.40 * avg_p20
            avg_mae = np.mean(mae_list) if mae_list else None

            results_list.append({
                "alpha_stgat": a,
                "weight_poisson": round(1.0 - a, 2),
                "p10": avg_p10,
                "p20": avg_p20,
                "composite_score": comp_score,
                "mae": avg_mae,
            })
        return pd.DataFrame(results_list)

    print("\n--- 1. BLEND MIN-MAX NORMALIZADO (Score = alpha * norm_stgat + (1-alpha) * norm_poisson) ---")
    df_minmax = evaluate_blend("minmax")
    print(df_minmax.to_string(index=False, formatters={
        "p10": "{:.2%}".format,
        "p20": "{:.2%}".format,
        "composite_score": "{:.2%}".format,
    }))

    print("\n--- 2. BLEND DE RANKING PERCENTIL (Score = alpha * rank_stgat + (1-alpha) * rank_poisson) ---")
    df_pct = evaluate_blend("percentile")
    print(df_pct.to_string(index=False, formatters={
        "p10": "{:.2%}".format,
        "p20": "{:.2%}".format,
        "composite_score": "{:.2%}".format,
    }))

    print("\n--- 3. BLEND DE CONTAGEM DIRETA (Score = alpha * mu_stgat + (1-alpha) * mu_poisson) ---")
    df_count = evaluate_blend("direct_count")
    print(df_count.to_string(index=False, formatters={
        "p10": "{:.2%}".format,
        "p20": "{:.2%}".format,
        "composite_score": "{:.2%}".format,
        "mae": lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
    }))

    print("\n--- 4. BLEND GEOMÉTRICO (Score = norm_stgat^alpha * norm_poisson^(1-alpha)) ---")
    df_geo = evaluate_blend("geometric")
    print(df_geo.to_string(index=False, formatters={
        "p10": "{:.2%}".format,
        "p20": "{:.2%}".format,
        "composite_score": "{:.2%}".format,
    }))

    # Comparação direta e melhor resultado
    best_minmax = df_minmax.loc[df_minmax["composite_score"].idxmax()]
    best_pct = df_pct.loc[df_pct["composite_score"].idxmax()]
    best_count = df_count.loc[df_count["composite_score"].idxmax()]
    best_geo = df_geo.loc[df_geo["composite_score"].idxmax()]

    print("\n" + "=" * 80)
    print("RESUMO COMPARATIVO DE MELHORES CONFIGURAÇÕES:")
    print("=" * 80)
    print(f"ST-GAT Puro (alpha=1.0):      P@10 = {df_minmax.iloc[-1]['p10']:.2%} | P@20 = {df_minmax.iloc[-1]['p20']:.2%} | Composto = {df_minmax.iloc[-1]['composite_score']:.2%}")
    print(f"Poisson Puro (alpha=0.0):     P@10 = {df_minmax.iloc[0]['p10']:.2%} | P@20 = {df_minmax.iloc[0]['p20']:.2%} | Composto = {df_minmax.iloc[0]['composite_score']:.2%}")
    print(f"Melhor Min-Max (alpha={best_minmax['alpha_stgat']}):    P@10 = {best_minmax['p10']:.2%} | P@20 = {best_minmax['p20']:.2%} | Composto = {best_minmax['composite_score']:.2%}")
    print(f"Melhor Percentil (alpha={best_pct['alpha_stgat']}):  P@10 = {best_pct['p10']:.2%} | P@20 = {best_pct['p20']:.2%} | Composto = {best_pct['composite_score']:.2%}")
    print(f"Melhor Direto (alpha={best_count['alpha_stgat']}):     P@10 = {best_count['p10']:.2%} | P@20 = {best_count['p20']:.2%} | Composto = {best_count['composite_score']:.2%}")
    print(f"Melhor Geométrico (alpha={best_geo['alpha_stgat']}):  P@10 = {best_geo['p10']:.2%} | P@20 = {best_geo['p20']:.2%} | Composto = {best_geo['composite_score']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
