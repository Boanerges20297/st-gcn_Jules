"""
Suite de benchmark para o padrao estocastico do CVLI.

- Analisa o ENRIQUECIDO bruto ate uma data de corte.
- Compara baselines classicos/count-aware com arquiteturas deep graph-temporal.
- Usa split temporal explicito: treino e validacao por datas.

Saidas:
  outputs/benchmarks/cvli_stochastic_suite_*.json
  outputs/benchmarks/cvli_stochastic_suite_*.csv
  outputs/benchmarks/cvli_stochastic_suite_*.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.architectures import (  # noqa: E402
    DeepSTGAT_32,
    DeepSTGAT_64,
    FortalezaHeteroSTGAT,
    PureSTGCN_64,
    ShallowGAT,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_HORIZON = 14
DEFAULT_CHANNELS = 41
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks"


@dataclass(frozen=True)
class RegionConfig:
    key: str
    processed_file: str
    window: int
    deep_candidates: tuple[str, ...]
    batch_size: int
    lr: float


REGION_CONFIGS: dict[str, RegionConfig] = {
    "fortaleza": RegionConfig(
        key="fortaleza",
        processed_file="processed_fortaleza.pkl",
        window=90,
        deep_candidates=("ShallowGAT", "DeepSTGAT_64", "PureSTGCN_64", "FortalezaHeteroSTGAT"),
        batch_size=4,
        lr=2e-4,
    ),
    "rmf": RegionConfig(
        key="rmf",
        processed_file="processed_rmf.pkl",
        window=14,
        deep_candidates=("ShallowGAT", "DeepSTGAT_32", "DeepSTGAT_64"),
        batch_size=8,
        lr=4e-4,
    ),
    "interior": RegionConfig(
        key="interior",
        processed_file="processed_interior.pkl",
        window=14,
        deep_candidates=("ShallowGAT", "DeepSTGAT_32", "DeepSTGAT_64"),
        batch_size=8,
        lr=4e-4,
    ),
}


DEEP_MODEL_REGISTRY = {
    "ShallowGAT": ShallowGAT,
    "DeepSTGAT_32": DeepSTGAT_32,
    "DeepSTGAT_64": DeepSTGAT_64,
    "PureSTGCN_64": PureSTGCN_64,
    "FortalezaHeteroSTGAT": FortalezaHeteroSTGAT,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII").upper().strip()


def clean_name(name) -> str:
    text = normalize_text(name)
    merges = [
        "CONJUNTO CEARA",
        "PRAIA DO FUTURO",
        "VILA MANOEL SATIRO",
        "ALTO ALEGRE",
        "EDSON QUEIROZ",
        "JOSE WALTER",
    ]
    for item in merges:
        if item in text:
            return item
    text = re.sub(r"\s+[IVXLCDM]+$", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    return text.strip()


def resolve_occurrence_location(bairro, cidade) -> str:
    bairro_clean = clean_name(bairro)
    cidade_clean = clean_name(cidade)
    if cidade_clean == "FORTALEZA":
        return bairro_clean or cidade_clean
    if cidade_clean:
        return cidade_clean
    return bairro_clean


def sanitize_qtd_mortes(value, event_type) -> float:
    try:
        qty = float(value)
    except Exception:
        qty = np.nan
    if not np.isfinite(qty) or qty <= 0:
        return 1.0 if str(event_type or "").lower() == "cvli" else 0.0
    return qty


def normalize_adj(adj: np.ndarray) -> torch.Tensor:
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    return torch.tensor(r_mat_inv.dot(adj), dtype=torch.float32, device=DEVICE)


def build_momentum_features(features: np.ndarray) -> np.ndarray:
    n_nodes, n_steps, _ = features.shape
    momentum_feat = np.zeros((n_nodes, n_steps, 4), dtype=np.float32)
    cold_streak = np.zeros(n_nodes, dtype=np.float32)
    for t in range(60, n_steps):
        recent_7 = features[:, t - 7 : t, 0].sum(axis=1)
        prev_7 = features[:, t - 14 : t - 7, 0].sum(axis=1)
        recent_14 = features[:, t - 14 : t, 0].sum(axis=1)
        prev_14 = features[:, t - 28 : t - 14, 0].sum(axis=1)
        recent_30 = features[:, t - 30 : t, 0].sum(axis=1)
        prev_30 = features[:, t - 60 : t - 30, 0].sum(axis=1)
        momentum_feat[:, t, 0] = recent_7 - prev_7
        momentum_feat[:, t, 1] = recent_14 - prev_14
        momentum_feat[:, t, 2] = recent_30 - prev_30
        cold_streak = np.where(features[:, t, 0] > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
    return momentum_feat


def inject_momentum_channels(features: np.ndarray) -> np.ndarray:
    enriched = features.copy().astype(np.float32)
    if enriched.shape[2] >= 37:
        enriched[:, :, 33:37] = build_momentum_features(enriched)
    return enriched


def trim_sequence(seq: list, max_items: int | None) -> list:
    if max_items is None or len(seq) <= max_items:
        return seq
    idx = np.linspace(0, len(seq) - 1, num=max_items, dtype=int)
    return [seq[i] for i in idx.tolist()]


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def precision_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    k_adj = min(k, len(scores))
    top_pred = np.argsort(scores)[::-1][:k_adj]
    return float((targets[top_pred] > 0).sum() / max(k_adj, 1))


def recall_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    positives = set(np.where(targets > 0)[0].tolist())
    if not positives:
        return 0.0
    top_pred = set(np.argsort(scores)[::-1][: min(k, len(scores))].tolist())
    return float(len(top_pred & positives) / len(positives))


def rank_overlap_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    k_adj = min(k, len(scores))
    top_pred = set(np.argsort(scores)[::-1][:k_adj].tolist())
    top_real = set(np.argsort(targets)[::-1][:k_adj].tolist())
    return float(len(top_pred & top_real) / max(k_adj, 1))


class ZeroModel:
    name = "zero_baseline"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ZeroModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class RollingMeanModel:
    def __init__(self, source_col: str, name: str):
        self.source_col = source_col
        self.name = name

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RollingMeanModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.source_col].to_numpy(dtype=float)


class HurdlePoissonModel:
    name = "hurdle_logit_poisson"

    def __init__(self) -> None:
        self.clf = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        self.reg = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", PoissonRegressor(alpha=1e-4, max_iter=400)),
            ]
        )
        self.positive_fallback = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HurdlePoissonModel":
        y_binary = (y > 0).astype(int)
        self.clf.fit(X, y_binary)
        positive_mask = y > 0
        if positive_mask.any():
            self.reg.fit(X.loc[positive_mask], y[positive_mask])
            self.positive_fallback = float(np.mean(y[positive_mask]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_event = self.clf.predict_proba(X)[:, 1]
        try:
            mu_pos = self.reg.predict(X)
        except Exception:
            mu_pos = np.full(len(X), self.positive_fallback, dtype=float)
        return np.clip(p_event * np.maximum(mu_pos, 1e-6), 0.0, None)


def build_classical_models(selected_names: set[str] | None = None) -> list[tuple[str, object, str]]:
    models = [
        ("zero_baseline", ZeroModel(), "score"),
        ("lag1_baseline", RollingMeanModel("lag_1", "lag1_baseline"), "score"),
        ("roll7_baseline", RollingMeanModel("mean_7", "roll7_baseline"), "score"),
        (
            "poisson_regressor",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", PoissonRegressor(alpha=1e-4, max_iter=400)),
                ]
            ),
            "count",
        ),
        (
            "histgb_classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=4,
                max_iter=200,
                min_samples_leaf=20,
                random_state=42,
            ),
            "binary",
        ),
        (
            "logit_classifier",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
                ]
            ),
            "binary",
        ),
        ("hurdle_logit_poisson", HurdlePoissonModel(), "count"),
    ]
    if not selected_names:
        return models
    return [item for item in models if item[0] in selected_names]


def analyze_enriched_raw(raw_path: Path, cutoff: pd.Timestamp) -> dict:
    df = pd.read_csv(raw_path, low_memory=False)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df[df["data"].notna()]
    df = df[df["data"] <= cutoff].copy()
    df["tipo"] = df["tipo"].astype(str).str.lower().str.strip()
    df = df[df["tipo"] == "cvli"].copy()
    df["qtd_mortes"] = [sanitize_qtd_mortes(v, "cvli") for v in df["qtd_mortes"]]
    df["loc"] = [resolve_occurrence_location(b, c) for b, c in zip(df["bairro"], df["cidade"])]
    df = df[df["loc"] != ""].copy()

    full_days = pd.date_range(df["data"].min(), cutoff, freq="D")
    daily_global = df.groupby("data")["qtd_mortes"].sum().reindex(full_days, fill_value=0.0)
    pos = daily_global[daily_global > 0]

    return {
        "rows_cvli": int(len(df)),
        "locations": int(df["loc"].nunique()),
        "date_min": str(df["data"].min().date()),
        "date_max": str(df["data"].max().date()),
        "daily_mean": float(daily_global.mean()),
        "daily_var": float(daily_global.var()),
        "daily_fano": float(daily_global.var() / max(daily_global.mean(), 1e-12)),
        "daily_zero_rate": float((daily_global == 0).mean()),
        "daily_acf_1": float(daily_global.autocorr(1)),
        "daily_acf_7": float(daily_global.autocorr(7)),
        "positive_days_share_eq_1": float((pos == 1).mean()) if len(pos) else 0.0,
        "positive_days_share_gt_1": float((pos > 1).mean()) if len(pos) else 0.0,
        "positive_days_p95": float(pos.quantile(0.95)) if len(pos) else 0.0,
        "positive_days_max": float(pos.max()) if len(pos) else 0.0,
    }


def load_processed_region(cfg: RegionConfig) -> dict:
    path = ROOT / "data" / "processed" / cfg.processed_file
    with path.open("rb") as f:
        return pickle.load(f)


def select_split(
    history_start: pd.Timestamp,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> str | None:
    if history_start >= train_start and target_end <= train_end:
        return "train"
    if target_start >= val_start and target_end <= val_end:
        return "val"
    return None


def build_region_datasets(
    cfg: RegionConfig,
    data: dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    horizon_days: int,
) -> dict:
    nf = data["node_features"].astype(np.float32)
    dates = pd.to_datetime(data["dates"])
    gdf = data["nodes_gdf"].copy()
    n_nodes, total_steps, channels = nf.shape
    enriched = inject_momentum_channels(nf)

    nodes_meta = {
        "tension_index": gdf["tension_index"].fillna(0).to_numpy(dtype=float) if "tension_index" in gdf.columns else np.zeros(n_nodes, dtype=float),
        "recent_cvli": gdf["recent_cvli"].fillna(0).to_numpy(dtype=float) if "recent_cvli" in gdf.columns else np.zeros(n_nodes, dtype=float),
        "total_cvli": gdf["total_cvli"].fillna(0).to_numpy(dtype=float) if "total_cvli" in gdf.columns else np.zeros(n_nodes, dtype=float),
        "node_name": gdf["name"].astype(str).tolist(),
    }

    deep_train: list[dict] = []
    deep_val: list[dict] = []
    classical_train_rows: list[dict] = []
    classical_val_rows: list[dict] = []
    daily_val_targets: dict[str, np.ndarray] = {}

    for t in range(cfg.window, total_steps - horizon_days + 1):
        history_start = dates[t - cfg.window]
        target_start = dates[t]
        target_end = dates[t + horizon_days - 1]
        split = select_split(history_start, target_start, target_end, train_start, train_end, val_start, val_end)
        if split is None:
            continue

        x_win = enriched[:, t - cfg.window : t, :].copy()
        if channels < DEFAULT_CHANNELS:
            x_win = np.pad(x_win, ((0, 0), (0, 0), (0, DEFAULT_CHANNELS - channels)), mode="constant")
        elif channels > DEFAULT_CHANNELS:
            x_win = x_win[:, :, :DEFAULT_CHANNELS]
        x_tensor = np.transpose(x_win, (2, 0, 1)).astype(np.float32)
        target = nf[:, t : t + horizon_days, 0].sum(axis=1).astype(np.float32)
        target_binary = (target > 0).astype(np.float32)

        sample = {
            "date": target_start.strftime("%Y-%m-%d"),
            "x": x_tensor,
            "y_count": target,
            "y_binary": target_binary,
        }
        if split == "train":
            deep_train.append(sample)
        else:
            deep_val.append(sample)
            daily_val_targets[sample["date"]] = target_binary

        day_of_week = target_start.dayofweek
        month = target_start.month
        sin_dow = math.sin(2.0 * math.pi * day_of_week / 7.0)
        cos_dow = math.cos(2.0 * math.pi * day_of_week / 7.0)
        sin_month = math.sin(2.0 * math.pi * month / 12.0)
        cos_month = math.cos(2.0 * math.pi * month / 12.0)
        hist_slice = nf[:, :t, 0]
        sum_7 = nf[:, max(0, t - 7) : t, 0].sum(axis=1)
        sum_14 = nf[:, max(0, t - 14) : t, 0].sum(axis=1)
        sum_30 = nf[:, max(0, t - 30) : t, 0].sum(axis=1)
        sum_60 = nf[:, max(0, t - 60) : t, 0].sum(axis=1)
        prev_7 = nf[:, max(0, t - 14) : max(0, t - 7), 0].sum(axis=1)

        frame_rows = classical_train_rows if split == "train" else classical_val_rows
        for node_idx in range(n_nodes):
            frame_rows.append(
                {
                    "sample_date": sample["date"],
                    "node_idx": node_idx,
                    "target_count": float(target[node_idx]),
                    "target_binary": float(target_binary[node_idx]),
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
                    "momentum_14": float(enriched[node_idx, t - 1, 34]) if DEFAULT_CHANNELS > 34 else 0.0,
                    "momentum_30": float(enriched[node_idx, t - 1, 35]) if DEFAULT_CHANNELS > 35 else 0.0,
                    "cold_streak_inv": float(enriched[node_idx, t - 1, 36]) if DEFAULT_CHANNELS > 36 else 0.0,
                    "tension_index": float(nodes_meta["tension_index"][node_idx]),
                    "recent_cvli_static": float(nodes_meta["recent_cvli"][node_idx]),
                    "total_cvli_static": float(nodes_meta["total_cvli"][node_idx]),
                    "sin_dow": sin_dow,
                    "cos_dow": cos_dow,
                    "sin_month": sin_month,
                    "cos_month": cos_month,
                    "is_weekend": float(day_of_week >= 5),
                }
            )

    return {
        "deep_train": deep_train,
        "deep_val": deep_val,
        "classical_train": pd.DataFrame(classical_train_rows),
        "classical_val": pd.DataFrame(classical_val_rows),
        "daily_val_targets": daily_val_targets,
        "node_count": n_nodes,
        "adj": [
            normalize_adj(data["adj_geo"]),
            normalize_adj(data["adj_conflict"]),
        ],
    }


def evaluate_daily_predictions(
    predictions_by_day: dict[str, np.ndarray],
    targets_by_day: dict[str, np.ndarray],
    model_type: str,
) -> dict:
    days = sorted(set(predictions_by_day) & set(targets_by_day))
    if not days:
        raise RuntimeError("Nenhum dia em comum entre predicoes e targets.")

    p1, p5, p10, p20 = [], [], [], []
    r10, r20 = [], []
    overlap10 = []
    reciprocal_ranks = []
    total_positive_days = 0
    for day in days:
        scores = np.asarray(predictions_by_day[day], dtype=float)
        target = np.asarray(targets_by_day[day], dtype=float)
        if target.sum() > 0:
            total_positive_days += 1
            p1.append(precision_at_k(scores, target, 1))
            p5.append(precision_at_k(scores, target, 5))
            p10.append(precision_at_k(scores, target, 10))
            p20.append(precision_at_k(scores, target, 20))
            r10.append(recall_at_k(scores, target, 10))
            r20.append(recall_at_k(scores, target, 20))
            overlap10.append(rank_overlap_at_k(scores, target, 10))
            ranked_idx = np.argsort(scores)[::-1]
            positive_positions = np.where(target[ranked_idx] > 0)[0]
            reciprocal_ranks.append(1.0 / (positive_positions[0] + 1) if len(positive_positions) else 0.0)

    return {
        "model_type": model_type,
        "days_scored": len(days),
        "positive_days": total_positive_days,
        "hit1_event": float(np.mean(p1)) if p1 else 0.0,
        "p5_event": float(np.mean(p5)) if p5 else 0.0,
        "p10_event": float(np.mean(p10)) if p10 else 0.0,
        "p20_event": float(np.mean(p20)) if p20 else 0.0,
        "recall10_event": float(np.mean(r10)) if r10 else 0.0,
        "recall20_event": float(np.mean(r20)) if r20 else 0.0,
        "overlap10_rank": float(np.mean(overlap10)) if overlap10 else 0.0,
        "mrr_event": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def run_classical_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets_by_day: dict[str, np.ndarray],
    selected_models: set[str] | None,
) -> list[dict]:
    feature_cols = [
        "lag_1",
        "sum_7",
        "sum_14",
        "sum_30",
        "sum_60",
        "mean_7",
        "mean_14",
        "mean_30",
        "mean_60",
        "hist_mean",
        "hist_sum",
        "momentum_7",
        "momentum_14",
        "momentum_30",
        "cold_streak_inv",
        "tension_index",
        "recent_cvli_static",
        "total_cvli_static",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "is_weekend",
    ]

    X_train = train_df[feature_cols]
    y_train_count = train_df["target_count"].to_numpy(dtype=float)
    y_train_binary = train_df["target_binary"].to_numpy(dtype=int)
    X_val = val_df[feature_cols]

    models = build_classical_models(selected_models)
    results = []
    started_at = time.time()
    for model_idx, (model_name, model, target_kind) in enumerate(models, start=1):
        print(
            f"  [classic] {progress_bar(model_idx - 1, len(models))} "
            f"{model_idx - 1}/{len(models)} concluídos | próximo={model_name}"
        )
        if target_kind == "binary":
            model.fit(X_train, y_train_binary)
            if hasattr(model, "predict_proba"):
                pred = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
            else:
                pred = np.asarray(model.predict(X_val), dtype=float)
        elif target_kind == "count":
            model.fit(X_train, y_train_count)
            pred = np.clip(np.asarray(model.predict(X_val), dtype=float), 0.0, None)
        else:
            model.fit(X_train, y_train_binary)
            pred = np.clip(np.asarray(model.predict(X_val), dtype=float), 0.0, None)
        pred_df = val_df[["sample_date", "node_idx"]].copy()
        pred_df["pred"] = pred

        predictions_by_day = {}
        for day, group in pred_df.groupby("sample_date"):
            predictions_by_day[day] = group.sort_values("node_idx")["pred"].to_numpy(dtype=float)

        metrics = evaluate_daily_predictions(predictions_by_day, targets_by_day, model_type="classic")
        metrics["model"] = model_name
        results.append(metrics)
        elapsed = time.time() - started_at
        avg = elapsed / model_idx
        eta = avg * (len(models) - model_idx)
        print(
            f"  [classic] {progress_bar(model_idx, len(models))} "
            f"{model_idx}/{len(models)} {model_name} | "
            f"hit1={metrics['hit1_event']:.3f} p10={metrics['p10_event']:.3f} "
            f"mrr={metrics['mrr_event']:.3f} | elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
        )
    return results


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for i in range(pred.shape[0]):
        pos_idx = torch.where(target[i] > 0.5)[0]
        neg_idx = torch.where(target[i] <= 0.5)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_idx = pos_idx[: min(len(pos_idx), 16)]
        perm = torch.randperm(len(neg_idx), device=pred.device)[: len(pos_idx)]
        sampled_neg = neg_idx[perm]
        margin = 1.0 - (pred[i, pos_idx] - pred[i, sampled_neg])
        losses.append(torch.relu(margin).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def predict_deep_scores(
    model: torch.nn.Module,
    samples: list[dict],
    adj: list[torch.Tensor],
    batch_size: int,
) -> dict[str, np.ndarray]:
    predictions_by_day: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            x = torch.tensor(np.stack([s["x"] for s in batch]), dtype=torch.float32, device=DEVICE)
            pred = model(x, adj).squeeze(-1)
            pred = torch.sigmoid(pred).cpu().numpy()
            for sample, row in zip(batch, pred):
                predictions_by_day[sample["date"]] = row.astype(float)
    return predictions_by_day


def train_deep_model(
    model_name: str,
    cfg: RegionConfig,
    train_samples: list[dict],
    val_samples: list[dict],
    adj: list[torch.Tensor],
    epochs: int,
) -> dict[str, np.ndarray]:
    print(f"  [deep] iniciando {model_name} | train_days={len(train_samples)} val_days={len(val_samples)} epochs={epochs}")
    model_cls = DEEP_MODEL_REGISTRY[model_name]
    num_nodes = train_samples[0]["x"].shape[1]
    model = model_cls(num_nodes=num_nodes, in_channels=DEFAULT_CHANNELS, time_steps=cfg.window, dropout=0.35).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    pos_total = sum(float(sample["y_binary"].sum()) for sample in train_samples)
    neg_total = sum(float(len(sample["y_binary"]) - sample["y_binary"].sum()) for sample in train_samples)
    pos_weight = torch.tensor(max(neg_total / max(pos_total, 1.0), 1.0), dtype=torch.float32, device=DEVICE)

    train_started_at = time.time()
    for epoch_idx in range(epochs):
        model.train()
        idx = list(range(len(train_samples)))
        random.shuffle(idx)
        epoch_started_at = time.time()
        epoch_loss_total = 0.0
        batch_count = 0
        for start in range(0, len(idx), cfg.batch_size):
            batch_idx = idx[start : start + cfg.batch_size]
            batch = [train_samples[i] for i in batch_idx]
            x = torch.tensor(np.stack([s["x"] for s in batch]), dtype=torch.float32, device=DEVICE)
            y = torch.tensor(np.stack([s["y_binary"] for s in batch]), dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            pred = model(x, adj).squeeze(-1)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred,
                y,
                pos_weight=pos_weight,
                reduction="mean",
            )
            loss = bce_loss + 0.20 * pairwise_rank_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_total += float(loss.item())
            batch_count += 1

        val_predictions = predict_deep_scores(model, val_samples, adj, cfg.batch_size)
        targets = {sample["date"]: sample["y_binary"] for sample in val_samples}
        epoch_metrics = evaluate_daily_predictions(val_predictions, targets, model_type="deep")
        score = epoch_metrics["p10_event"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        elapsed = time.time() - train_started_at
        avg_epoch = elapsed / (epoch_idx + 1)
        eta = avg_epoch * (epochs - (epoch_idx + 1))
        print(
            f"  [deep] {model_name} {progress_bar(epoch_idx + 1, epochs)} "
            f"epoch {epoch_idx + 1}/{epochs} | loss={epoch_loss_total / max(batch_count, 1):.4f} "
            f"hit1={epoch_metrics['hit1_event']:.3f} p10={epoch_metrics['p10_event']:.3f} "
            f"mrr={epoch_metrics['mrr_event']:.3f} | epoch={format_seconds(time.time() - epoch_started_at)} "
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
        )

    model.load_state_dict(best_state)
    return predict_deep_scores(model, val_samples, adj, cfg.batch_size)


def run_deep_benchmark(
    cfg: RegionConfig,
    train_samples: list[dict],
    val_samples: list[dict],
    targets_by_day: dict[str, np.ndarray],
    adj: list[torch.Tensor],
    epochs: int,
    selected_models: set[str] | None,
) -> list[dict]:
    results = []
    candidate_names = cfg.deep_candidates if not selected_models else tuple(name for name in cfg.deep_candidates if name in selected_models)
    started_at = time.time()
    for model_idx, model_name in enumerate(candidate_names, start=1):
        print(
            f"  [deep] {progress_bar(model_idx - 1, len(candidate_names))} "
            f"{model_idx - 1}/{len(candidate_names)} concluídos | próximo={model_name}"
        )
        predictions = train_deep_model(model_name, cfg, train_samples, val_samples, adj, epochs=epochs)
        metrics = evaluate_daily_predictions(predictions, targets_by_day, model_type="deep")
        metrics["model"] = model_name
        results.append(metrics)
        elapsed = time.time() - started_at
        avg = elapsed / model_idx
        eta = avg * (len(candidate_names) - model_idx)
        print(
            f"  [deep] {progress_bar(model_idx, len(candidate_names))} "
            f"{model_idx}/{len(candidate_names)} {model_name} | "
            f"hit1={metrics['hit1_event']:.3f} p10={metrics['p10_event']:.3f} "
            f"mrr={metrics['mrr_event']:.3f} | elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
        )
    return results


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_sem dados_"
    cols = list(df.columns)
    rows = [[str(item) for item in row] for row in df.to_numpy()]
    widths = [len(str(col)) for col in cols]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"
    sep = "|-" + "-|-".join("-" * widths[idx] for idx in range(len(cols))) + "-|"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    raw_summary: dict,
    combined_df: pd.DataFrame,
) -> None:
    best_by_region = combined_df.sort_values(["region", "hit1_event", "p10_event", "mrr_event"], ascending=[True, False, False, False]).groupby("region").head(3)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# CVLI Stochastic Benchmark Suite\n\n")
        f.write(f"- Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Device: {DEVICE}\n")
        f.write(f"- Cutoff bruto: {args.cutoff}\n")
        f.write(f"- Treino: {args.train_start} -> {args.train_end}\n")
        f.write(f"- Validacao: {args.val_start} -> {args.val_end}\n")
        f.write(f"- Horizonte: {args.horizon_days} dias\n")
        f.write(f"- Deep epochs: {args.deep_epochs}\n")
        f.write(f"- Regioes: {', '.join(args.regions)}\n\n")
        f.write("## Sinal Estocastico no ENRIQUECIDO\n\n")
        f.write(dataframe_to_markdown(pd.DataFrame([raw_summary])))
        f.write("\n\n## Top-3 por Regiao\n\n")
        f.write(dataframe_to_markdown(best_by_region))
        f.write("\n\n## Benchmark Completo\n\n")
        f.write(dataframe_to_markdown(combined_df))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", default=["fortaleza", "rmf", "interior"])
    parser.add_argument("--cutoff", default="2025-12-31")
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-start", default="2025-01-01")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--deep-epochs", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--classical-models", nargs="*", default=None)
    parser.add_argument("--deep-models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_started_at = time.time()

    cutoff = pd.Timestamp(args.cutoff)
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)

    raw_summary = analyze_enriched_raw(ROOT / "data" / "raw" / "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv", cutoff)
    classical_selection = set(args.classical_models) if args.classical_models else None
    deep_selection = set(args.deep_models) if args.deep_models else None

    all_results = []
    total_regions = len(args.regions)
    for region_idx, region in enumerate(args.regions, start=1):
        cfg = REGION_CONFIGS[region]
        print("")
        print(f"[region] {progress_bar(region_idx - 1, total_regions)} {region_idx - 1}/{total_regions} concluídas | iniciando={region}")
        data = load_processed_region(cfg)
        datasets = build_region_datasets(
            cfg=cfg,
            data=data,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            horizon_days=args.horizon_days,
        )
        deep_train = trim_sequence(datasets["deep_train"], args.max_train_samples)
        deep_val = trim_sequence(datasets["deep_val"], args.max_val_samples)
        val_days_allowed = {sample["date"] for sample in deep_val}
        targets_by_day = {day: datasets["daily_val_targets"][day] for day in val_days_allowed}

        classical_train = datasets["classical_train"]
        classical_val = datasets["classical_val"]
        if args.max_train_samples is not None:
            keep_train_days = {sample["date"] for sample in deep_train}
            classical_train = classical_train[classical_train["sample_date"].isin(keep_train_days)].copy()
        classical_val = classical_val[classical_val["sample_date"].isin(val_days_allowed)].copy()
        print(
            f"[region] {region} | train_days={len(deep_train)} val_days={len(deep_val)} "
            f"train_rows={len(classical_train)} val_rows={len(classical_val)}"
        )

        region_results = []
        region_results.extend(run_classical_benchmark(classical_train, classical_val, targets_by_day, classical_selection))
        region_results.extend(
            run_deep_benchmark(
                cfg=cfg,
                train_samples=deep_train,
                val_samples=deep_val,
                targets_by_day=targets_by_day,
                adj=datasets["adj"],
                epochs=args.deep_epochs,
                selected_models=deep_selection,
            )
        )
        for row in region_results:
            row["region"] = region
            row["train_samples"] = len(deep_train)
            row["val_samples"] = len(deep_val)
        all_results.extend(region_results)
        region_df = pd.DataFrame(region_results).sort_values(["hit1_event", "p10_event", "mrr_event"], ascending=[False, False, False])
        top = region_df.iloc[0]
        elapsed = time.time() - run_started_at
        avg_region = elapsed / region_idx
        eta = avg_region * (total_regions - region_idx)
        print(
            f"[region] {progress_bar(region_idx, total_regions)} {region_idx}/{total_regions} {region} concluída | "
            f"best={top['model']} hit1={top['hit1_event']:.3f} p10={top['p10_event']:.3f} "
            f"mrr={top['mrr_event']:.3f} | elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
        )

    combined_df = pd.DataFrame(all_results).sort_values(
        ["region", "hit1_event", "p10_event", "mrr_event"],
        ascending=[True, False, False, False],
    )

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = DEFAULT_OUT_DIR / f"cvli_stochastic_suite_{stamp}.json"
    csv_path = DEFAULT_OUT_DIR / f"cvli_stochastic_suite_{stamp}.csv"
    md_path = DEFAULT_OUT_DIR / f"cvli_stochastic_suite_{stamp}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "device": str(DEVICE),
        "args": vars(args),
        "raw_summary": raw_summary,
        "results": combined_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    combined_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_report(md_path, args, raw_summary, combined_df)

    print(f"Raw summary: {json.dumps(raw_summary, ensure_ascii=False)}")
    print("\nTop results:")
    print(combined_df.groupby("region").head(3).to_string(index=False))
    print(f"\nArquivos gerados:\n  {json_path}\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
