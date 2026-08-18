from __future__ import annotations

import math
import pickle
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
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


def normalize_name(text) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII").upper().strip()
    text = re.sub(r"\s*-\s*AIS.*$", "", text)
    return text.strip()


def _build_momentum_features(node_features: np.ndarray) -> np.ndarray:
    n_nodes, n_steps, _ = node_features.shape
    momentum_feat = np.zeros((n_nodes, n_steps, 4), dtype=np.float32)
    cold_streak = np.zeros(n_nodes, dtype=np.float32)
    for t in range(60, n_steps):
        recent_7 = node_features[:, t - 7 : t, 0].sum(axis=1)
        prev_7 = node_features[:, t - 14 : t - 7, 0].sum(axis=1)
        recent_14 = node_features[:, t - 14 : t, 0].sum(axis=1)
        prev_14 = node_features[:, t - 28 : t - 14, 0].sum(axis=1)
        recent_30 = node_features[:, t - 30 : t, 0].sum(axis=1)
        prev_30 = node_features[:, t - 60 : t - 30, 0].sum(axis=1)
        momentum_feat[:, t, 0] = recent_7 - prev_7
        momentum_feat[:, t, 1] = recent_14 - prev_14
        momentum_feat[:, t, 2] = recent_30 - prev_30
        cold_streak = np.where(node_features[:, t, 0] > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
    return momentum_feat


def inject_momentum_channels(node_features: np.ndarray) -> np.ndarray:
    enriched = node_features.copy().astype(np.float32)
    if enriched.shape[2] >= 37:
        enriched[:, :, 33:37] = _build_momentum_features(enriched)
    return enriched


def build_training_frame(
    data: dict,
    train_start: str = "2022-01-01",
    train_end: str = "2024-12-31",
    horizon_days: int = 30,
    window: int = 90,
) -> pd.DataFrame:
    nf = np.asarray(data["node_features"], dtype=np.float32)
    enriched = inject_momentum_channels(nf)
    nodes = data["nodes_gdf"].reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(data["dates"]))
    n_nodes, total_steps, _ = nf.shape
    rows = []

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    tension_index = nodes["tension_index"].fillna(0.0).to_numpy(dtype=float)
    recent_cvli = nodes.get("recent_cvli", pd.Series(np.zeros(n_nodes))).fillna(0.0).to_numpy(dtype=float)
    total_cvli = nodes.get("total_cvli", pd.Series(np.zeros(n_nodes))).fillna(0.0).to_numpy(dtype=float)

    for t in range(window, total_steps - horizon_days + 1):
        target_start = dates[t]
        target_end = dates[t + horizon_days - 1]
        if not (train_start_ts <= target_start <= train_end_ts and target_end <= train_end_ts):
            continue

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
        target_count = nf[:, t : t + horizon_days, 0].sum(axis=1).astype(float)

        for node_idx in range(n_nodes):
            rows.append(
                {
                    "sample_date": target_start.strftime("%Y-%m-%d"),
                    "node_idx": node_idx,
                    "target_count": float(target_count[node_idx]),
                    "target_binary": float(target_count[node_idx] > 0),
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
                }
            )

    return pd.DataFrame(rows)


def build_current_frame(data: dict, window: int = 90) -> pd.DataFrame:
    nf = np.asarray(data["node_features"], dtype=np.float32)
    enriched = inject_momentum_channels(nf)
    nodes = data["nodes_gdf"].reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(data["dates"]))
    n_nodes = nf.shape[0]
    t = len(dates)
    target_start = dates[-1] + pd.Timedelta(days=1)
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

    rows = []
    for node_idx in range(n_nodes):
        rows.append(
            {
                "sample_date": target_start.strftime("%Y-%m-%d"),
                "node_idx": node_idx,
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
                "tension_index": float(nodes.iloc[node_idx].get("tension_index", 0.0) or 0.0),
                "recent_cvli_static": float(nodes.iloc[node_idx].get("recent_cvli", 0.0) or 0.0),
                "total_cvli_static": float(nodes.iloc[node_idx].get("total_cvli", 0.0) or 0.0),
                "sin_dow": sin_dow,
                "cos_dow": cos_dow,
                "sin_month": sin_month,
                "cos_month": cos_month,
                "is_weekend": float(day_of_week >= 5),
            }
        )
    return pd.DataFrame(rows)


def build_poisson_pipeline(alpha: float = 1e-4, max_iter: int = 400) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=alpha, max_iter=max_iter)),
        ]
    )


def train_poisson_payload(
    data: dict,
    region: str = "fortaleza",
    train_start: str = "2022-01-01",
    train_end: str = "2024-12-31",
    horizon_days: int = 30,
    window: int = 90,
    benchmark_metrics: dict | None = None,
) -> dict:
    train_df = build_training_frame(
        data=data,
        train_start=train_start,
        train_end=train_end,
        horizon_days=horizon_days,
        window=window,
    )
    pipeline = build_poisson_pipeline()
    pipeline.fit(train_df[FEATURE_COLS], train_df["target_count"].to_numpy(dtype=float))
    return {
        "backend_type": "regional_poisson_regressor",
        "region": region,
        "generated_at": datetime.now().isoformat(),
        "window": int(window),
        "horizon_days": int(horizon_days),
        "feature_cols": list(FEATURE_COLS),
        "train_range": {"start": train_start, "end": train_end},
        "pipeline": pipeline,
        "benchmark_metrics": benchmark_metrics or {},
        "train_rows": int(len(train_df)),
    }


def save_payload(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def load_payload(path: str | Path) -> dict:
    path = Path(path)

    class NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with path.open("rb") as fh:
        return NumpyCompatUnpickler(fh).load()


@dataclass
class FortalezaPoissonRuntime:
    payload: dict
    data: dict

    @property
    def pipeline(self) -> Pipeline:
        return self.payload["pipeline"]

    @property
    def window(self) -> int:
        return int(self.payload.get("window", 90))

    @property
    def feature_cols(self) -> list[str]:
        return list(self.payload.get("feature_cols", FEATURE_COLS))

    @property
    def region(self) -> str:
        return str(self.payload.get("region", "fortaleza")).lower()

    @property
    def territorial_level(self) -> str:
        return "bairro" if self.region == "fortaleza" else "cidade"

    def predict_scores(self, exogenous_shocks: dict | None = None) -> tuple[dict[str, float], dict[str, dict]]:
        nodes = self.data["nodes_gdf"].reset_index(drop=True)
        frame = build_current_frame(self.data, window=self.window)
        raw_pred = np.clip(self.pipeline.predict(frame[self.feature_cols]), 0.0, None)
        raw_pred = np.asarray(raw_pred, dtype=float)

        if raw_pred.max() - raw_pred.min() > 1e-9:
            norm_model = (raw_pred - raw_pred.min()) / (raw_pred.max() - raw_pred.min())
        else:
            norm_model = np.zeros_like(raw_pred)

        tension_vec = nodes["tension_index"].fillna(0.0).to_numpy(dtype=float)
        if tension_vec.max() - tension_vec.min() > 1e-9:
            norm_tension = (tension_vec - tension_vec.min()) / (tension_vec.max() - tension_vec.min())
        else:
            norm_tension = np.zeros_like(tension_vec)

        current_cvli_recent = frame["sum_14"].to_numpy(dtype=float)
        current_cvli_30d = frame["sum_30"].to_numpy(dtype=float)
        historical_cvli = nodes.get("total_cvli", pd.Series(np.zeros(len(nodes)))).fillna(0.0).to_numpy(dtype=float)

        historical_support = np.clip((historical_cvli - 20.0) / 40.0, 0, 1)
        live_support = np.maximum(
            np.clip(current_cvli_recent / 1.0, 0, 1),
            np.clip(current_cvli_30d / 2.0, 0, 1),
        )
        recent_activity_gate = np.maximum(
            np.clip(current_cvli_recent / 1.0, 0, 1),
            np.clip(current_cvli_30d / 3.0, 0, 1),
        )
        historical_carry = historical_support * (0.20 + (0.80 * recent_activity_gate))
        territorial_support = np.maximum(historical_carry, live_support)
        norm_tension = norm_tension * territorial_support

        sim_impact = np.zeros(len(nodes), dtype=float)
        sim_relief = np.zeros(len(nodes), dtype=float)
        if exogenous_shocks:
            for loc_name, info in exogenous_shocks.items():
                norm_target = normalize_name(loc_name)
                if isinstance(info, dict):
                    # conflict_intensity é o campo gravado pelo app; intensity é o campo legado
                    intensity = float(info.get("conflict_intensity") or info.get("intensity") or 0.0)
                    suppression = float(info.get("suppression_intensity") or 0.0)
                    impact_value = (2.5 if self.region == "fortaleza" else 3.0) * intensity
                    relief_value = (1.3 if self.region == "fortaleza" else 1.6) * suppression
                    for i, row in nodes.iterrows():
                        if normalize_name(row["name"]) == norm_target:
                            if intensity > 0:
                                sim_impact[i] = impact_value
                            if suppression > 0:
                                sim_relief[i] = relief_value

        adj_geo = np.asarray(self.data["adj_geo"], dtype=float)
        recent_crime_signal = np.clip(current_cvli_recent, 0, 2) / 2.0

        # Normaliza pelo impacto máximo possível (HIGH = 0.9 * multiplicador regional)
        # para preservar a graduação de intensidade em vez de binarizar.
        impact_max = 2.5 * 0.9 if self.region == "fortaleza" else 3.0 * 0.9
        relief_max = 1.3 * 1.0 if self.region == "fortaleza" else 1.6 * 1.0  # supressão acumula, cap em 1.0 post-clip
        own_event_signal = np.clip(sim_impact / impact_max, 0.0, 1.0)
        neighbor_signal = np.clip(adj_geo.dot(own_event_signal), 0.0, 1.0)
        own_suppression_signal = np.clip(sim_relief / relief_max, 0.0, 1.0)
        suppression_neighbor_signal = np.clip(adj_geo.dot(own_suppression_signal), 0.0, 1.0)
        suppression_signal = np.clip(np.maximum(suppression_neighbor_signal, own_suppression_signal), 0, 1)
        raw_inclusion_signal = np.maximum.reduce([recent_crime_signal, neighbor_signal, own_event_signal])
        inclusion_signal = np.clip(raw_inclusion_signal - (0.85 * suppression_signal), 0, 1)
        calm_signal = np.clip(-frame["cold_streak_inv"].to_numpy(dtype=float), 0, 30) / 30.0
        decay_factor = 1.0 - (calm_signal * 0.35)

        final_logic = (
            (0.65 * norm_model * decay_factor)
            + (0.10 * norm_tension)
            + (0.25 * inclusion_signal)
        )
        suppression_discount = (0.22 * own_suppression_signal) + (0.10 * suppression_neighbor_signal)
        out_norm = np.clip(final_logic - suppression_discount, 0, 1) * 100.0

        scores = {}
        details = {}
        for idx, row in nodes.iterrows():
            name_key = normalize_name(str(row["name"]))
            scores[name_key] = float(out_norm[idx])
            details[name_key] = {
                "name": str(row["name"]),
                "name_normalized": name_key,
                "region": self.region,
                "territorial_level": self.territorial_level,
                "score_raw": float(out_norm[idx]),
                "neural_score": float(norm_model[idx] * 100.0),
                "model_signal_score": float(norm_model[idx] * 100.0),
                "tension_score": float(norm_tension[idx] * 100.0),
                "inclusion_score": float(inclusion_signal[idx] * 100.0),
                "shock_conflict_intensity": float(own_event_signal[idx] * 100.0),
                "shock_suppression_intensity": float(own_suppression_signal[idx] * 100.0),
                "shock_neighbor_signal": float(neighbor_signal[idx] * 100.0),
                "calm_penalty": float(calm_signal[idx] * 100.0),
                "recent_cvli_14d": int(current_cvli_recent[idx]),
                "recent_cvli_30d": int(current_cvli_30d[idx]),
                "historical_cvli": float(historical_cvli[idx]),
                "territorial_support_pct": float(territorial_support[idx] * 100.0),
                "historical_support_pct": float(historical_support[idx] * 100.0),
                "live_support_pct": float(live_support[idx] * 100.0),
                "dynamic_window": self.window,
                "base_window": self.window,
                "historical_fallback": False,
                "trend": "stable",
                "expected_cvli_30d": float(raw_pred[idx]),
                "primary_signal_label": "Sinal Poisson do ranking operacional",
                "model_family": "Poisson Ranker",
            }
        return scores, details
