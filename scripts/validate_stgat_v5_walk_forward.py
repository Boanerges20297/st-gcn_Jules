from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.architectures import get_model_class  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_VALUES = (5, 10, 20)


@dataclass(frozen=True)
class RegionConfig:
    key: str
    model_file: str
    processed_file: str
    window: int
    channels: int = 41


REGIONS = (
    RegionConfig("fortaleza", "fortaleza_model_active.pth", "processed_fortaleza.pkl", 120),
    RegionConfig("rmf", "rmf_model.pth", "processed_rmf.pkl", 14),
    RegionConfig("interior", "interior_model.pth", "processed_interior.pkl", 14),
)


def norm_adj(geo: np.ndarray, conf: np.ndarray) -> list[torch.Tensor]:
    def normalize(a: np.ndarray) -> torch.Tensor:
        s = np.array(a.sum(1))
        d = np.zeros_like(s, dtype=float).flatten()
        positive = s.flatten() > 0
        d[positive] = np.power(s.flatten()[positive], -0.5)
        m = np.diag(d)
        return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(DEVICE)

    return [normalize(geo), normalize(conf)]


def add_momentum(raw_window: np.ndarray) -> np.ndarray:
    n_nodes, n_steps, _ = raw_window.shape
    momentum = np.zeros((n_nodes, n_steps, 4), dtype=np.float32)
    cold_streak = np.zeros(n_nodes, dtype=np.float32)
    for t in range(60, n_steps):
        momentum[:, t, 0] = raw_window[:, t - 7 : t, 0].sum(axis=1) - raw_window[:, t - 14 : t - 7, 0].sum(axis=1)
        momentum[:, t, 1] = raw_window[:, t - 14 : t, 0].sum(axis=1) - raw_window[:, t - 28 : t - 14, 0].sum(axis=1)
        momentum[:, t, 2] = raw_window[:, t - 30 : t, 0].sum(axis=1) - raw_window[:, t - 60 : t - 30, 0].sum(axis=1)
        cold_streak = np.where(raw_window[:, t, 0] > 0, 0, cold_streak + 1)
        momentum[:, t, 3] = -np.clip(cold_streak, 0, 30)
    if raw_window.shape[2] >= 37:
        raw_window[:, :, 33:37] = momentum
    return raw_window


def build_input(features: np.ndarray, ti: int, window: int, channels: int) -> np.ndarray:
    total_window = window + 60
    raw_extended = features[:, ti - total_window : ti, :].copy()
    raw_extended = add_momentum(raw_extended)
    x_final = raw_extended[:, -window:, :channels].copy()
    if x_final.shape[2] < channels:
        x_final = np.pad(x_final, ((0, 0), (0, 0), (0, channels - x_final.shape[2])), mode="constant")
    for c in (0, 1, 2, 24, 27, 28, 31, 33, 34, 35, 36):
        if c < x_final.shape[2]:
            x_final[:, :, c] = (x_final[:, :, c] - x_final[:, :, c].mean()) / (x_final[:, :, c].std() + 1e-6)
    return x_final


def load_region(cfg: RegionConfig):
    data = pickle.loads((ROOT / "data" / "processed" / cfg.processed_file).read_bytes())
    ckpt = torch.load(ROOT / "models" / "active" / "legacy_torch" / cfg.model_file, map_location=DEVICE, weights_only=False)
    meta = ckpt if isinstance(ckpt, dict) else {}
    window = int(meta.get("window") or cfg.window)
    channels = int(meta.get("in_channels") or cfg.channels)
    model_class = get_model_class(meta.get("model_class") or meta.get("architecture") or "DeepSTGAT_v5")
    model = model_class(num_nodes=len(data["nodes_gdf"]), in_channels=channels, time_steps=window).to(DEVICE)
    model.load_state_dict(meta.get("model_state_dict", ckpt), strict=False)
    model.eval()
    return data, model, norm_adj(data["adj_geo"], data["adj_conflict"]), RegionConfig(cfg.key, cfg.model_file, cfg.processed_file, window, channels)


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)


def score_stgat_v5(data: dict, model: torch.nn.Module, adj: list[torch.Tensor], cfg: RegionConfig, ti: int) -> np.ndarray:
    features = data["node_features"]
    x_final = build_input(features, ti, cfg.window, cfg.channels)
    x = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw = model(x, adj).squeeze().cpu()
    norm_neural = normalize(F.softplus(raw).numpy())
    gdf = data["nodes_gdf"]
    tension = gdf["tension_index"].fillna(0).to_numpy(dtype=float)
    norm_tension = normalize(tension)
    current_cvli_30d = features[:, max(0, ti - 30) : ti, 0].sum(axis=1)
    live_support = np.clip(current_cvli_30d / 2.0, 0, 1)
    return np.clip(((0.50 * norm_neural) + (0.25 * live_support) + (0.25 * norm_tension * live_support)) * 100.0, 0, 100)


def score_ewma(data: dict, ti: int) -> np.ndarray:
    features = data["node_features"]
    score = np.zeros(features.shape[0], dtype=float)
    for window, weight in ((7, 0.40), (14, 0.30), (30, 0.20), (60, 0.10)):
        score += weight * features[:, max(0, ti - window) : ti, 0].sum(axis=1)
    return normalize(score) * 100.0


def target(features: np.ndarray, ti: int, horizon: int) -> np.ndarray | None:
    end = ti + horizon
    if end >= features.shape[1]:
        return None
    return features[:, ti + 1 : end + 1, 0].sum(axis=1)


def metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    row = {}
    positives = set(np.where(y > 0)[0].tolist())
    for k in K_VALUES:
        top = set(np.argsort(scores)[::-1][: min(k, len(scores))].tolist())
        hits = len(top & positives)
        row[f"p{k}"] = hits / min(k, len(scores))
        row[f"r{k}"] = hits / len(positives) if positives else 0.0
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith(("p", "r")) and c[1:].isdigit()]
    return (
        df.groupby(["model", "scope"], as_index=False)
        .agg(windows=("date", "count"), active_locations_avg=("active_locations", "mean"), total_cvli=("total_cvli", "sum"), **{c: (c, "mean") for c in metric_cols})
        .round(4)
    )


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-07-03")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--stride", type=int, default=7)
    args = parser.parse_args()

    rows = []
    global_by_date: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for base_cfg in REGIONS:
        data, model, adj, cfg = load_region(base_cfg)
        dates = pd.to_datetime(data["dates"])
        min_ti = cfg.window + 60
        indices = [i for i, d in enumerate(dates) if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end) and i >= min_ti]
        indices = indices[:: max(args.stride, 1)]
        print(f"{cfg.key}: {len(indices)} janelas, window={cfg.window}, channels={cfg.channels}")
        for ti in indices:
            y = target(data["node_features"], ti, args.horizon)
            if y is None or y.sum() <= 0:
                continue
            date = dates[ti].strftime("%Y-%m-%d")
            for model_name, scores in (("ST-GAT_v5", score_stgat_v5(data, model, adj, cfg, ti)), ("EWMA_baseline", score_ewma(data, ti))):
                row = {"date": date, "scope": cfg.key, "model": model_name, "active_locations": int((y > 0).sum()), "total_cvli": int(y.sum())}
                row.update(metrics(scores, y))
                rows.append(row)
                bucket = global_by_date.setdefault((date, model_name), {"scores": [], "targets": []})
                bucket["scores"].append(scores)
                bucket["targets"].append(y)

    for (date, model_name), parts in global_by_date.items():
        scores = np.concatenate(parts["scores"])
        y = np.concatenate(parts["targets"])
        row = {"date": date, "scope": "global", "model": model_name, "active_locations": int((y > 0).sum()), "total_cvli": int(y.sum())}
        row.update(metrics(scores, y))
        rows.append(row)

    out_dir = ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_df = pd.DataFrame(rows)
    summary_df = summarize(rows_df)
    rows_path = out_dir / f"stgat_v5_walk_forward_rows_{stamp}.csv"
    summary_path = out_dir / f"stgat_v5_walk_forward_summary_{stamp}.csv"
    report_path = out_dir / f"stgat_v5_walk_forward_report_{stamp}.md"
    rows_df.to_csv(rows_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        "# Validação walk-forward ST-GAT v5\n\n"
        f"- Gerado em: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"- Corte: {args.start} a {args.end}\n"
        f"- Horizonte futuro: {args.horizon} dias\n"
        f"- Passo: {args.stride} dia(s)\n"
        "- Observação: avalia o checkpoint ativo sem retreino; se o checkpoint foi treinado com dados posteriores ao corte, isto mede inferência temporal, não prova prospectiva estrita.\n\n"
        + markdown_table(summary_df)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / f"stgat_v5_walk_forward_summary_{stamp}.json").write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"\nArquivos:\n{rows_path}\n{summary_path}\n{report_path}")


if __name__ == "__main__":
    main()
