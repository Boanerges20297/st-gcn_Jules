"""
Benchmark dos modelos ativos em producao.

Avalia os checkpoints em models/active contra o CVLI real por horizonte de 14 dias,
separando periodo de avaliacao e validacao por datas.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.architectures import ShallowGAT, get_model_class  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZON_DAYS = 14
K_VALUES = (5, 10, 20)


@dataclass(frozen=True)
class RegionConfig:
    key: str
    model_file: str
    processed_file: str
    window: int = 14
    channels: int = 41


REGIONS = (
    RegionConfig("fortaleza", "fortaleza_model_active.pth", "processed_fortaleza.pkl"),
    RegionConfig("rmf", "rmf_model.pth", "processed_rmf.pkl"),
    RegionConfig("interior", "interior_model.pth", "processed_interior.pkl"),
)


def norm_adj(geo: np.ndarray, conf: np.ndarray) -> list[torch.Tensor]:
    def n(a: np.ndarray) -> torch.Tensor:
        s = np.array(a.sum(1))
        d = np.zeros_like(s, dtype=float).flatten()
        positive = s.flatten() > 0
        d[positive] = np.power(s.flatten()[positive], -0.5)
        m = np.diag(d)
        return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(DEVICE)

    return [n(geo), n(conf)]


def add_momentum(raw_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replica os canais de momentum 33:37 usados pelo orquestrador."""
    num_nodes, total_window, _ = raw_window.shape
    momentum = np.zeros((num_nodes, total_window, 4), dtype=np.float32)
    cold_streak = np.zeros(num_nodes, dtype=np.float32)

    for t in range(60, total_window):
        r7 = raw_window[:, t - 7 : t, 0].sum(axis=1)
        p7 = raw_window[:, t - 14 : t - 7, 0].sum(axis=1)
        momentum[:, t, 0] = r7 - p7
        momentum[:, t, 1] = raw_window[:, t - 14 : t, 0].sum(axis=1) - raw_window[:, t - 28 : t - 14, 0].sum(axis=1)
        momentum[:, t, 2] = raw_window[:, t - 30 : t, 0].sum(axis=1) - raw_window[:, t - 60 : t - 30, 0].sum(axis=1)
        cold_streak = np.where(raw_window[:, t, 0] > 0, 0, cold_streak + 1)
        momentum[:, t, 3] = -np.clip(cold_streak, 0, 30)

    if raw_window.shape[2] >= 37:
        raw_window[:, :, 33:37] = momentum[:, :, :4]
    return raw_window, momentum


def build_input(features: np.ndarray, ti: int, window: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    extra_history = 60
    total_window = window + extra_history
    if ti < total_window:
        raise ValueError(f"Indice temporal {ti} insuficiente para janela {total_window}")

    raw_extended = features[:, ti - total_window : ti, :].copy()
    raw_extended, momentum = add_momentum(raw_extended)
    x_final = raw_extended[:, -window:, :channels].copy()

    if x_final.shape[2] < channels:
        pad_width = channels - x_final.shape[2]
        x_final = np.pad(x_final, ((0, 0), (0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    for c in (0, 1, 2, 24, 27, 28, 31, 33, 34, 35, 36):
        if c < x_final.shape[2]:
            mean = x_final[:, :, c].mean()
            std = x_final[:, :, c].std() + 1e-6
            x_final[:, :, c] = (x_final[:, :, c] - mean) / std

    return x_final, momentum


def load_region(cfg: RegionConfig) -> tuple[dict, torch.nn.Module, list[torch.Tensor], RegionConfig]:
    data_path = ROOT / "data" / "processed" / cfg.processed_file
    model_path = ROOT / "models" / "active" / cfg.model_file

    with data_path.open("rb") as f:
        data = pickle.load(f)

    num_nodes = len(data["nodes_gdf"])
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    ckpt_meta = ckpt if isinstance(ckpt, dict) else {}
    runtime_cfg = RegionConfig(
        cfg.key,
        cfg.model_file,
        cfg.processed_file,
        window=int(ckpt_meta.get("window", cfg.window)),
        channels=int(ckpt_meta.get("in_channels", cfg.channels)),
    )
    model_class = get_model_class(ckpt_meta.get("model_class")) if ckpt_meta.get("model_class") else ShallowGAT
    model = model_class(num_nodes=num_nodes, in_channels=runtime_cfg.channels, time_steps=runtime_cfg.window).to(DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return data, model, norm_adj(data["adj_geo"], data["adj_conflict"]), runtime_cfg


def horizon_target(features: np.ndarray, ti: int, horizon: int, allow_partial: bool) -> np.ndarray | None:
    end = ti + horizon
    if end >= features.shape[1]:
        if not allow_partial or ti + 1 >= features.shape[1]:
            return None
        end = features.shape[1] - 1
    return features[:, ti + 1 : end + 1, 0].sum(axis=1)


def precision_against_events(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    k_adj = min(k, len(scores))
    top_pred = np.argsort(scores)[::-1][:k_adj]
    return float((targets[top_pred] > 0).sum() / k_adj)


def topk_overlap(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    k_adj = min(k, len(scores))
    top_pred = set(np.argsort(scores)[::-1][:k_adj].tolist())
    top_real = set(np.argsort(targets)[::-1][:k_adj].tolist())
    return float(len(top_pred & top_real) / k_adj)


def recall_against_events(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    positives = set(np.where(targets > 0)[0].tolist())
    if not positives:
        return 0.0
    k_adj = min(k, len(scores))
    top_pred = set(np.argsort(scores)[::-1][:k_adj].tolist())
    return float(len(top_pred & positives) / len(positives))


def score_day(data: dict, model: torch.nn.Module, adj: list[torch.Tensor], cfg: RegionConfig, ti: int) -> np.ndarray:
    features = data["node_features"]
    x_final, momentum = build_input(features, ti, cfg.window, cfg.channels)
    x = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x, adj).squeeze().cpu().numpy()

    norm_neural = (out - out.min()) / (out.max() - out.min() + 1e-6)
    gdf = data["nodes_gdf"]
    tension = gdf["tension_index"].fillna(0).values.astype(float)
    norm_tension = (tension - tension.min()) / (tension.max() - tension.min() + 1e-6)

    inclusion_horizon = 28 if cfg.key == "interior" else 14
    current_cvli_recent = features[:, max(0, ti - inclusion_horizon) : ti, 0].sum(axis=1)
    current_cvli_30d = features[:, max(0, ti - 30) : ti, 0].sum(axis=1)
    historical_col = "total_cvli" if "total_cvli" in gdf.columns else "recent_cvli"
    historical_cvli = gdf[historical_col].fillna(0).values.astype(float)

    historical_support = np.clip((historical_cvli - 20.0) / 40.0, 0, 1)
    live_support = np.maximum(np.clip(current_cvli_recent / 1.0, 0, 1), np.clip(current_cvli_30d / 2.0, 0, 1))
    territorial_support = np.maximum(historical_support, live_support)
    norm_tension = norm_tension * territorial_support

    recent_crime_signal = np.clip(current_cvli_recent, 0, 2) / 2.0
    calm_signal = np.clip(-momentum[:, -1, 3], 0, 30) / 30.0
    decay_factor = 1.0 - (calm_signal * 0.5)

    neural_weight = 0.50 if cfg.key != "interior" else 0.35
    final_logic = (neural_weight * norm_neural * decay_factor) + (0.10 * norm_tension) + (0.40 * recent_crime_signal)
    return np.clip(final_logic, 0, 1) * 100.0


def date_indices(dates: list[pd.Timestamp], start: str, end: str) -> list[int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [i for i, d in enumerate(pd.to_datetime(dates)) if start_ts <= d <= end_ts]


def evaluate_period(cfg: RegionConfig, data: dict, model: torch.nn.Module, adj: list[torch.Tensor], label: str, start: str, end: str, allow_partial: bool) -> list[dict]:
    rows: list[dict] = []
    features = data["node_features"]
    dates = pd.to_datetime(data["dates"])

    for ti in date_indices(list(dates), start, end):
        targets = horizon_target(features, ti, HORIZON_DAYS, allow_partial=allow_partial)
        if targets is None or targets.sum() <= 0:
            continue

        scores = score_day(data, model, adj, cfg, ti)
        row = {
            "period": label,
            "region": cfg.key,
            "date": dates[ti].strftime("%Y-%m-%d"),
            "target_end": dates[min(ti + HORIZON_DAYS, len(dates) - 1)].strftime("%Y-%m-%d"),
            "full_horizon": bool(ti + HORIZON_DAYS < len(dates)),
            "active_locations": int((targets > 0).sum()),
            "total_cvli_horizon": int(targets.sum()),
        }
        for k in K_VALUES:
            row[f"p{k}_event"] = precision_against_events(scores, targets, k)
            row[f"p{k}_rank"] = topk_overlap(scores, targets, k)
            row[f"recall{k}_event"] = recall_against_events(scores, targets, k)
        rows.append(row)

    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if re.match(r"^(p\d+_|recall\d+_)", c)]
    summary = (
        df.groupby(["period", "region", "full_horizon"], dropna=False)
        .agg(
            days=("date", "count"),
            active_locations_avg=("active_locations", "mean"),
            total_cvli_horizon_sum=("total_cvli_horizon", "sum"),
            **{c: (c, "mean") for c in metric_cols},
        )
        .reset_index()
    )
    for c in metric_cols + ["active_locations_avg"]:
        summary[c] = summary[c].round(4)
    return summary


def write_report(summary: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark modelos ativos\n\n")
        f.write(f"- Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Device: {DEVICE}\n")
        f.write(f"- Horizonte alvo: {HORIZON_DAYS} dias\n")
        f.write(f"- Avaliacao: {args.eval_start} a {args.eval_end}\n")
        f.write(f"- Validacao: {args.val_start} a {args.val_end}\n")
        f.write(f"- Validacao parcial habilitada: {args.allow_partial_validation}\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-04-30")
    parser.add_argument("--val-start", default="2026-05-01")
    parser.add_argument("--val-end", default="2026-05-23")
    parser.add_argument("--allow-partial-validation", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    print(f"Device: {DEVICE}")
    for cfg in REGIONS:
        print(f"\n[{cfg.key}] carregando {cfg.model_file}...")
        data, model, adj, runtime_cfg = load_region(cfg)
        dates = pd.to_datetime(data["dates"])
        print(f"  grid: {data['node_features'].shape} | datas: {dates[0].date()} -> {dates[-1].date()}")
        all_rows.extend(evaluate_period(runtime_cfg, data, model, adj, "avaliacao", args.eval_start, args.eval_end, allow_partial=False))
        all_rows.extend(evaluate_period(runtime_cfg, data, model, adj, "validacao", args.val_start, args.val_end, allow_partial=args.allow_partial_validation))

    df = pd.DataFrame(all_rows)
    summary = summarize(all_rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_path = out_dir / f"active_models_benchmark_rows_{stamp}.csv"
    summary_path = out_dir / f"active_models_benchmark_summary_{stamp}.csv"
    json_path = out_dir / f"active_models_benchmark_summary_{stamp}.json"
    report_path = out_dir / f"active_models_benchmark_report_{stamp}.md"

    df.to_csv(rows_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, report_path, args)

    print("\nResumo:")
    print(summary.to_string(index=False))
    print(f"\nArquivos gerados:\n  {rows_path}\n  {summary_path}\n  {json_path}\n  {report_path}")


if __name__ == "__main__":
    main()
