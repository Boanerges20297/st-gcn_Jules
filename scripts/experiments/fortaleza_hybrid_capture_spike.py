import argparse
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FREEZE_SCRIPT = os.path.join(BASE_DIR, "tests", "Sentinela", "freeze_total_v3.py")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments")
BAIRRO_POLYGONS_PATH = os.path.join(BASE_DIR, "data", "static", "AIS - CAPITAL.geojson")
_BAIRRO_POLYGONS = None
_BAIRRO_AREAS = None


def load_freeze():
    spec = importlib.util.spec_from_file_location("freeze_total_v3_local", FREEZE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(values):
    arr = np.asarray(values, dtype=float)
    low, high = arr.min(), arr.max()
    if high - low < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def decayed_cvli_score(cvli_raw, ti, recent_half_life=30.0, old_half_life=180.0, recent_weight=0.75):
    history = cvli_raw[:, : ti + 1]
    if history.shape[1] == 0:
        return np.zeros(cvli_raw.shape[0], dtype=float)
    age = np.arange(history.shape[1] - 1, -1, -1, dtype=float)
    recent = history @ np.exp(-age / recent_half_life)
    old = history @ np.exp(-age / old_half_life)
    return recent_weight * normalize(recent) + (1.0 - recent_weight) * normalize(old)


def apply_eligibility(scores, eligibility_scores, top_n):
    if not top_n or top_n >= len(scores):
        return scores
    gated = np.full_like(scores, -1.0, dtype=float)
    keep = np.argsort(eligibility_scores)[::-1][:top_n]
    gated[keep] = scores[keep]
    return gated


def score_families(feats, cvli_raw, ti, eligible_top=None):
    structural = 0.55 * normalize(feats["hist_pct"][:, ti]) + 0.45 * normalize(feats["target_enc"][:, ti])
    tactical = (
        0.45 * normalize(feats["cvli_ewma_7d"][:, ti])
        + 0.35 * normalize(feats["cvli_ewma_14d"][:, ti])
        + 0.20 * normalize(feats["cvli_ewma_30d"][:, ti])
    )
    contagion = 0.75 * tactical + 0.25 * normalize(feats["nbr_cvli_30d"][:, ti])
    recency = decayed_cvli_score(cvli_raw, ti)
    families = {
        "HISTORICO_RECORRENTE": structural,
        "CVLI_TATICO": contagion,
        "HIBRIDO_BAIRRO": 0.60 * structural + 0.40 * contagion,
        "HIBRIDO_RECENTE": 0.45 * structural + 0.35 * contagion + 0.20 * recency,
    }
    if eligible_top:
        families = {
            f"{name}_TOP{eligible_top}": apply_eligibility(scores, recency, eligible_top)
            for name, scores in families.items()
        }
    return families


def bairro_score_map(bairros, scores):
    return {str(bairro).upper().strip(): float(score) for bairro, score in zip(bairros, normalize(scores))}


def load_bairro_polygons():
    global _BAIRRO_POLYGONS
    if _BAIRRO_POLYGONS is not None:
        return _BAIRRO_POLYGONS
    with open(BAIRRO_POLYGONS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    polygons = {}
    for feature in data.get("features", []):
        name = str((feature.get("properties") or {}).get("Name") or "").split(" - AIS")[0].upper().strip()
        geom = shape(feature.get("geometry"))
        if name and not geom.is_empty:
            polygons[name] = geom
    _BAIRRO_POLYGONS = polygons
    return polygons


def polygon_area_km2(geom):
    lat0 = geom.centroid.y
    coords = list(geom.exterior.coords) if geom.geom_type == "Polygon" else []
    if not coords:
        return 0.0
    xy = [
        (lon * 111.32 * math.cos(math.radians(lat0)), lat * 110.54)
        for lon, lat in coords
    ]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(xy, xy[1:]))) / 2.0


def bairro_radius_km(bairro, default_radius=1.0):
    global _BAIRRO_AREAS
    if _BAIRRO_AREAS is None:
        _BAIRRO_AREAS = {name: polygon_area_km2(geom) for name, geom in load_bairro_polygons().items()}
    area = _BAIRRO_AREAS.get(str(bairro).upper().strip(), 0.0)
    if area <= 3.0:
        return 0.5
    if area <= 8.0:
        return 0.75
    if area <= 15.0:
        return 1.0
    return max(1.25, float(default_radius))


def metric_row(family, pred_date, scores, targets, k):
    order = np.argsort(scores)[::-1][:k]
    total_cvli = float(targets.sum())
    captured = float(targets[order].sum())
    positives = int((targets > 0).sum())
    return {
        "pred_date": str(pd.Timestamp(pred_date).date()),
        "family": family,
        "k": k,
        "future_positive_bairros": positives,
        "future_cvli_total": total_cvli,
        "selected_positive_bairros": int((targets[order] > 0).sum()),
        "captured_cvli": captured,
        "capture_rate": captured / total_cvli if total_cvli else 0.0,
        "precision_at_k": float((targets[order] > 0).sum() / k),
        "recall_at_k": float((targets[order] > 0).sum() / max(positives, 1)),
    }


def top_selection_rows(family, pred_date, bairros, scores, targets, k):
    rows = []
    for rank, idx in enumerate(np.argsort(scores)[::-1][:k], 1):
        rows.append({
            "pred_date": str(pd.Timestamp(pred_date).date()),
            "family": family,
            "rank": rank,
            "bairro": bairros[idx],
            "score": round(float(scores[idx]), 6),
            "future_cvli": float(targets[idx]),
        })
    return rows


def latest_microzones(selected_bairros, top_n_per_bairro):
    path = os.path.join(BASE_DIR, "outputs", "hermes", "top_30_micronodes_capital.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if "bairro" not in df.columns:
        return []
    df["bairro_norm"] = df["bairro"].astype(str).str.upper().str.strip()

    rows = []
    for bairro in selected_bairros:
        subset = df[df["bairro_norm"] == str(bairro).upper().strip()].head(top_n_per_bairro)
        for _, row in subset.iterrows():
            rows.append({
                "bairro": bairro,
                "micronode_id": row.get("micronode_id"),
                "micro_score": row.get("score"),
                "longitude": row.get("longitude"),
                "latitude": row.get("latitude"),
                "nearby_streets": row.get("nearby_streets"),
            })
    return rows


def latest_microzone_greedy(selected_bairros, limit=20, max_per_bairro=3):
    micro_path = os.path.join(BASE_DIR, "outputs", "hermes", "top_30_micronodes_capital.csv")
    hist_path = os.path.join(BASE_DIR, "outputs", "hermes", "total_cvli_micronodo.csv")
    if not os.path.exists(micro_path):
        return []

    df = pd.read_csv(micro_path)
    if "bairro" not in df.columns:
        return []
    df["bairro_norm"] = df["bairro"].astype(str).str.upper().str.strip()
    df = df[df["bairro_norm"].isin({str(b).upper().strip() for b in selected_bairros})].copy()
    if df.empty:
        return []

    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        if {"micronodo", "area_oficial"}.issubset(hist.columns):
            hist["join_key"] = hist["micronodo"].astype(str).str.upper().str.strip() + "||" + hist["area_oficial"].astype(str).str.upper().str.strip()
            df["join_key"] = df["micronode_id"].astype(str).str.upper().str.strip() + "||" + df["bairro"].astype(str).str.upper().str.strip()
            df = df.merge(hist[["join_key", "cvli_count_1km", "cvli_count_2km"]], on="join_key", how="left")

    df["score_norm"] = normalize(df["score"].fillna(0).to_numpy(dtype=float))
    df["hist_norm"] = normalize(df.get("cvli_count_1km", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float))
    df["street_norm"] = normalize(df.get("local_street_pressure", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy(dtype=float))
    df["greedy_score"] = 0.55 * df["score_norm"] + 0.30 * df["hist_norm"] + 0.15 * df["street_norm"]

    picked = []
    per_bairro = {}
    for _, row in df.sort_values("greedy_score", ascending=False).iterrows():
        bairro = row["bairro_norm"]
        if per_bairro.get(bairro, 0) >= max_per_bairro:
            continue
        picked.append({
            "rank": len(picked) + 1,
            "bairro": row.get("bairro"),
            "micronode_id": row.get("micronode_id"),
            "greedy_score": round(float(row.get("greedy_score", 0.0)), 6),
            "micro_score": row.get("score"),
            "cvli_count_1km": row.get("cvli_count_1km"),
            "longitude": row.get("longitude"),
            "latitude": row.get("latitude"),
            "nearby_streets": row.get("nearby_streets"),
        })
        per_bairro[bairro] = per_bairro.get(bairro, 0) + 1
        if len(picked) >= limit:
            break
    return picked


def load_cvli_points():
    path = os.path.join(BASE_DIR, "data", "raw", "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
    df = pd.read_csv(
        path,
        usecols=["data", "tipo", "cidade", "bairro", "latitude", "longitude", "qtd_mortes"],
        low_memory=False,
    )
    df = df[df["tipo"].astype(str).str.lower().eq("cvli")].copy()
    df = df[df["cidade"].astype(str).str.upper().eq("FORTALEZA")].copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["qtd_mortes"] = pd.to_numeric(df["qtd_mortes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    df["bairro_norm"] = df["bairro"].astype(str).str.upper().str.strip()
    return df.dropna(subset=["data", "latitude", "longitude"])


def haversine_matrix_km(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)))
    lat1 = np.radians(points_a[:, 0])[:, None]
    lon1 = np.radians(points_a[:, 1])[:, None]
    lat2 = np.radians(points_b[:, 0])[None, :]
    lon2 = np.radians(points_b[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def spatial_greedy_metrics(points_df, pred_date, eligible_bairros, horizon, limit=20, radius_km=1.0, cell_km=1.0):
    pred_date = pd.Timestamp(pred_date)
    eligible = {str(b).upper().strip() for b in eligible_bairros}
    past = points_df[(points_df["data"] <= pred_date) & points_df["bairro_norm"].isin(eligible)].copy()
    future = points_df[
        (points_df["data"] > pred_date)
        & (points_df["data"] <= pred_date + pd.Timedelta(days=horizon))
    ].copy()
    if past.empty or future.empty:
        return None

    lat_cell = cell_km / 110.54
    lon_cell = cell_km / (111.32 * max(0.15, np.cos(np.radians(float(past["latitude"].mean())))))
    past["cell_y"] = np.floor(past["latitude"] / lat_cell).astype(int)
    past["cell_x"] = np.floor(past["longitude"] / lon_cell).astype(int)
    age_days = (pred_date - past["data"]).dt.days.clip(lower=0).astype(float)
    past["decayed_weight"] = past["qtd_mortes"] * (
        0.75 * np.exp(-age_days / 30.0) + 0.25 * np.exp(-age_days / 180.0)
    )

    cells = (
        past.groupby(["cell_y", "cell_x", "bairro_norm"], as_index=False)
        .agg(
            score=("decayed_weight", "sum"),
            historical_cvli=("qtd_mortes", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .sort_values("score", ascending=False)
    )

    picked = []
    picked_coords = []
    for _, row in cells.iterrows():
        candidate = np.array([[float(row["latitude"]), float(row["longitude"])]])
        if picked_coords and float(haversine_matrix_km(candidate, np.array(picked_coords)).min()) < radius_km * 0.75:
            continue
        picked.append(row)
        picked_coords.append([float(row["latitude"]), float(row["longitude"])])
        if len(picked) >= limit:
            break

    if not picked:
        return None

    selected = pd.DataFrame(picked)
    future_coords = future[["latitude", "longitude"]].to_numpy(dtype=float)
    selected_coords = selected[["latitude", "longitude"]].to_numpy(dtype=float)
    hit_mask = haversine_matrix_km(future_coords, selected_coords).min(axis=1) <= radius_km
    captured = float(future.loc[hit_mask, "qtd_mortes"].sum())
    total = float(future["qtd_mortes"].sum())
    return {
        "pred_date": str(pred_date.date()),
        "future_cvli_total": total,
        "captured_cvli": captured,
        "capture_rate": captured / total if total else 0.0,
        "selected_cells": len(selected),
        "radius_km": radius_km,
        "area_km2_naive": float(len(selected) * np.pi * (radius_km ** 2)),
        "cell_km": cell_km,
    }


def build_spatial_cells(points_df, pred_date, eligible_bairros, bairro_scores=None, radius_km=1.0, local_weight=0.0):
    rows = []
    polygons = load_bairro_polygons()
    scores = bairro_scores or {}
    for bairro in [str(b).upper().strip() for b in eligible_bairros]:
        geom = polygons.get(bairro)
        if geom is None:
            continue
        cell_radius = bairro_radius_km(bairro, radius_km)
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        lat_step = (1.5 * cell_radius) / 110.54
        lon_step = (math.sqrt(3.0) * cell_radius) / (111.32 * max(0.15, math.cos(math.radians((min_lat + max_lat) / 2.0))))
        y = 0
        lat = min_lat
        while lat <= max_lat:
            x = 0
            lon = min_lon + (y % 2) * lon_step / 2.0
            while lon <= max_lon:
                if geom.contains(Point(lon, lat)):
                    rows.append({
                        "cell_y": y,
                        "cell_x": x,
                        "bairro_norm": bairro,
                        "latitude": lat,
                        "longitude": lon,
                        "radius_km": cell_radius,
                        "score": float(scores.get(bairro, 0.0)),
                    })
                lon += lon_step
                x += 1
            lat += lat_step
            y += 1
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells
    if local_weight > 0:
        pred_date = pd.Timestamp(pred_date)
        past = points_df[(points_df["data"] <= pred_date) & points_df["bairro_norm"].isin(set(cells["bairro_norm"]))].copy()
        if not past.empty:
            age_days = (pred_date - past["data"]).dt.days.clip(lower=0).astype(float)
            past["local_weight"] = past["qtd_mortes"] * np.exp(-age_days / 90.0)
            local_scores = []
            past_coords = past[["latitude", "longitude"]].to_numpy(dtype=float)
            for _, row in cells.iterrows():
                distances = haversine_matrix_km(np.array([[float(row["latitude"]), float(row["longitude"])]]), past_coords)[0]
                local_scores.append(float(past.loc[distances <= float(row["radius_km"]), "local_weight"].sum()))
            cells["local_pred_score"] = normalize(local_scores)
            cells["score"] = (1.0 - local_weight) * cells["score"] + local_weight * cells["local_pred_score"]
    return cells.sort_values("score", ascending=False).head(240).reset_index(drop=True)


def select_spatial_cells(cells, genes, spacing_factor):
    radius_km, limit, _, _ = genes
    selected = []
    selected_coords = []
    selected_radii = []
    for _, row in cells.sort_values("score", ascending=False).iterrows():
        candidate = [float(row["latitude"]), float(row["longitude"])]
        candidate_radius = float(row.get("radius_km", radius_km))
        if selected_coords:
            distances = haversine_matrix_km(np.array([candidate]), np.array(selected_coords))[0]
            min_allowed = spacing_factor * (np.array(selected_radii) + candidate_radius) / 2.0
            if bool((distances < min_allowed).any()):
                continue
        selected.append(row)
        selected_coords.append(candidate)
        selected_radii.append(candidate_radius)
        if len(selected) >= int(limit):
            break
    selected_df = pd.DataFrame(selected)
    if not selected_df.empty:
        selected_df["radius_km"] = selected_radii
    return selected_df


def evaluate_spatial_selection(cells, future, genes, zone_shape="hex", objective="balanced"):
    if cells.empty or future.empty:
        return 0.0, 0.0, 0.0
    _, _, area_penalty, spacing = genes
    selected_df = select_spatial_cells(cells, genes, spacing)
    if selected_df.empty:
        return 0.0, 0.0, 0.0

    distances = haversine_matrix_km(
        future[["latitude", "longitude"]].to_numpy(dtype=float),
        selected_df[["latitude", "longitude"]].to_numpy(dtype=float),
    )
    hit_mask = (distances <= selected_df["radius_km"].to_numpy(dtype=float)[None, :]).any(axis=1)
    captured = float(future.loc[hit_mask, "qtd_mortes"].sum())
    total = float(future["qtd_mortes"].sum())
    capture_rate = captured / total if total else 0.0
    area_km2 = float(sum(operational_area_km2(float(radius), 1, zone_shape) for radius in selected_df["radius_km"]))
    capture_per_100km2 = capture_rate * 100.0 / area_km2 if area_km2 else 0.0
    if objective == "capture":
        fitness = capture_rate
    elif objective == "efficiency":
        fitness = capture_per_100km2
    else:
        fitness = capture_rate + area_penalty * capture_per_100km2
    return fitness, capture_rate, area_km2


def operational_spacing_factor(zone_shape):
    return 2.0 if zone_shape == "circle" else math.sqrt(3.0)


def operational_area_km2(radius_km, count, zone_shape):
    unit = math.pi * radius_km ** 2 if zone_shape == "circle" else (3.0 * math.sqrt(3.0) / 2.0) * radius_km ** 2
    return float(count * unit)


def regular_polygon(lon, lat, radius_km, vertices, rotation=0.0):
    coords = []
    for i in range(vertices + 1):
        angle = rotation + 2 * math.pi * i / vertices
        dy = radius_km * math.sin(angle)
        dx = radius_km * math.cos(angle)
        coords.append([
            lon + dx / (111.32 * max(0.15, math.cos(math.radians(lat)))),
            lat + dy / 110.54,
        ])
    return coords


def latest_ga_geojson(points_df, dates, cvli_raw, bairros, feats, gene, zone_shape="circle", local_weight=0.0):
    latest_date = pd.Timestamp(dates[-1])
    ti = len(dates) - 1
    scores = score_families(feats, cvli_raw, ti, eligible_top=30)["HIBRIDO_RECENTE_TOP30"]
    score_map = bairro_score_map(bairros, scores)
    eligible_bairros = [bairros[idx] for idx in np.argsort(scores)[::-1][:10]]
    polygons = load_bairro_polygons()
    recent = points_df[
        (points_df["data"] <= latest_date)
        & (points_df["data"] >= latest_date - pd.Timedelta(days=365))
    ].copy()
    features = []
    for bairro in eligible_bairros:
        bairro_key = str(bairro).upper().strip()
        geom = polygons.get(bairro_key)
        subset = recent[recent["bairro_norm"].eq(bairro_key)].dropna(subset=["latitude", "longitude"])
        if geom is None or subset.empty:
            continue
        coords = subset[["latitude", "longitude"]].to_numpy(dtype=float)
        best_idx = 0
        best_count = -1.0
        for idx, candidate in enumerate(coords):
            count = float(subset.loc[haversine_matrix_km(np.array([candidate]), coords)[0] <= 0.75, "qtd_mortes"].sum())
            if count > best_count:
                best_idx = idx
                best_count = count
        lat, lon = map(float, coords[best_idx])
        if not geom.contains(Point(lon, lat)):
            continue
        radius_km = min(1.25, max(0.45, 0.35 + 0.18 * math.sqrt(max(best_count, 1.0))))
        features.append({
            "type": "Feature",
            "properties": {
                "rank": len(features) + 1,
                "bairro": bairro_key,
                "score": round(float(score_map.get(bairro_key, 0.0)), 4),
                "local_cvli_count": round(best_count, 2),
                "radius_km": round(radius_km, 3),
                "style_class": "cvli_historical_focus",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def genetic_spatial_search(points_df, dates, eval_dates, cvli_raw, bairros, feats, horizon, zone_shape="hex", objective="balanced", local_weight=0.0, population=12, generations=6, seed=42):
    rng = random.Random(seed)
    date_map = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    sample_dates = list(eval_dates[:: max(1, len(eval_dates) // 8)])
    spacing = operational_spacing_factor(zone_shape)

    def random_gene():
        return [
            rng.choice([0.5, 0.75, 1.0, 1.25, 1.5]),  # honeycomb radius_km
            rng.choice([8, 10, 12, 15, 18, 20, 24, 28, 30]),  # selected cells
            rng.uniform(0.15, 0.75),        # spatial efficiency weight
            spacing,                        # operational no-overlap factor
        ]

    def score_gene(gene):
        rows = []
        for pred_date in sample_dates:
            ti = date_map[pd.Timestamp(pred_date)]
            scores = score_families(feats, cvli_raw, ti, eligible_top=30)["HIBRIDO_RECENTE_TOP30"]
            eligible_bairros = [bairros[idx] for idx in np.argsort(scores)[::-1][:30]]
            cells = build_spatial_cells(points_df, pred_date, eligible_bairros, bairro_score_map(bairros, scores), float(gene[0]), local_weight)
            future = points_df[
                (points_df["data"] > pd.Timestamp(pred_date))
                & (points_df["data"] <= pd.Timestamp(pred_date) + pd.Timedelta(days=horizon))
            ].copy()
            if future.empty:
                continue
            rows.append(evaluate_spatial_selection(cells, future, gene, zone_shape, objective))
        if not rows:
            return {"fitness": 0.0, "capture_rate": 0.0, "area_km2": 0.0}
        arr = np.array(rows, dtype=float)
        return {
            "fitness": float(arr[:, 0].mean()),
            "capture_rate": float(arr[:, 1].mean()),
            "area_km2": float(arr[:, 2].mean()),
        }

    population_genes = [random_gene() for _ in range(population)]
    for _ in range(generations):
        scored = sorted(((score_gene(g), g) for g in population_genes), key=lambda item: item[0]["fitness"], reverse=True)
        elites = [g for _, g in scored[: max(4, population // 4)]]
        next_gen = elites[:]
        while len(next_gen) < population:
            a, b = rng.sample(elites, 2)
            child = [
                rng.choice([a[0], b[0], random_gene()[0]]),
                rng.choice([a[1], b[1], random_gene()[1]]),
                max(0.0, (a[2] + b[2]) / 2 + rng.uniform(-0.06, 0.06)),
                spacing,
            ]
            next_gen.append(child)
        population_genes = next_gen

    final = sorted(((score_gene(g), g) for g in population_genes), key=lambda item: item[0]["fitness"], reverse=True)
    rows = []
    for rank, (metrics, gene) in enumerate(final[:10], 1):
        rows.append({
            "rank": rank,
            "objective": objective,
            "local_weight": local_weight,
            "fitness": metrics["fitness"],
            "capture_rate": metrics["capture_rate"],
            "area_km2": metrics["area_km2"],
            "capture_per_100km2": metrics["capture_rate"] * 100.0 / metrics["area_km2"] if metrics["area_km2"] else 0.0,
            "radius_km": gene[0],
            "selected_cells": int(gene[1]),
            "area_penalty": gene[2],
            "overlap_penalty": gene[3],
            "sample_windows": len(sample_dates),
        })
    return pd.DataFrame(rows)


def champion_rows(summary, spatial_summary, ga_summary):
    rows = []
    if not summary.empty:
        for _, row in summary.iterrows():
            rows.append({
                "method": row["family"],
                "objective": "bairro_rank",
                "k_or_cells": int(row["k"]),
                "capture_rate": float(row["capture_rate"]),
                "area_km2": None,
                "capture_per_100km2": None,
                "notes": "baseline por bairro",
            })
    if spatial_summary is not None and not spatial_summary.empty:
        for _, row in spatial_summary.iterrows():
            area = float(row["area_km2_naive"])
            rows.append({
                "method": "SPATIAL_GREEDY_TOP30",
                "objective": f"radius_{row['radius_km']}",
                "k_or_cells": int(row["selected_cells"]),
                "capture_rate": float(row["capture_rate"]),
                "area_km2": area,
                "capture_per_100km2": float(row["capture_rate"]) * 100.0 / area if area else None,
                "notes": "historico espacial, nao operacional final",
            })
    if ga_summary is not None and not ga_summary.empty:
        for _, row in ga_summary.iterrows():
            rows.append({
                "method": "HONEYCOMB_GA",
                "objective": f"{row['objective']}_local{row.get('local_weight', 0.0)}",
                "k_or_cells": int(row["selected_cells"]),
                "capture_rate": float(row["capture_rate"]),
                "area_km2": float(row["area_km2"]),
                "capture_per_100km2": float(row["capture_per_100km2"]),
                "notes": f"hex radius {row['radius_km']} km",
            })
    return pd.DataFrame(rows).sort_values(
        ["capture_rate", "capture_per_100km2"],
        ascending=[False, False],
        na_position="last",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2026-02-28")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--eligible-top", type=int, nargs="*", default=[20, 30])
    parser.add_argument("--latest-top-bairros", type=int, default=10)
    parser.add_argument("--microzones-per-bairro", type=int, default=3)
    parser.add_argument("--greedy-microzones", type=int, default=20)
    parser.add_argument("--spatial-cells", type=int, nargs="+", default=[20])
    parser.add_argument("--spatial-radius-km", type=float, nargs="+", default=[1.0])
    parser.add_argument("--run-ga", action="store_true")
    parser.add_argument("--ga-objective", choices=["balanced", "capture", "efficiency", "all"], default="balanced")
    parser.add_argument("--hex-local-weight", type=float, nargs="+", default=[0.0])
    parser.add_argument("--zone-shape", choices=["hex", "circle"], default="hex")
    parser.add_argument("--keep-history", action="store_true", help="preserva arquivos carimbados da rodada")
    parser.add_argument("--write-details", action="store_true", help="exporta CSVs diarios e selecoes por janela")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    freeze = load_freeze()
    feats, _, dates, cvli_raw, bairros = freeze.build_all()
    date_map = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff not in date_map:
        raise SystemExit(f"cutoff fora da serie: {args.cutoff}")

    eval_dates = pd.date_range(cutoff + pd.Timedelta(days=1), dates[-1] - pd.Timedelta(days=args.horizon), freq="D")
    metrics = []
    selections = []
    spatial_rows = []
    points_df = load_cvli_points()
    for pred_date in eval_dates:
        ti = date_map[pd.Timestamp(pred_date)]
        targets = cvli_raw[:, ti + 1 : ti + args.horizon + 1].sum(axis=1)
        if targets.sum() == 0:
            continue
        for eligible_top in ([None] + list(args.eligible_top or [])):
            for family, scores in score_families(feats, cvli_raw, ti, eligible_top=eligible_top).items():
                for k in args.k:
                    metrics.append(metric_row(family, pred_date, scores, targets, min(k, len(bairros))))
                selections.extend(top_selection_rows(family, pred_date, bairros, scores, targets, max(args.k)))

        recency_scores = decayed_cvli_score(cvli_raw, ti)
        eligible_bairros = [bairros[idx] for idx in np.argsort(recency_scores)[::-1][:30]]
        for spatial_cells in args.spatial_cells:
            for radius_km in args.spatial_radius_km:
                spatial = spatial_greedy_metrics(
                    points_df,
                    pred_date,
                    eligible_bairros,
                    horizon=args.horizon,
                    limit=spatial_cells,
                    radius_km=radius_km,
                )
                if spatial:
                    spatial_rows.append(spatial)

    metrics_df = pd.DataFrame(metrics)
    selections_df = pd.DataFrame(selections)
    spatial_df = pd.DataFrame(spatial_rows)
    summary = (
        metrics_df.groupby(["family", "k"])[["capture_rate", "precision_at_k", "recall_at_k", "captured_cvli"]]
        .mean()
        .reset_index()
        .sort_values(["k", "capture_rate", "precision_at_k"], ascending=[True, False, False])
    )

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") if args.keep_history else "latest"
    prefix = f"fortaleza_hybrid_capture_h{args.horizon}_{suffix}"
    metrics_path = os.path.join(OUT_DIR, f"{prefix}_daily.csv")
    selections_path = os.path.join(OUT_DIR, f"{prefix}_selections.csv")
    summary_path = os.path.join(OUT_DIR, f"{prefix}_summary.csv")
    json_path = os.path.join(OUT_DIR, f"{prefix}_summary.json")
    spatial_path = os.path.join(OUT_DIR, f"{prefix}_spatial_greedy_daily.csv")
    spatial_summary_path = os.path.join(OUT_DIR, f"{prefix}_spatial_greedy_summary.csv")
    ga_path = os.path.join(OUT_DIR, f"{prefix}_spatial_ga_summary.csv")
    champions_path = os.path.join(OUT_DIR, f"{prefix}_champions.csv")

    if args.write_details:
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        selections_df.to_csv(selections_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if args.write_details:
        spatial_df.to_csv(spatial_path, index=False, encoding="utf-8-sig")
    if spatial_df.empty:
        spatial_summary = pd.DataFrame()
    else:
        spatial_summary = (
            spatial_df.groupby(["selected_cells", "radius_km"], as_index=False)
            .agg(
                capture_rate=("capture_rate", "mean"),
                captured_cvli=("captured_cvli", "mean"),
                future_cvli_total=("future_cvli_total", "mean"),
                area_km2_naive=("area_km2_naive", "mean"),
                windows=("pred_date", "count"),
            )
            .sort_values(["capture_rate", "area_km2_naive"], ascending=[False, True])
        )
        spatial_summary.insert(0, "family", "SPATIAL_GREEDY_TOP30")
        spatial_summary.insert(1, "horizon", args.horizon)
    spatial_summary.to_csv(spatial_summary_path, index=False, encoding="utf-8-sig")
    ga_summary = pd.DataFrame()
    if args.run_ga:
        objectives = ["capture", "efficiency", "balanced"] if args.ga_objective == "all" else [args.ga_objective]
        ga_summary = pd.concat([
            genetic_spatial_search(points_df, dates, eval_dates, cvli_raw, bairros, feats, args.horizon, args.zone_shape, objective, local_weight)
            for objective in objectives
            for local_weight in args.hex_local_weight
        ], ignore_index=True)
        ga_summary.to_csv(ga_path, index=False, encoding="utf-8-sig")
        if not ga_summary.empty:
            latest_objective = "balanced" if "balanced" in set(ga_summary["objective"]) else str(ga_summary.iloc[0]["objective"])
            best = ga_summary[ga_summary["objective"].eq(latest_objective)].sort_values("capture_rate", ascending=False).iloc[0]
            gene = [
                float(best["radius_km"]),
                int(best["selected_cells"]),
                float(best["area_penalty"]),
                float(best["overlap_penalty"]),
            ]
            ga_geojson_path = os.path.join(OUT_DIR, f"{prefix}_ga_zones.geojson")
            with open(ga_geojson_path, "w", encoding="utf-8") as handle:
                json.dump(latest_ga_geojson(points_df, dates, cvli_raw, bairros, feats, gene, args.zone_shape, float(best.get("local_weight", 0.0))), handle, ensure_ascii=False, indent=2)
    champions = champion_rows(summary, spatial_summary, ga_summary)
    champions.to_csv(champions_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict("records"), handle, ensure_ascii=False, indent=2)

    latest_ti = len(dates) - 1
    latest_scores = score_families(feats, cvli_raw, latest_ti, eligible_top=30)["HIBRIDO_RECENTE_TOP30"]
    latest_bairros = [bairros[idx] for idx in np.argsort(latest_scores)[::-1][: args.latest_top_bairros]]
    micro_rows = latest_microzones(latest_bairros, args.microzones_per_bairro)
    micro_path = os.path.join(OUT_DIR, f"{prefix}_latest_microzones.csv")
    if args.write_details:
        pd.DataFrame(micro_rows).to_csv(micro_path, index=False, encoding="utf-8-sig")
    greedy_rows = latest_microzone_greedy(latest_bairros, limit=args.greedy_microzones, max_per_bairro=args.microzones_per_bairro)
    greedy_path = os.path.join(OUT_DIR, f"{prefix}_latest_microzone_greedy.csv")
    if args.write_details:
        pd.DataFrame(greedy_rows).to_csv(greedy_path, index=False, encoding="utf-8-sig")

    print("\n=== Fortaleza hybrid capture spike ===")
    print(summary.to_string(index=False))
    if args.write_details:
        print(f"\n[OK] daily: {metrics_path}")
        print(f"[OK] selections: {selections_path}")
    else:
        print("\n[OK] details: skipped (use --write-details)")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] spatial greedy: {spatial_summary_path}")
    print(f"[OK] champions: {champions_path}")
    if args.run_ga:
        print(f"[OK] spatial GA: {ga_path}")
        if 'ga_geojson_path' in locals():
            print(f"[OK] latest GA zones: {ga_geojson_path}")
    if args.write_details:
        print(f"[OK] latest microzones: {micro_path}")
        print(f"[OK] greedy microzones: {greedy_path}")


if __name__ == "__main__":
    main()
