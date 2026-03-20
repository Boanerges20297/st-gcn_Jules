import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as report_app


DEFAULT_OUTPUT_DIR = ROOT_DIR / "static_export" / "data"
REGION_EXPORTS = (
    ("fortaleza", "top30_capital.geojson"),
    ("rmf", "top30_rmf.geojson"),
    ("interior", "top30_interior.geojson"),
)
MOMENTUM_WINDOW_DAYS = 14
RECENT_EXOGENOUS_WINDOW_DAYS = 14


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_flask_json(response: Any) -> Tuple[Dict[str, Any], int]:
    status_code = 200
    if isinstance(response, tuple):
        flask_response = response[0]
        if len(response) > 1:
            status_code = int(response[1])
    else:
        flask_response = response
        status_code = int(getattr(flask_response, "status_code", 200))

    if hasattr(flask_response, "get_json"):
        payload = flask_response.get_json()
    else:
        payload = flask_response

    if payload is None:
        payload = {}
    return payload, status_code


def _request_json(path: str, handler) -> Dict[str, Any]:
    with report_app.app.test_request_context(path):
        payload, status_code = _extract_flask_json(handler())
    if status_code != 200:
        raise RuntimeError(f"Falha ao carregar {path}: HTTP {status_code} -> {payload}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(float(value))
    except Exception:
        return default


def _normalize_region(region: str) -> str:
    region_key = (region or "fortaleza").lower()
    if region_key == "capital":
        return "fortaleza"
    return region_key


def _region_display_name(region: str) -> str:
    mapping = {
        "fortaleza": "Fortaleza",
        "rmf": "RMF",
        "interior": "Interior",
    }
    return mapping.get(region, region.title())


def _normalize_polygon_lookup_name(value: Any) -> str:
    text = str(value or "")
    text = report_app.normalize_name(text)
    return report_app.normalize_name(text.split("- AIS")[0].strip())


def _municipality_for_item(item: Dict[str, Any]) -> str:
    region = _normalize_region(str(item.get("region_type") or item.get("region") or "fortaleza"))
    if region == "fortaleza":
        return "Fortaleza"
    return str(item.get("municipality") or item.get("name") or "Desconhecido")


def _snapshot_item_id(region: str, clean_name: str) -> str:
    return f"{_normalize_region(region)}:{clean_name}"


def _has_accents(value: str) -> bool:
    return any(ord(char) > 127 for char in value or "")


def _prefer_display_name(current_name: str, candidate_name: str) -> str:
    current_name = str(current_name or "")
    candidate_name = str(candidate_name or "")
    if not current_name:
        return candidate_name
    if not candidate_name:
        return current_name
    if _has_accents(candidate_name) and not _has_accents(current_name):
        return candidate_name
    if len(candidate_name) > len(current_name):
        return candidate_name
    return current_name


def _dedupe_risk_items(risk_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}

    for item in sorted(risk_items, key=lambda row: _safe_float(row.get("risk_score")), reverse=True):
        region = _normalize_region(str(item.get("region_type") or item.get("region") or "fortaleza"))
        clean_name = str(item.get("clean_name") or report_app.normalize_name(str(item.get("name") or "")))
        item_id = _snapshot_item_id(region, clean_name)
        normalized_item = dict(item)
        normalized_item["region_type"] = region
        normalized_item["clean_name"] = clean_name

        existing = deduped.get(item_id)
        if existing is None:
            deduped[item_id] = normalized_item
            continue

        merged = dict(existing)
        merged["name"] = _prefer_display_name(str(existing.get("name") or ""), str(normalized_item.get("name") or ""))
        merged["faction"] = existing.get("faction") or normalized_item.get("faction")
        merged["municipality"] = existing.get("municipality") or normalized_item.get("municipality")
        merged["node_id"] = existing.get("node_id") or normalized_item.get("node_id")

        existing_metrics = dict(existing.get("metrics") or {})
        candidate_metrics = dict(normalized_item.get("metrics") or {})
        for key, value in candidate_metrics.items():
            if key not in existing_metrics or existing_metrics.get(key) in (None, "", [], {}):
                existing_metrics[key] = value
        merged["metrics"] = existing_metrics
        deduped[item_id] = merged

    return list(deduped.values())


def _load_manager_cache() -> Dict[str, Any]:
    if not os.path.exists(report_app.CACHE_FILE):
        return {}
    try:
        with open(report_app.CACHE_FILE, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file) or {}
    except Exception:
        return {}


def _load_exogenous_events() -> List[Dict[str, Any]]:
    events_path = ROOT_DIR / "data" / "exogenous_events.json"
    if not events_path.exists():
        return []
    try:
        return json.loads(events_path.read_text(encoding="utf-8")) or []
    except Exception:
        return []


def _count_recent_exogenous_by_node(risk_items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    events = _load_exogenous_events()
    if not events:
        return {}

    cutoff = datetime.now().date() - timedelta(days=RECENT_EXOGENOUS_WINDOW_DAYS)
    nodes_index = []
    for item in risk_items:
        region = _normalize_region(str(item.get("region_type") or item.get("region") or "fortaleza"))
        name = str(item.get("name") or "")
        clean_name = str(item.get("clean_name") or report_app.normalize_name(name))
        nodes_index.append(
            {
                "id": _snapshot_item_id(region, clean_name),
                "name_norm": clean_name,
                "municipality_norm": report_app.normalize_name(_municipality_for_item(item)),
            }
        )

    last_date_base = None
    if report_app.orchestrator is not None and hasattr(report_app.orchestrator, "dates"):
        last_date_base = report_app.orchestrator.dates[-1]

    for event in events:
        try:
            event_date_str = event.get("date") or event.get("event_date")
            if not report_app.verify_date_consistency(event_date_str, last_date_base):
                continue
            event_dt = report_app.parse_event_datetime(event)
            if event_dt and event_dt.date() < cutoff:
                continue

            event_bairro = report_app.normalize_name(str(event.get("bairro") or ""))
            event_municipio = report_app.normalize_name(str(event.get("municipio") or ""))
            event_title = report_app.normalize_name(str(event.get("title") or ""))
            event_location = report_app.normalize_name(str(event.get("location") or ""))

            for node in nodes_index:
                name_norm = node["name_norm"]
                municipality_norm = node["municipality_norm"]
                if not name_norm:
                    continue
                if event_bairro and (
                    event_bairro == name_norm
                    or name_norm in event_title
                    or name_norm in event_location
                ):
                    counts[node["id"]] += 1
                    continue
                if not event_bairro and event_municipio and municipality_norm == event_municipio:
                    counts[node["id"]] += 1
        except Exception:
            continue

    return dict(counts)


def _build_momentum_index() -> Dict[str, Dict[str, float]]:
    momentum_index: Dict[str, Dict[str, float]] = {}
    specialists = getattr(report_app.orchestrator, "specialists", {}) or {}

    for region_key, specialist in specialists.items():
        spec_region = _normalize_region(region_key)
        try:
            spec_nodes = specialist["data"]["nodes_gdf"]
            features = specialist["data"]["node_features"]
        except Exception:
            continue

        for spec_idx, row in spec_nodes.iterrows():
            try:
                name = str(row.get("name") or row.get("bairro") or row.get("municipio") or "")
                clean_name = report_app.normalize_name(name)
                if not clean_name:
                    continue
                item_id = _snapshot_item_id(spec_region, clean_name)
                recent_7 = int(features[spec_idx, -7:, 0].sum()) if features.shape[1] >= 7 else 0
                previous_7 = int(features[spec_idx, -14:-7, 0].sum()) if features.shape[1] >= 14 else 0
                recent_14 = int(features[spec_idx, -14:, 0].sum()) if features.shape[1] >= 14 else recent_7
                previous_14 = int(features[spec_idx, -28:-14, 0].sum()) if features.shape[1] >= 28 else previous_7
                momentum_index[item_id] = {
                    "momentum_7d": float(recent_7 - previous_7),
                    "momentum_14d": float(recent_14 - previous_14),
                    "recent_cvli": recent_7,
                }
            except Exception:
                continue

    return momentum_index


def _summarize_item(item: Dict[str, Any], metrics: Dict[str, Any], manager_cache: Dict[str, Any]) -> str:
    clean_name = str(item.get("clean_name") or "")
    cache_entry = manager_cache.get(clean_name) or manager_cache.get(item.get("name") or "") or {}
    for key in ("summary", "resumo", "text", "texto"):
        value = cache_entry.get(key) if isinstance(cache_entry, dict) else None
        if value:
            return str(value).strip()

    recent_cvli = _safe_int(metrics.get("recent_cvli"))
    recent_exogenous = _safe_int(metrics.get("recent_exogenous"))
    momentum_14d = _safe_float(metrics.get("momentum_14d"))
    tension = _safe_float(metrics.get("tension_index"))
    status = str(item.get("status_label") or "monitorado").lower()

    fragments = [f"Território em nível {status}."]
    if recent_cvli > 0:
        fragments.append(f"CVLI recente: {recent_cvli} nos últimos 7 dias.")
    if recent_exogenous > 0:
        fragments.append(f"Eventos exógenos recentes: {recent_exogenous}.")
    if momentum_14d > 0:
        fragments.append("Momentum de 14 dias em aceleração.")
    elif momentum_14d < 0:
        fragments.append("Momentum de 14 dias em arrefecimento.")
    if tension > 0:
        fragments.append(f"Tensão territorial atual em {tension:.2f}.")
    return " ".join(fragments)


def _build_explainability(risk_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    explainability: Dict[str, Dict[str, Any]] = {}
    for item in risk_items:
        node_id = item.get("node_id")
        if node_id is None:
            continue
        try:
            with report_app.app.test_request_context(f"/api/explain/{node_id}"):
                payload, status_code = _extract_flask_json(report_app.explain_node(int(node_id)))
            if status_code == 200:
                explainability[_snapshot_item_id(item["region_type"], item["clean_name"])] = payload
        except Exception:
            continue
    return explainability


def _copy_top_layer(region: str, target_path: Path) -> Dict[str, Any]:
    payload = _request_json(
        f"/api/top20_micro_nodes?region={region}&limit=30",
        report_app.get_top20_micro_nodes,
    )
    features = payload.get("features", [])
    for index, feature in enumerate(features, start=1):
        props = dict(feature.get("properties") or {})
        props["rank"] = index
        props["region"] = _normalize_region(str(props.get("region") or region))
        props.setdefault("score", _safe_float(props.get("score") or props.get("risk_score")))
        props.setdefault("municipality", props.get("municipality") or _region_display_name(props["region"]))
        feature["properties"] = props
    payload["features"] = features
    _write_json(target_path, payload)
    return payload


def _build_dashboard_summary(risk_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    global_top = max(risk_items, key=lambda item: _safe_float(item.get("risk_score")), default=None)
    summary = {
        "global": {
            "total_nodes": len(risk_items),
            "active_locations": sum(1 for item in risk_items if _safe_float(item.get("risk_score")) >= 50.0),
            "top_region": global_top.get("region_type") if global_top else None,
            "top_name": global_top.get("name") if global_top else None,
            "avg_risk": round(
                sum(_safe_float(item.get("risk_score")) for item in risk_items) / max(1, len(risk_items)),
                2,
            ),
        },
        "regions": {},
    }

    for region in ("fortaleza", "rmf", "interior"):
        region_items = [item for item in risk_items if _normalize_region(str(item.get("region_type"))) == region]
        if not region_items:
            continue
        top_item = max(region_items, key=lambda item: _safe_float(item.get("risk_score")))
        summary["regions"][region] = {
            "total_nodes": len(region_items),
            "avg_risk": round(
                sum(_safe_float(item.get("risk_score")) for item in region_items) / max(1, len(region_items)),
                2,
            ),
            "max_risk": round(_safe_float(top_item.get("risk_score")), 2),
            "top_name": top_item.get("name"),
        }
    return summary


def _enrich_polygons_with_risk(polygons_payload: Dict[str, Any], risk_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    risk_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in risk_items:
        region = _normalize_region(str(item.get("region_type") or item.get("region") or "fortaleza"))
        clean_name = str(item.get("clean_name") or report_app.normalize_name(str(item.get("name") or "")))
        risk_by_key[(clean_name, region)] = item

    features = []
    for feature in list(polygons_payload.get("features") or []):
        props = dict(feature.get("properties") or {})
        region = _normalize_region(str(props.get("region_type") or "fortaleza"))
        raw_name = props.get("Name") or props.get("name") or props.get("NOME") or props.get("bairro") or props.get("municipio") or ""
        clean_name = _normalize_polygon_lookup_name(raw_name)
        matched = risk_by_key.get((clean_name, region))

        props["name"] = matched.get("name") if matched else raw_name
        props["clean_name"] = clean_name
        props["region_type"] = region
        props["node_id"] = matched.get("node_id") if matched else None
        props["risk_score"] = _safe_float(matched.get("risk_score")) if matched else None
        props["risk_score_cvli"] = _safe_float(matched.get("risk_score")) if matched else None
        props["faction"] = matched.get("faction") if matched else props.get("faction")
        props["status_label"] = matched.get("status_label") if matched else None
        props["trend"] = matched.get("trend") if matched else None
        props["metrics"] = dict(matched.get("metrics") or {}) if matched else dict(props.get("metrics") or {})

        enriched = dict(feature)
        enriched["properties"] = props
        features.append(enriched)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _build_manifest(snapshot_id: str, generated_at: str) -> Dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "generated_at": generated_at,
        "source_repo": "Report Preview",
        "source_commit": _git_commit_sha(),
        "model_label": "ELITE P10",
        "model_architecture": getattr(report_app, "RISK_MODEL_NAME", None)
        or "Deep ST-GAT Elite (Regionalizado)",
        "momentum_window_days": MOMENTUM_WINDOW_DAYS,
        "regions": ["fortaleza", "rmf", "interior"],
        "notes": "Snapshot estático publicado manualmente",
    }


def _git_commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def export_snapshot(output_dir: Path) -> Path:
    if report_app.nodes_gdf is None or report_app.orchestrator is None:
        report_app.load_data_and_models()

    if report_app.nodes_gdf is None or report_app.orchestrator is None:
        raise RuntimeError("Aplicação ainda não inicializada; exporter não pode continuar.")

    _ensure_dir(output_dir)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    snapshot_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    risk_payload = _request_json("/api/risk", report_app.get_risk)
    polygons_payload = _request_json("/api/polygons", report_app.get_polygons)
    micronodes_payload = _request_json("/api/micronodes", report_app.get_micronodes)

    risk_items = _dedupe_risk_items(list(risk_payload.get("data", [])))
    risk_items.sort(key=lambda item: _safe_float(item.get("risk_score")), reverse=True)

    momentum_index = _build_momentum_index()
    exogenous_index = _count_recent_exogenous_by_node(risk_items)
    manager_cache = _load_manager_cache()
    explainability_index = _build_explainability(risk_items)

    territory_details: Dict[str, Any] = {}
    risk_snapshot_items: List[Dict[str, Any]] = []
    region_rank_counters = Counter()

    for global_rank, item in enumerate(risk_items, start=1):
        region = _normalize_region(str(item.get("region_type") or "fortaleza"))
        clean_name = str(item.get("clean_name") or report_app.normalize_name(str(item.get("name") or "")))
        item_id = _snapshot_item_id(region, clean_name)
        region_rank_counters[region] += 1
        momentum = momentum_index.get(item_id, {})
        metrics = dict(item.get("metrics") or {})
        recent_cvli = _safe_int(momentum.get("recent_cvli", metrics.get("cvli_7d")))
        recent_exogenous = _safe_int(exogenous_index.get(item_id, 0))

        combined_metrics = {
            "critical_streets": metrics.get("critical_streets", "Sem logradouros críticos registrados"),
            "cvli_7d": _safe_int(metrics.get("cvli_7d")),
            "tension": _safe_float(metrics.get("tension")),
            "events_count": _safe_int(metrics.get("events_count")),
            "event_types": list(metrics.get("event_types") or []),
            "recent_cvli": recent_cvli,
            "recent_exogenous": recent_exogenous,
            "momentum_7d": _safe_float(momentum.get("momentum_7d")),
            "momentum_14d": _safe_float(momentum.get("momentum_14d")),
            "tension_index": _safe_float(item.get("tension_score", item.get("risk_score"))),
        }

        summary = _summarize_item(item, combined_metrics, manager_cache)
        explanation = explainability_index.get(item_id) or {
            "node_id": item.get("node_id"),
            "name": item.get("name"),
            "risk_score_pct": _safe_float(item.get("risk_score")),
            "summary": summary,
            "factors": [],
            "caveats": [],
            "explanation_available": False,
            "source": "snapshot_fallback",
        }

        municipality = _municipality_for_item(item)
        territory_details[item_id] = {
            "name": item.get("name"),
            "municipality": municipality,
            "region": region,
            "faction": item.get("faction"),
            "recent_cvli": recent_cvli,
            "recent_exogenous": recent_exogenous,
            "momentum_7d": combined_metrics["momentum_7d"],
            "momentum_14d": combined_metrics["momentum_14d"],
            "critical_streets": combined_metrics["critical_streets"],
            "summary": summary,
            "risk_score": _safe_float(item.get("risk_score")),
            "status": item.get("status_label"),
        }

        risk_snapshot_items.append(
            {
                "id": item_id,
                "node_id": item.get("node_id"),
                "name": item.get("name"),
                "clean_name": clean_name,
                "region": region,
                "municipality": municipality,
                "score": round(_safe_float(item.get("risk_score")), 2),
                "rank_region": region_rank_counters[region],
                "rank_global": global_rank,
                "momentum_7d": combined_metrics["momentum_7d"],
                "momentum_14d": combined_metrics["momentum_14d"],
                "recent_cvli": recent_cvli,
                "recent_exogenous": recent_exogenous,
                "faction": item.get("faction"),
                "tension_index": combined_metrics["tension_index"],
                "status": item.get("status_label"),
                "trend": item.get("trend"),
                "summary": explanation.get("summary") or summary,
            }
        )
        explainability_index[item_id] = explanation

    top_layers = {}
    for region, filename in REGION_EXPORTS:
        layer_payload = _copy_top_layer(region, output_dir / filename)
        top_layers[region] = layer_payload

    _write_json(output_dir / "manifest.json", _build_manifest(snapshot_id, generated_at))
    _write_json(output_dir / "dashboard_summary.json", _build_dashboard_summary(risk_items))
    _write_json(
        output_dir / "risk_snapshot.json",
        {
            "meta": risk_payload.get("meta", {}),
            "items": risk_snapshot_items,
        },
    )
    _write_json(output_dir / "territory_details.json", territory_details)
    _write_json(output_dir / "explainability.json", explainability_index)
    _write_json(output_dir / "polygons.geojson", _enrich_polygons_with_risk(polygons_payload, risk_items))
    _write_json(output_dir / "micronodes.geojson", micronodes_payload)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta um snapshot estático JSON/GeoJSON para o frontend Vite.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório de saída para os artefatos estáticos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    try:
        export_snapshot(output_dir)
    except Exception as exc:
        print(f"ERRO: {exc}")
        return 1

    print(f"Snapshot exportado em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())