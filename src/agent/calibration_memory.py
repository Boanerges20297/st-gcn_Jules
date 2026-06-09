import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


LEARNING_MEMORY_FILE = os.path.join("logs", "calibration_learning_memory.json")
AGENT_HISTORY_FILE = os.path.join("logs", "agent_calibrations_history.json")
STATE_FILE = os.path.join("data", "calibration_state.json")


def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return default
    return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _metric_from_trigger(trigger: str) -> Optional[str]:
    if not trigger:
        return None
    for metric in ("p10", "p20", "faction_coverage"):
        if f".{metric}=" in trigger or metric in trigger:
            return metric
    return None


def _compact_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    keys = (
        "tension_factor",
        "min_risk",
        "tag_bias_direct",
        "tag_bias_neighbor",
        "norm_neural_weight",
        "dynamic_window",
        "use_historical_fallback",
    )
    compact = {k: params.get(k) for k in keys if k in params}
    top10 = params.get("historical_top10")
    if top10:
        compact["historical_top10_sample"] = list(top10[:5])
    return compact


def _semantic_tags(entry: Dict[str, Any]) -> List[str]:
    tags = []
    metric = entry.get("metric") or _metric_from_trigger(entry.get("trigger", ""))
    if metric:
        tags.append(f"metric:{metric}")

    params = entry.get("new_params") or entry.get("weights") or {}
    if params.get("use_historical_fallback"):
        tags.append("fallback_historico")
    if params.get("dynamic_window"):
        tags.append(f"janela:{params.get('dynamic_window')}")

    neural = params.get("norm_neural_weight")
    if isinstance(neural, (int, float)):
        if neural >= 0.75:
            tags.append("rede_neural_dominante")
        elif neural <= 0.35:
            tags.append("rede_neural_baixa")

    tension = params.get("tension_factor")
    if isinstance(tension, (int, float)):
        if tension >= 2.5:
            tags.append("tensao_alta")
        elif tension <= 0.5:
            tags.append("tensao_reduzida")

    direct = params.get("tag_bias_direct")
    if isinstance(direct, (int, float)) and direct >= 4.0:
        tags.append("intel_trigger_forte")

    analysis = entry.get("data_analysis") or entry.get("semantic_review") or {}
    if isinstance(analysis, dict):
        if analysis.get("anomalies_detected"):
            tags.append("anomalia_semantica")
        if analysis.get("geographical_drift"):
            tags.append("deriva_geografica")
        hotspot = analysis.get("next_probable_cvli_hotspot")
        if hotspot:
            tags.append(f"hotspot:{str(hotspot)[:60]}")

    return tags


def append_learning_event(base_dir: str, event: Dict[str, Any], max_events: int = 300) -> None:
    """Append a compact calibration learning event for future agent prompts."""
    path = os.path.join(base_dir, LEARNING_MEMORY_FILE)
    history = _load_json(path, [])
    if not isinstance(history, list):
        history = []

    enriched = dict(event)
    enriched.setdefault("timestamp", datetime.now().isoformat())
    enriched["semantic_tags"] = sorted(set(_semantic_tags(enriched)))
    history.append(enriched)
    _save_json(path, history[-max_events:])


def record_agent_decision(base_dir: str, result: Dict[str, Any], source: str = "agent") -> None:
    if not isinstance(result, dict):
        return
    append_learning_event(
        base_dir,
        {
            "event_type": "calibration_decision",
            "source": source,
            "region": str(result.get("target_region") or "global").lower(),
            "metric": result.get("affected_metric"),
            "weights": _compact_params(result.get("calibrated_weights")),
            "explanations": result.get("explanations"),
            "data_analysis": result.get("data_analysis") or {},
            "should_intervene": result.get("should_intervene"),
            "outcome": "pending_validation",
        },
    )


def record_validation_outcome(base_dir: str, adjustment: Dict[str, Any], validation: Dict[str, Any]) -> None:
    if not isinstance(adjustment, dict) or not isinstance(validation, dict):
        return
    append_learning_event(
        base_dir,
        {
            "event_type": "calibration_validation",
            "source": "auto_calibrator",
            "region": str(adjustment.get("region") or "global").lower(),
            "metric": adjustment.get("metric"),
            "trigger": (
                f"{adjustment.get('region')}.{adjustment.get('metric')}="
                f"{float(adjustment.get('current_value', 0.0)) * 100:.1f}%"
            ),
            "old_params": _compact_params(adjustment.get("old_params")),
            "new_params": _compact_params(adjustment.get("new_params")),
            "semantic_review": adjustment.get("semantic_review") or {},
            "old_value": validation.get("old_value"),
            "new_value": validation.get("new_value"),
            "improvement_pct": validation.get("improvement_pct"),
            "outcome": validation.get("status"),
        },
    )


class CalibrationSemanticMemory:
    """Builds a compact, prompt-safe view of calibration experience."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def build_context(self, region: str = "global", metric: Optional[str] = None, limit: int = 8) -> Dict[str, Any]:
        region_key = (region or "global").lower()
        memory = _load_json(os.path.join(self.base_dir, LEARNING_MEMORY_FILE), [])
        agent_history = _load_json(os.path.join(self.base_dir, AGENT_HISTORY_FILE), [])
        state = _load_json(os.path.join(self.base_dir, STATE_FILE), {})

        events = []
        if isinstance(memory, list):
            events.extend(memory)

        if isinstance(agent_history, list):
            for item in agent_history[-80:]:
                events.append(
                    {
                        "event_type": "agent_history",
                        "timestamp": item.get("timestamp"),
                        "region": str(item.get("target_region") or "global").lower(),
                        "weights": _compact_params(item.get("weights")),
                        "explanations": item.get("explanations"),
                        "outcome": "historical_unvalidated",
                    }
                )

        if isinstance(state, dict):
            for reg, info in state.items():
                if region_key not in ("global", str(reg).lower()):
                    continue
                for item in (info.get("history") or [])[-50:]:
                    events.append(
                        {
                            "event_type": item.get("event") or "model_calibrator_step",
                            "timestamp": item.get("timestamp"),
                            "region": str(reg).lower(),
                            "metric": _metric_from_trigger(item.get("trigger", "")),
                            "trigger": item.get("trigger"),
                            "old_params": _compact_params(item.get("old_params") or item.get("params_before")),
                            "new_params": _compact_params(item.get("new_params") or item.get("params_after")),
                            "outcome": "rollback" if item.get("event") == "full_rollback" else "applied",
                        }
                    )

        filtered = []
        for event in events:
            event_region = str(event.get("region") or "global").lower()
            event_metric = event.get("metric") or _metric_from_trigger(event.get("trigger", ""))
            if region_key not in ("global", event_region):
                continue
            if metric and event_metric and metric != event_metric:
                continue
            event = dict(event)
            event["semantic_tags"] = sorted(set(event.get("semantic_tags") or _semantic_tags(event)))
            filtered.append(event)

        filtered.sort(key=lambda e: str(e.get("timestamp") or ""))
        recent = filtered[-limit:]

        positive = [
            e for e in filtered
            if e.get("outcome") in ("improved", "rollback")
            or (isinstance(e.get("improvement_pct"), (int, float)) and e.get("improvement_pct") > 0)
        ]
        negative = [
            e for e in filtered
            if e.get("outcome") in ("degraded", "no_improvement", "critical")
            or (isinstance(e.get("improvement_pct"), (int, float)) and e.get("improvement_pct") <= 0)
        ]

        tag_scores = Counter()
        tag_outcomes = defaultdict(lambda: {"positive": 0, "negative": 0})
        for e in positive:
            for tag in e.get("semantic_tags", []):
                tag_scores[tag] += 1
                tag_outcomes[tag]["positive"] += 1
        for e in negative:
            for tag in e.get("semantic_tags", []):
                tag_scores[tag] -= 1
                tag_outcomes[tag]["negative"] += 1

        return {
            "region": region_key,
            "metric": metric,
            "events_considered": len(filtered),
            "recent_events": recent,
            "positive_patterns": self._compact_patterns(positive[-limit:]),
            "negative_patterns": self._compact_patterns(negative[-limit:]),
            "semantic_tag_scores": [
                {"tag": tag, "score": score, **tag_outcomes[tag]}
                for tag, score in tag_scores.most_common(12)
            ],
        }

    def format_for_prompt(self, region: str = "global", metric: Optional[str] = None) -> str:
        context = self.build_context(region=region, metric=metric)
        if context["events_considered"] == 0:
            return "Sem memoria historica validada para esta regiao/metrica."
        return json.dumps(context, ensure_ascii=False)

    @staticmethod
    def _compact_patterns(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns = []
        for e in events:
            params = e.get("new_params") or e.get("weights") or {}
            patterns.append(
                {
                    "timestamp": e.get("timestamp"),
                    "region": e.get("region"),
                    "metric": e.get("metric"),
                    "outcome": e.get("outcome"),
                    "improvement_pct": e.get("improvement_pct"),
                    "tags": e.get("semantic_tags", [])[:6],
                    "params": _compact_params(params),
                    "summary": str(e.get("explanations") or e.get("trigger") or "")[:220],
                }
            )
        return patterns
