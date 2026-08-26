from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np

import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

# --- BLINDAGEM DE ENCODING PARA WINDOWS ---
import builtins
_original_print = builtins.print
def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                new_args.append(arg.encode('ascii', 'ignore').decode('ascii'))
            else:
                new_args.append(arg)
        _original_print(*new_args, **kwargs)
builtins.print = safe_print
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import pickle
import json
import shutil
import subprocess
import warnings
import logging
import unicodedata
import math
import difflib
from datetime import datetime, timedelta
import re
from shapely.geometry import Point
from shapely.geometry import mapping
from shapely.geometry import shape
from shapely.ops import unary_union

# --- Orquestrador Regional Híbrido ---
try:
    from src.core.orchestrator import StateOrchestrator, normalize_name
    from src.core.efficiency_monitor import EfficiencyMonitor
    from src.core.validation_logger import append_validation_log
    orchestrator = None 
except ImportError:
    # Fallback se o PYTHONPATH não incluir a raiz corretamente
    import sys
    sys.path.append(os.getcwd())
    from src.core.orchestrator import StateOrchestrator
    from src.core.efficiency_monitor import EfficiencyMonitor
    from src.core.validation_logger import append_validation_log
    def normalize_name(text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
        import re
        return re.sub(r'\s*-\s*AIS.*$', '', text).strip()

# --- Champion/Challenger LGBM Lean (Sentinela V3) ---
try:
    from src.core.champion_challenger import ChampionChallenger
except ImportError:
    ChampionChallenger = None

# --- API V4 (Inteligência Granular 500m) ---
try:
    from src.core.api_v4_routes import create_v4_api_blueprint
except ImportError:
    create_v4_api_blueprint = None

warnings.filterwarnings('ignore')
# Configurando logs para garantir visibilidade no terminal
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_NAME = "Poisson Ranker Estadual"
DEFAULT_MODEL_MODE = "stgat_v5"
DEFAULT_MODEL_LABEL = "ST-GAT v5 (DeepSTGAT_v5 Ativo)"
STATIC_EXPORT_SCRIPT = os.path.join(BASE_DIR, 'scripts', 'export_static_snapshot.py')
STATIC_EXPORT_OUTPUT_DIR = os.path.normpath(
    os.environ.get('STATIC_EXPORT_OUTPUT_DIR', os.path.join(BASE_DIR, 'static_export', 'data'))
)
STATIC_SCREENSHOT_REPO_DIR = os.path.normpath(
    os.environ.get('STATIC_SCREENSHOT_REPO_DIR', os.path.join(BASE_DIR, '..', 'screenshot-report_preview'))
)
STATIC_SCREENSHOT_PUBLIC_DATA_DIR = os.path.join(STATIC_SCREENSHOT_REPO_DIR, 'public', 'data')

# Cache file for manager-harmonized explanations
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'manager_explanations_cache.json')

# Metadados de região — populados ao carregar o orquestrador (sem hardcode)
_RMF_NODES: set = set()          # nomes normalizados dos nós RMF
_ALL_REGIONS: list = ['fortaleza', 'rmf', 'interior']  # atualizado após load
REGION_LABELS: dict = {          # rótulos de exibição por região
    'fortaleza': 'FORTALEZA (CAPITAL)',
    'rmf': 'REGIÃO METROPOLITANA',
    'interior': 'INTERIOR DO ESTADO',
}

# Champion/Challenger — inicializado no startup
champion_challenger = None

RISK_SCORE_THRESHOLDS = {
    'critical_min': 71.0,
    'high_min': 51.0,
    'moderate_min': 31.0,
}

EXOGENOUS_WINDOW_DAYS = 7
SUPPRESSION_HALF_LIFE_HOURS = 36.0

QUALIFIED_SUPPRESSION_KEYWORDS = (
    'TRAFICO', 'TRÁFICO', 'ENTORPEC', 'DROGA', 'APREENS', 'ARMA', 'FUZIL',
    'PISTOLA', 'REVOLV', 'PORTE ILEGAL', 'MANDADO DE PRIS', 'FLAGRANTE',
    'DESARTIC', 'LIDER', 'LIDERANCA', 'CHEFE', 'COMANDO'
)

ADMIN_POLICE_KEYWORDS = (
    'VEICULO LOCALIZADO', 'VEÍCULO LOCALIZADO', 'RECUPERAD', 'LOCALIZAD',
    'CELULAR COM ALERTA', 'CELULAR COM QUEIXA', 'RECEPTA', 'CONDUZID',
    'PESSOA SITUACAO SUSPEITA', 'PESSOA SITUAÇÃO SUSPEITA',
    'DIRECAO PERIGOSA', 'DIREÇÃO PERIGOSA', 'CLONADO', 'QUEIXA DE ROUBO'
)

CONFLICT_EVENT_KEYWORDS = (
    'HOMICID', 'CHACINA', 'EXECUC', 'LESAO A BALA', 'LESÃO A BALA',
    'TORTURA', 'DESLOCAMENTO FORCADO', 'DESLOCAMENTO FORÇADO',
    'EXPULSAO DE MORADORES', 'EXPULSÃO DE MORADORES', 'ACHADO DE CADAVER',
    'ACHADO DE CADÁVER'
)

RISK_STYLE_BY_LEVEL = {
    'crítico': ('CRÍTICO', 'risk-critico', '#8B0000'),
    'alto': ('ALTO', 'risk-alto', '#E63946'),
    'moderado': ('MODERADO', 'risk-moderado', '#F4A261'),
    'baixo': ('BAIXO', 'risk-baixo', '#A8DADC'),
}


def normalize_risk_score(score) -> float:
    """Normaliza score percentual para evitar artefatos de precisão próximos aos cortes."""
    try:
        value = float(score)
    except Exception:
        value = 0.0
    if np.isnan(value) or np.isinf(value):
        value = 0.0
    return round(max(0.0, min(100.0, value)), 1)


def classify_risk_score(score):
    """Classificação oficial de risco: baixo <=30, moderado 31-50, alto 51-70, crítico >=71."""
    normalized = normalize_risk_score(score)
    if normalized >= RISK_SCORE_THRESHOLDS['critical_min']:
        level = 'crítico'
    elif normalized >= RISK_SCORE_THRESHOLDS['high_min']:
        level = 'alto'
    elif normalized >= RISK_SCORE_THRESHOLDS['moderate_min']:
        level = 'moderado'
    else:
        level = 'baixo'

    status, css, color = RISK_STYLE_BY_LEVEL[level]
    return level, status, css, color, normalized


def get_risk_thresholds_meta():
    return {
        'critical_min': RISK_SCORE_THRESHOLDS['critical_min'],
        'high_min': RISK_SCORE_THRESHOLDS['high_min'],
        'moderate_min': RISK_SCORE_THRESHOLDS['moderate_min'],
        'low_max': RISK_SCORE_THRESHOLDS['moderate_min'] - 1,
    }


def parse_event_datetime(event: dict):
    for field in ('date', 'event_date', 'ingested_at'):
        value = event.get(field)
        if not value:
            continue
        try:
            return datetime.fromisoformat(str(value)[:19])
        except Exception:
            try:
                return datetime.strptime(str(value)[:10], '%Y-%m-%d')
            except Exception:
                continue
    return None


def classify_exogenous_event(event: dict):
    text = ' '.join([
        str(event.get('type') or ''),
        str(event.get('natureza') or ''),
        str(event.get('description') or ''),
        str(event.get('descricao') or ''),
        str(event.get('resumo') or ''),
        str(event.get('raw_text') or ''),
    ]).upper()
    base_suppression = bool(event.get('is_suppression', False))
    is_conflict = any(keyword in text for keyword in CONFLICT_EVENT_KEYWORDS)
    has_qualified_signal = any(keyword in text for keyword in QUALIFIED_SUPPRESSION_KEYWORDS)
    has_admin_signal = any(keyword in text for keyword in ADMIN_POLICE_KEYWORDS)
    has_arrest_only = any(keyword in text for keyword in ('PRESO', 'PRESA', 'PRISAO', 'PRISÃO', 'DETIDO'))

    if is_conflict:
        return {
            'is_conflict': True,
            'is_suppression': False,
            'is_qualified_suppression': False,
            'signal_class': 'conflict',
        }

    if has_qualified_signal or ((base_suppression or has_arrest_only) and not has_admin_signal):
        return {
            'is_conflict': False,
            'is_suppression': True,
            'is_qualified_suppression': True,
            'signal_class': 'qualified_suppression',
        }

    if has_admin_signal or base_suppression:
        return {
            'is_conflict': False,
            'is_suppression': False,
            'is_qualified_suppression': False,
            'signal_class': 'administrative_police',
        }

    return {
        'is_conflict': False,
        'is_suppression': False,
        'is_qualified_suppression': False,
        'signal_class': 'neutral',
    }


def suppression_decay_factor(event_dt):
    if event_dt is None:
        return 1.0
    age_hours = max(0.0, (datetime.now() - event_dt).total_seconds() / 3600.0)
    return 0.5 ** (age_hours / SUPPRESSION_HALF_LIFE_HOURS)

import threading
import time

nodes_gdf = None
orchestrator = None
efficiency_monitor = None
health_monitor = None
confidence_tracker = None

_MICRONODE_POLYGON_CACHE = None
_MICRONODE_REFERENCE_CACHE = None
_TOP_MICRONODE_FACTION_CACHE = None
_PEAK_HOURS_CACHE = None  # {bairro_norm: "Entre XHs e YHs"}
_MICRONODE_EXPORT_BUILT_AT = None
_MICRONODE_EXPORT_LOCK = threading.Lock()

_API_RISK_CACHE = {}
_API_RISK_CACHE_LOCK = threading.Lock()
_CVLI_OPTIONAL_MODELS_CACHE = None

def invalidate_api_risk_cache():
    global _API_RISK_CACHE
    with _API_RISK_CACHE_LOCK:
        _API_RISK_CACHE = {}


def normalize_model_mode(value) -> str:
    mode = str(value or DEFAULT_MODEL_MODE).strip().lower()
    aliases = {
        'stgat': 'stgat',
        'default': DEFAULT_MODEL_MODE,
        'stgat_v5': DEFAULT_MODEL_MODE,
        'deepstgat_v5': DEFAULT_MODEL_MODE,
        'v5': DEFAULT_MODEL_MODE,
        'cvli_tactical': 'cvli_tactical',
        'cvli_tactical_only': 'cvli_tactical',
        'tactical': 'cvli_tactical',
        'short20_mix': 'short20_mix',
        'short20': 'short20_mix',
        'shortlist20': 'short20_mix',
    }
    return aliases.get(mode, DEFAULT_MODEL_MODE)


def _load_optional_cvli_models():
    global _CVLI_OPTIONAL_MODELS_CACHE
    if _CVLI_OPTIONAL_MODELS_CACHE is not None:
        return _CVLI_OPTIONAL_MODELS_CACHE

    rankings_path = os.path.join(BASE_DIR, 'outputs', 'cvli_first_candidate_rankings.json')
    metadata_path = os.path.join(BASE_DIR, 'outputs', 'cvli_first_candidate_metadata.json')
    tactical_summary_path = os.path.join(BASE_DIR, 'outputs', 'cvli_first_architecture_summary.json')
    shortlist_summary_path = os.path.join(BASE_DIR, 'outputs', 'cvli_shortlist_rerank_summary.json')

    optional_models = {
        'stgat_v5': {
            'mode': 'stgat_v5',
            'label': DEFAULT_MODEL_LABEL,
            'description': 'Modelo padrão DeepSTGAT_v5 retreinado, com atenção por aresta e Binomial Negativa.',
            'kind': 'champion',
            'fortaleza_scores': {},
            'metrics': None,
        },
        'stgat': {
            'mode': 'stgat',
            'label': 'Poisson Ranker (Legacy)',
            'description': 'Ranking Poisson legado disponível para comparação operacional.',
            'kind': 'legacy',
            'fortaleza_scores': {},
            'metrics': None,
        }
    }

    if not os.path.exists(rankings_path):
        _CVLI_OPTIONAL_MODELS_CACHE = optional_models
        return _CVLI_OPTIONAL_MODELS_CACHE

    try:
        with open(rankings_path, 'r', encoding='utf-8') as fh:
            rankings = json.load(fh) or []
    except Exception:
        rankings = []

    try:
        with open(metadata_path, 'r', encoding='utf-8') as fh:
            metadata = json.load(fh) or {}
    except Exception:
        metadata = {}

    summary_by_family = {}
    for path in (tactical_summary_path, shortlist_summary_path):
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as fh:
                    for row in (json.load(fh) or []):
                        family = str(row.get('family') or '').strip().upper()
                        if family:
                            summary_by_family[family] = row
        except Exception:
            continue

    family_to_mode = {
        'CVLI_TACTICAL_ONLY': 'cvli_tactical',
        'SHORT20_MIX': 'short20_mix',
    }
    mode_meta = {
        'cvli_tactical': {
            'label': 'CVLI Tático',
            'description': 'Foca no momentum recente de CVLI para maximizar acerto no top 10.',
            'kind': 'experimental',
        },
        'short20_mix': {
            'label': 'CVLI Shortlist 20',
            'description': 'Shortlist tática com ajuste estrutural leve para equilibrar topo e cobertura.',
            'kind': 'experimental',
        },
    }

    grouped_scores = {mode: [] for mode in mode_meta}
    for row in rankings:
        family = str(row.get('modelo') or '').strip().upper()
        mode = family_to_mode.get(family)
        if not mode:
            continue
        bairro = normalize_name(row.get('bairro') or '')
        if not bairro:
            continue
        grouped_scores[mode].append({
            'bairro': bairro,
            'rank': int(row.get('rank') or 999),
            'raw_score': float(row.get('score') or 0.0),
        })

    for family, mode in family_to_mode.items():
        items = grouped_scores.get(mode) or []
        if not items:
            continue
        raw_values = [item['raw_score'] for item in items]
        raw_min = min(raw_values)
        raw_max = max(raw_values)
        fortaleza_scores = {}
        for item in items:
            if raw_max > raw_min:
                scaled_raw = 15.0 + 85.0 * ((item['raw_score'] - raw_min) / (raw_max - raw_min))
            else:
                scaled_raw = 50.0
            rank_score = max(18.0, 100.0 - ((item['rank'] - 1) * 2.1))
            score_pct = round((0.7 * scaled_raw) + (0.3 * rank_score), 1)
            fortaleza_scores[item['bairro']] = score_pct

        metrics = summary_by_family.get(family, {})
        optional_models[mode] = {
            'mode': mode,
            'label': mode_meta[mode]['label'],
            'description': mode_meta[mode]['description'],
            'kind': mode_meta[mode]['kind'],
            'fortaleza_scores': fortaleza_scores,
            'metrics': {
                'p10': metrics.get('p10'),
                'p20': metrics.get('p20'),
                'r10': metrics.get('r10'),
                'r20': metrics.get('r20'),
            },
            'reference_date': metadata.get('reference_date'),
            'prediction_window': {
                'start': metadata.get('prediction_start'),
                'end': metadata.get('prediction_end'),
            },
        }

    _CVLI_OPTIONAL_MODELS_CACHE = optional_models
    return _CVLI_OPTIONAL_MODELS_CACHE


def _get_model_selection_meta(selected_mode: str):
    optional_models = _load_optional_cvli_models()
    selected = optional_models.get(selected_mode) or optional_models[DEFAULT_MODEL_MODE]
    available = []
    for mode in (DEFAULT_MODEL_MODE, 'stgat', 'cvli_tactical', 'short20_mix'):
        model = optional_models.get(mode)
        if not model:
            continue
        available.append({
            'mode': mode,
            'label': model.get('label'),
            'description': model.get('description'),
            'kind': model.get('kind'),
            'metrics': model.get('metrics'),
        })
    return selected, available


def _resolve_optional_model_score(name_norm: str, fortaleza_scores: dict):
    if not fortaleza_scores:
        return None
    exact = fortaleza_scores.get(name_norm)
    if exact is not None:
        return exact
    matches = difflib.get_close_matches(name_norm, list(fortaleza_scores.keys()), n=1, cutoff=0.88)
    if matches:
        return fortaleza_scores.get(matches[0])
    return None


def _score_map_for_model_mode(model_mode: str, exogenous_shocks=None, return_trends=False):
    if model_mode == 'stgat_v5' and hasattr(orchestrator, 'get_combined_risk_stgat_v5'):
        return orchestrator.get_combined_risk_stgat_v5(exogenous_shocks, return_trends=return_trends)
    return orchestrator.get_combined_risk(exogenous_shocks, return_trends=return_trends)


# Cache global para explicabilidade para evitar redundância de I/O em loops
_EXOGENOUS_EVENTS_CACHE = None
_RUAS_CRITICAS_FORTALEZA_CACHE = None
_STREETS_BY_MUNICIPIO_CACHE = None

# Status da exportação de snapshot assíncrona
_SNAPSHOT_EXPORT_STATUS = {
    'status': 'idle',
    'error': None,
    'last_run': None,
    'copied_count': 0
}
_SNAPSHOT_EXPORT_LOCK = threading.Lock()
_MODEL_UPDATE_STATUS_LOCK = threading.Lock()
_MODEL_UPDATE_STATUS = {
    'status': 'idle',
    'progress': 0,
    'message': None,
    'error': None,
    'revision': 0,
    'updated_at': None,
    'expires_at': None,
}


def _set_model_update_status(status='idle', progress=0, message=None,
                             error=None, bump_revision=False, ttl_seconds=20):
    now = datetime.now()
    with _MODEL_UPDATE_STATUS_LOCK:
        _MODEL_UPDATE_STATUS['status'] = status
        _MODEL_UPDATE_STATUS['progress'] = progress
        _MODEL_UPDATE_STATUS['message'] = message
        _MODEL_UPDATE_STATUS['error'] = error
        _MODEL_UPDATE_STATUS['updated_at'] = now.isoformat()
        _MODEL_UPDATE_STATUS['expires_at'] = (now.timestamp() + ttl_seconds) if status != 'idle' else None
        if bump_revision:
            _MODEL_UPDATE_STATUS['revision'] += 1


def _get_model_update_status_payload():
    with _MODEL_UPDATE_STATUS_LOCK:
        payload = dict(_MODEL_UPDATE_STATUS)

    expires_at = payload.get('expires_at')
    if payload['status'] != 'idle' and expires_at and datetime.now().timestamp() > expires_at:
        _set_model_update_status(status='idle', progress=0, message=None, error=None, ttl_seconds=0)
        with _MODEL_UPDATE_STATUS_LOCK:
            payload = dict(_MODEL_UPDATE_STATUS)
    return payload

# === REGISTRAR HEALTH MONITOR BLUEPRINT (antes de load_data_and_models) ===
model_calibrator = None
auto_calibrator_daemon = None
try:
    from src.core.health_monitor import HealthMonitor, ConfidenceTracker
    from src.core.admin_health_routes import create_admin_health_blueprint
    from src.core.model_calibrator import ModelCalibrator
    
    health_monitor = HealthMonitor(base_dir=BASE_DIR)
    confidence_tracker = ConfidenceTracker(base_dir=BASE_DIR)
    model_calibrator = ModelCalibrator(base_dir=BASE_DIR, health_monitor=health_monitor)
    # Popular confidence_tracker com histórico já existente do efficiency_monitor
    efficiency_history_path = os.path.join(BASE_DIR, 'logs', 'efficiency_history.json')
    confidence_tracker.seed_from_efficiency_history(efficiency_history_path)
    admin_bp = create_admin_health_blueprint(
        health_monitor, confidence_tracker, model_calibrator,
        auto_calibrator_daemon=None,
        get_orchestrator=lambda: orchestrator
    )
    app.register_blueprint(admin_bp)
    print("✅ Admin Dashboard Registrado em /api/admin/health")
    
    print("Auto-ajuste deterministico ativo; agente Ollama desativado.")

    # Registro API V4 (Sentinela Granular)
    if create_v4_api_blueprint:
        v4_bp = create_v4_api_blueprint(BASE_DIR)
        app.register_blueprint(v4_bp)
        print("🚀 API Sentinela V4 (500m) Registrada em /api/v4")
    
    # Thread de checagem periódica de saúde (a cada 5 minutos)
    def _run_health_checks():
        time.sleep(30)  # aguarda o app inicializar
        while True:
            try:
                health_monitor.check_system_health()
            except Exception:
                pass
            time.sleep(300)  # 5 minutos
    
    threading.Thread(target=_run_health_checks, daemon=True).start()
except ImportError:
    print("⚠️ Health Monitor não disponível. Instale psutil: pip install psutil")
except Exception as e:
    print(f"⚠️ Erro ao registrar Health Monitor: {e}")

# Limiar de cobertura: % mínima de territórios de facção que deve estar no top-20% do ranking.
# Se cair abaixo disso, o modelo está "esquecendo" zonas de tensão conhecidas.
_FACTION_COVERAGE_MIN = 0.80  # 80% dos territórios de facção devem aparecer no top-20%


def _resolve_territorial_alerts_for_region(region_name: str, now: datetime, reason: str):
    if health_monitor is None:
        return 0
    stale_types = {
        f"faction_coverage_{region_name}",
        f"calibration_maxed_{region_name}",
        f"auto_calibration_{region_name}",
    }
    resolved_count = 0
    for alert in health_monitor.alerts_history:
        if alert.get('type') in stale_types and not alert.get('resolved'):
            alert['resolved'] = True
            alert['resolved_at'] = now.isoformat()
            alert['resolved_reason'] = reason
            resolved_count += 1
    if resolved_count:
        health_monitor._save_history()
    return resolved_count


def _select_faction_targets(nodes_df, target_count: int):
    faction_candidates = []
    for _, row in nodes_df.iterrows():
        faction = str(row.get('faction', 'NEUTRO')).upper()
        if faction in ('NEUTRO', 'N/A', '', 'NAN', 'NONE'):
            continue
        tension_score = float(row.get('tension_index', 0.0) or 0.0)
        historical_cvli = float(row.get('total_cvli', row.get('recent_cvli', 0.0)) or 0.0)
        faction_candidates.append((normalize_name(str(row['name'])), tension_score, historical_cvli))

    if not faction_candidates:
        return []

    faction_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    unique_targets = []
    seen = set()
    for name_norm, _, _ in faction_candidates:
        if name_norm in seen:
            continue
        seen.add(name_norm)
        unique_targets.append(name_norm)
        if len(unique_targets) >= target_count:
            break
    return unique_targets


def _effective_faction_coverage_threshold(target_count: int) -> float:
    if target_count <= 0:
        return _FACTION_COVERAGE_MIN
    required_hits = max(2, round(target_count * _FACTION_COVERAGE_MIN))
    required_hits = min(target_count, required_hits)
    return required_hits / target_count

def _check_faction_coverage_alerts(metrics: dict):
    """
    Régua territorial por facção desativada.

    Alertas do health dashboard devem ser guiados pelas métricas regionais de
    performance (P10/P20/Recall), não pela simples presença de territórios
    faccionados no topo do ranking. Território faccionado calmo não é, por si só,
    sinal de degradação do modelo.

    Esta rotina agora apenas resolve alertas/calibrações legadas desse domínio.
    """
    from datetime import datetime
    if orchestrator is None or health_monitor is None:
        return

    now = datetime.now()
    for r_name in orchestrator.specialists.keys():
        resolved_count = _resolve_territorial_alerts_for_region(
            r_name,
            now,
            f"Região {r_name.upper()} reavaliada: alertas territoriais por facção foram desativados. Health dashboard agora deve seguir métricas regionais P10/P20."
        )
        if resolved_count:
            print(f"✅ [Cobertura Territorial] {r_name.upper()}: {resolved_count} alerta(s) legados resolvidos.")
        if model_calibrator is not None:
            reg_state = model_calibrator.state.get(r_name, {})
            if reg_state.get('steps', 0) > 0:
                model_calibrator.on_recovery(orchestrator, r_name, 'faction_coverage', 1.0)


def run_background_efficiency_monitor():
    """Tarefa em background que executa a cada 7 dias ou no start."""
    global efficiency_monitor
    # Aguarda o sistema inicializar completamente
    time.sleep(10)
    while True:
        if efficiency_monitor is not None:
            try:
                num_loc = len(nodes_gdf) if nodes_gdf is not None else 0
                print(f"\n" + "="*60)
                print(f"🛡️  [MONITOR DE EFICIÊNCIA] Iniciando Avaliação ({num_loc} localidades)")
                print("="*60)
                
                metrics = efficiency_monitor.run_evaluation()
                
                if metrics:
                    print(f"📅 Data da Avaliação: {metrics.get('date')}")
                    print(f"📊 Eventos Detectados: {metrics.get('total_events', 0)} ({metrics.get('brute_cvli', 0)} Brutos + {metrics.get('exogenous', 0)} Exógenos)")
                    
                    # Atualizar confidence_tracker com os novos dados
                    if confidence_tracker is not None:
                        try:
                            eval_date = metrics.get('date', datetime.now().date().isoformat())
                            global_data = metrics.get('global', {})
                            global_metrics = {
                                'p10': global_data.get('p10') or 0,
                                'p20': global_data.get('p20') or 0,
                                'precision': global_data.get('p10') or 0,
                                'recall': global_data.get('recall20') or global_data.get('p20') or 0,
                                'recall10': global_data.get('recall10') or 0,
                                'recall20': global_data.get('recall20') or 0,
                                'active_locations': global_data.get('active_locations') or 0,
                                'total_nodes': global_data.get('total_nodes') or 0,
                                'total_events': global_data.get('total_events') or 0,
                                'assigned_total_events': metrics.get('assigned_total_events') or 0,
                                'unmapped_total_events': metrics.get('unmapped_total_events') or 0,
                                'f1_score': 0.0
                            }
                            p, r = global_metrics['precision'], global_metrics['recall']
                            if p + r > 0:
                                global_metrics['f1_score'] = round(2 * p * r / (p + r), 4)
                            region_metrics = {}
                            for reg in (orchestrator.specialists.keys() if orchestrator else _ALL_REGIONS):
                                reg_data = metrics.get(reg, {})
                                if reg_data and isinstance(reg_data, dict):
                                    p10_score = reg_data.get('p10') or 0
                                    region_metrics[reg] = {
                                        'p10': p10_score,
                                        'p20': reg_data.get('p20') or 0,
                                        'precision': p10_score,
                                        'recall': reg_data.get('recall20') or reg_data.get('p20') or 0,
                                        'recall10': reg_data.get('recall10') or 0,
                                        'recall20': reg_data.get('recall20') or 0,
                                        'active_locations': reg_data.get('active_locations') or 0,
                                        'total_nodes': reg_data.get('total_nodes') or 0,
                                        'total_events': reg_data.get('total_events') or 0,
                                        'f1_score': 0.0
                                    }
                                    
                                    # --- ATUALIZAÇÃO: AUTO-CURRICULUM TEMPORAL (Shrinkage) ---
                                    # Orquestrador reage dinamicamente a quedas de eficiência em tempo de produção
                                    if orchestrator is not None:
                                        orchestrator.adjust_temporal_focus(reg, p10_score)
                                        
                            confidence_tracker.record_evaluation(eval_date, global_metrics, region_metrics)
                            
                            # === ALERTAS DE COBERTURA TERRITORIAL (termômetro de tensão) ===
                            if health_monitor is not None:
                                _check_faction_coverage_alerts(metrics)
                        except Exception as ct_err:
                            import traceback
                            traceback.print_exc()
                            print(f"⚠️ Erro ao atualizar confidence_tracker: {ct_err}")
                    print(f"📅 Data da Avaliação: {metrics.get('date')}")
                    print(f"📊 Eventos Detectados: {metrics.get('total_events') or 0} ({(metrics.get('brute_cvli') or 0)} Brutos + {(metrics.get('exogenous') or 0)} Exógenos)")
                    
                    # Exibir Global
                    if 'global' in metrics:
                        m = metrics['global']
                        print(f"\n🌍 REGIONALIZAÇÃO: GLOBAL")
                        print(f"   P5:  {(m.get('p5') or 0)*100:.1f}% | Hits: {', '.join(m.get('hits5') or [])}")
                        print(f"   P10: {(m.get('p10') or 0)*100:.1f}% | Hits: {', '.join(m.get('hits10') or [])}")
                        print(f"   P20: {(m.get('p20') or 0)*100:.1f}% | R20: {(m.get('recall20') or 0)*100:.1f}% | Hits: {', '.join(m.get('hits20') or [])}")
                    
                    # Exibir Fortaleza
                    if 'fortaleza' in metrics:
                        m = metrics['fortaleza']
                        print(f"\n🏙️  REGIONALIZAÇÃO: FORTALEZA")
                        print(f"   P10: {(m.get('p10') or 0)*100:.1f}% | Hits: {', '.join(m.get('hits10') or [])}")
                        print(f"   P20: {(m.get('p20') or 0)*100:.1f}% | R20: {(m.get('recall20') or 0)*100:.1f}% | Hits: {', '.join(m.get('hits20') or [])}")
                    
                    # Exibir demais regiões se houver acertos
                    for reg in (orchestrator.specialists.keys() if orchestrator else _ALL_REGIONS):
                        if reg == 'fortaleza': continue
                        if reg in metrics and (metrics[reg].get('p10') or 0) > 0:
                            m = metrics[reg]
                            reg_name = REGION_LABELS.get(reg, reg.upper())
                            print(f"\n📍 REGIONALIZAÇÃO: {reg_name}")
                            print(f"   P10: {(m.get('p10') or 0)*100:.1f}% | Hits: {', '.join(m.get('hits10') or [])}")
                    
                    print("\n" + "="*60 + "\n")
                else:
                    print("📊 [Monitor] Sem eventos suficientes para avaliação hoje.")
            except Exception as e:
                print(f"⚠️ [Monitor] Erro na thread de eficiência: {e}")
        
        # Reavaliação a cada 30 minutos — auto-ajuste reativo em tempo de produção
        time.sleep(1800)

def verify_date_consistency(event_date_str, last_base_date):
    """
    Verifica a consistência temporal.
    
    MODO PROTÓTIPO (Data Lag Tolerance):
    Aceita o evento se:
    1. For anterior ou igual à base do modelo (Consistência Histórica)
    2. OU Se for posterior à base mas anterior/igual a HOJE (Preenchimento do Gap de Atraso)
    
    Rejeita apenas se for > HOJE (Futuro Real).
    """
    if not event_date_str:
        return True # Sem data, aceita por segurança
        
    try:
        # Conversão robusta de strings
        if isinstance(event_date_str, str):
            e_date = datetime.strptime(event_date_str[:10], '%Y-%m-%d').date()
        elif hasattr(event_date_str, 'date'):
            e_date = event_date_str.date()
        else:
            e_date = event_date_str
            
        # Data de Hoje (Limite do Real)
        today = datetime.now().date()
        
        # Se o evento é futuro em relação ao tempo real, rejeita sempre
        if e_date > today:
            return False
            
        # Se não temos last_base_date, aceitamos pois é <= today
        if not last_base_date:
            return True
            
        # Lógica Original Estrita (Comentada para o Protótipo)
        # if isinstance(last_base_date, str):
        #     b_date = datetime.strptime(last_base_date[:10], '%Y-%m-%d').date()
        # elif hasattr(last_base_date, 'date'):
        #     b_date = last_base_date.date()
        # else:
        #     b_date = last_base_date
        # return e_date <= b_date
        
        # Lógica de Tolerância de Atraso (Prototype Mode)
        # Aceitamos o evento pois ele representa a realidade atual sobreposta ao modelo defasado
        return True 

    except Exception:
        return True # Em caso de erro, permitimos a inclusão

def archive_old_exogenous_events():
    """
    Cria um 'arquivo morto' dos eventos exógenos que ultrapassam os últimos 7 dias.
    O arquivo é salvo como 'data/exogenous_events_(data_limite).json'.

    Correção: alguns registros não tinham campo `date` mas tinham `ingested_at` e `timestamp` (hora).
    Neste caso, combinamos a data de `ingested_at` com o `timestamp` da ocorrência para obter a data
    do evento. Se não for possível extrair uma data, consideramos o evento recente e NÃO o arquivamos.
    """
    exogenous_file = os.path.join(BASE_DIR, "data", "exogenous_events.json")
    if not os.path.exists(exogenous_file):
        return

    try:
        with open(exogenous_file, 'r', encoding='utf-8') as f:
            events = json.load(f)

        if not events:
            return

        cutoff_date = (datetime.now() - timedelta(days=7)).date()

        old_events = []
        current_events = []

        for e in events:
            event_date = None

            # 1) Prefer explicit 'date' field if it's a full date string (YYYY-MM-DD)
            dval = e.get('date') or e.get('event_date')
            if isinstance(dval, str):
                try:
                    # if it's only a time like '22:10', skip here
                    if re.match(r'^\d{2}:\d{2}$', dval.strip()):
                        dval = None
                    else:
                        # accept ISO-like strings, take leading YYYY-MM-DD
                        event_date = datetime.strptime(dval.strip()[:10], '%Y-%m-%d').date()
                except Exception:
                    event_date = None

            # 2) If no full date, try combining 'ingested_at' (datetime) with 'timestamp' (HH:MM)
            if event_date is None:
                ing = e.get('ingested_at')
                ts = e.get('timestamp')  # e.g. '22:10'
                try:
                    ing_dt = None
                    if isinstance(ing, str) and ing:
                        try:
                            ing_dt = datetime.strptime(ing.strip(), '%Y-%m-%d %H:%M:%S')
                        except Exception:
                            try:
                                ing_dt = datetime.fromisoformat(ing.strip())
                            except Exception:
                                ing_dt = None

                    if ing_dt:
                        # If timestamp present and looks like HH:MM, we could combine,
                        # but for archiving we only need the date portion.
                        event_date = ing_dt.date()
                except Exception:
                    event_date = None

            # 3) If still no date, treat as recent (do not archive)
            if event_date is None:
                current_events.append(e)
                continue

            # Decide to archive or keep
            if event_date < cutoff_date:
                old_events.append(e)
            else:
                current_events.append(e)

        if old_events:
            # Write archive files into a dedicated directory to avoid cluttering
            archives_dir = os.path.join(BASE_DIR, 'data', 'archives')
            os.makedirs(archives_dir, exist_ok=True)

            archive_filename = f"exogenous_events_{cutoff_date.isoformat()}.json"
            archive_path = os.path.join(archives_dir, archive_filename)

            # Write atomically: write to a temp file then replace
            try:
                tmp_path = archive_path + '.tmp'
                with open(tmp_path, 'w', encoding='utf-8') as af:
                    json.dump(old_events, af, indent=2, ensure_ascii=False)
                os.replace(tmp_path, archive_path)
            except Exception as e:
                print(f"⚠️ Erro ao gravar arquivo morto: {e}")

            # Update canonical events file atomically as well
            try:
                tmp_main = exogenous_file + '.tmp'
                with open(tmp_main, 'w', encoding='utf-8') as f:
                    json.dump(current_events, f, indent=2, ensure_ascii=False)
                os.replace(tmp_main, exogenous_file)
            except Exception as e:
                print(f"⚠️ Erro ao atualizar arquivo principal de eventos: {e}")

            print(f"📦 Arquivo morto criado: data/archives/{archive_filename} ({len(old_events)} eventos)")
            print(f"✅ Arquivo principal atualizado ({len(current_events)} eventos ativos)")

    except Exception as e:
        print(f"⚠️ Erro ao arquivar eventos exógenos: {e}")

def generate_daily_ranking_report():
    """
    Gera um relatório Markdown diário com o Top 20 de cada região.
    Utilizado para acompanhamento manual de eficiência e auditoria.
    """
    if orchestrator is None or nodes_gdf is None:
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    base_log_dir = os.path.join(BASE_DIR, "logs", "rankings")
    os.makedirs(base_log_dir, exist_ok=True)

    # Calculamos o risco atual (sem shocks para servir de baseline estável ou com os atuais)
    scores_map = orchestrator.get_combined_risk()
    
    regions = {reg: REGION_LABELS.get(reg, reg.upper()) for reg in (orchestrator.specialists.keys() if orchestrator else _ALL_REGIONS)}

    for reg_key, reg_name in regions.items():
        filename = f"ranking_{today_str}_{reg_key}.md"
        filepath = os.path.join(base_log_dir, filename)

        # Se já existir o relatório de hoje, não sobrescrevemos (mantém o snapshot inicial)
        if os.path.exists(filepath):
            continue

        try:
            # Filtrar e ordenar bairros da região
            reg_results = []
            for i, row in nodes_gdf.iterrows():
                # Lógica de identificação de região similar ao api/risk
                r = str(row.get('regiao', 'fortaleza')).lower()
                if r == 'capital': r = 'fortaleza'
                
                name = str(row['name'])
                name_norm = normalize_name(name)
                
                # Sincronização RMF Oficial (via índice dinâmico do orquestrador)
                if name_norm in _RMF_NODES: r = 'rmf'
                
                if r == reg_key:
                    score = float(scores_map.get(name_norm, 20.0))
                    reg_results.append({
                        'name': name,
                        'score': score,
                        'faction': str(row.get('faction', 'N/A'))
                    })

            # Ordenar por Score (Top 50)
            reg_results.sort(key=lambda x: x['score'], reverse=True)
            top_20 = reg_results[:50]

            # Escrever o arquivo Markdown
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# 🛡️ Relatório de Risco Diário - {reg_name}\n")
                f.write(f"**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"**Estado da Base Histórica:** {orchestrator.dates[-1] if hasattr(orchestrator, 'dates') else 'N/A'}\n\n")
                f.write("| Pos | Localidade | Risco (%) | Facção Predominante |\n")
                f.write("|:---:|:---|:---:|:---:|\n")
                for idx, res in enumerate(top_20):
                    f.write(f"| {idx+1} | {res['name']} | {res['score']:.2f}% | {res['faction']} |\n")
                
                f.write(f"\n\n*Nota: Este ranking reflete o estado de inteligência do modelo no início do dia operacional.*")

            print(f"📄 Relatório gerado: {filename}")
        except Exception as e:
            print(f"⚠️ Erro ao gerar relatório {reg_key}: {e}")

def load_data_and_models():
    global nodes_gdf, orchestrator, efficiency_monitor, health_monitor, confidence_tracker
    global _EXOGENOUS_EVENTS_CACHE, _RUAS_CRITICAS_FORTALEZA_CACHE, _STREETS_BY_MUNICIPIO_CACHE
    export_worker = os.environ.get('REPORT_PREVIEW_EXPORT_MODE') == '1'
    
    # Invalida caches de explicabilidade para garantir recarga de novos dados
    _EXOGENOUS_EVENTS_CACHE = None
    _RUAS_CRITICAS_FORTALEZA_CACHE = None
    _STREETS_BY_MUNICIPIO_CACHE = None

    # Limpeza de eventos exógenos antigos
    if not export_worker:
        archive_old_exogenous_events()

    # --- ATUALIZAÇÃO OFICIAL DO ORCRIMS NO STARTUP (background) ---
    def _refresh_orcrim_background():
        try:
            from data.raw.inteligencia.import_orcrim_kml import refresh_orcrim_from_official
            refresh_result = refresh_orcrim_from_official()
            print(f"🧭 [ORCRIMS] Resultado do refresh em background: {refresh_result}")
        except Exception as e:
            print(f"⚠️ [ORCRIMS] Falha ao atualizar ORCRIMS em background: {e}")

    import threading
    if not export_worker:
        threading.Thread(target=_refresh_orcrim_background, daemon=True, name="orcrims-refresh").start()

    # --- ATUALIZAÇÃO DINÂMICA DE RUAS CRÍTICAS (CACHE GEO) ---
    try:
        if export_worker:
            raise RuntimeError('export worker skips street cache refresh')
        from scripts.gerar_geo_ruas_criticas import generate_geo_streets_dynamic
        print("🌐 Atualizando Cache de Ruas Críticas (Dinâmico 30 dias)...")
        generate_geo_streets_dynamic()
        print("✅ Cache Geográfico de Ruas Atualizado com Sucesso.")
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao atualizar cache de ruas: {e}")

    # --- LOCALIDADES CRÍTICAS POR MUNICÍPIO (RMF / INTERIOR) ---
    # Roda em background para não bloquear o startup.
    # Modo rápido (bairros + exógenos, sem geocoding) → resultado imediato.
    # O geocoding completo é disparado em thread separada e enriquece o arquivo incrementalmente.
    def _build_municipio_streets_bg():
        try:
            from scripts.gerar_streets_municipios import build as _build_mun
            mun_path = os.path.join(BASE_DIR, 'data', 'streets_by_municipio.json')
            # Primeiro passo: rápido (sem geocoding) — popula o arquivo imediatamente
            if not os.path.exists(mun_path):
                print("🗺️  Gerando localidades por município (modo rápido, 30 dias)...")
                _build_mun(fast_mode=True, days=30)
                print("✅ Localidades por município (rápido) geradas.")
            # Segundo passo: enriquecimento com geocoding — apenas últimos 30 dias
            print("🌐 Enriquecendo localidades com geocoding (background, 30 dias)...")
            _build_mun(fast_mode=False, days=30)
            print("✅ Localidades por município enriquecidas com geocoding.")
        except Exception as _e:
            print(f"⚠️ Erro ao gerar localidades por município: {_e}")

    if not export_worker:
        threading.Thread(target=_build_municipio_streets_bg, daemon=True).start()

    # Load all regional metadata (auto-discovery via glob)
    import glob as _glob
    dfs = []
    for path in sorted(_glob.glob(os.path.join(BASE_DIR, "data", "processed", "processed_*.pkl"))):
        reg = os.path.basename(path).replace('processed_', '').replace('.pkl', '')
        # Ignorar arquivo legado global (gerado com numpy 2.x incompatível); os três regionais o substituem
        if reg in ['graph_data_global', 'graph_data']:
            continue
        if os.path.exists(path):
            try:
                # Carregamento robusto para evitar falhas de StringDtype (NotImplementedError)
                def _robust_pickle_load(p):
                    import pickle
                    try: return pd.read_pickle(p)
                    except:
                        class RobustUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module.startswith('numpy._core'):
                                    module = module.replace('numpy._core', 'numpy.core', 1)
                                if 'pandas' in module and 'StringDtype' in name:
                                    try:
                                        from pandas import StringDtype
                                        return StringDtype
                                    except: return object
                                return super().find_class(module, name)
                        try:
                            with open(p, 'rb') as f: return RobustUnpickler(f).load()
                        except: return None

                data = _robust_pickle_load(path)
                if data is not None:
                    reg_gdf = data.get("nodes_gdf")
                    if reg_gdf is not None:
                        dfs.append(reg_gdf)
                else:
                    print(f"❌ Falha crítica ao carregar {path}.")
                    continue
            except Exception as e:
                print(f"❌ Erro crítico ao carregar {path}: {e}")
        else:
            print(f"❌ Erro: Metadados não encontrados em {path}.")
            
    if dfs:
        nodes_gdf = pd.concat(dfs, ignore_index=True)
        print(f"✅ Metadados Regionais Unificados: {len(nodes_gdf)} localidades.")
        
        # === ENRIQUECER faction A PARTIR DE inteligencia_faccoes.csv ===
        try:
            faccoes_path = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv')
            if os.path.exists(faccoes_path):
                import unicodedata
                def _norm(s):
                    s = unicodedata.normalize('NFKD', str(s).upper())
                    return ''.join(c for c in s if not unicodedata.combining(c)).strip()
                fac_df = pd.read_csv(faccoes_path, encoding='utf-8')
                fac_df['_key'] = fac_df['local'].apply(_norm)
                fac_map = dict(zip(fac_df['_key'], fac_df['faccao_predominante'].str.upper()))
                nodes_gdf['_key'] = nodes_gdf['name'].apply(_norm)
                nodes_gdf['faction'] = nodes_gdf['_key'].map(fac_map).fillna(
                    nodes_gdf['faction'] if 'faction' in nodes_gdf.columns else 'NEUTRO'
                )
                nodes_gdf['faction'] = nodes_gdf['faction'].fillna('NEUTRO')
                nodes_gdf.drop(columns=['_key'], inplace=True)
                matched = (nodes_gdf['faction'] != 'NEUTRO').sum()
                print(f"✅ Facções carregadas: {matched}/{len(nodes_gdf)} nós com facção ativa.")
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer facções: {e}")
    else:
        print("❌ Erro Crítico: Nenhum dado regional encontrado.")

    try:
        invalidate_api_risk_cache()
        orchestrator = StateOrchestrator(BASE_DIR)
        if not export_worker:
            start_stgcn_street_warmup()
        print(f"✅ Motor de Inteligência Ativo: {RISK_MODEL_NAME}.")

        # Champion/Challenger — inicializa após o orchestrator
        global champion_challenger
        if ChampionChallenger is not None:
            try:
                champion_challenger = orchestrator.champion_challenger
            except Exception as cc_err:
                print(f"⚠️ [CC] Falha ao inicializar champion_challenger: {cc_err}")

        # Sincronizar metadados de região do orquestrador (elimina hardcode)
        global _RMF_NODES, _ALL_REGIONS, REGION_LABELS
        _ALL_REGIONS = list(orchestrator.specialists.keys())
        if 'rmf' in orchestrator.specialists:
            _RMF_NODES = set(
                normalize_name(r['name'])
                for _, r in orchestrator.specialists['rmf']['data']['nodes_gdf'].iterrows()
            )
            print(f"✅ RMF nodes sincronizados: {len(_RMF_NODES)} municípios.")
        REGION_LABELS.update({reg: REGION_LABELS.get(reg, reg.upper()) for reg in _ALL_REGIONS})

        # Reaplica estado de calibração persistido (evita reset ao reiniciar)
        if model_calibrator is not None:
            model_calibrator.reapply_on_startup(orchestrator)
        
        # Iniciar Monitor de Eficiência e Relatórios
        efficiency_monitor = EfficiencyMonitor(BASE_DIR, orchestrator, nodes_gdf, model_mode=DEFAULT_MODEL_MODE)
        if not export_worker:
            generate_daily_ranking_report()

        try:
            if export_worker:
                raise RuntimeError('export worker skips startup validation')
            enriched_path = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
            if os.path.exists(enriched_path):
                df_validation = pd.read_csv(enriched_path, low_memory=False)
                append_validation_log(
                    df_eval=df_validation,
                    project_root=BASE_DIR,
                    window_days=30,
                    source_label='startup',
                    orchestrator=orchestrator,
                    model_label=DEFAULT_MODEL_LABEL,
                    model_mode=DEFAULT_MODEL_MODE,
                )
        except Exception as validation_exc:
            print(f"⚠️ Falha ao registrar VALIDATION_LOG no startup: {validation_exc}")

        # Validar perfis temporais em background; /api/risk recalcula na chamada.
        if not export_worker:
            threading.Thread(target=_compute_peak_hours_cache, daemon=True).start()

        # Exportar base para o projeto Crime-Predict em background (não bloqueia startup)
        def _run_crime_predict_export():
            try:
                import subprocess
                export_script = os.path.join(BASE_DIR, 'scripts', 'export_to_crime_predict.py')
                if os.path.exists(export_script):
                    print("--- [EXPORT] Iniciando sincronização em background com o Crime-Predict...")
                    completed = subprocess.run(
                        [sys.executable, export_script],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace'
                    )
                    if completed.returncode == 0:
                        print("--- [EXPORT] Sincronização com o Crime-Predict concluída com sucesso.")
                    else:
                        print(f"--- [EXPORT] Falha na sincronização. Código: {completed.returncode}\nstdout: {completed.stdout}\nstderr: {completed.stderr}")
                else:
                    print("--- [EXPORT] Script export_to_crime_predict.py não encontrado.")
            except Exception as e:
                print(f"--- [EXPORT] Erro inesperado ao rodar exportação em background: {e}")

        if not export_worker:
            threading.Thread(target=_run_crime_predict_export, daemon=True, name="crime-predict-export").start()

        # Regenerar micronodos dinâmicos no startup para alinhar a camada ao mapa.
        if not export_worker:
            rebuild_dynamic_micronode_exports(force=True)
        
        # Disparar Monitor em Segundo Plano (Thread Paralela)
        # Guard: não iniciar no processo filho do Flask reloader
        import os as _os
        if not export_worker and (_os.environ.get('WERKZEUG_RUN_MAIN') != 'true' or not app.debug):
            threading.Thread(target=run_background_efficiency_monitor, daemon=True).start()
    except Exception as e:
        print(f"❌ Erro Motor: {e}")

@app.route('/')
def index(): return render_template('index.html')

# === MIDDLEWARE PARA RASTREAMENTO DE REQUISIÇÕES ===
@app.before_request
def track_request_start():
    """Marca o início de uma requisição para rastreamento de latência."""
    request.start_time = time.time()

@app.after_request
def track_request_end(response):
    """Rastreia latência e status de cada requisição no health monitor."""
    if hasattr(request, 'start_time') and health_monitor:
        path = request.path
        # Ignorar arquivos estáticos e favicon (não são indicadores de saúde da API)
        if path.startswith('/static/') or path in ('/favicon.ico', '/robots.txt'):
            return response
        try:
            latency_ms = (time.time() - request.start_time) * 1000
            success = response.status_code < 400
            health_monitor.track_api_request(
                endpoint=path,
                latency_ms=latency_ms,
                success=success
            )
        except Exception as e:
            logging.warning(f"Erro ao rastrear requisição: {e}")
    return response


@app.route('/connections')
def connections(): return render_template('connections.html')

_RMF_CITIES = {
    'CAUCAIA','MARACANAU','MARACANAÚ','PACATUBA','MARANGUAPE','AQUIRAZ',
    'EUSEBIO','EUSÉBIO','HORIZONTE','ITAITINGA','GUAIUBA','GUAIÚBA',
    'BEBERIBE','CASCAVEL','CHOROZINHO','PACAJUS','PINDORETAMA',
    'SAO LUIS DO CURU','PARACURU','PARAIPABA','TRAIRI','GENERAL SAMPAIO',
    'ITAPIPOCA','ACARAPE','REDENÇÃO','REDENCAO','PALMACIA','PALMÁCIA',
    'SAO GONCALO DO AMARANTE', 'SÃO GONÇALO DO AMARANTE'
}

def _classify_region(props: dict) -> str:
    """Infere região (fortaleza/rmf/interior) a partir das propriedades do micronodo."""
    mn  = str(props.get('micronodo') or '').upper()
    lat = props.get('lat') or 0
    lng = props.get('long') or 0
    # Fortaleza por bbox
    if -3.86 <= lat <= -3.69 and -38.64 <= lng <= -38.40:
        return 'fortaleza'
    # Extrair município do campo micronodo: formato 'LOC - CIDADE / CE'
    parts = re.split(r'\s*-\s*', mn)
    if len(parts) > 1:
        cidade_raw = parts[-1].split('/')[0].strip()
        cidade_norm = normalize_name(cidade_raw)
        if cidade_norm in ('FORTALEZA',):
            return 'fortaleza'
        if cidade_raw in _RMF_CITIES or normalize_name(cidade_raw) in {normalize_name(c) for c in _RMF_CITIES}:
            return 'rmf'
        if cidade_raw:
            return 'interior'
    # RMF por coordenadas aproximadas (bbox Grande Fortaleza)
    if -4.20 <= lat <= -3.60 and -38.90 <= lng <= -38.20:
        return 'rmf'
    return 'interior'


def _normalize_polygon_lookup_name(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', normalize_name(text or '')).strip()
    return normalized.split('- AIS')[0].strip()


def _build_micronode_polygon_lookup_keys(micronodo=None, bairro=None, name=None):
    micronodo_key = _normalize_polygon_lookup_name(micronodo)
    bairro_key = _normalize_polygon_lookup_name(bairro)
    name_key = _normalize_polygon_lookup_name(name)

    keys = []
    for key in (
        f'{micronodo_key}||{bairro_key}' if micronodo_key and bairro_key else '',
        f'{name_key}||{bairro_key}' if name_key and bairro_key else '',
        micronodo_key,
        name_key,
        bairro_key,
    ):
        if key and key not in keys:
            keys.append(key)
    return keys


def _load_micronode_reference_centroids():
    global _MICRONODE_REFERENCE_CACHE
    if _MICRONODE_REFERENCE_CACHE is not None:
        return _MICRONODE_REFERENCE_CACHE

    cache_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
    if not os.path.exists(cache_path):
        _MICRONODE_REFERENCE_CACHE = {}
        return _MICRONODE_REFERENCE_CACHE

    grouped = {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as file_obj:
            data = json.load(file_obj) or []

        for item in data:
            bairro_key = _normalize_polygon_lookup_name(item.get('bairro'))
            lat = item.get('lat')
            lng = item.get('lng')
            if not bairro_key or lat is None or lng is None:
                continue
            try:
                grouped.setdefault(bairro_key, []).append((float(lng), float(lat)))
            except Exception:
                continue

        _MICRONODE_REFERENCE_CACHE = {
            bairro_key: (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            for bairro_key, points in grouped.items()
            if points
        }
    except Exception as error:
        print(f"⚠️ [Top Micronodes] Falha ao carregar centroides de referência: {error}")
        _MICRONODE_REFERENCE_CACHE = {}

    return _MICRONODE_REFERENCE_CACHE


def _select_micronode_polygon_geometry(polygon_cache, lookup_keys, bairro=None):
    bairro_key = _normalize_polygon_lookup_name(bairro)
    ref_centroid = _load_micronode_reference_centroids().get(bairro_key)
    max_distance_sq = 0.08 ** 2

    for key in lookup_keys:
        entry = polygon_cache.get(key)
        if not entry:
            continue

        candidates = entry.get('candidates') or []
        if ref_centroid and candidates:
            best = min(
                candidates,
                key=lambda candidate: ((candidate['centroid'][0] - ref_centroid[0]) ** 2) + ((candidate['centroid'][1] - ref_centroid[1]) ** 2),
            )
            best_distance_sq = ((best['centroid'][0] - ref_centroid[0]) ** 2) + ((best['centroid'][1] - ref_centroid[1]) ** 2)
            if best_distance_sq <= max_distance_sq:
                return best.get('geometry')
            continue

        return entry.get('merged')

    return None


def _load_micronode_polygon_cache():
    global _MICRONODE_POLYGON_CACHE
    if _MICRONODE_POLYGON_CACHE is not None:
        return _MICRONODE_POLYGON_CACHE

    polygon_path = os.path.join(app.root_path, 'data', 'raw', 'inteligencia', 'micronodos_faccoes_2026.geojson')
    polygon_groups = {}
    if not os.path.exists(polygon_path):
        _MICRONODE_POLYGON_CACHE = {}
        return _MICRONODE_POLYGON_CACHE

    try:
        with open(polygon_path, 'r', encoding='utf-8') as file_obj:
            data = json.load(file_obj)

        for feature in data.get('features', []):
            geometry = feature.get('geometry') or {}
            if geometry.get('type') not in ('Polygon', 'MultiPolygon'):
                continue

            properties = feature.get('properties') or {}
            keys = _build_micronode_polygon_lookup_keys(
                micronodo=properties.get('micronodo'),
                bairro=properties.get('area_oficial'),
                name=properties.get('name'),
            )
            if not keys:
                continue

            geometry_obj = shape(geometry)
            if geometry_obj.is_empty:
                continue

            for key in keys:
                polygon_groups.setdefault(key, []).append(geometry_obj)

        cache = {}
        for key, geometries in polygon_groups.items():
            merged = unary_union(geometries)
            if not merged.is_empty:
                candidates = []
                for geometry_obj in geometries:
                    centroid = geometry_obj.centroid
                    candidates.append({
                        'geometry': mapping(geometry_obj),
                        'centroid': (centroid.x, centroid.y),
                    })
                cache[key] = {
                    'merged': mapping(merged),
                    'candidates': candidates,
                }
        _MICRONODE_POLYGON_CACHE = cache
    except Exception as error:
        print(f"⚠️ [Top Micronodes] Falha ao carregar cache de polígonos ORCRIMS: {error}")
        _MICRONODE_POLYGON_CACHE = {}

    return _MICRONODE_POLYGON_CACHE


def _load_top_micronode_faction_cache():
    global _TOP_MICRONODE_FACTION_CACHE
    if _TOP_MICRONODE_FACTION_CACHE is not None:
        return _TOP_MICRONODE_FACTION_CACHE

    factions_path = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv')
    if not os.path.exists(factions_path):
        _TOP_MICRONODE_FACTION_CACHE = {}
        return _TOP_MICRONODE_FACTION_CACHE

    try:
        fac_df = pd.read_csv(factions_path, encoding='utf-8')
        cache = {}
        for _, row in fac_df.iterrows():
            area_name = row.get('local')
            faction = str(row.get('faccao_predominante') or '').strip().upper()
            if not area_name or not faction:
                continue
            cache[_normalize_polygon_lookup_name(area_name)] = faction
        _TOP_MICRONODE_FACTION_CACHE = cache
    except Exception as error:
        print(f"⚠️ [Top Micronodes] Falha ao carregar cache de facções: {error}")
        _TOP_MICRONODE_FACTION_CACHE = {}

    return _TOP_MICRONODE_FACTION_CACHE

def _temporal_profile_key(region, name):
    region_key = str(region or 'interior').lower()
    if region_key == 'capital':
        region_key = 'fortaleza'
    return f"{region_key}:{normalize_name(name)}"


def _build_predictive_temporal_profiles(reference_date=None, horizon_days=30):
    """
    Calcula dia da semana e faixa horaria de pico a partir do CSV bruto.
    Nao usa cache: deve refletir a chamada atual de previsao de risco.
    """
    csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
    if not os.path.exists(csv_path):
        return {}

    weekday_labels = {
        0: 'Segunda-feira',
        1: 'Terca-feira',
        2: 'Quarta-feira',
        3: 'Quinta-feira',
        4: 'Sexta-feira',
        5: 'Sabado',
        6: 'Domingo',
    }

    def _extract_hour(value):
        match = re.search(r'(\d{1,2})', str(value or ''))
        if not match:
            return None
        hour = int(match.group(1))
        return hour if 0 <= hour <= 23 else None

    try:
        df = pd.read_csv(csv_path, usecols=['bairro', 'cidade', 'hora', 'tipo', 'data'], low_memory=False)
        df = df[df['tipo'].astype(str).str.lower().eq('cvli')].copy()
        df = df.dropna(subset=['hora', 'data'])
        df['hour'] = df['hora'].map(_extract_hour)
        df['date'] = pd.to_datetime(df['data'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['hour', 'date'])
        if df.empty:
            return {}

        ref = pd.to_datetime(reference_date, errors='coerce') if reference_date is not None else pd.NaT
        if pd.isna(ref):
            ref = df['date'].max()
        ref = pd.Timestamp(ref).normalize()
        future_weekdays = {
            int((ref + pd.Timedelta(days=offset)).weekday())
            for offset in range(1, int(horizon_days) + 1)
        }
        df = df[df['date'] <= ref]
        if df.empty:
            return {}

        df['hour'] = df['hour'].astype(int)
        df['weekday'] = df['date'].dt.weekday.astype(int)
        df['cidade_norm'] = df['cidade'].map(normalize_name)
        df['bairro_norm'] = df['bairro'].map(normalize_name)
        df['region_type'] = np.where(
            df['cidade_norm'].eq('FORTALEZA'),
            'fortaleza',
            np.where(df['cidade_norm'].isin(_RMF_NODES), 'rmf', 'interior'),
        )
        df['local_norm'] = np.where(df['region_type'].eq('fortaleza'), df['bairro_norm'], df['cidade_norm'])
        df = df[df['local_norm'].astype(bool)]

        profiles = {}
        window = 5
        for (region_type, local_norm), grp in df.groupby(['region_type', 'local_norm']):

            total = len(grp)
            best_weekday = None
            best_start = 0
            best_count = -1
            weekday_counts = grp['weekday'].value_counts()
            for weekday in future_weekdays:
                hour_counts = [0] * 24
                for hour in grp.loc[grp['weekday'].eq(weekday), 'hour']:
                    hour_counts[int(hour) % 24] += 1
                for start in range(24):
                    count = sum(hour_counts[(start + i) % 24] for i in range(window))
                    if count > best_count:
                        best_weekday = weekday
                        best_start = start
                        best_count = count
            if best_weekday is None or best_count <= 0:
                continue

            end_hour = (best_start + window) % 24
            peak_hours = f"Entre {best_start:02d}hs e {end_hour:02d}hs"
            peak_weekday = weekday_labels.get(best_weekday, '')
            profiles[_temporal_profile_key(region_type, local_norm)] = {
                'peak_hours': peak_hours,
                'peak_weekday': peak_weekday,
                'peak_time_label': f"{peak_weekday}, {peak_hours.lower()}" if peak_weekday else peak_hours,
                'peak_hour_start': best_start,
                'peak_hour_end': end_hour,
                'peak_hour_share': round(best_count / total, 4),
                'peak_weekday_share': round(int(weekday_counts.max()) / len(grp), 4),
                'temporal_sample_size': int(len(grp)),
                'temporal_horizon_days': int(horizon_days),
                'temporal_reference_date': ref.strftime('%Y-%m-%d'),
            }
        return profiles
    except Exception as e:
        print(f"Erro ao calcular perfis temporais preditivos: {e}")
        return {}


def _compute_peak_hours_cache():
    profiles = _build_predictive_temporal_profiles()
    return {name: data.get('peak_hours', '') for name, data in profiles.items()}

def build_current_exogenous_shocks():
    """Builds the live exogenous shock map used by risk and micronode overlays."""
    exogenous_shocks = {}
    try:
        exo_files = ['exogenous_events.json']
        all_raw_events = []
        for f_name in exo_files:
            f_path = os.path.join(BASE_DIR, 'data', f_name)
            if os.path.exists(f_path):
                try:
                    with open(f_path, 'r', encoding='utf-8') as xf:
                        all_raw_events.extend(json.load(xf) or [])
                except Exception:
                    pass

        cutoff = datetime.now().date() - timedelta(days=EXOGENOUS_WINDOW_DAYS)
        last_date_base = None
        if orchestrator is not None and hasattr(orchestrator, 'dates') and orchestrator.dates is not None:
            last_date_base = orchestrator.dates[-1]

        critical_types = [
            'leader_transfer', 'faction_conflict', 'territory_dispute',
            'confronto', 'execucao', 'chacina', 'tortura', 'homicidio_com_sinais_de_faccao'
        ]

        for ev in all_raw_events:
            try:
                ev_date_str = ev.get('date') or ev.get('event_date')
                if not verify_date_consistency(ev_date_str, last_date_base):
                    continue

                event_dt = parse_event_datetime(ev)
                if event_dt and event_dt.date() < cutoff:
                    continue

                bairro_raw = (ev.get('bairro') or '').strip()
                municipio_raw = (ev.get('municipio') or '').strip()
                targets = []

                if bairro_raw:
                    targets = [normalize_name(str(bairro_raw))]
                elif municipio_raw:
                    mun_norm = normalize_name(municipio_raw)
                    if 'FORTALEZA' in municipio_raw.upper():
                        region_key = 'fortaleza'
                    elif mun_norm in _RMF_NODES or any(n in mun_norm for n in _RMF_NODES):
                        region_key = 'rmf'
                    else:
                        region_key = 'interior'

                    try:
                        for _, row in nodes_gdf.iterrows():
                            if str(row.get('regiao', '')).lower() == region_key:
                                targets.append(normalize_name(row['name']))
                    except Exception:
                        targets = [mun_norm]
                else:
                    continue

                ev_type = str(ev.get('type') or ev.get('natureza') or '').lower()
                description = ' '.join([
                    str(ev.get('description') or ''),
                    str(ev.get('descricao') or ''),
                    str(ev.get('resumo') or ''),
                    str(ev.get('raw_text') or ''),
                ]).lower()
                conflict_severity = str(ev.get('conflict_severity', '')).upper()
                classification = classify_exogenous_event(ev)

                is_supp = classification['is_qualified_suppression']
                is_conflict = classification['is_conflict']
                if classification['signal_class'] == 'administrative_police':
                    continue

                if is_supp:
                    if any(w in description for w in ['fuzil', 'metralhadora', 'fuzi', '7.62', '5.56']):
                        intensity = 1.0
                    elif any(w in description for w in ['lider', 'chefe', 'frente', 'comando']):
                        intensity = 0.9
                    elif any(w in description for w in ['pistola', 'revolver', 'arma de fogo']):
                        intensity = 0.7
                    elif any(w in description for w in ['quilos', 'kg', 'grande quantidade', 'deposito']):
                        intensity = 0.6
                    elif any(w in description for w in ['veiculo', 'carro', 'moto', 'recuperad']):
                        intensity = 0.4
                    else:
                        intensity = float(ev.get('intensity', 0.3))
                    intensity *= suppression_decay_factor(event_dt)
                else:
                    if conflict_severity == 'HIGH':
                        intensity = 0.9
                    elif conflict_severity == 'MEDIUM':
                        intensity = 0.6
                    elif conflict_severity == 'LOW':
                        intensity = 0.3
                    else:
                        intensity = float(ev.get('intensity', 0.5))

                is_city_wide = not bairro_raw and bool(municipio_raw)
                if is_city_wide and len(targets) > 1:
                    intensity = intensity / min(len(targets), 10.0)

                is_critical = (
                    ev_type in critical_types or
                    (is_conflict and intensity > 0.7) or
                    ('execuc' in description) or
                    ('facç' in description) or
                    ('morte' in description and 'facç' in description)
                )

                for loc_norm in targets:
                    if loc_norm not in exogenous_shocks:
                        exogenous_shocks[loc_norm] = {
                            'conflict_intensity': 0.0,
                            'suppression_intensity': 0.0,
                            'is_critical': False,
                            'events_count': 0,
                            'event_types': set()
                        }

                    exogenous_shocks[loc_norm]['events_count'] += 1
                    if ev_type:
                        exogenous_shocks[loc_norm]['event_types'].add(ev_type)

                    if is_supp:
                        exogenous_shocks[loc_norm]['suppression_intensity'] += intensity
                    elif is_conflict:
                        exogenous_shocks[loc_norm]['conflict_intensity'] += intensity
                        if is_critical:
                            exogenous_shocks[loc_norm]['is_critical'] = True
            except Exception:
                continue
    except Exception as e:
        print(f"Erro ao processar shocks no app.py: {e}")
        return None, {}

    return (exogenous_shocks or None), exogenous_shocks


def rebuild_dynamic_micronode_exports(force=False):
    """Ensures micronode overlay files match the current model state."""
    global _MICRONODE_EXPORT_BUILT_AT, _MICRONODE_GEOMETRY_CACHE
    output_files = [
        os.path.join(BASE_DIR, 'outputs', 'visible_micronodes_capital.geojson'),
        os.path.join(BASE_DIR, 'outputs', 'visible_micronodes_rmf.geojson'),
        os.path.join(BASE_DIR, 'outputs', 'visible_micronodes_interior.geojson'),
        os.path.join(BASE_DIR, 'outputs', 'visible_micronodes.geojson'),
    ]

    if not force and _MICRONODE_EXPORT_BUILT_AT is not None and all(os.path.exists(path) for path in output_files):
        return True

    with _MICRONODE_EXPORT_LOCK:
        if not force and _MICRONODE_EXPORT_BUILT_AT is not None and all(os.path.exists(path) for path in output_files):
            return True

        try:
            from scripts.nodes.extract_top30_sentinela_micronodes import build_all_micronode_exports

            build_all_micronode_exports(ensure_runtime=False)
            _MICRONODE_EXPORT_BUILT_AT = datetime.now()
            _MICRONODE_GEOMETRY_CACHE = None
            print("✅ Micronodos dinâmicos regenerados.")
            return True
        except Exception as error:
            print(f"⚠️ [Micronodes] Falha ao regenerar overlay dinâmico: {error}")
            return False

@app.route('/api/micronodes')
def get_micronodes():
    path = os.path.join(app.root_path, 'data', 'raw', 'inteligencia', 'micronodos_faccoes_2026.geojson')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'features' in data:
            for feat in data['features']:
                feat['properties']['region'] = _classify_region(feat['properties'])
        return jsonify(data)
    return jsonify({"type": "FeatureCollection", "features": []})

@app.route('/api/all_streets')
def get_all_streets():
    """Retorna a malha completa de ruas geolocalizadas para exibição no mapa."""
    path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])

@app.route('/api/cvli_points')
def get_cvli_points():
    """Retorna pontos CVLI recentes geocodificados para calibragem de densidade."""
    region = request.args.get('region', 'all').lower()
    if region == 'capital':
        region = 'fortaleza'
    try:
        days = max(1, min(365, int(float(request.args.get('days', 90)))))
    except Exception:
        days = 90

    path = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
    if not os.path.exists(path):
        return jsonify({'type': 'FeatureCollection', 'features': [], 'metadata': {'total': 0}})

    try:
        cols = [
            'id', 'id_evento', 'data', 'hora', 'tipo', 'bairro', 'cidade',
            'latitude', 'longitude', 'tipo_evento', 'name', 'qtd_mortes',
            'arma', 'sexo', 'idade'
        ]
        df = pd.read_csv(path, usecols=lambda col: col in cols, low_memory=False)
        df = df[df['tipo'].astype(str).str.lower().eq('cvli')].copy()
        df['event_date'] = pd.to_datetime(df['data'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['event_date', 'latitude', 'longitude'])
        df = df[
            (df['latitude'].between(-8.0, -2.0)) &
            (df['longitude'].between(-42.0, -37.0))
        ].copy()

        latest_available = df['event_date'].max()
        if pd.isna(latest_available):
            return jsonify({'type': 'FeatureCollection', 'features': [], 'metadata': {'total': 0}})
        today = pd.Timestamp(datetime.now().date())
        reference_date = min(latest_available.normalize(), today)
        cutoff = reference_date - pd.Timedelta(days=days - 1)
        df = df[(df['event_date'] >= cutoff) & (df['event_date'] <= reference_date)].copy()

        features = []
        
        # Load google maps api cache from data_processing.py logic if available
        cache_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
        cache_coords = {}
        if os.path.exists(cache_path):
            try:
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    streets_data = json.load(f)
                    cache_coords = {(round(float(c['lat']), 3), round(float(c['lng']), 3)): c for c in streets_data}
            except Exception as e:
                logging.warning(f"Erro ao carregar cache de ruas: {e}")

        df = df.sort_values('event_date', ascending=False).copy()
        df['hora_norm'] = df['hora'].astype(str).fillna('').str.strip()
        df['bairro_norm'] = df['bairro'].astype(str).fillna('').str.strip().str.upper()
        df['cidade_norm'] = df['cidade'].astype(str).fillna('').str.strip().str.upper()
        df['tipo_evento_norm'] = df['tipo_evento'].astype(str).fillna('CVLI').str.strip().str.upper()
        df['occurrence_key'] = (
            df['event_date'].dt.strftime('%Y-%m-%d').fillna('') + '|' +
            df['hora_norm'] + '|' +
            df['bairro_norm'] + '|' +
            df['cidade_norm'] + '|' +
            df['tipo_evento_norm']
        )

        grouped = df.groupby('occurrence_key', as_index=False).agg(
            event_date=('event_date', 'min'),
            hora=('hora_norm', 'first'),
            bairro=('bairro', 'first'),
            cidade=('cidade', 'first'),
            local=('name', 'first'),
            tipo_evento=('tipo_evento', 'first'),
            latitude=('latitude', 'mean'),
            longitude=('longitude', 'mean'),
            qtd_mortes_max=('qtd_mortes', lambda s: pd.to_numeric(s, errors='coerce').fillna(0).max()),
            row_count=('id', 'count'),
            related_ids=('id_evento', lambda s: [str(v) for v in s.dropna().tolist() if str(v).strip()]),
            fallback_ids=('id', lambda s: [str(v) for v in s.dropna().tolist() if str(v).strip()]),
        )

        for _, row in grouped.iterrows():
            lat = float(row['latitude'])
            lng = float(row['longitude'])
            
            cidade_raw = str(row.get('cidade') or '').strip()
            bairro_raw = str(row.get('bairro') or '').strip()
            local = str(row.get('local') or '').strip()
            
            # Use Nominatim API data from cache if present
            cached = cache_coords.get((round(lat, 3), round(lng, 3)))
            if cached and cached.get('source') == 'auto_update':
                if cached.get('cidade'): cidade_raw = cached['cidade']
                if cached.get('bairro'): bairro_raw = cached['bairro']
                if cached.get('rua'): local = cached['rua']
            
            # Saneamento de coordenadas / cidade para o filtro regional
            declared_city_norm = normalize_name(cidade_raw)
            try:
                actual_city_norm = _municipality_from_lnglat(lng, lat)
            except Exception:
                actual_city_norm = ""

            # Se o ponto cai fora do local declarado mas bate com um município válido
            if actual_city_norm and declared_city_norm and actual_city_norm != declared_city_norm:
                # Se for Fortaleza, RMF ou Interior, confia no mapa e sobrescreve a cidade
                cidade_raw = actual_city_norm

            item_region = _street_region_from_point({
                'lat': lat,
                'lng': lng,
                'cidade': cidade_raw,
            })
            if region != 'all' and item_region != region:
                continue

            event_date = row['event_date']
            deaths_reported = int(row.get('qtd_mortes_max') or 0)
            occurrence_rows = int(row.get('row_count') or 1)
            # Regra robusta: se o dataset já traz total de vítimas (>1), respeita esse valor.
            # Caso contrário, usa a quantidade de linhas da mesma ocorrência para não subcontar vítimas múltiplas.
            victims = deaths_reported if deaths_reported > 1 else max(1, occurrence_rows)
            ids = row.get('related_ids') or []
            if not ids:
                ids = row.get('fallback_ids') or []
            primary_id = ids[0] if ids else ''
            features.append({
                'type': 'Feature',
                'properties': {
                    'id': primary_id,
                    'related_ids': ids,
                    'data': event_date.strftime('%Y-%m-%d'),
                    'hora': str(row.get('hora') or ''),
                    'local': local,
                    'rua': local,
                    'bairro': bairro_raw,
                    'cidade': cidade_raw,
                    'vitimas': victims,
                    'arma': '',
                    'sexo': '',
                    'idade': '',
                    'tipo_evento': str(row.get('tipo_evento') or 'CVLI').strip(),
                    'region': item_region,
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lng, lat],
                },
            })

        return jsonify({
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'total': len(features),
                'days': days,
                'region': region,
                'latest_available_date': latest_available.strftime('%Y-%m-%d'),
                'reference_date': reference_date.strftime('%Y-%m-%d'),
                'cutoff_date': cutoff.strftime('%Y-%m-%d'),
            }
        })
    except Exception as e:
        logging.exception("Erro ao carregar pontos CVLI")
        return jsonify({'error': str(e), 'type': 'FeatureCollection', 'features': []}), 500

_STREET_FOCI_CACHE = {}
_STREET_FOCI_CACHE_LOCK = threading.Lock()
_STREET_FOCI_WARMUP_STATUS = {
    'running': False,
    'completed': [],
    'errors': [],
    'started_at': None,
    'finished_at': None,
}
_MUNICIPALITY_SHAPES_CACHE = None
_MICRONODE_GEOMETRY_CACHE = None

def _load_municipality_shapes_for_street_foci():
    global _MUNICIPALITY_SHAPES_CACHE
    if _MUNICIPALITY_SHAPES_CACHE is not None:
        return _MUNICIPALITY_SHAPES_CACHE

    shapes = []
    path = os.path.join(BASE_DIR, 'data', 'static', 'municipios_ceara.geojson')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        for feature in payload.get('features', []):
            props = feature.get('properties') or {}
            name = props.get('name') or props.get('NAME') or props.get('nome') or props.get('NM_MUN') or ''
            geom = feature.get('geometry')
            if name and geom:
                shapes.append((normalize_name(str(name)), shape(geom)))
    except Exception:
        logging.exception("Falha ao carregar municipios_ceara.geojson para filtro de focos")
    _MUNICIPALITY_SHAPES_CACHE = shapes
    return shapes

def _municipality_from_lnglat(lng, lat):
    try:
        point = Point(float(lng), float(lat))
        for name_norm, geom in _load_municipality_shapes_for_street_foci():
            minx, miny, maxx, maxy = geom.bounds
            if not (minx <= lng <= maxx and miny <= lat <= maxy):
                continue
            if geom.contains(point) or geom.touches(point):
                return name_norm
    except Exception:
        pass
    return ''

def _street_region_from_point(street):
    try:
        lat = float(street.get('lat'))
        lng = float(street.get('lng'))
    except Exception:
        return 'interior'

    # Resolve municipality by physical coordinates first to prevent outliers
    municipality_norm = _municipality_from_lnglat(lng, lat)
    if municipality_norm == 'FORTALEZA':
        return 'fortaleza'
    if municipality_norm in {normalize_name(city) for city in _RMF_CITIES}:
        return 'rmf'
    if municipality_norm:
        return 'interior'

    # Fallback to textual city if coordinates yielded nothing
    city_raw = street.get('cidade') or street.get('municipio') or street.get('municipality') or ''
    if city_raw:
        city_norm = normalize_name(str(city_raw))
        if city_norm == 'FORTALEZA':
            return 'fortaleza'
        if city_norm in {normalize_name(city) for city in _RMF_CITIES}:
            return 'rmf'
        return 'interior'

    if -3.92 <= lat <= -3.68 and -38.70 <= lng <= -38.36:
        return 'fortaleza'
    if -4.25 <= lat <= -3.45 and -39.05 <= lng <= -38.05:
        return 'rmf'
    return 'interior'
def _hexagon_grid_geometry(center_x, center_y, radius_m, origin_lat):
    """Hexagonos de uma mesma grade plana compartilham exatamente as bordas."""
    coords = []
    for i in range(6):
        angle = math.radians(60 * i + 30)
        x = center_x + math.cos(angle) * radius_m
        y = center_y + math.sin(angle) * radius_m
        coords.append([
            x / (111320.0 * max(0.15, math.cos(math.radians(origin_lat)))),
            y / 110540.0,
        ])
    coords.append(coords[0])
    return {'type': 'Polygon', 'coordinates': [coords]}


def _round_hex_axial(q, r):
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def _ga_honeycomb_radius_m():
    path = os.path.join(BASE_DIR, 'outputs', 'experiments', 'fortaleza_hybrid_capture_h30_latest_spatial_ga_summary.csv')
    try:
        summary = pd.read_csv(path)
        radius_km = float(summary.iloc[0]['radius_km'])
        if 0.1 <= radius_km <= 2.0:
            return round(radius_km * 1000)
    except Exception:
        logging.warning('Resumo GA indisponivel; usando a malha operacional de 500m.')
    return 500

def _load_micronode_geometries():
    """Carrega geometrias dos micronodos para evitar sobreposição com focos ST-GCN."""
    global _MICRONODE_GEOMETRY_CACHE
    if _MICRONODE_GEOMETRY_CACHE is not None:
        return _MICRONODE_GEOMETRY_CACHE

    geometries = []
    candidates = [
        os.path.join(BASE_DIR, 'outputs', 'visible_micronodes.geojson'),
        os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia', 'micronodos_faccoes_2026.geojson'),
    ]
    try:
        for path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            for feature in (payload.get('features') or []):
                geom = feature.get('geometry')
                if not geom:
                    continue
                try:
                    geometries.append(shape(geom))
                except Exception:
                    continue
            if geometries:
                break
    except Exception:
        logging.exception("Falha ao carregar geometrias de micronodos para filtro de sobreposição")

    _MICRONODE_GEOMETRY_CACHE = geometries
    return geometries

def _feature_intersects_existing_micronodes(feature):
    geom = feature.get('geometry')
    if not geom:
        return False
    try:
        focus_geom = shape(geom)
    except Exception:
        return False

    focus_area = float(getattr(focus_geom, 'area', 0.0) or 0.0)
    focus_centroid = focus_geom.centroid
    for mn_geom in _load_micronode_geometries():
        try:
            if mn_geom.contains(focus_centroid):
                return True
            if focus_area > 0 and focus_geom.intersects(mn_geom):
                overlap_ratio = focus_geom.intersection(mn_geom).area / focus_area
                if overlap_ratio >= 0.35:
                    return True
            elif focus_geom.intersects(mn_geom):
                return True
        except Exception:
            continue
    return False
def _get_area_risk_scores_for_street_foci():
    if orchestrator is None:
        return {}

    try:
        exogenous_shocks, _ = build_current_exogenous_shocks()
        scores_map = orchestrator.get_combined_risk(exogenous_shocks)
    except Exception:
        try:
            scores_map = orchestrator.get_combined_risk()
        except Exception:
            logging.exception("Erro ao obter scores das areas para ST-GCN de ruas")
            return {}

    return {normalize_name(str(key)): float(value) for key, value in (scores_map or {}).items()}

def _apply_stgcn_street_predictions(features):
    global stgcn_engine
    if not features:
        return features

    try:
        if stgcn_engine is None:
            from src.core.stgcn_escape_engine import STGCNEscapeEngine
            stgcn_engine = STGCNEscapeEngine(data_dir=os.path.join(BASE_DIR, 'data', 'static'), base_dir=BASE_DIR)

        area_risk_scores = _get_area_risk_scores_for_street_foci()
        return stgcn_engine.score_street_foci(
            features,
            area_risk_scores=area_risk_scores,
            neighbor_distance=1000,
            propagation_steps=2,
        )
    except Exception:
        logging.exception("Falha ao aplicar ST-GCN nos focos de ruas; usando score historico")
        return features

def _slice_street_foci_payload(payload, limit):
    sliced = dict(payload)
    features = list(payload.get('features') or [])
    sliced['features'] = features[:limit]
    metadata = dict(payload.get('metadata') or {})
    metadata['total'] = len(sliced['features'])
    metadata['total_available'] = len(features)
    sliced['metadata'] = metadata
    return sliced

def _is_point_in_requested_street_region(lng, lat, region_norm, item_region):
    if region_norm == 'all':
        return True
    municipality_norm = _municipality_from_lnglat(lng, lat)
    if region_norm == 'fortaleza':
        return municipality_norm == 'FORTALEZA'
    if municipality_norm:
        municipality_region = 'rmf' if municipality_norm in {normalize_name(city) for city in _RMF_CITIES} else 'interior'
        return municipality_region == region_norm
    return item_region == region_norm

def _build_street_foci_payload(region='all', radius_m=1000, shape_kind='hex', min_points=2, limit=100):
    cache_key = ('street-foci-v4-ga-honeycomb', region, radius_m, shape_kind, min_points)
    with _STREET_FOCI_CACHE_LOCK:
        cached = _STREET_FOCI_CACHE.get(cache_key)
    if cached:
        return _slice_street_foci_payload(cached, limit)

    path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
    if not os.path.exists(path):
        return {'type': 'FeatureCollection', 'features': [], 'metadata': {'total': 0}}

    with open(path, 'r', encoding='utf-8') as f:
        raw_streets = json.load(f) or []

    points = []
    region_norm = str(region or 'all').lower()
    if region_norm == 'capital':
        region_norm = 'fortaleza'

    for idx, item in enumerate(raw_streets):
        try:
            lat = float(item.get('lat'))
            lng = float(item.get('lng'))
        except Exception:
            continue
        if not (-8.0 <= lat <= -2.0 and -42.0 <= lng <= -37.0):
            continue

        item_region = _street_region_from_point(item)
        if not _is_point_in_requested_street_region(lng, lat, region_norm, item_region):
            continue

        points.append({
            'idx': idx,
            'lat': lat,
            'lng': lng,
            'rua': str(item.get('rua') or item.get('street') or 'Logradouro sem nome').strip(),
            'bairro': str(item.get('bairro') or '').strip(),
            'cidade': str(item.get('cidade') or item.get('municipio') or '').strip(),
            'region': item_region,
            'ocorrencias': max(1, int(float(item.get('ocorrencias') or item.get('occurrences') or 1))),
        })

    # O raio vem do melhor gene GA do experimento h30; a malha inteira usa a
    # mesma origem e o mesmo tamanho, portanto as celulas nao se sobrepoem.
    honeycomb_radius_m = _ga_honeycomb_radius_m() if shape_kind == 'hex' else radius_m
    origin_lat = sum(p['lat'] for p in points) / len(points) if points else 0.0
    meters_per_lon = 111320.0 * max(0.15, math.cos(math.radians(origin_lat)))
    groups = {}
    for p in points:
        x, y = p['lng'] * meters_per_lon, p['lat'] * 110540.0
        if shape_kind == 'hex':
            q = (math.sqrt(3) * x / 3 - y / 3) / honeycomb_radius_m
            r = (2 * y / 3) / honeycomb_radius_m
            key = _round_hex_axial(q, r)
        else:
            key = (int(math.floor(y / radius_m)), int(math.floor(x / radius_m)))
        groups.setdefault(key, []).append(p)

    features = []
    for key, group in groups.items():
        if len(group) < min_points:
            continue

        total_occ = sum(p['ocorrencias'] for p in group)
        streets_rank, bairros_rank, cities_rank, region_rank = {}, {}, {}, {}
        for p in group:
            streets_rank[p['rua']] = streets_rank.get(p['rua'], 0) + p['ocorrencias']
            if p['bairro']:
                bairros_rank[p['bairro']] = bairros_rank.get(p['bairro'], 0) + p['ocorrencias']
            if p['cidade']:
                cities_rank[p['cidade']] = cities_rank.get(p['cidade'], 0) + p['ocorrencias']
            region_rank[p['region']] = region_rank.get(p['region'], 0) + p['ocorrencias']

        top_streets = [name for name, _ in sorted(streets_rank.items(), key=lambda item: item[1], reverse=True)[:6]]
        bairro = next(iter(sorted(bairros_rank.items(), key=lambda item: item[1], reverse=True)), ('', 0))[0]
        cidade = next(iter(sorted(cities_rank.items(), key=lambda item: item[1], reverse=True)), ('', 0))[0]
        focus_region = next(iter(sorted(region_rank.items(), key=lambda item: item[1], reverse=True)), ('interior', 0))[0]
        if shape_kind == 'hex':
            q, r = key
            center_x = honeycomb_radius_m * math.sqrt(3) * (q + r / 2)
            center_y = honeycomb_radius_m * 1.5 * r
            geometry = _hexagon_grid_geometry(center_x, center_y, honeycomb_radius_m, origin_lat)
        else:
            lng = sum(p['lng'] * p['ocorrencias'] for p in group) / total_occ
            lat = sum(p['lat'] * p['ocorrencias'] for p in group) / total_occ
            geometry = {
            'type': 'Point',
            'coordinates': [lng, lat],
            }

        features.append({
            'type': 'Feature',
            'properties': {
                'name': f"Celula GA - {bairro or cidade or 'CE'}",
                'focus_id': '',
                'region': focus_region,
                'bairro': bairro,
                'cidade': cidade,
                'radius_m': honeycomb_radius_m,
                'cluster_distance_m': honeycomb_radius_m,
                'street_count': len(group),
                'total_occurrences': total_occ,
                'top_streets': top_streets,
                'score': total_occ,
                'risk_score': total_occ,
                'is_street_focus': True,
            },
            'geometry': geometry,
        })

    features.sort(
        key=lambda feat: (
            (feat.get('properties') or {}).get('total_occurrences', 0),
            (feat.get('properties') or {}).get('street_count', 0),
        ),
        reverse=True,
    )
    max_candidates = 300
    features = features[:max_candidates]
    features = _apply_stgcn_street_predictions(features)

    # Viés preditivo conservador (tende para cima) e filtragem para focos de alto risco.
    scored = []
    for feat in features:
        props = feat.get('properties') or {}
        base_pred = float(props.get('predicted_cvli_probability') or props.get('stgcn_score') or props.get('risk_score') or 0.0)
        conservative_pred = max(base_pred, min(100.0, base_pred + 8.0))
        props['predicted_cvli_probability_raw'] = round(base_pred, 2)
        props['predicted_cvli_probability'] = round(conservative_pred, 2)
        props['stgcn_score'] = round(conservative_pred, 2)
        props['risk_score'] = round(conservative_pred, 2)
        props['score'] = round(conservative_pred, 2)
        scored.append(feat)

    non_overlapping = [f for f in scored if not _feature_intersects_existing_micronodes(f)]
    sorted_non_overlapping = sorted(
        non_overlapping,
        key=lambda feat: float((feat.get('properties') or {}).get('predicted_cvli_probability') or 0),
        reverse=True
    )
    features = sorted_non_overlapping[:100]
    max_occ = max((feat['properties']['total_occurrences'] for feat in features), default=1)
    for rank, feat in enumerate(features, 1):
        props = feat['properties']
        props['rank'] = rank
        props['focus_id'] = f"FOCO-{props['region'].upper()}-{rank:04d}"
        props['intensity_pct'] = round(100.0 * props['total_occurrences'] / max_occ, 1)

    payload = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'total': len(features),
            'total_available': len(features),
            'source': 'GA h30 honeycomb over geo_streets_cache',
            'model': 'ST-GCN Rua/Foco GA',
            'radius_m': honeycomb_radius_m,
            'ga_honeycomb_radius_m': honeycomb_radius_m,
            'selection': 'top_100_predicted_stgcn',
            'shape': shape_kind,
            'min_points': min_points,
            'region': region_norm,
            'generated_at': datetime.now().isoformat(),
        }
    }
    with _STREET_FOCI_CACHE_LOCK:
        _STREET_FOCI_CACHE[cache_key] = payload
    return _slice_street_foci_payload(payload, limit)

def _warm_stgcn_street_pipeline():
    global stgcn_engine
    if _STREET_FOCI_WARMUP_STATUS.get('running'):
        return

    _STREET_FOCI_WARMUP_STATUS.update({
        'running': True,
        'completed': [],
        'errors': [],
        'started_at': datetime.now().isoformat(),
        'finished_at': None,
    })
    try:
        if stgcn_engine is None:
            from src.core.stgcn_escape_engine import STGCNEscapeEngine
            stgcn_engine = STGCNEscapeEngine(data_dir=os.path.join(BASE_DIR, 'data', 'static'), base_dir=BASE_DIR)

        print("🧠 ST-GCN ruas/focos: aquecimento em paralelo iniciado.")
        for region in ('fortaleza', 'rmf', 'interior', 'all'):
            try:
                _build_street_foci_payload(region, radius_m=1000, shape_kind='hex', min_points=2, limit=50)
                _STREET_FOCI_WARMUP_STATUS['completed'].append(region)
                print(f"✅ ST-GCN ruas/focos aquecido: {region}")
            except Exception as exc:
                msg = f"{region}: {exc}"
                _STREET_FOCI_WARMUP_STATUS['errors'].append(msg)
                logging.exception("Falha no aquecimento ST-GCN de ruas (%s)", region)
    finally:
        _STREET_FOCI_WARMUP_STATUS['running'] = False
        _STREET_FOCI_WARMUP_STATUS['finished_at'] = datetime.now().isoformat()

def start_stgcn_street_warmup():
    threading.Thread(target=_warm_stgcn_street_pipeline, daemon=True).start()

@app.route('/api/street_foci')
def get_street_foci():
    """Agrupa ruas/pontos georreferenciados em focos táticos de até 1km."""
    region = request.args.get('region', 'all').lower()
    shape_kind = request.args.get('shape', 'hex').lower()
    if shape_kind not in ('hex', 'circle'):
        shape_kind = 'hex'
    try:
        radius_m = max(100, min(1000, int(float(request.args.get('radius_m', 1000)))))
    except Exception:
        radius_m = 1000
    try:
        min_points = max(1, min(20, int(float(request.args.get('min_points', 2)))))
    except Exception:
        min_points = 2
    try:
        limit = max(1, min(100, int(float(request.args.get('limit', 100)))))
    except Exception:
        limit = 100

    try:
        return jsonify(_build_street_foci_payload(region, radius_m, shape_kind, min_points, limit))
    except Exception as e:
        logging.exception("Erro ao gerar focos de ruas")
        return jsonify({'error': str(e), 'type': 'FeatureCollection', 'features': []}), 500

@app.route('/api/visible_micronodes')
@app.route('/api/top20_micro_nodes')
def get_visible_micronodes():
    region = request.args.get('region', 'fortaleza').lower()
    limit_raw = request.args.get('limit')
    limit = None
    if limit_raw not in (None, ''):
        try:
            limit = max(1, int(limit_raw))
        except Exception:
            pass
    force_refresh = request.args.get('refresh', '0').lower() in ('1', 'true', 'yes')
    # Mapear regiao para o arquivo correspondente na pasta outputs
    filename_map = {
        'fortaleza': 'visible_micronodes_capital.geojson',
        'rmf': 'visible_micronodes_rmf.geojson',
        'interior': 'visible_micronodes_interior.geojson',
        'all': 'visible_micronodes.geojson'
    }
    legacy_filename_map = {
        'fortaleza': 'top20_micro_nodes_capital.geojson',
        'rmf': 'top20_micro_nodes_rmf.geojson',
        'interior': 'top20_micro_nodes_interior.geojson',
        'all': 'top20_micro_nodes.geojson'
    }
    
    filename = filename_map.get(region, 'visible_micronodes_capital.geojson')
    path = os.path.join(app.root_path, 'outputs', filename)
    legacy_path = os.path.join(app.root_path, 'outputs', legacy_filename_map.get(region, 'top20_micro_nodes_capital.geojson'))

    rebuild_dynamic_micronode_exports(force=force_refresh)

    def _decorate_top_features(payload):
        polygon_cache = _load_micronode_polygon_cache()
        faction_cache = _load_top_micronode_faction_cache()
        # Perfil preditivo calculado uma vez para a resposta inteira, nao por micronodo.
        peak_cache = _build_predictive_temporal_profiles() or {}
        features = sorted(
            payload.get('features', []),
            key=lambda feature: float((feature.get('properties') or {}).get('score') or (feature.get('properties') or {}).get('risk_score') or 0),
            reverse=True,
        )
        
        # --- FILTRO DE DEDUP POR MICRONODO (SENTINELA CLEAN MODE) ---
        # Cada micronodo tem rank unico - nao eliminar pontos taticos distintos.
        decorated = []
        seen_areas = set()
        rmf_municipalities = {normalize_name(city) for city in _RMF_CITIES}
        
        for feature in features:
            props = dict(feature.get('properties') or {})
            try:
                point = shape(feature.get('geometry') or {}).centroid
                municipality_norm = _municipality_from_lnglat(point.x, point.y)
            except Exception:
                municipality_norm = ''
            if region == 'fortaleza' and municipality_norm != 'FORTALEZA':
                continue
            if region == 'rmf' and municipality_norm not in rmf_municipalities:
                continue
            if region == 'interior' and (not municipality_norm or municipality_norm == 'FORTALEZA' or municipality_norm in rmf_municipalities):
                continue
            
            # Usar regiao + rank + nome como chave unica (micronodos distintos podem compartilhar nome)
            rank = props.get('rank', '')
            region_key = props.get('region', '')
            area_raw = props.get('name') or props.get('micronodo') or props.get('bairro') or "DESCONHECIDO"
            area_key = f"{region_key}:{rank}:{_normalize_polygon_lookup_name(str(area_raw))}"
            
            if area_key in seen_areas:
                continue
            seen_areas.add(area_key)

            lookup_keys = _build_micronode_polygon_lookup_keys(
                micronodo=props.get('micronodo') or props.get('name'),
                bairro=props.get('bairro') or props.get('parent_area'),
                name=props.get('name'),
            )
            polygon_geometry = _select_micronode_polygon_geometry(
                polygon_cache,
                lookup_keys,
                bairro=props.get('bairro') or props.get('parent_area'),
            )
            faction = props.get('faction') or next((faction_cache.get(key) for key in lookup_keys if faction_cache.get(key)), None)
            if polygon_geometry:
                feature['geometry'] = polygon_geometry
                props['geometry_type'] = polygon_geometry.get('type', 'Polygon')
                props['source_geometry_type'] = polygon_geometry.get('type', 'Polygon')
                props['is_centroid'] = False
            if faction:
                props['faction'] = faction
            # Inject temporal prediction pattern
            bairro_lookup = _normalize_polygon_lookup_name(props.get('bairro') or props.get('name') or props.get('micronodo') or '')
            profile_key = _temporal_profile_key(props.get('region_type') or props.get('region'), bairro_lookup)
            if bairro_lookup and profile_key in peak_cache:
                props.update(peak_cache[profile_key])
            feature['properties'] = props
            decorated.append(feature)
            
            # A camada do mapa precisa cobrir todas as areas plotadas.
            if limit is not None and len(decorated) >= limit:
                break
                
        payload['features'] = decorated
        return payload
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'features' in data:
                data = _decorate_top_features(data)
            return jsonify(data)
    if os.path.exists(legacy_path):
        with open(legacy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'features' in data:
                data = _decorate_top_features(data)
            return jsonify(data)
    
    # Fallback se o regional nao existir
    fallback_path = os.path.join(app.root_path, 'outputs', 'visible_micronodes.geojson')
    if os.path.exists(fallback_path):
        with open(fallback_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'features' in data:
                data = _decorate_top_features(data)
            return jsonify(data)
    legacy_fallback_path = os.path.join(app.root_path, 'outputs', 'top20_micro_nodes.geojson')
    if os.path.exists(legacy_fallback_path):
        with open(legacy_fallback_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'features' in data:
                data = _decorate_top_features(data)
            return jsonify(data)
            
    return jsonify({"type": "FeatureCollection", "features": []})


# Backward-compatible module attribute used by snapshot/export scripts.
get_top20_micro_nodes = get_visible_micronodes


def _apply_temporal_profiles_to_risk_payload(results, meta, profiles):
    if not profiles:
        return
    by_name = {}
    for row in results:
        name_key = normalize_name(row.get('clean_name') or row.get('name') or '')
        profile_key = _temporal_profile_key(row.get('region_type') or row.get('region'), name_key)
        profile = profiles.get(profile_key)
        if not profile:
            continue
        metrics = row.setdefault('metrics', {})
        metrics.update(profile)
        row.update({
            'peak_hours': profile.get('peak_hours', ''),
            'peak_weekday': profile.get('peak_weekday', ''),
            'peak_time_label': profile.get('peak_time_label', ''),
        })
        by_name[profile_key] = profile

    for bucket in list((meta or {}).get('top10_by_region', {}).values()) + [(meta or {}).get('top10', [])]:
        for item in bucket or []:
            profile_key = _temporal_profile_key(item.get('region_type') or item.get('region'), item.get('name') or '')
            profile = by_name.get(profile_key) or profiles.get(profile_key)
            if profile:
                item.update({
                    'peak_hours': profile.get('peak_hours', ''),
                    'peak_weekday': profile.get('peak_weekday', ''),
                    'peak_time_label': profile.get('peak_time_label', ''),
                })


@app.route('/api/risk')
def get_risk():
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
        
    target_region = request.args.get('region', 'global').lower()
    selected_model_mode = normalize_model_mode(request.args.get('model_mode', 'stgat_v5'))
    selected_model_meta, available_model_modes = _get_model_selection_meta(selected_model_mode)
    
    global _API_RISK_CACHE
    with _API_RISK_CACHE_LOCK:
        cache_entry = _API_RISK_CACHE.get(selected_model_mode)
        if cache_entry is not None:
            import copy
            meta = copy.deepcopy(cache_entry['meta'])
            results = copy.deepcopy(cache_entry['data'])
            temporal_profiles = _build_predictive_temporal_profiles(
                reference_date=orchestrator.dates[-1] if getattr(orchestrator, 'dates', None) is not None else None,
                horizon_days=30,
            )
            _apply_temporal_profiles_to_risk_payload(results, meta, temporal_profiles)
            if target_region != 'global' and target_region in meta.get('counts_by_region', {}):
                meta['counts'] = meta['counts_by_region'][target_region]
                results = [r for r in results if r.get('region_type') == target_region]
            return jsonify({'meta': meta, 'data': results})

    try:
        exogenous_shocks, exogenous_shocks_map = build_current_exogenous_shocks()
        temporal_profiles = _build_predictive_temporal_profiles(
            reference_date=orchestrator.dates[-1] if getattr(orchestrator, 'dates', None) is not None else None,
            horizon_days=30,
        )

        scores_map, trends_map = _score_map_for_model_mode(selected_model_mode, exogenous_shocks, return_trends=True)
            
        results = []
        meta = {'counts': {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0}}
        all_scores = []
        
        # Prepare per-region accumulators
        region_buckets = {r: [] for r in _ALL_REGIONS}
        # Contadores por região para o frontend
        region_stats = {r: {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0} for r in _ALL_REGIONS}

        # Carregar cache de explicações do gestor para uso no dashboard
        manager_cache = {}
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                    manager_cache = json.load(cf) or {}
        except: pass

        # Carregar Inteligência de Ruas Críticas
        streets_cache = {}
        try:
            # Primeiro tenta carregar de ruas_criticas_por_bairro.json (formato dicionário por bairro)
            streets_path = os.path.join(BASE_DIR, 'data', 'raw', 'ruas_criticas_por_bairro.json')
            if os.path.exists(streets_path):
                with open(streets_path, 'r', encoding='utf-8') as sf:
                    streets_cache = json.load(sf)
                # Criar versao normalizada do cache para match garantido
                streets_cache = {normalize_name(k): v for k, v in streets_cache.items() if k}
                print(f"✅ Inteligência de ruas (via bairro.json): {len(streets_cache)} bairros.")
            else:
                # Fallback: carregar geo_streets_cache.json (formato array) e agrupar por bairro
                geo_streets_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
                if os.path.exists(geo_streets_path):
                    with open(geo_streets_path, 'r', encoding='utf-8') as sf:
                        geo_streets_array = json.load(sf) or []
                    
                    # Agrupar ruas por bairro e contar ocorrências
                    streets_by_bairro = {}
                    for item in geo_streets_array:
                        bairro = (item.get('bairro') or item.get('area') or 'DESCONHECIDO').strip()
                        rua = (item.get('rua') or item.get('street') or 'AREA SEM NOME').strip()
                        ocorrencias = item.get('ocorrencias', 1)
                        
                        if not bairro or bairro.upper() == 'DESCONHECIDO':
                            continue  # Pula bairros desconhecidos
                        
                        bairro_norm = normalize_name(bairro)
                        if bairro_norm not in streets_by_bairro:
                            streets_by_bairro[bairro_norm] = []
                        
                        # Adicionar rua com peso de ocorrências
                        streets_by_bairro[bairro_norm].append({
                            'name': rua,
                            'occurrences': ocorrencias
                        })
                    
                    # Ordenar ruas por ocorrências e manter os top N por bairro
                    streets_cache = {}
                    for bairro_norm, streets_list in streets_by_bairro.items():
                        # Ordena por ocorrências (maior primeiro) e pega top 8
                        sorted_streets = sorted(streets_list, key=lambda x: x.get('occurrences', 0), reverse=True)
                        top_streets = [s['name'] for s in sorted_streets[:8]]
                        streets_cache[bairro_norm] = top_streets
                    
                    print(f"✅ Inteligência de ruas (via geo_streets_cache.json): {len(streets_cache)} bairros com top ruas.")
        except Exception as e: 
            print(f"❌ Erro ao carregar ruas: {e}")

        # Índice de localidades/ruas para RMF e Interior
        # Fonte primária: streets_by_municipio.json (gerado por gerar_streets_municipios.py)
        # Fallback inline: bairros + exógenos sem geocoding se o arquivo ainda não existir
        exo_streets_by_municipio = {}
        try:
            _mun_streets_path = os.path.join(BASE_DIR, 'data', 'streets_by_municipio.json')
            if os.path.exists(_mun_streets_path):
                with open(_mun_streets_path, 'r', encoding='utf-8') as _msf:
                    exo_streets_by_municipio = json.load(_msf)
                print(f"✅ Localidades por município: {len(exo_streets_by_municipio)} municípios carregados.")
            else:
                # Fallback: bairros do CSV + exógenos, sem geocoding (para o primeiro boot)
                from collections import defaultdict, Counter as _Counter
                _mun_locs: dict = defaultdict(_Counter)
                _INVALID = ['HOMICIDIO','BALA','FOGO','LESAO','MORTE','CADAVER',
                            'LATROCINIO','TIRO','EXECUCAO','ACHADO','CVLI','CVP']
                _csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
                if os.path.exists(_csv_path):
                    import pandas as _pd
                    _df = _pd.read_csv(_csv_path, usecols=['cidade','bairro','tipo'], low_memory=False)
                    _df_nf = _df[
                        _df['cidade'].notna() &
                        ~_df['cidade'].str.upper().str.contains('FORTALEZA', na=True) &
                        _df['bairro'].notna() & (_df['bairro'].str.len() > 2)
                    ]
                    for _, _rw in _df_nf.iterrows():
                        _ck = normalize_name(str(_rw['cidade']))
                        _b  = str(_rw['bairro']).strip().upper()
                        if _b and not any(t in _b for t in _INVALID):
                            _peso = 3 if str(_rw.get('tipo','')).lower() == 'cvli' else 1
                            _mun_locs[_ck][_b] += _peso
                _exo_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
                if os.path.exists(_exo_path):
                    with open(_exo_path, 'r', encoding='utf-8') as _ef:
                        _exo_evs = json.load(_ef)
                    for _ev in _exo_evs:
                        _mun = normalize_name(_ev.get('municipio') or '')
                        if not _mun or _mun == 'fortaleza': continue
                        _b = str(_ev.get('bairro') or '').strip().upper()
                        if _b and len(_b) > 2 and not any(t in _b for t in _INVALID):
                            _mun_locs[_mun][_b] += 5
                for _mk, _ctr in _mun_locs.items():
                    _top = [l for l, _ in _ctr.most_common(8) if len(l) > 3]
                    if _top:
                        exo_streets_by_municipio[_mk] = _top
                print(f"✅ Localidades fallback (bairros): {len(exo_streets_by_municipio)} municípios.")
        except Exception as _exo_err:
            print(f"⚠️ Erro ao carregar localidades por município: {_exo_err}")

        fortaleza_override_scores = selected_model_meta.get('fortaleza_scores') or {}

        for i, row in nodes_gdf.iterrows():
            try:
                name = str(row['name'])
                name_norm = normalize_name(name)
                # SCORE REAL E DIRETO (Sem Amortecimento)
                score = normalize_risk_score(scores_map.get(name_norm, 20.0))
                trend = trends_map.get(name_norm, 'stable')
                
                # Identificação de Região
                reg = str(row.get('regiao', 'fortaleza')).lower()
                if reg == 'capital': reg = 'fortaleza'
                if name_norm in _RMF_NODES: reg = 'rmf'
                
                if reg not in region_buckets: region_buckets[reg] = []

                if reg == 'fortaleza' and selected_model_mode != 'stgat':
                    override_score = _resolve_optional_model_score(name_norm, fortaleza_override_scores)
                    if override_score is not None:
                        score = normalize_risk_score(override_score)

                level, status, css, color, score = classify_risk_score(score)

                if reg in region_stats:
                    region_stats[reg][level] += 1
                meta['counts'][level] += 1

                # Inteligência de Ruas Críticas
                critical_streets_info = streets_cache.get(name_norm, None)
                if critical_streets_info is None and reg == 'fortaleza':
                    # Match parcial só para nós de Fortaleza (bairros no cache)
                    for k, v in streets_cache.items():
                        if name_norm in k or k in name_norm:
                            critical_streets_info = v
                            break

                # Para RMF/Interior: índice de localidades por município
                # O name_norm para esses nodes É o nome do município (ex: 'CAUCAIA', 'MARACANAU')
                if critical_streets_info is None and reg in ('rmf', 'interior'):
                    exo_locs = exo_streets_by_municipio.get(name_norm)
                    if exo_locs:
                        critical_streets_info = exo_locs

                # Se ainda não encontrou, usar fallback
                if critical_streets_info is None:
                    critical_streets_info = 'Sem logradouros críticos recentes'
                
                events_info = (exogenous_shocks_map or {}).get(name_norm, {}) if 'exogenous_shocks_map' in locals() else {}
                ev_count = events_info.get('events_count', 0)
                ev_types = list(events_info.get('event_types', set()))

                # Horário de pico padrão-baseado para este bairro
                temporal_profile = temporal_profiles.get(_temporal_profile_key(reg, name_norm), {})

                node_metrics = {
                    'cvli_7d': 0,
                    'tension': round(float(np.nan_to_num(row.get('tension_index', 0))), 2),
                    'events_count': ev_count,
                    'event_types': ev_types[:3],
                    'critical_streets': critical_streets_info,
                    'spatial_influence': score >= 80,
                }
                node_metrics.update(temporal_profile)
                
                # Crimes Reais
                current_spec = orchestrator.specialists.get(reg)
                if current_spec:
                    try:
                        local_idx = next(idx for idx, r in current_spec['data']['nodes_gdf'].iterrows() if normalize_name(r['name']) == name_norm)
                        node_metrics['cvli_7d'] = int(current_spec['data']['node_features'][local_idx, -14:, 0].sum())
                    except: pass

                all_scores.append(score)
                row_cidade = row.get('cidade')
                if not isinstance(row_cidade, str) or pd.isna(row_cidade) or not row_cidade:
                    row_cidade = 'Fortaleza' if reg == 'fortaleza' else name
                node_result = {
                    'node_id': i, 'name': name, 'clean_name': name_norm,
                    'tension_score': score, 'risk_score': score,
                    'status_label': status, 'css_class': css,
                    'color': color, 'trend': trend, 
                    'metrics': node_metrics,
                    'peak_hours': temporal_profile.get('peak_hours', ''),
                    'peak_weekday': temporal_profile.get('peak_weekday', ''),
                    'peak_time_label': temporal_profile.get('peak_time_label', ''),
                    'faction': str(row.get('faction', 'N/A')), 'region_type': reg,
                    'cidade': str(row_cidade)
                }
                results.append(node_result)
                region_buckets[reg].append(node_result)
            except Exception as e:
                print(f"Erro no nó {i}: {e}")
                continue

        # Adicionar Ranking Info para o Frontend
        if all_scores:
            meta['stats_overall_mean'] = float(np.mean(all_scores))
            meta['ranking_info'] = {
                'top_1_percent_threshold': float(np.percentile(all_scores, 99)),
                'top_5_percent_threshold': float(np.percentile(all_scores, 95)),
                    'top_10_percent_threshold': float(np.percentile(all_scores, 90)),
                    'risk_bands': get_risk_thresholds_meta()
            }
        else:
                meta['ranking_info'] = {
                    'top_1_percent_threshold': 99,
                    'top_5_percent_threshold': 95,
                    'top_10_percent_threshold': 90,
                    'risk_bands': get_risk_thresholds_meta()
                }

        # --- Métricas focadas no gestor: confiança do ranking e "temperatura do estado" ---
        try:
            scores_arr = np.array(all_scores) if all_scores else np.array([20.0])
            s_mean = float(np.mean(scores_arr))
            s_std = float(np.std(scores_arr))
            s_min = float(np.min(scores_arr))
            s_max = float(np.max(scores_arr))

            # Ordenar scores para estatísticas de topo
            sorted_scores_arr = np.sort(scores_arr)[::-1]
            
            # 1. Pressão nos Hotspots (Top 5 Mean)
            top5_scores = sorted_scores_arr[:5]
            meta['stats_top5_mean'] = float(np.mean(top5_scores)) if len(top5_scores) > 0 else s_mean
            
            # 2. Alerta do Top 10 (Top 10 Mean)
            top10_scores = sorted_scores_arr[:10]
            meta['stats_top10_mean'] = float(np.mean(top10_scores)) if len(top10_scores) > 0 else s_mean
            
            # 3. Corte de Prioridade (Mínimo do Top 5)
            meta['stats_top5_min'] = float(np.min(top5_scores)) if len(top5_scores) > 0 else s_min
            
            # 4. Volatilidade Geral (STD)
            meta['stats_overall_std'] = s_std

            # Separação entre top10 e média geral — indica clareza do ranking
            top10_threshold = int(np.percentile(scores_arr, 90)) if len(scores_arr) > 1 else s_mean
            top10_mean = float(np.mean([v for v in scores_arr if v >= top10_threshold])) if len(scores_arr) > 0 else s_mean
            separation = top10_mean - s_mean

            # Confiança heurística recalibrada
            denom = (s_max - s_min) if (s_max - s_min) > 0 else 1.0
            std_norm = min(1.0, s_std / (s_mean + 1e-6)) # Relativo à média
            sep_norm = min(1.0, separation / (s_std + 1e-6)) # Quantos desvios o top 10 está acima

            # Novo cálculo: Base de 65%, bônus por separação, penalidade leve por volatilidade
            confidence_score = 0.65 + (0.30 * sep_norm) - (0.15 * std_norm)
            confidence_score = max(0.4, min(0.98, confidence_score))
            confidence_pct = round(confidence_score * 100.0, 1)

            if confidence_pct >= 80:
                confidence_label = 'Alta'
            elif confidence_pct >= 60:
                confidence_label = 'Moderada'
            elif confidence_pct >= 40:
                confidence_label = 'Baixa'
            else:
                confidence_label = 'Muito baixa'

            confidence_explanation = (
                f"Cobertura territorial estimada em {confidence_pct}%: separação dos top {max(1,int(len(scores_arr)*0.1))}% territórios "
                f"em relação à média (desvio padrão {s_std:.2f}). Consultar Cov@20 no dashboard admin para métrica real."
            )

            # Temperatura do estado (visão gerencial): mapeia média para níveis claros
            state_pct = round(s_mean, 1)
            if state_pct >= RISK_SCORE_THRESHOLDS['critical_min']:
                temp_label = 'Crítico'
                temp_color = '#8B0000'
                recommendation = 'Intervenção imediata e mobilização de recursos.'
            elif state_pct >= RISK_SCORE_THRESHOLDS['high_min']:
                temp_label = 'Alto'
                temp_color = '#E63946'
                recommendation = 'Aumentar vigilância e priorizar ações no top 10.'
            elif state_pct >= RISK_SCORE_THRESHOLDS['moderate_min']:
                temp_label = 'Moderado'
                temp_color = '#F4A261'
                recommendation = 'Reforçar monitoramento e revisar alocação de recursos.'
            else:
                temp_label = 'Baixo'
                temp_color = '#A8DADC'
                recommendation = 'Manter operações regulares e monitorar tendências.'

            meta['manager_view'] = {
                'confidence_pct': confidence_pct,
                'confidence_label': confidence_label,
                'confidence_explanation': confidence_explanation,
                'state_temperature_pct': state_pct,
                'state_temperature_label': temp_label,
                'state_temperature_color': temp_color,
                'recommendation': recommendation,
                'source': 'computed'
            }
        except Exception:
            meta['manager_view'] = {
                'confidence_pct': 50.0,
                'confidence_label': 'Moderada',
                'state_temperature_pct': meta.get('stats_overall_mean', 30.0),
                'state_temperature_label': 'Baixo',
                'recommendation': 'Monitorar',
                'source': 'fallback'
            }

        meta['risk_thresholds'] = get_risk_thresholds_meta()
        meta['selected_model_mode'] = selected_model_mode
        meta['selected_model_label'] = selected_model_meta.get('label')
        meta['selected_model_kind'] = selected_model_meta.get('kind')
        meta['selected_model_description'] = selected_model_meta.get('description')
        meta['available_model_modes'] = available_model_modes
        if selected_model_meta.get('metrics'):
            meta['selected_model_metrics'] = selected_model_meta.get('metrics')

        # Build counts by region and top10 by region
        try:
            meta['counts_by_region'] = {}
            meta['top10_by_region'] = {}
            for region_key, items in region_buckets.items():
                c = {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0}
                for it in items:
                    level, _, _, _, _ = classify_risk_score(it.get('risk_score', 0))
                    c[level] += 1
                meta['counts_by_region'][region_key] = c

                sorted_region = sorted(items, key=lambda x: x.get('risk_score', 0), reverse=True)
                
                deduped_region = []
                seen_names = set()
                for r in sorted_region:
                    name_norm = normalize_name(r.get('name'))
                    if name_norm not in seen_names:
                        seen_names.add(name_norm)
                        deduped_region.append(r)
                
                meta['top10_by_region'][region_key] = [{
                    'name': r.get('name'), 'node_id': r.get('node_id'), 'risk_score': r.get('risk_score'),
                    'status_label': r.get('status_label'), 'region_type': r.get('region_type'),
                    'peak_hours': (r.get('metrics') or {}).get('peak_hours', ''),
                    'peak_weekday': (r.get('metrics') or {}).get('peak_weekday', ''),
                    'peak_time_label': (r.get('metrics') or {}).get('peak_time_label', ''),
                    'cidade': r.get('cidade')
                } for r in deduped_region[:10]]
        except Exception:
            meta['counts_by_region'] = {}
            meta['top10_by_region'] = {}

        # Build Top10 list server-side so frontend doesn't need to re-derive ranking.
        try:
            sorted_results = sorted(results, key=lambda x: x.get('risk_score', 0), reverse=True)
            meta['top10'] = []
            seen_names_all = set()
            for r in sorted_results:
                name_norm = normalize_name(r.get('name'))
                if name_norm not in seen_names_all:
                    seen_names_all.add(name_norm)
                    meta['top10'].append({
                        'name': r.get('name'),
                        'node_id': r.get('node_id'),
                        'risk_score': r.get('risk_score'),
                        'status_label': r.get('status_label'),
                        'region_type': r.get('region_type'),
                        'peak_hours': (r.get('metrics') or {}).get('peak_hours', ''),
                        'peak_weekday': (r.get('metrics') or {}).get('peak_weekday', ''),
                        'peak_time_label': (r.get('metrics') or {}).get('peak_time_label', ''),
                        'cidade': r.get('cidade')
                    })
                    if len(meta['top10']) >= 10:
                        break
        except Exception:
            meta['top10'] = []

            # --- CORREÇÃO: Adicionar Datas da Janela de Inteligência (Projeção 30 dias) ---
        try:
            if orchestrator is not None and hasattr(orchestrator, 'dates') and orchestrator.dates is not None:
                last_db_date = orchestrator.dates[-1]
                if isinstance(last_db_date, str):
                    last_db_dt = datetime.strptime(last_db_date[:10], '%Y-%m-%d')
                else:
                    last_db_dt = last_db_date
                
                # Início e fim da projeção de 30 dias à frente da base.
                start_pred = last_db_dt + timedelta(days=1)
                end_pred = last_db_dt + timedelta(days=30)
                
                meta['start_cvli'] = str(orchestrator.dates[0])
                meta['last_date_base'] = last_db_dt.strftime('%d/%m/%Y')
                meta['prediction_window'] = f"{start_pred.strftime('%d/%m')} a {end_pred.strftime('%d/%m')}"
                meta['intelligence_label'] = f"Janela de Inteligência: {meta['prediction_window']} (Atualizada com Eventos de Hoje)"
                meta['window_cvli'] = len(orchestrator.dates)
                meta['model_architecture'] = RISK_MODEL_NAME
                meta['model_window_cvli'] = 120 # Nova janela de 120 dias para todos
                
            else:
                meta['intelligence_label'] = "Janela de Inteligência: Projeção 30 dias (Tempo Real)"
                meta['last_date_base'] = 'N/A'
                meta['model_architecture'] = RISK_MODEL_NAME
                meta['model_window_cvli'] = 120
                
            # Adicionar métricas de eficiência separadamente para que frontend (Cov@20) não fique com N/A
            if 'efficiency_monitor' in globals() and efficiency_monitor:
                _latest_metrics = efficiency_monitor.get_latest_metrics()
                if _latest_metrics:
                    meta['efficiency_metrics'] = _latest_metrics
        except Exception as e:
            print(f"Erro ao calcular datas de inteligência: {e}")
            meta['intelligence_label'] = "Janela de Inteligência: Ativa"

        if selected_model_mode != 'stgat':
            metrics = selected_model_meta.get('metrics') or {}
            metric_bits = []
            if isinstance(metrics.get('p10'), (int, float)):
                metric_bits.append(f"P@10 hist.: {metrics['p10'] * 100:.1f}%")
            if isinstance(metrics.get('p20'), (int, float)):
                metric_bits.append(f"P@20 hist.: {metrics['p20'] * 100:.1f}%")
            meta['model_architecture'] = f"{selected_model_meta.get('label')} (Comparativo Fortaleza)"
            meta['intelligence_label'] = (
                f"{meta.get('intelligence_label', 'Janela de Inteligência: Ativa')} • "
                f"Modo opcional aplicado somente em Fortaleza"
            )
            if metric_bits:
                meta['selected_model_validation_text'] = ' | '.join(metric_bits)

        # Cache the unfiltered results
        import copy
        cache_payload = {
            'meta': copy.deepcopy(meta),
            'data': copy.deepcopy(results)
        }
        with _API_RISK_CACHE_LOCK:
            _API_RISK_CACHE[selected_model_mode] = cache_payload

        # --- CORREÇÃO: Respeitar Filtro de Região nas caixas de resumo ---
        if target_region != 'global' and target_region in meta.get('counts_by_region', {}):
            meta['counts'] = meta['counts_by_region'][target_region]
            # Envia apenas os resultados daquela região
            results = region_buckets.get(target_region, [])

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/territory')
def get_territory():
    """Retorna informações detalhadas de um território/nó específico (incluindo ruas críticas)."""
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
    
    try:
        name_param = request.args.get('name', '').strip().upper()
        if not name_param:
            return jsonify({'error': 'Parâmetro name é obrigatório'}), 400
        
        # Normalizar o nome
        name_norm = normalize_name(name_param)
        
        # Obter as respostas da API /api/risk para usar os dados já processados
        risk_response = get_risk()
        if risk_response.status_code != 200:
            return jsonify({'error': 'Erro ao obter dados de risco'}), 500
        
        risk_data = risk_response.get_json()
        results = risk_data.get('data', [])
        
        # Procurar pelo nó específico
        node_result = None
        for item in results:
            if normalize_name(item.get('name', '')) == name_norm:
                node_result = item
                break
        
        if not node_result:
            # Se não encontrar, retornar um objeto vazio com estrutura básica
            return jsonify({
                'name': name_param,
                'node_id': None,
                'risk_score': 0,
                'metrics': {
                    'critical_streets': 'Sem logradouros críticos registrados',
                    'cvli_7d': 0,
                    'tension': 0,
                    'events_count': 0,
                    'event_types': []
                }
            })
        
        return jsonify({
            'name': node_result.get('name'),
            'node_id': node_result.get('node_id'),
            'clean_name': node_result.get('clean_name'),
            'risk_score': node_result.get('risk_score'),
            'tension_score': node_result.get('tension_score'),
            'status_label': node_result.get('status_label'),
            'faction': node_result.get('faction'),
            'region_type': node_result.get('region_type'),
            'metrics': node_result.get('metrics', {
                'critical_streets': 'Indispoível',
                'cvli_7d': 0,
                'tension': 0,
                'events_count': 0,
                'event_types': []
            })
        })
    except Exception as e:
        print(f"Erro em /api/territory: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_risk():
    """Simula um cenário de supressão ou conflito em pontos geográficos específicos."""
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
    try:
        payload = request.get_json(force=True) or {}
        points = payload.get('points', []) # List of [lat, lng]
        sim_type = payload.get('type', 'suppression')
        
        if not points:
            return jsonify({'error': 'Nenhum ponto fornecido'}), 400

        # 1. Mapear pontos [lat, lng] para nomes canônicos da malha especialista
        def _has_specialist_owner(name_norm):
            try:
                owners = getattr(orchestrator, '_node_owners', {}) or {}
                return name_norm in owners
            except Exception:
                return False

        def _nearest_specialist_name(lat_p, lng_p):
            best_name = None
            best_dist = None
            try:
                for _, spec in orchestrator.specialists.items():
                    spec_nodes = spec.get('data', {}).get('nodes_gdf') if isinstance(spec, dict) else None
                    if spec_nodes is None or spec_nodes.empty:
                        continue
                    for _, srow in spec_nodes.iterrows():
                        lat_s = srow.get('lat')
                        lng_s = srow.get('long')
                        if lat_s is None or lng_s is None:
                            continue
                        try:
                            lat_s = float(lat_s)
                            lng_s = float(lng_s)
                        except (TypeError, ValueError):
                            continue
                        dist = (lat_s - lat_p) ** 2 + (lng_s - lng_p) ** 2
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_name = normalize_name(str(srow.get('name', '')))
            except Exception:
                return None
            return best_name

        temp_shocks = {}
        intensity_per_point = 0.25 # Cada equipe/conflito contribui com 25% de intensidade
        
        for pt in points:
            try:
                if len(pt) < 2: continue
                lat_p, lng_p = float(pt[0]), float(pt[1])
                
                dists = np.sqrt((nodes_gdf['lat'] - lat_p)**2 + (nodes_gdf['long'] - lng_p)**2)
                nearest_idx = dists.idxmin()
                row = nodes_gdf.loc[nearest_idx]
                name_norm = normalize_name(str(row['name']))

                # Se o nó mais próximo não participa da malha especialista, aproximar para o nó canônico válido
                if not _has_specialist_owner(name_norm):
                    fallback_name = _nearest_specialist_name(lat_p, lng_p)
                    if fallback_name:
                        name_norm = fallback_name
                
                # Configurar Shock Simulado (CUMULATIVO)
                is_supp = (sim_type == 'suppression')
                
                if name_norm not in temp_shocks:
                    temp_shocks[name_norm] = {
                        'intensity': 0.0,
                        'suppression_intensity': 0.0,
                        'is_critical': not is_supp,
                        'is_suppression': is_supp
                    }
                
                # Incrementa intensidade por tipo (mais pontos no mesmo bairro = mais força)
                if is_supp:
                    temp_shocks[name_norm]['suppression_intensity'] += intensity_per_point
                    # Cap de 1.0 (100%) para evitar valores irreais
                    if temp_shocks[name_norm]['suppression_intensity'] > 1.0:
                        temp_shocks[name_norm]['suppression_intensity'] = 1.0
                else:
                    temp_shocks[name_norm]['intensity'] += intensity_per_point
                    # Cap de 1.0 (100%) para evitar valores irreais
                    if temp_shocks[name_norm]['intensity'] > 1.0:
                        temp_shocks[name_norm]['intensity'] = 1.0
                    
            except Exception as e:
                print(f"Erro ao processar ponto de simulação {pt}: {e}")

        if not temp_shocks:
            return jsonify({'error': 'Não foi possível mapear pontos para a malha'}), 400

        # 2. Obter risco combinado com os shocks temporários
        scores_map, trends_map = orchestrator.get_combined_risk(temp_shocks, return_trends=True)
        
        # 3. Formatar retorno idêntico ao /api/risk
        results = []
        meta = {'counts': {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0}}
        all_scores = []
        
        # Copiar lógica de métricas reais do /api/risk para manter o dashboard funcional
        for i, row in nodes_gdf.iterrows():
            name = str(row['name'])
            name_norm = normalize_name(name)
            score = normalize_risk_score(scores_map.get(name_norm, 20.0))
            trend = trends_map.get(name_norm, 'stable')
            
            # Identificação de Região
            reg = str(row.get('regiao', 'fortaleza')).lower()
            if reg == 'capital': reg = 'fortaleza'
            if name_norm in _RMF_NODES: reg = 'rmf'

            level, status, css, color, score = classify_risk_score(score)
            meta['counts'][level] += 1
            
            # Métricas Reais (Mesma lógica do get_risk)
            node_metrics = {
                'cvli_7d': 0,
                'tension': round(float(np.nan_to_num(row.get('tension_index', 0))), 2),
                'events_count': 0,
                'event_types': [],
                'spatial_influence': score >= 80
            }
            
            # Crimes Reais
            current_spec = orchestrator.specialists.get(reg)
            if current_spec:
                try:
                    local_idx = next(idx for idx, r in current_spec['data']['nodes_gdf'].iterrows() if normalize_name(r['name']) == name_norm)
                    node_metrics['cvli_7d'] = int(current_spec['data']['node_features'][local_idx, -7:, 0].sum())
                except: pass

            # Se este ponto está sendo simulado, marcar nas métricas para o frontend saber
            if name_norm in temp_shocks:
                node_metrics['simulated_event'] = True
                node_metrics['sim_type'] = sim_type

            all_scores.append(score)
            results.append({
                'node_id': i, 'name': name, 'clean_name': name_norm,
                'tension_score': score, 'risk_score': score,
                'risk_score_cvli': score,
                'status_label': status, 'css_class': css,
                'color': color, 'trend': trend, 
                'metrics': node_metrics,
                'faction': str(row.get('faction', 'N/A')), 'region_type': reg
            })

        # Adicionar Top 10 Simulado e Stats
        sorted_results = sorted(results, key=lambda x: x['tension_score'], reverse=True)
        meta['top10'] = [{
            'name': r['name'], 'node_id': r['node_id'],
            'tension_score': r['tension_score'], 'risk_score': r['tension_score'],
            'status_label': r['status_label'], 'region_type': r['region_type']
        } for r in sorted_results[:10]]
        
        meta['stats_overall_mean'] = float(np.mean(all_scores))
        meta['simulated'] = True
        meta['intelligence_label'] = f"SIMULAÇÃO ATIVA: Cenário de {sim_type.upper()}"

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/polygons')
def get_polygons():
    features = []
    polygon_files = [
        ('fortaleza', 'nodes_polygons.geojson'),
        ('rmf', 'AIS - METROPOLITANA.geojson'),
        ('interior', 'AIS - INTERIOR.geojson'),
    ]
    for reg, fname in polygon_files:
        path = os.path.join(BASE_DIR, 'data', 'static', fname)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for feat in data.get('features', []):
                        feat.setdefault('properties', {})
                        if reg == 'fortaleza':
                            source_region = str(feat['properties'].get('region_type') or '').strip().lower()
                            if source_region and source_region not in ('capital', 'fortaleza'):
                                continue
                        feat['properties']['region_type'] = reg
                        if reg == 'fortaleza':
                            raw_name = feat['properties'].get('bairro') or feat['properties'].get('NOME') or feat['properties'].get('name') or feat['properties'].get('Name')
                            clean_name = _normalize_polygon_lookup_name(raw_name)
                            if clean_name:
                                feat['properties']['bairro'] = clean_name
                        features.append(feat)
            except: pass
    return jsonify({"type": "FeatureCollection", "features": features})


def _is_valid_screenshot_repo(repo_dir: str) -> bool:
    return os.path.isdir(repo_dir) and os.path.exists(os.path.join(repo_dir, 'package.json'))


def _subprocess_env() -> dict:
    temp_dir = os.path.expandvars(os.environ.get('TEMP') or os.environ.get('TMP') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp'))
    os.makedirs(temp_dir, exist_ok=True)
    return {
        **os.environ,
        'PYTHONIOENCODING': 'utf-8',
        'TEMP': temp_dir,
        'TMP': temp_dir,
        'TMPDIR': temp_dir,
    }


def _run_git_command(repo_dir: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=180,
        check=False,
        env=_subprocess_env(),
    )


def _ensure_screenshot_git_identity(repo_dir: str) -> None:
    defaults = {
        'user.email': os.environ.get('SCREENSHOT_GIT_USER_EMAIL', 'boanergesteixeiraalmeida@gmail.com'),
        'user.name': os.environ.get('SCREENSHOT_GIT_USER_NAME', 'Boanerges Teixeira Almeida'),
    }
    for key, value in defaults.items():
        current = _run_git_command(repo_dir, ['config', '--local', '--get', key])
        if current.returncode != 0 or not current.stdout.strip():
            result = _run_git_command(repo_dir, ['config', '--local', key, value])
            if result.returncode != 0:
                raise RuntimeError(f'Falha ao configurar {key}: {result.stderr.strip() or result.stdout.strip()}')


def _sync_static_snapshot_to_screenshot_app(target_data_dir: str):
    if not os.path.exists(STATIC_EXPORT_SCRIPT):
        raise FileNotFoundError(f'Exporter não encontrado em {STATIC_EXPORT_SCRIPT}')

    target_repo_dir = os.path.dirname(os.path.dirname(target_data_dir))
    if not _is_valid_screenshot_repo(target_repo_dir):
        raise FileNotFoundError(
            f'Repositório screenshot-report_preview não encontrado ou inválido em {target_repo_dir}'
        )

    os.makedirs(target_data_dir, exist_ok=True)
    logging.info('[SCREENSHOT EXPORT] Iniciando exportação estática')
    logging.info('[SCREENSHOT EXPORT] Export script: %s', STATIC_EXPORT_SCRIPT)
    logging.info('[SCREENSHOT EXPORT] Output dir: %s', STATIC_EXPORT_OUTPUT_DIR)
    logging.info('[SCREENSHOT EXPORT] Target repo: %s', target_repo_dir)
    logging.info('[SCREENSHOT EXPORT] Target data dir: %s', target_data_dir)

    # Reutiliza os modelos ja carregados pelo Flask em vez de abrir outro processo.
    from pathlib import Path
    sys.modules.setdefault('app', sys.modules[__name__])
    from scripts.export_static_snapshot import export_snapshot
    export_snapshot(Path(STATIC_EXPORT_OUTPUT_DIR))
    logging.info('[SCREENSHOT EXPORT] Exportação concluída com sucesso')

    copied_files = []
    for filename in sorted(os.listdir(STATIC_EXPORT_OUTPUT_DIR)):
        if not filename.endswith(('.json', '.geojson')):
            continue
        source_path = os.path.join(STATIC_EXPORT_OUTPUT_DIR, filename)
        if not os.path.isfile(source_path):
            continue
        destination_path = os.path.join(target_data_dir, filename)
        shutil.copy2(source_path, destination_path)
        copied_files.append(filename)
    logging.info('[SCREENSHOT EXPORT] %s arquivos sincronizados para a app screenshot', len(copied_files))
    if copied_files:
        logging.info('[SCREENSHOT EXPORT] Arquivos: %s', ', '.join(copied_files))

    return {
        'target_repo_dir': target_repo_dir,
        'export_output_dir': STATIC_EXPORT_OUTPUT_DIR,
        'target_data_dir': target_data_dir,
        'copied_files': copied_files,
        'stdout': '',
    }


def _pull_and_merge_remote(repo_dir: str, data_subdir: str = 'public/data') -> None:
    """
    Executa git pull integrando alterações remotas. Em caso de conflito,
    preserva a versão local dos arquivos de snapshot (public/data).
    """
    logging.info('[SCREENSHOT EXPORT] Executando git pull origin main...')
    pull_result = _run_git_command(repo_dir, ['pull', 'origin', 'main', '--no-rebase', '-X', 'ours', '--no-edit'])
    if pull_result.returncode == 0:
        logging.info('[SCREENSHOT EXPORT] Git pull/merge concluído com sucesso.')
        return

    pull_err = (pull_result.stderr or pull_result.stdout or '').strip()
    logging.warning('[SCREENSHOT EXPORT] Git pull retornou aviso/erro: %s', pull_err)

    # Verifica se o repositório ficou em estado de conflito não resolvido
    status_result = _run_git_command(repo_dir, ['status', '--porcelain'])
    unmerged = [
        line for line in status_result.stdout.splitlines()
        if line.startswith(('UU', 'AA', 'DD', 'DU', 'UD', 'AU', 'UA'))
    ]

    if unmerged:
        logging.warning('[SCREENSHOT EXPORT] Conflitos de merge detectados em %d arquivos. Resolvendo com a versão local...', len(unmerged))
        # Força checkout da versão local (--ours) para os arquivos de dados
        _run_git_command(repo_dir, ['checkout', '--ours', data_subdir])
        _run_git_command(repo_dir, ['add', data_subdir])
        commit_res = _run_git_command(
            repo_dir,
            ['commit', '-m', f'chore: resolve merge conflicts in {data_subdir} using local snapshot'],
        )
        if commit_res.returncode == 0 or 'nothing to commit' in (commit_res.stderr or commit_res.stdout or '').lower():
            logging.info('[SCREENSHOT EXPORT] Conflitos resolvidos e commit de merge finalizado com sucesso.')
            return

    # Caso o pull falhe por razões graves ou não resolvidas, aborta o merge para manter a WC limpa
    logging.error('[SCREENSHOT EXPORT] Abortando merge devido a falha no git pull.')
    _run_git_command(repo_dir, ['merge', '--abort'])
    raise RuntimeError(f'Falha ao integrar alterações remotas via git pull: {pull_err}')


def _publish_screenshot_repo(repo_dir: str, data_subdir: str = 'public/data') -> dict[str, any]:
    logging.info('[SCREENSHOT EXPORT] Iniciando publicação git do repositório screenshot')
    _ensure_screenshot_git_identity(repo_dir)

    # 1. Verificar se há alterações no working tree da subpasta
    status_result = _run_git_command(repo_dir, ['status', '--porcelain', data_subdir])
    if status_result.returncode != 0:
        raise RuntimeError(f'Falha ao consultar status git: {status_result.stderr.strip() or status_result.stdout.strip()}')

    changed_entries = [line for line in status_result.stdout.splitlines() if line.strip()]
    commit_created = False
    commit_message = f'chore: sync static snapshot {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    if changed_entries:
        # Adiciona e realiza o commit das alterações locais
        add_result = _run_git_command(repo_dir, ['add', data_subdir])
        if add_result.returncode != 0:
            raise RuntimeError(f'Falha no git add: {add_result.stderr.strip() or add_result.stdout.strip()}')

        commit_result = _run_git_command(repo_dir, ['commit', '-m', commit_message])
        if commit_result.returncode != 0:
            combined_output = (commit_result.stderr or commit_result.stdout or '').strip()
            if 'nothing to commit' not in combined_output.lower():
                raise RuntimeError(f'Falha no git commit: {combined_output}')
            logging.info('[SCREENSHOT EXPORT] Nenhum commit novo gerado (nothing to commit)')
        else:
            commit_created = True
            logging.info('[SCREENSHOT EXPORT] Commit local criado com sucesso: %s', commit_message)

    # 2. Tentar realizar o push com retentativas integrando git pull em caso de divergência/rejeição
    max_push_attempts = 3
    push_executed = False

    for attempt in range(1, max_push_attempts + 1):
        push_result = _run_git_command(repo_dir, ['push', 'origin', 'main'])
        if push_result.returncode == 0:
            push_executed = True
            stdout_lower = (push_result.stdout or '').lower()
            if 'everything up-to-date' in stdout_lower and not commit_created:
                logging.info('[SCREENSHOT EXPORT] Repositório já está atualizado (Everything up-to-date)')
                return {
                    'published': False,
                    'commit_created': False,
                    'push_executed': True,
                    'message': f'Nenhuma alteração detectada em {data_subdir} no repositório screenshot.',
                }
            logging.info('[SCREENSHOT EXPORT] Push para origin/main realizado com sucesso (tentativa %d)', attempt)
            break

        push_err = (push_result.stderr or push_result.stdout or '').strip()
        logging.warning(
            '[SCREENSHOT EXPORT] Falha no git push (tentativa %d/%d): %s',
            attempt,
            max_push_attempts,
            push_err,
        )

        if attempt < max_push_attempts:
            logging.info('[SCREENSHOT EXPORT] Executando git pull para integrar alterações remotas antes de tentar push novamente...')
            _pull_and_merge_remote(repo_dir, data_subdir)
        else:
            raise RuntimeError(f'Falha no git push após {max_push_attempts} tentativas: {push_err}')

    logging.info('[SCREENSHOT EXPORT] Publicação git concluída com sucesso')
    return {
        'published': True,
        'commit_created': commit_created,
        'push_executed': push_executed,
        'commit_message': commit_message,
        'message': f'Snapshot de {data_subdir} sincronizado e publicado no repositório screenshot.',
    }


def _bg_export_thread(target_data_dir, publish_repo):
    global _SNAPSHOT_EXPORT_STATUS
    with _SNAPSHOT_EXPORT_LOCK:
        _SNAPSHOT_EXPORT_STATUS['status'] = 'running'
        _SNAPSHOT_EXPORT_STATUS['error'] = None
        try:
            sync_info = _sync_static_snapshot_to_screenshot_app(target_data_dir)
            publish_info = None
            if publish_repo:
                publish_info = _publish_screenshot_repo(sync_info['target_repo_dir'])
            
            _SNAPSHOT_EXPORT_STATUS['status'] = 'success'
            _SNAPSHOT_EXPORT_STATUS['copied_count'] = len(sync_info.get('copied_files', []))
            _SNAPSHOT_EXPORT_STATUS['last_run'] = datetime.now().isoformat()
            logging.info('[SCREENSHOT EXPORT] Exportação assíncrona concluída com sucesso')
        except Exception as e:
            _SNAPSHOT_EXPORT_STATUS['status'] = 'error'
            _SNAPSHOT_EXPORT_STATUS['error'] = str(e)
            _SNAPSHOT_EXPORT_STATUS['last_run'] = datetime.now().isoformat()
            logging.exception('[SCREENSHOT EXPORT] Erro na exportação assíncrona: %s', e)


@app.route('/api/export_static_snapshot', methods=['POST'])
def export_static_snapshot_to_screenshot_app():
    global _SNAPSHOT_EXPORT_STATUS
    if _SNAPSHOT_EXPORT_LOCK.locked() or _SNAPSHOT_EXPORT_STATUS['status'] == 'running':
        return jsonify({
            'ok': False,
            'message': 'Uma exportação já está em andamento. Por favor, aguarde.'
        }), 409

    try:
        payload = request.get_json(silent=True) or {}
        target_repo_dir = payload.get('target_repo_dir') or STATIC_SCREENSHOT_REPO_DIR
        target_data_dir = payload.get('target_data_dir') or os.path.join(target_repo_dir, 'public', 'data')
        publish_repo = bool(payload.get('publish_repo', False))

        # Dispara thread em background
        threading.Thread(
            target=_bg_export_thread,
            args=(target_data_dir, publish_repo),
            daemon=True
        ).start()

        return jsonify({
            'ok': True,
            'status': 'processing',
            'message': 'Exportação iniciada em segundo plano. Os arquivos serão gerados e sincronizados.'
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/export_static_snapshot/status', methods=['GET'])
def export_static_snapshot_status():
    global _SNAPSHOT_EXPORT_STATUS
    return jsonify({
        'ok': True,
        'status': _SNAPSHOT_EXPORT_STATUS['status'],
        'error': _SNAPSHOT_EXPORT_STATUS['error'],
        'last_run': _SNAPSHOT_EXPORT_STATUS['last_run'],
        'copied_count': _SNAPSHOT_EXPORT_STATUS['copied_count']
    })

@app.route('/api/geocode')
def geocode_search():
    """Geolocaliza uma rua, bairro ou localidade via Nominatim (OpenStreetMap).
    Restringe busca ao Estado do Ceará para resultados mais relevantes.
    Parâmetro: ?q=<texto>
    Retorna: lista de {name, lat, lon, type}
    """
    q = request.args.get('q', '').strip()
    if not q or len(q) < 3:
        return jsonify([])
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError

        geolocator = Nominatim(
            user_agent='report_preview_app/1.0',
            timeout=6
        )
        # Restringe ao Ceará para evitar resultados de outros estados
        query = q + ', Ceará, Brasil'
        locations = geolocator.geocode(query, exactly_one=False, limit=6, language='pt') or []

        results = []
        seen = set()
        for loc in locations:
            raw = loc.raw or {}
            display = loc.address or ''
            # Remove duplicatas por display_name truncado
            key = display[:60]
            if key in seen:
                continue
            seen.add(key)
            # Tipo legível
            loc_type = raw.get('type') or raw.get('class') or 'lugar'
            results.append({
                'name':    display,
                'short':   (raw.get('namedetails') or {}).get('name') or q,
                'lat':     float(loc.latitude),
                'lon':     float(loc.longitude),
                'type':    loc_type,
                'source':  'nominatim'
            })
        return jsonify(results)
    except Exception as e:
        logging.warning(f'Geocode error: {e}')
        return jsonify([])


@app.route('/api/streets/critical')
def get_geo_critical_streets():
    """Retorna as ruas geolocalizadas mais críticas para um bairro/cidade."""
    bairro = request.args.get('bairro', '').upper()
    cidade = request.args.get('cidade', '').upper()
    
    cache_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
    if not os.path.exists(cache_path):
        return jsonify([])
        
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            all_streets = json.load(f)
            
        # Normalizar busca
        bairro_norm = normalize_name(bairro)
        cidade_norm = normalize_name(cidade)
        
        filtered = []
        for s in all_streets:
            s_bairro_norm = normalize_name(s.get('bairro', ''))
            s_cidade_norm = normalize_name(s.get('cidade', ''))

            # Para micronodos de inteligência sem cidade, tentar derivar do padrão "NOME - CIDADE"
            if not s_cidade_norm and s.get('source') == 'intelligence':
                rua_name = s.get('rua', '')
                if ' - ' in rua_name:
                    s_cidade_norm = normalize_name(rua_name.rsplit(' - ', 1)[1].strip())

            # Match robusto: Bairro deve bater (se fornecido) e cidade deve ser compatível
            match_bairro = False
            if bairro_norm and s_bairro_norm:
                if bairro_norm == s_bairro_norm or s_bairro_norm in bairro_norm or bairro_norm in s_bairro_norm:
                    match_bairro = True

            match_cidade = False
            if cidade_norm and s_cidade_norm:
                if cidade_norm == s_cidade_norm or s_cidade_norm in cidade_norm or cidade_norm in s_cidade_norm:
                    match_cidade = True
            elif not s_cidade_norm: # Se o cache não tem cidade (e não foi possível derivar), aceita se bairro bateu
                match_cidade = True

            if (bairro_norm and match_bairro and match_cidade) or (not bairro_norm and cidade_norm and match_cidade):
                rua_val = normalize_name(s.get('rua', ''))
                if rua_val and rua_val != 'AREA SEM NOME':
                    filtered.append(s)

        # Ordenar por ocorrências e limitar às 10 mais críticas
        filtered.sort(key=lambda x: x.get('ocorrencias', 0), reverse=True)
        if filtered:
            return jsonify(filtered[:10])

        # ── Fallback: streets_by_municipio.json (RMF / Interior) ─────────────
        # Para nós RMF/Interior o bairroParam é o nome do município.
        # Tentamos: normalize(bairro) e normalize(cidade) como chaves.
        mun_streets_path = os.path.join(BASE_DIR, 'data', 'streets_by_municipio.json')
        if os.path.exists(mun_streets_path):
            try:
                with open(mun_streets_path, 'r', encoding='utf-8') as _mf:
                    mun_data = json.load(_mf)
                candidates = []
                for key in (bairro_norm, cidade_norm):
                    if key and key in mun_data:
                        candidates = mun_data[key]
                        break
                # Match parcial se exato não encontrou
                if not candidates:
                    for mk, locs in mun_data.items():
                        if bairro_norm and (bairro_norm == mk or bairro_norm in mk or mk in bairro_norm):
                            candidates = locs
                            break
                if candidates:
                    # Suporta formato novo [{"loc": ..., "score": ...}] e legado [str]
                    result = []
                    for entry in candidates[:10]:
                        if isinstance(entry, dict):
                            # cvli = contagem bruta; score = peso ponderado (desempate)
                            result.append({'rua': entry['loc'], 'bairro': bairro,
                                           'cidade': bairro,
                                           'ocorrencias': entry.get('cvli', 0),
                                           'source': 'intelligence'})
                        else:
                            result.append({'rua': entry, 'bairro': bairro,
                                           'cidade': bairro, 'ocorrencias': 0,
                                           'source': 'intelligence'})
                    # Já vem ordenado por cvli desc do JSON; garante ordem aqui também
                    result.sort(key=lambda x: x['ocorrencias'], reverse=True)
                    return jsonify(result)
            except Exception:
                pass

        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-update-status')
def model_status():
    return jsonify(_get_model_update_status_payload())

@app.route('/api/anomaly_status')
def anomaly_status():
    """Retorna o status de anomalias calculado em tempo real sobre o estado atual do modelo."""
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'monitoring_active': False, 'error': 'Inicializando...'}), 503
    
    try:
        # 1. Obter scores atuais para cálculo de tensão
        scores_map = orchestrator.get_combined_risk()
        scores = list(scores_map.values())
        
        # 2. Cálculo da Tensão Estadual (Escala 0-10)
        # Baseado na volatilidade (STD) e na média do Top 5%
        if scores:
            s_mean = np.mean(scores)
            s_std = np.std(scores)
            top_mean = np.percentile(scores, 95)
            # Tensão sobe se a média do topo for alta e houver muita variação
            tension = min(10.0, (top_mean / 20.0) + (s_std / 10.0))
        else:
            tension = 0.0
            
        # Determinar Label
        if tension >= 7.5: label = 'CRÍTICO'
        elif tension >= 5.0: label = 'ALERTA'
        else: label = 'ESTÁVEL'

        # 3. Listar Eventos Ativos Reais (Janela exógena canônica de 7 dias)
        active_events = []
        try:
            exo_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
            if os.path.exists(exo_path):
                with open(exo_path, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                
                cutoff = (datetime.now() - timedelta(days=EXOGENOUS_WINDOW_DAYS)).date()
                last_date_base = orchestrator.dates[-1] if (orchestrator and hasattr(orchestrator, 'dates')) else None
                for e in events:
                    # Tenta pegar a data
                    try:
                        dstr = e.get('date', '') or e.get('event_date', '')
                        if not verify_date_consistency(dstr, last_date_base):
                            continue # Pula evento futuro

                        classification = classify_exogenous_event(e)
                        if classification['signal_class'] == 'administrative_police':
                            continue

                        if dstr and datetime.strptime(dstr[:10], '%Y-%m-%d').date() >= cutoff:
                            active_events.append({
                                'description': e.get('description') or e.get('descricao') or e.get('resumo', 'Evento Crítico'),
                                'severity': float(e.get('intensity', 0.5)),
                                'is_suppression': classification['is_suppression'],
                                'is_qualified_suppression': classification['is_qualified_suppression'],
                            })
                    except: continue
        except: pass

        # 4. Cobertura Territorial (Recall@K do último ciclo de avaliação)
        # Usa Cov@20 global do efficiency_history; fallback para heurística estatística
        confidence = 0.5
        try:
            eff_path = os.path.join(BASE_DIR, 'logs', 'efficiency_history.json')
            if os.path.exists(eff_path):
                with open(eff_path, 'r', encoding='utf-8') as ef:
                    eff_hist = json.load(ef)
                if eff_hist:
                    latest = eff_hist[-1]
                    g = latest.get('global', {})
                    # Prefere Cov@20 (Recall@20) = % das zonas de tensão no top-20
                    cov = g.get('p20')
                    if cov is not None:
                        confidence = float(cov)
        except Exception:
            pass

        if confidence == 0.5 and scores and s_std > 0:
            separation = (np.max(scores) - np.mean(scores)) / s_std
            confidence = min(0.98, 0.6 + (separation / 10.0))

        return jsonify({
            'monitoring_active': True,
            'anomaly_level': float(tension),
            'anomaly_risk_level': label,
            'active_events': active_events[:5], # Top 5 eventos
            'model_confidence': float(confidence),
            'last_check': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'monitoring_active': True, 'error': str(e), 'anomaly_level': 0.0})


@app.route('/api/explain/<int:node_id>')
def explain_node(node_id, scores_map_override=None, temporal_profile_override=None):
    """Retorna uma explicação resumida dos motivos de criticidade para um nó (região/localidade).
    Implementação leve que responde mesmo sem o gerador de explicações completo disponível.
    """
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
    try:
        if node_id not in list(nodes_gdf.index):
            return jsonify({'error': 'node not found'}), 404

        row = nodes_gdf.loc[node_id]
        # Tentar obter nome real do bairro ou cidade
        name = str(row.get('name') or row.get('bairro') or row.get('municipio') or 'Localidade Desconhecida')
        name_norm = normalize_name(name)

        # ... (mantendo lógica de scores e ranking) ...
        selected_model_mode = normalize_model_mode(request.args.get('model_mode', DEFAULT_MODEL_MODE))
        selected_model_meta, _ = _get_model_selection_meta(selected_model_mode)
        scores_map = scores_map_override if scores_map_override is not None else _score_map_for_model_mode(selected_model_mode)
        if selected_model_mode != 'stgat':
            fortaleza_override_scores = selected_model_meta.get('fortaleza_scores') or {}
            if fortaleza_override_scores:
                scores_map = dict(scores_map)
                for _, override_row in nodes_gdf.iterrows():
                    override_name = normalize_name(str(override_row.get('name') or override_row.get('bairro') or ''))
                    override_region = str(override_row.get('regiao') or override_row.get('region_type') or '').lower()
                    if override_region == 'capital':
                        override_region = 'fortaleza'
                    if override_region == 'fortaleza':
                        override_score = _resolve_optional_model_score(override_name, fortaleza_override_scores)
                        if override_score is not None:
                            scores_map[override_name] = normalize_risk_score(override_score)
        score_pct = float(scores_map.get(name_norm, 20.0))
        score_10 = score_pct / 10.0
        component_details = {}
        try:
            if selected_model_mode == 'stgat':
                component_details = orchestrator.get_last_component_details()
        except Exception:
            component_details = {}
        component_meta = component_details.get(name_norm, {}) if isinstance(component_details, dict) else {}
        if selected_model_mode != 'stgat':
            component_meta = {
                **component_meta,
                'model_family': selected_model_meta.get('label') or DEFAULT_MODEL_LABEL,
                'primary_signal_label': selected_model_meta.get('label') or DEFAULT_MODEL_LABEL,
            }
        
        # (pulei blocos intermediários de ranking para brevidade no replace)
        all_scores = []
        node_score_pairs = []
        for i, r in nodes_gdf.iterrows():
            nname = normalize_name(str(r.get('name') or r.get('bairro') or ''))
            s = float(scores_map.get(nname, 20.0))
            all_scores.append(s)
            node_score_pairs.append((i, s))
            
        # ... (lógica de rank e tier mantida) ...
        sorted_by_score = sorted(node_score_pairs, key=lambda x: x[1], reverse=True)
        ranks = {nid: idx + 1 for idx, (nid, _) in enumerate(sorted_by_score)}
        rank_pos = ranks.get(node_id, len(sorted_by_score))
        total_nodes = len(sorted_by_score)
        score_mean = float(np.mean(all_scores)) if all_scores else 0.0
        score_std = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
        score_median = float(np.median(all_scores)) if all_scores else 0.0
        score_gap_pct = float(score_pct - score_mean)
        score_zscore = float(score_gap_pct / score_std) if score_std > 1e-6 else 0.0
        top_slice_pct = float((rank_pos / max(1, total_nodes)) * 100.0)

        pct_rank = rank_pos / max(1, len(sorted_by_score))
        if rank_pos <= 5: tier = 'top_5'
        elif pct_rank <= 0.2: tier = 'long_tail_20'
        elif pct_rank <= 0.5: tier = 'long_tail_50'
        else: tier = 'tail'

        nearby = []
        try:
            region_type = str(row.get('region_type', '')).lower()
            peers = [nid for nid, s in node_score_pairs if nid != node_id and str(nodes_gdf.loc[nid].get('region_type','')).lower() == region_type]
            if not peers: peers = [nid for nid, s in sorted_by_score if nid != node_id]
            nearby = peers[:3]
        except: nearby = []

        global _EXOGENOUS_EVENTS_CACHE
        events = []
        events_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        last_date_base = orchestrator.dates[-1] if (orchestrator and hasattr(orchestrator, 'dates')) else None
        try:
            if _EXOGENOUS_EVENTS_CACHE is None:
                if os.path.exists(events_path):
                    with open(events_path, 'r', encoding='utf-8') as ef:
                        _EXOGENOUS_EVENTS_CACHE = json.load(ef) or []
                else:
                    _EXOGENOUS_EVENTS_CACHE = []
            
            for e in _EXOGENOUS_EVENTS_CACHE:
                e_date_str = e.get('date') or e.get('event_date')
                if not verify_date_consistency(e_date_str, last_date_base): continue

                # Match robusto por bairro ou município
                evt_bairro = normalize_name(str(e.get('bairro', '')))
                evt_mun = normalize_name(str(e.get('municipio', '')))
                evt_title = normalize_name(str(e.get('title', '')))
                evt_loc = normalize_name(str(e.get('location', '')))

                if (name_norm and (name_norm == evt_bairro or name_norm in evt_title or name_norm in evt_loc)) or \
                   (not evt_bairro and evt_mun and (evt_mun == name_norm or name_norm in evt_mun)):
                    events.append(e)
        except Exception as e_evt:
            logging.warning(f"Erro ao filtrar eventos exógenos: {e_evt}")
            events = []

        region_type = str(row.get('region_type') or row.get('regiao') or '').lower()
        faction = str(row.get('faction') or 'NEUTRO').upper()
        tension_index = float(row.get('tension_index', 0.0) or 0.0)

        event_types = []
        total_event_intensity = 0.0
        critical_event_count = 0
        suppression_event_count = 0
        conflict_intensity = 0.0
        suppression_intensity = 0.0
        for event in events:
            raw_intensity = event.get('intensity') or event.get('severity') or 1.0
            try:
                intensity = float(raw_intensity)
            except (TypeError, ValueError):
                intensity = 1.0
            total_event_intensity += intensity

            is_suppression = bool(event.get('is_qualified_suppression') is True or event.get('is_suppression') is True)
            if is_suppression:
                suppression_event_count += 1
                suppression_intensity += intensity
            else:
                critical_event_count += 1
                conflict_intensity += intensity

            event_type = str(event.get('natureza') or event.get('tipo') or event.get('title') or event.get('descricao') or '').strip()
            if event_type and event_type not in event_types:
                event_types.append(event_type)

        global _RUAS_CRITICAS_FORTALEZA_CACHE, _STREETS_BY_MUNICIPIO_CACHE
        critical_streets = 'Sem logradouros críticos recentes'
        critical_streets_count = 0
        try:
            if region_type == 'fortaleza':
                if _RUAS_CRITICAS_FORTALEZA_CACHE is None:
                    streets_path = os.path.join(BASE_DIR, 'data', 'raw', 'ruas_criticas_por_bairro.json')
                    if os.path.exists(streets_path):
                        with open(streets_path, 'r', encoding='utf-8') as sf:
                            streets_cache = json.load(sf) or {}
                        _RUAS_CRITICAS_FORTALEZA_CACHE = {normalize_name(k): v for k, v in streets_cache.items() if k}
                    else:
                        _RUAS_CRITICAS_FORTALEZA_CACHE = {}
                
                critical_streets = _RUAS_CRITICAS_FORTALEZA_CACHE.get(name_norm)
                if critical_streets is None:
                    for cache_key, cache_value in _RUAS_CRITICAS_FORTALEZA_CACHE.items():
                        if name_norm in cache_key or cache_key in name_norm:
                            critical_streets = cache_value
                            break
            else:
                if _STREETS_BY_MUNICIPIO_CACHE is None:
                    streets_path = os.path.join(BASE_DIR, 'data', 'streets_by_municipio.json')
                    if os.path.exists(streets_path):
                        with open(streets_path, 'r', encoding='utf-8') as sf:
                            _STREETS_BY_MUNICIPIO_CACHE = json.load(sf) or {}
                    else:
                        _STREETS_BY_MUNICIPIO_CACHE = {}
                critical_streets = _STREETS_BY_MUNICIPIO_CACHE.get(name_norm)
        except Exception as e:
            logging.warning(f"Erro ao carregar ruas críticas para explicação de {name}: {e}")
            critical_streets = 'Sem logradouros críticos recentes'

        if isinstance(critical_streets, list):
            critical_streets = [str(item).strip() for item in critical_streets if str(item).strip()]
            critical_streets_count = len(critical_streets)
            if not critical_streets:
                critical_streets = 'Sem logradouros críticos recentes'
        else:
            critical_streets = str(critical_streets or 'Sem logradouros críticos recentes').strip()
            if critical_streets and not critical_streets.lower().startswith('sem logradouros'):
                critical_streets_count = len([part for part in re.split(r',|;', critical_streets) if part.strip()])

        temporal_pattern = 'Increasing' if score_pct > score_mean else 'Stable'
        heuristic_confidence = min(0.95, 0.6 + (score_10 / 20.0))

        # --- EXTRAÇÃO DE DADOS REAIS DOS TENSORES (PARA EXPLICABILIDADE) ---
        cvli_recent_7 = 0
        cvli_prev_7 = 0
        cvli_recent = 0
        cvli_prev = 0
        cvli_recent_30 = 0
        cvli_prev_30 = 0
        vehicles_recent_14 = 0.0
        intel_recent_14 = 0.0
        rolling_cvli_7d = 0.0
        global_cvli_latest = 0.0
        rain_acc_14 = 0.0
        rainy_days_14 = 0
        holiday_days_14 = 0
        hot_days_14 = 0
        weekend_days_14 = 0
        geo_neighbor_count = 0
        conflict_neighbor_count = 0
        high_risk_neighbor_count = 0
        neighbor_mean_score = 0.0
        neighbor_max_score = 0.0
        nearby_names = []

        try:
            reg_key = str(row.get('regiao', 'fortaleza')).lower()
            if reg_key == 'capital': reg_key = 'fortaleza'
            # Sincronização RMF Oficial (via índice dinâmico)
            if name_norm in _RMF_NODES:
                reg_key = 'rmf'            
            spec = orchestrator.specialists.get(reg_key)
            if spec:
                # 1. Encontrar o índice do nó no especialista
                spec_nodes = spec['data']['nodes_gdf']
                spec_idx = next((idx for idx, r in spec_nodes.iterrows() if normalize_name(r['name']) == name_norm), None)
                
                if spec_idx is not None:
                    features = spec['data']['node_features'] # (N, T, F)
                    # Janela Recente (Últimos 14 dias) vs Anterior (14 dias antes disso)
                    cvli_recent_7 = int(features[spec_idx, -7:, 0].sum())
                    cvli_prev_7 = int(features[spec_idx, -14:-7, 0].sum()) if features.shape[1] >= 14 else 0
                    cvli_recent = int(features[spec_idx, -14:, 0].sum())
                    cvli_prev = int(features[spec_idx, -28:-14, 0].sum())
                    cvli_recent_30 = int(features[spec_idx, -30:, 0].sum()) if features.shape[1] >= 30 else cvli_recent
                    cvli_prev_30 = int(features[spec_idx, -60:-30, 0].sum()) if features.shape[1] >= 60 else 0
                    vehicles_recent_14 = float(features[spec_idx, -14:, 1].sum()) if features.shape[2] > 1 else 0.0
                    intel_recent_14 = float(features[spec_idx, -14:, 27].sum()) if features.shape[2] > 27 else 0.0
                    rolling_cvli_7d = float(features[spec_idx, -1, 24]) if features.shape[2] > 24 else 0.0
                    global_cvli_latest = float(features[spec_idx, -1, 28]) if features.shape[2] > 28 else 0.0
                    holiday_days_14 = int(features[spec_idx, -14:, 29].sum()) if features.shape[2] > 29 else 0
                    hot_days_14 = int(features[spec_idx, -14:, 30].sum()) if features.shape[2] > 30 else 0
                    rain_acc_14 = float(features[spec_idx, -14:, 31].sum()) if features.shape[2] > 31 else 0.0
                    rainy_days_14 = int(features[spec_idx, -14:, 32].sum()) if features.shape[2] > 32 else 0
                    weekend_days_14 = int(features[spec_idx, -14:, 22].sum()) if features.shape[2] > 22 else 0
                    
                    # 2. Vizinhos Geográficos Reais (via Matriz de Adjacência)
                    adj_geo = spec['data']['adj_geo']
                    neighbor_indices = np.where(adj_geo[spec_idx] > 0)[0]
                    geo_neighbor_count = max(0, len(neighbor_indices) - 1)
                    adj_conflict = spec['data']['adj_conflict']
                    conflict_neighbor_count = max(0, int(np.sum(adj_conflict[spec_idx] > 0)) - 1)
                    
                    # Pegar os 3 vizinhos com maior risco atual para o "efeito de contágio"
                    n_scores = []
                    for n_idx in neighbor_indices:
                        if n_idx == spec_idx: continue
                        n_name = normalize_name(spec_nodes.iloc[n_idx]['name'])
                        n_score = float(scores_map.get(n_name, 0))
                        n_scores.append((n_name, n_score))
                    
                    # Ordenar por risco e pegar nomes
                    n_scores.sort(key=lambda x: x[1], reverse=True)
                    nearby_names = [x[0] for x in n_scores[:3]]
                    if n_scores:
                        neighbor_values = [item[1] for item in n_scores]
                        neighbor_mean_score = float(np.mean(neighbor_values))
                        neighbor_max_score = float(np.max(neighbor_values))
                        high_risk_neighbor_count = sum(1 for _, n_score in n_scores if n_score >= 50.0)
                    
                    logging.info(f"📊 EXPLAIN [{name}]: recent={cvli_recent}, prev={cvli_prev}, neighbors={nearby_names}")
        except Exception as e:
            logging.warning(f"Erro ao extrair métricas reais para {name}: {e}")

        temporal_profile = {}
        if temporal_profile_override is not None:
            temporal_profile = temporal_profile_override
        elif request.args.get('include_temporal', '1') != '0':
            try:
                temporal_profiles = _build_predictive_temporal_profiles(
                    reference_date=orchestrator.dates[-1] if getattr(orchestrator, 'dates', None) is not None else None,
                    horizon_days=30,
                )
                temporal_profile = temporal_profiles.get(_temporal_profile_key(region_type, name_norm), {})
            except Exception as e:
                logging.warning(f"Erro ao extrair perfil temporal preditivo para {name}: {e}")

        # Criar contexto esperado por ExplanationGenerator
        try:
            from src.explanation_generator import ExplanationGenerator
            gen = ExplanationGenerator()
            
            context = {
                'node_id': int(node_id),
                'name': name,
                'score': score_10,
                'score_pct': float(score_pct),
                'avg_score_pct': score_mean,
                'median_score_pct': score_median,
                'score_gap_pct': score_gap_pct,
                'score_zscore': score_zscore,
                'temporal_pattern': 'Increasing' if cvli_recent > cvli_prev else 'Stable',
                'cvli_recent_7': cvli_recent_7,
                'cvli_prev_7': cvli_prev_7,
                'cvli_count_recent': cvli_recent,
                'cvli_count_prev': cvli_prev,
                'cvli_recent_30': cvli_recent_30,
                'vehicles_recent_14': vehicles_recent_14,
                'intel_recent_14': intel_recent_14,
                'rolling_cvli_7d': rolling_cvli_7d,
                'global_cvli_latest': global_cvli_latest,
                'rain_acc_14': rain_acc_14,
                'rainy_days_14': rainy_days_14,
                'holiday_days_14': holiday_days_14,
                'hot_days_14': hot_days_14,
                'weekend_days_14': weekend_days_14,
                'nearby_nodes': nearby,
                'nearby_impact_names': nearby_names,
                'geo_neighbor_count': geo_neighbor_count,
                'conflict_neighbor_count': conflict_neighbor_count,
                'high_risk_neighbor_count': high_risk_neighbor_count,
                'neighbor_mean_score': neighbor_mean_score,
                'neighbor_max_score': neighbor_max_score,
                'events': events,
                'events_count_total': len(events),
                'critical_event_count': critical_event_count,
                'suppression_event_count': suppression_event_count,
                'event_types': event_types,
                'event_types_count': len(event_types),
                'total_event_intensity': total_event_intensity,
                'conflict_intensity': conflict_intensity,
                'suppression_intensity': suppression_intensity,
                'confidence': float(heuristic_confidence),
                'heuristic_confidence_seed': float(heuristic_confidence),
                'tier': tier,
                'total_nodes': total_nodes,
                'top_slice_pct': top_slice_pct,
                'region_type': region_type,
                'faction': faction,
                'tension_index': tension_index,
                'trend_label': str(temporal_pattern).lower(),
                'critical_streets': critical_streets,
                'critical_streets_count': critical_streets_count,
                'model_family': str(component_meta.get('model_family') or 'Poisson Ranker'),
                'model_architecture': RISK_MODEL_NAME,
                'primary_signal_label': str(component_meta.get('primary_signal_label') or 'Sinal Poisson do ranking operacional'),
                'model_signal_score': float(component_meta.get('model_signal_score', component_meta.get('neural_score', 0.0)) or 0.0),
                'territorial_support_pct': float(component_meta.get('territorial_support_pct', 0.0) or 0.0),
                'historical_support_pct': float(component_meta.get('historical_support_pct', 0.0) or 0.0),
                'live_support_pct': float(component_meta.get('live_support_pct', 0.0) or 0.0),
                'expected_cvli_30d': float(component_meta.get('expected_cvli_30d', 0.0) or 0.0),
                'peak_hours': temporal_profile.get('peak_hours', ''),
                'peak_weekday': temporal_profile.get('peak_weekday', ''),
                'peak_time_label': temporal_profile.get('peak_time_label', ''),
                'peak_hour_share': temporal_profile.get('peak_hour_share', 0.0),
                'peak_weekday_share': temporal_profile.get('peak_weekday_share', 0.0),
                'temporal_sample_size': temporal_profile.get('temporal_sample_size', 0),
            }

            explanation = gen.explain_node_ranking(int(node_id), int(rank_pos), context)
            explanation['risk_score_pct'] = float(score_pct)
            explanation['model_family'] = context['model_family']
            explanation['model_architecture'] = context['model_architecture']
            explanation['primary_signal_label'] = context['primary_signal_label']
            explanation['model_signal_score'] = context['model_signal_score']
            explanation['expected_cvli_30d'] = context['expected_cvli_30d']
            explanation.update(temporal_profile)
            explanation['territorial_support_pct'] = context['territorial_support_pct']
            explanation['historical_support_pct'] = context['historical_support_pct']
            explanation['live_support_pct'] = context['live_support_pct']
            explanation['model_drivers'] = [
                {
                    'label': context['primary_signal_label'],
                    'value_pct': round(float(context['model_signal_score']), 1),
                },
                {
                    'label': 'Suporte territorial',
                    'value_pct': round(float(context['territorial_support_pct']), 1),
                },
                {
                    'label': 'Atividade recente e vizinhança',
                    'value_pct': round(float(component_meta.get('inclusion_score', 0.0) or 0.0), 1),
                },
            ]

            # Indicador executivo de tendência futura para o horizonte de 30 dias.
            delta_30d = int(cvli_recent_30 - cvli_prev_30)
            if delta_30d >= 2:
                trend_direction = 'up'
                trend_label = 'Alta de Risco'
                trend_message = 'Pressão criminal em aceleração nas últimas janelas comparáveis.'
            elif delta_30d <= -2:
                trend_direction = 'down'
                trend_label = 'Queda de Risco'
                trend_message = 'Sinal de arrefecimento recente, mantendo monitoramento ativo.'
            else:
                trend_direction = 'stable'
                trend_label = 'Estável'
                trend_message = 'Sem ruptura relevante de tendência no curto prazo.'

            explanation['future_trend'] = {
                'direction': trend_direction,
                'label': trend_label,
                'delta_30d': delta_30d,
                'cvli_recent_30': int(cvli_recent_30),
                'cvli_prev_30': int(cvli_prev_30),
                'message': trend_message,
                'horizon_days': 30,
            }

            # Percentil de confiança na previsão
            conf_pct = float(explanation.get('confidence_pct', round(float(explanation.get('confidence', heuristic_confidence)) * 100.0, 1)))
            if conf_pct >= 80:
                conf_label = 'Alta'
            elif conf_pct >= 60:
                conf_label = 'Moderada'
            elif conf_pct >= 40:
                conf_label = 'Baixa'
            else:
                conf_label = 'Muito baixa'
            explanation['confidence_pct'] = conf_pct
            explanation['confidence_label'] = explanation.get('confidence_label') or conf_label

            # ENVIAR DIRETAMENTE PARA O FRONTEND (Sem normalização que apaga campos)
            return jsonify(explanation)
        except Exception as e:
            # Fail-safe: do not return HTTP error. Provide a consistent JSON response
            # indicating the detailed explanation is unavailable while preserving
            # the node id, name and score so the frontend can continue rendering.
            safe_resp = {
                'node_id': node_id,
                'name': name,
                'risk_score_pct': float(score_pct),
                'confidence': float(heuristic_confidence),
                'summary': 'Métricas e explicabilidade indisponíveis',
                'factors': [],
                'caveats': [],
                'explanation_available': False,
                'source': 'unavailable',
                'model_family': str(component_meta.get('model_family') or 'Poisson Ranker'),
                'model_architecture': RISK_MODEL_NAME,
            }
            # Log the underlying exception for diagnostics without crashing
            logging.exception('ExplanationGenerator failed: %s', e)
            return jsonify(safe_resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    """Parse raw CIOPS-like text into structured events using llm_service.process_exogenous_text
    Expects JSON: { text: '...raw lines...' }
    Returns: { points: [ {bairro, municipio, resumo, ...}, ... ] }
    """
    try:
        payload = request.get_json(force=True) or {}
        text = payload.get('text') or payload.get('raw') or ''
        if not text or not text.strip():
            return jsonify({'error': 'empty_text'}), 400

        try:
            from src.llm_service import process_exogenous_text
        except Exception as e:
            # Friendly JSON response that frontend can render in a user-facing modal
            friendly = {
                'error': 'llm_service_unavailable',
                'title': 'Serviço de extração temporariamente indisponível',
                'message': 'Não foi possível processar automaticamente o texto de origem. Tente novamente em alguns minutos ou entre com os pontos manualmente.',
                'detail': str(e)
            }
            return jsonify(friendly), 503

        try:
            parsed = process_exogenous_text(text)
        except Exception as e:
            friendly = {
                'error': 'llm_processing_failed',
                'title': 'Erro ao processar o texto',
                'message': 'O servidor encontrou um problema ao tentar interpretar os dados. Tente novamente ou informe o problema ao suporte.',
                'detail': str(e)
            }
            return jsonify(friendly), 503

        # Return parsed items as 'points' for frontend compatibility
        return jsonify({'points': parsed, 'count': len(parsed)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
    """Save parsed exogenous events to disk for downstream processing.
    Expects JSON: { points: [...], original_text: '...' }
    Writes to `data/exogenous_events.json` and `data/exogenous_events_geocoded.json`.
    """
    try:
        payload = request.get_json(force=True) or {}
        points = payload.get('points') or []
        original = payload.get('original_text', '')

        if not isinstance(points, list) or len(points) == 0:
            return jsonify({'error': 'no_points'}), 400

        # Normalize minimal fields and add ingest metadata
        for p in points:
            if isinstance(p, dict):
                p.setdefault('ingested_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                date_val = p.get('date')
                ts_val = p.get('timestamp') or p.get('time') or p.get('hora')
                raw_text = (p.get('raw_text') or p.get('descricao') or '')

                def _normalize_time_part(t):
                    if not t or not isinstance(t, str):
                        return "00:00:00"
                    t = t.strip()
                    m = re.match(r'^(\d{1,2}:\d{2})(:?\d{0,2})$', t)
                    if m:
                        hhmm = m.group(1)
                        rest = m.group(2) or ''
                        if rest and rest.startswith(':') and len(rest) == 3:
                            return hhmm + rest
                        return hhmm + ':00'
                    return "00:00:00"

                # 1. Tentar extrair data do raw_text se não houver date_val
                extracted_date = None
                if raw_text and isinstance(raw_text, str):
                    mdate = re.search(r"(\d{4}-\d{2}-\d{2})", raw_text)
                    if not mdate:
                        mdate = re.search(r"(\d{2}/\d{2}/\d{4})", raw_text)
                    if mdate:
                        extracted_date = mdate.group(1)
                        if '/' in extracted_date:
                            dparts = extracted_date.split('/')
                            extracted_date = f"{dparts[2]}-{dparts[1]}-{dparts[0]}"

                # 1. Definir a parte da DATA (Prioridade: date_val > extracted_date > ingested_at)
                final_date_part = None
                if date_val and isinstance(date_val, str):
                    # Se vier "YYYY-MM-DD HH:MM:SS", extrair apenas a data
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", date_val)
                    if m:
                        final_date_part = m.group(1)
                
                if not final_date_part and extracted_date:
                    final_date_part = extracted_date
                
                if not final_date_part:
                    ing = p.get('ingested_at')
                    final_date_part = ing[:10] if ing else datetime.now().strftime('%Y-%m-%d')

                # 2. Definir a parte do HORÁRIO
                # Se date_val já continha hora, tenta extrair dela primeiro
                final_time_part = None
                if date_val and isinstance(date_val, str) and len(date_val) > 10:
                    mt = re.search(r"(\d{2}:\d{2}(?::\d{2})?)", date_val)
                    if mt: final_time_part = mt.group(1)
                
                if not final_time_part:
                    final_time_part = _normalize_time_part(ts_val)
                
                # 3. Combinar e Garantir HH:MM:SS
                if len(final_time_part) == 5: final_time_part += ":00"
                
                # Reconstruir o dicionário para que 'date' seja a última chave (Python 3.7+ mantém a ordem de inserção)
                event_data = {}
                for key, val in p.items():
                    if key != 'date':
                        event_data[key] = val
                
                event_data['date'] = f"{final_date_part} {final_time_part}"
                
                if 'raw_text' not in event_data and original:
                    event_data['raw_text'] = original
                
                # Substituir p pelo novo dicionário ordenado
                p.clear()
                p.update(event_data)

        # Save raw parsed to exogenous_events.json
        raw_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        try:
            if os.path.exists(raw_path):
                with open(raw_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f) or []
                if isinstance(existing, dict) and 'events' in existing:
                    existing_list = existing['events']
                elif isinstance(existing, list):
                    existing_list = existing
                else:
                    existing_list = []
            else:
                existing_list = []
        except Exception:
            existing_list = []

        # Append and write back
        existing_list.extend(points)
        try:
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(existing_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return jsonify({'error': 'write_failed', 'detail': str(e)}), 500

        # Note: do not create a separate 'geocoded' file here. The system
        # should maintain a single canonical `exogenous_events.json` file.
        # Other components that require geocoded/enriched data should read
        # this file and perform their own enrichment rather than relying
        # on a duplicate file being written here.

        invalidate_api_risk_cache()
        return jsonify({'saved': len(points)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/manager_explanations/cache', methods=['GET'])
def get_manager_cache():
    """Return the full manager_explanations cache JSON (for debugging/inspection)."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                data = json.load(cf) or {}
        else:
            data = {}
        return jsonify({'cache': data})
    except Exception as e:
        logging.exception('Failed reading manager_explanations cache: %s', e)
        return jsonify({'error': 'cache_read_failed', 'detail': str(e)}), 500


@app.route('/api/manager_explanations/cache/<node_id>', methods=['DELETE'])
def delete_manager_cache_node(node_id):
    """Invalidate cached manager text for a specific node_id."""
    try:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                    cache = json.load(cf) or {}
            except Exception:
                cache = {}
        else:
            cache = {}

        if str(node_id) in cache:
            del cache[str(node_id)]
            try:
                with open(CACHE_FILE, 'w', encoding='utf-8') as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.exception('Failed saving cache after delete: %s', e)
                return jsonify({'error': 'cache_write_failed', 'detail': str(e)}), 500
            return jsonify({'deleted': node_id})
        return jsonify({'deleted': None, 'reason': 'not_found'})
    except Exception as e:
        logging.exception('Error deleting cache node: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/manager_explanations/cache/clear', methods=['POST'])
def clear_manager_cache():
    """Clear the entire manager_explanations cache."""
    try:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as cf:
                json.dump({}, cf, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.exception('Failed clearing cache: %s', e)
            return jsonify({'error': 'cache_clear_failed', 'detail': str(e)}), 500
        return jsonify({'cleared': True})
    except Exception as e:
        logging.exception('Error clearing cache: %s', e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/exogenous-events')
def get_exogenous_events_list():
    """Retorna a lista de eventos exógenos para o dashboard estratégico."""
    try:
        path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f) or []
            # Inverter para mostrar os mais recentes primeiro
            return jsonify(list(reversed(data)))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exogenous/sync-sheets', methods=['POST'])
def sync_exogenous_from_sheets():
    """Sync exogenous events from Google Sheets CSV URL"""
    try:
        from src.google_sheets_sync import sync_google_sheets
        from dotenv import load_dotenv
        load_dotenv()
        
        csv_url = os.environ.get("GOOGLE_SHEETS_CSV_URL")
        payload = request.get_json(silent=True) or {}
        if not csv_url:
             csv_url = payload.get("csv_url")
             
        if not csv_url:
            return jsonify({"status": "error", "message": "GOOGLE_SHEETS_CSV_URL não configurada no .env e não enviada no payload."}), 400
            
        exogenous_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        result = sync_google_sheets(csv_url, exogenous_path)
        
        invalidate_api_risk_cache()
        return jsonify(result), 200 if result.get("status") == "success" else 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exogenous/pending-count')
def get_pending_exogenous_count():
    """Check how many new events in Google Sheets haven't been imported yet."""
    try:
        import csv as csv_mod
        from io import StringIO
        from dotenv import load_dotenv
        from src.google_sheets_sync import _download_sheets_csv
        load_dotenv()
        
        csv_url = os.environ.get("GOOGLE_SHEETS_CSV_URL")
        if not csv_url:
            return jsonify({"pending": 0, "total_sheet": 0, "total_local": 0}), 200

        csv_text = _download_sheets_csv(csv_url, timeout=10)
        reader = csv_mod.reader(StringIO(csv_text))
        rows = list(reader)

        # Detectar se a planilha tem coluna de ID explícita ou começa direto com Data/Hora.
        # Se o cabeçalho da primeira coluna não for numérico/id-like, a "chave" é o
        # conteúdo normalizado da linha inteira (mesmo critério usado no sync).
        from src.google_sheets_sync import normalize_name as _norm
        from datetime import datetime as _dt, timedelta as _td
        _now = _dt.now()
        _cutoff = (_now - _td(days=7)).date()

        # Construir conjunto de todas as chaves já conhecidas (local + archives)
        known_keys = set()
        exogenous_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        if os.path.exists(exogenous_path):
            with open(exogenous_path, 'r', encoding='utf-8') as f:
                try:
                    for ev in json.load(f):
                        if ev.get("id"): known_keys.add(str(ev["id"]).strip())
                        if ev.get("raw_text"): known_keys.add(_norm(ev["raw_text"]))
                except json.JSONDecodeError:
                    pass

        archives_dir = os.path.join(BASE_DIR, 'data', 'archives')
        if os.path.exists(archives_dir):
            for arch_file in os.listdir(archives_dir):
                if arch_file.endswith('.json'):
                    try:
                        with open(os.path.join(archives_dir, arch_file), 'r', encoding='utf-8') as af:
                            for ev in json.load(af):
                                if ev.get("id"): known_keys.add(str(ev["id"]).strip())
                                if ev.get("raw_text"): known_keys.add(_norm(ev["raw_text"]))
                    except Exception:
                        pass

        # Contar eventos pendentes por linha (não por chave) para evitar dupla contagem
        pending = 0
        for r in rows[1:]:
            # Ignorar linhas sem conteúdo além da data/hora (linhas fantasma)
            if not r or not any(col.strip() for col in r[1:]):
                continue
            # Ignorar linhas fora da janela de 7 dias (mesmo critério do sync)
            try:
                date_val = _dt.fromisoformat(r[0].strip().replace('Z', '+00:00')).date()
                if date_val < _cutoff:
                    continue
            except Exception:
                pass
            ev_id = r[0].strip()
            desc = r[4].strip() if len(r) > 4 else ""
            desc_norm = _norm(desc) if desc else _norm(" ".join(c.strip() for c in r if c.strip()))
            # Evento é conhecido se qualquer uma das suas chaves já existir
            if ev_id in known_keys or (desc_norm and desc_norm in known_keys):
                continue
            pending += 1
        total_sheet = sum(
            1 for r in rows[1:]
            if r and any(col.strip() for col in r[1:])
        )
        total_local = len(known_keys)
        return jsonify({
            "pending": pending,
            "total_sheet": total_sheet,
            "total_local": total_local,
        }), 200
    except Exception as e:
        return jsonify({"pending": 0, "error": str(e)}), 200

stgcn_engine = None

@app.route('/api/stgcn/escape-routes', methods=['GET'])
def predict_escape():
    """Calcula vetores de fuga prováveis (ST-GCN) a partir de um ponto geolocalizado."""
    global stgcn_engine
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except (TypeError, ValueError):
        return jsonify({"error": "Parâmetros lat e lon são necessários e devem ser números."}), 400
    try:
        max_distance = max(250, min(2000, int(float(request.args.get('max_distance', 1000)))))
    except Exception:
        max_distance = 1000
        
    try:
        if stgcn_engine is None:
            from src.core.stgcn_escape_engine import STGCNEscapeEngine
            stgcn_engine = STGCNEscapeEngine(data_dir=os.path.join(BASE_DIR, 'data', 'static'), base_dir=BASE_DIR)
        
        result = stgcn_engine.predict_escape_routes(lat, lon, max_distance=max_distance)
        return jsonify(result), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/efficiency-latest')
def get_efficiency_latest():
    """Retorna as métricas mais recentes do monitor de eficiência."""
    try:
        if efficiency_monitor:
            latest = efficiency_monitor.get_latest_metrics()
            return jsonify(latest if latest else {})
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cache em memória e em arquivo para pesos de calibração do agente de background
AGENT_CALIBRATION_CACHE_FILE = os.path.join(BASE_DIR, 'data', 'agent_calibrated_weights.json')
_agent_calibration_state = {
    "status": "idle",
    "last_calibration": None,
    "error": None
}


def _persist_agent_result(result: dict):
    global _agent_calibration_state

    _agent_calibration_state["status"] = "success"
    _agent_calibration_state["error"] = None
    _agent_calibration_state["last_calibration"] = result

    os.makedirs(os.path.dirname(AGENT_CALIBRATION_CACHE_FILE), exist_ok=True)
    with open(AGENT_CALIBRATION_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    hist_file = os.path.join(BASE_DIR, 'logs', 'agent_calibrations_history.json')
    hist_data = []
    if os.path.exists(hist_file):
        with open(hist_file, 'r', encoding='utf-8') as hf:
            hist_data = json.load(hf)
    hist_data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights": result.get("calibrated_weights"),
        "explanations": result.get("explanations"),
        "target_region": result.get("target_region"),
    })
    hist_data = hist_data[-50:]
    with open(hist_file, 'w', encoding='utf-8') as hf:
        json.dump(hist_data, hf, indent=2, ensure_ascii=False)

    try:
        from src.agent.calibration_memory import record_agent_decision
        record_agent_decision(BASE_DIR, result, source='app_persist')
    except Exception as exc:
        print(f"⚠️ [Calibration Memory] Falha ao registrar decisão: {exc}")


def _build_live_agent_payload(region: str = 'global'):
    if orchestrator is None:
        return None, None

    scores_map = orchestrator.get_combined_risk()
    top_predictions = [
        {'name': name, 'score': round(float(score), 4)}
        for name, score in sorted(scores_map.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    confidence = confidence_tracker.get_current_confidence(region=region) if confidence_tracker else {}
    raw_stgcn_data = {
        'timestamp': datetime.now().isoformat(),
        'region': region,
        'confidence_scores': confidence,
        'top_predictions': top_predictions,
        'combined_risk_sample_size': len(scores_map),
    }
    user_profile = {
        'region': region.upper(),
        'focus': 'CVLI',
        'historical_alerts': len(health_monitor.get_active_alerts()) if health_monitor else 0,
    }
    return raw_stgcn_data, user_profile


def _handle_agent_intervention(event: dict):
    result = {
        'status': event.get('status', 'success'),
        'manager_decision': 'Intervenção automática aplicada ao modelo por desvio semântico ou erro de convergência.',
        'calibrated_weights': event.get('new_params'),
        'explanations': ((event.get('agent_review') or {}).get('explanations')
                         or ((event.get('agent_review') or {}).get('data_analysis') or {}).get('technical_summary')
                         or f"{event.get('region', '').upper()}: ajuste automático aplicado para {event.get('metric', '')}."),
        'data_analysis': (event.get('agent_review') or {}).get('data_analysis', {}),
        'calibrated_at': event.get('timestamp'),
        'target_region': event.get('region'),
        'affected_metric': event.get('metric'),
        'operational_params': event.get('new_params'),
    }
    _persist_agent_result(result)
    invalidate_api_risk_cache()
    _set_model_update_status(
        status='updating_models',
        progress=100,
        message='Intervenção automática aplicada. Recalculando risco e sincronizando dashboards...',
        error=None,
        bump_revision=True,
        ttl_seconds=30,
    )

@app.route('/api/agent/calibrate-report', methods=['POST'])
def run_agent_calibration():
    """
    Endpoint legado do agente local. Desativado: o sistema usa apenas o
    auto-ajuste deterministico do orquestrador.
    """
    return jsonify({
        'status': 'disabled',
        'message': 'Agente Ollama desativado. Auto-ajuste deterministico permanece no orquestrador.'
    }), 410

@app.route('/api/agent/calibration-status', methods=['GET'])
def get_agent_calibration_status():
    """Retorna status do agente local legado."""
    return jsonify({
        "status": "disabled",
        "error": None,
        "last_calibration": None,
    }), 200

if __name__ == "__main__":
    import atexit
    
    # Registrar cleanup ao desligar
    def _cleanup_on_shutdown():
        """Para daemons e faz cleanup ao desligar o app."""
        print("\n[SHUTDOWN] Parando daemons...")
        
        # Sinaliza para o monitor de agentes (listen_agents.py) que a aplicação foi encerrada
        try:
            shutdown_file = os.path.join(BASE_DIR, 'logs', '.app_shutdown')
            with open(shutdown_file, 'w') as f:
                f.write('shutdown')
            print("[SHUTDOWN] ✅ Sinal de parada enviado ao Monitor de Diálogos")
        except Exception:
            pass
    atexit.register(_cleanup_on_shutdown)
    
    load_data_and_models()
    print("\n" + "="*50)
    print("DASHBOARD CPRAIO PRONTO")
    app_port = int(os.environ.get('APP_PORT', os.environ.get('PORT', '5050')))
    debug_mode = os.environ.get('FLASK_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'on')
    print(f"ACESSE: http://localhost:{app_port}")
    print("="*50 + "\n")
    # Usando 0.0.0.0 para maior compatibilidade, mas o link impresso é localhost
    app.run(host='0.0.0.0', port=app_port, debug=debug_mode, use_reloader=False)

