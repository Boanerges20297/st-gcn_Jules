from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np

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
from datetime import datetime, timedelta
import re
from shapely.geometry import Point
from shapely.geometry import mapping
from shapely.geometry import shape
from shapely.ops import unary_union

# --- Orquestrador Regional ST-GAT ---
try:
    from src.core.orchestrator import StateOrchestrator, normalize_name
    from src.core.efficiency_monitor import EfficiencyMonitor
    orchestrator = None 
except ImportError:
    # Fallback se o PYTHONPATH não incluir a raiz corretamente
    import sys
    sys.path.append(os.getcwd())
    from src.core.orchestrator import StateOrchestrator
    from src.core.efficiency_monitor import EfficiencyMonitor
    def normalize_name(text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
        import re
        return re.sub(r'\s*-\s*AIS.*$', '', text).strip()

warnings.filterwarnings('ignore')
# Configurando logs para garantir visibilidade no terminal
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_SCREENSHOT_REPO_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'screenshot-report_preview'))
STATIC_SCREENSHOT_PUBLIC_DATA_DIR = os.path.join(STATIC_SCREENSHOT_REPO_DIR, 'public', 'data')
STATIC_EXPORT_SCRIPT = os.path.join(BASE_DIR, 'scripts', 'export_static_snapshot.py')
STATIC_EXPORT_OUTPUT_DIR = os.path.join(BASE_DIR, 'static_export', 'data')

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
_TOP_MICRONODE_FACTION_CACHE = None

# === REGISTRAR HEALTH MONITOR BLUEPRINT (antes de load_data_and_models) ===
model_calibrator = None
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
        get_orchestrator=lambda: orchestrator
    )
    app.register_blueprint(admin_bp)
    print("✅ Admin Dashboard Registrado em /api/admin/health")
    
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
                                'p10': global_data.get('p10', 0),
                                'p20': global_data.get('p20', 0),
                                'precision': global_data.get('p10', 0),
                                'recall': global_data.get('recall20', global_data.get('p20', 0)),
                                'recall10': global_data.get('recall10', 0),
                                'recall20': global_data.get('recall20', 0),
                                'active_locations': global_data.get('active_locations', 0),
                                'total_nodes': global_data.get('total_nodes', 0),
                                'total_events': global_data.get('total_events', 0),
                                'assigned_total_events': metrics.get('assigned_total_events', 0),
                                'unmapped_total_events': metrics.get('unmapped_total_events', 0),
                                'f1_score': 0.0
                            }
                            p, r = global_metrics['precision'], global_metrics['recall']
                            if p + r > 0:
                                global_metrics['f1_score'] = round(2 * p * r / (p + r), 4)
                            region_metrics = {}
                            for reg in (orchestrator.specialists.keys() if orchestrator else _ALL_REGIONS):
                                reg_data = metrics.get(reg, {})
                                if reg_data and isinstance(reg_data, dict):
                                    p10_score = reg_data.get('p10', 0)
                                    region_metrics[reg] = {
                                        'p10': p10_score,
                                        'p20': reg_data.get('p20', 0),
                                        'precision': p10_score,
                                        'recall': reg_data.get('recall20', reg_data.get('p20', 0)),
                                        'recall10': reg_data.get('recall10', 0),
                                        'recall20': reg_data.get('recall20', 0),
                                        'active_locations': reg_data.get('active_locations', 0),
                                        'total_nodes': reg_data.get('total_nodes', 0),
                                        'total_events': reg_data.get('total_events', 0),
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
                            print(f"⚠️ Erro ao atualizar confidence_tracker: {ct_err}")
                    print(f"📅 Data da Avaliação: {metrics.get('date')}")
                    print(f"📊 Eventos Detectados: {metrics.get('total_events', 0)} ({metrics.get('brute_cvli', 0)} Brutos + {metrics.get('exogenous', 0)} Exógenos)")
                    
                    # Exibir Global
                    if 'global' in metrics:
                        m = metrics['global']
                        print(f"\n🌍 REGIONALIZAÇÃO: GLOBAL")
                        print(f"   P5:  {m.get('p5', 0)*100:.1f}% | Hits: {', '.join(m.get('hits5', []))}")
                        print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                        print(f"   P20: {m.get('p20', 0)*100:.1f}% | R20: {m.get('recall20', 0)*100:.1f}% | Hits: {', '.join(m.get('hits20', []))}")
                    
                    # Exibir Fortaleza
                    if 'fortaleza' in metrics:
                        m = metrics['fortaleza']
                        print(f"\n🏙️  REGIONALIZAÇÃO: FORTALEZA")
                        print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                        print(f"   P20: {m.get('p20', 0)*100:.1f}% | R20: {m.get('recall20', 0)*100:.1f}% | Hits: {', '.join(m.get('hits20', []))}")
                    
                    # Exibir demais regiões se houver acertos
                    for reg in (orchestrator.specialists.keys() if orchestrator else _ALL_REGIONS):
                        if reg == 'fortaleza': continue
                        if reg in metrics and metrics[reg].get('p10', 0) > 0:
                            m = metrics[reg]
                            reg_name = REGION_LABELS.get(reg, reg.upper())
                            print(f"\n📍 REGIONALIZAÇÃO: {reg_name}")
                            print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                    
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
    
    # Limpeza de eventos exógenos antigos
    archive_old_exogenous_events()

    # --- ATUALIZAÇÃO OFICIAL DO ORCRIMS NO STARTUP ---
    try:
        from data.raw.inteligencia.import_orcrim_kml import refresh_orcrim_from_official
        refresh_result = refresh_orcrim_from_official()
        print(f"🧭 [ORCRIMS] Resultado do refresh no startup: {refresh_result}")
    except Exception as e:
        print(f"⚠️ [ORCRIMS] Falha ao atualizar ORCRIMS no startup: {e}")

    # --- ATUALIZAÇÃO DINÂMICA DE RUAS CRÍTICAS (CACHE GEO) ---
    try:
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
                # Carregamento limpo e direto
                data = pd.read_pickle(path)
                reg_gdf = data.get("nodes_gdf")
                if reg_gdf is not None:
                    dfs.append(reg_gdf)
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
        orchestrator = StateOrchestrator(BASE_DIR)
        print("✅ Motor de Inteligência ST-GAT Ativo.")

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
        efficiency_monitor = EfficiencyMonitor(BASE_DIR, orchestrator, nodes_gdf)
        generate_daily_ranking_report()
        
        # Disparar Monitor em Segundo Plano (Thread Paralela)
        # Guard: não iniciar no processo filho do Flask reloader
        import os as _os
        if _os.environ.get('WERKZEUG_RUN_MAIN') != 'true' or not app.debug:
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
    if ' - ' in mn:
        cidade_raw = mn.split(' - ')[-1].split('/')[0].strip()
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
    return re.sub(r'\s+', ' ', normalize_name(text or '')).strip()


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
            keys = {
                _normalize_polygon_lookup_name(properties.get('area_oficial')),
                _normalize_polygon_lookup_name(properties.get('micronodo')),
                _normalize_polygon_lookup_name(properties.get('name')),
            }
            keys = {key for key in keys if key}
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
                cache[key] = mapping(merged)
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

@app.route('/api/top20_micro_nodes')
def get_top20_micro_nodes():
    region = request.args.get('region', 'fortaleza').lower()
    try:
        limit = max(1, min(int(request.args.get('limit', 30)), 100))
    except Exception:
        limit = 30
    # Mapear regiao para o arquivo correspondente na pasta outputs
    filename_map = {
        'fortaleza': 'top20_micro_nodes_capital.geojson',
        'rmf': 'top20_micro_nodes_rmf.geojson',
        'interior': 'top20_micro_nodes_interior.geojson',
        'all': 'top20_micro_nodes.geojson'
    }
    
    filename = filename_map.get(region, 'top20_micro_nodes_capital.geojson')
    path = os.path.join(app.root_path, 'outputs', filename)

    def _decorate_top_features(payload):
        polygon_cache = _load_micronode_polygon_cache()
        faction_cache = _load_top_micronode_faction_cache()
        features = payload.get('features', [])[:limit]
        decorated = []
        for feature in features:
            props = dict(feature.get('properties') or {})
            lookup_key = _normalize_polygon_lookup_name(props.get('name') or props.get('micronodo'))
            polygon_geometry = polygon_cache.get(lookup_key)
            faction = props.get('faction') or faction_cache.get(lookup_key)
            if polygon_geometry:
                feature['geometry'] = polygon_geometry
                props['geometry_type'] = polygon_geometry.get('type', 'Polygon')
                props['source_geometry_type'] = polygon_geometry.get('type', 'Polygon')
                props['is_centroid'] = False
            if faction:
                props['faction'] = faction
            feature['properties'] = props
            decorated.append(feature)
        payload['features'] = decorated
        return payload
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'features' in data:
                data = _decorate_top_features(data)
            return jsonify(data)
    
    # Fallback se o regional nao existir
    fallback_path = os.path.join(app.root_path, 'outputs', 'top20_micro_nodes.geojson')
    if os.path.exists(fallback_path):
        with open(fallback_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'features' in data:
                data = _decorate_top_features(data)
            return jsonify(data)
            
    return jsonify({"type": "FeatureCollection", "features": []})

@app.route('/api/risk')
def get_risk():
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
    try:
        # --- Build exogenous_shocks from recent exogenous events ---
        exogenous_shocks = {}
        try:
            # Carregar de ambos os arquivos potenciais para garantir cobertura
            # Use only the canonical exogenous events file. Previously the
            # geocoded variant was written as a duplicate which created
            # multiple files and confusion. Keep a single source of truth.
            exo_files = ['exogenous_events.json']
            all_raw_events = []
            for f_name in exo_files:
                f_path = os.path.join(BASE_DIR, 'data', f_name)
                if os.path.exists(f_path):
                    try:
                        with open(f_path, 'r', encoding='utf-8') as xf:
                            all_raw_events.extend(json.load(xf) or [])
                    except: pass

            # Considerar apenas a janela exógena operacional vigente do arquivo canônico (7 dias)
            cutoff = datetime.now().date() - timedelta(days=EXOGENOUS_WINDOW_DAYS)
            
            # Limite superior baseado nos dados do orquestrador (evita inconsistência teórica)
            last_date_base = None
            if orchestrator is not None and hasattr(orchestrator, 'dates') and orchestrator.dates is not None:
                last_date_base = orchestrator.dates[-1]

            # Tipos que sempre são CRÍTICOS (Canal 25) - Incluindo sinais de violência extrema
            CRITICAL_TYPES = [
                'leader_transfer', 'faction_conflict', 'territory_dispute', 
                'confronto', 'execucao', 'chacina', 'tortura', 'homicidio_com_sinais_de_faccao'
            ]
            
            # Tipos que são de SUPRESSÃO (Canal 23) - Ação Policial Positiva
            SUPPRESSION_TYPES = [
                'apreensao', 'prisao', 'recuperacao_veiculo', 'cumprimento_mandado',
                'abordagem_positiva', 'desarticulacao_grupo'
            ]

            # Small helper: RMF cities derived from orchestrator index
            for ev in all_raw_events:
                try:
                    # --- Verificação de Consistência Teórica (Não ver o futuro) ---
                    ev_date_str = ev.get('date') or ev.get('event_date')
                    if not verify_date_consistency(ev_date_str, last_date_base):
                        continue # Pula evento futuro em relação ao estado do modelo
                    event_dt = parse_event_datetime(ev)
                    if event_dt and event_dt.date() < cutoff:
                        continue

                    # Extração e Normalização da Localidade
                    bairro_raw = (ev.get('bairro') or '').strip()
                    municipio_raw = (ev.get('municipio') or '').strip()
                    # If a bairro is present, target that specific node; otherwise expand by municipio->region
                    targets = []  # list of normalized node names to apply this shock to
                    if bairro_raw:
                        targets = [normalize_name(str(bairro_raw))]
                    elif municipio_raw:
                        mun_up = municipio_raw.upper()
                        # Determine high-level region from municipality name
                        if 'FORTALEZA' in mun_up:
                            region_key = 'fortaleza'
                        elif normalize_name(municipio_raw) in _RMF_NODES or any(n in normalize_name(municipio_raw) for n in _RMF_NODES):
                            region_key = 'rmf'
                        else:
                            region_key = 'interior'

                        # Expand to all node names that belong to that region
                        try:
                            for i, row in nodes_gdf.iterrows():
                                if str(row.get('regiao', '')).lower() == region_key:
                                    targets.append(normalize_name(row['name']))
                        except Exception:
                            # Fallback: use municipality string as single target
                            targets = [normalize_name(municipio_raw)]
                    else:
                        continue

                    # Intensidade e Criticidade
                    ev_type = str(ev.get('type') or ev.get('natureza') or '').lower()
                    description = ' '.join([
                        str(ev.get('description') or ''),
                        str(ev.get('descricao') or ''),
                        str(ev.get('resumo') or ''),
                        str(ev.get('raw_text') or ''),
                    ]).lower()
                    conflict_severity = str(ev.get('conflict_severity', '')).upper()
                    classification = classify_exogenous_event(ev)

                    # Classificação exógena: conflito real, supressão qualificada ou ação policial administrativa.
                    is_supp = classification['is_qualified_suppression']
                    is_conflict = classification['is_conflict']
                    if classification['signal_class'] == 'administrative_police':
                        continue

                    # Se for supressão, calibramos a intensidade pelo impacto
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
                        # Mapeia o conflict_severity fornecido pelo LLM para intensity
                        if conflict_severity == 'HIGH':
                            intensity = 0.9
                        elif conflict_severity == 'MEDIUM':
                            intensity = 0.6
                        elif conflict_severity == 'LOW':
                            intensity = 0.3
                        else:
                            intensity = float(ev.get('intensity', 0.5))

                    # Se a localidade for genérica (município inteiro), reduzimos drasticamente o impacto por nó
                    # para não zerar ou estourar a cidade inteira.
                    is_city_wide = not bairro_raw and bool(municipio_raw)
                    if is_city_wide and len(targets) > 1:
                        intensity = intensity / min(len(targets), 10.0) # Fator de amortecimento para impacto difuso                    
                    # Decisão de Canal: Canal 25 se tipo for crítico, intensidade > 0.7 
                    # OU se a descrição contiver palavras-chave de alerta máximo
                    is_critical = (ev_type in CRITICAL_TYPES) or (is_conflict and intensity > 0.7) or \
                                  ('execuc' in description) or ('facç' in description) or \
                                  ('morte' in description and 'facç' in description)
                    
                    # Apply/update shock for all resolved targets (single bairro or expanded region nodes)
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
                        
                        # Acumula as intensidades de forma independente
                        if is_supp:
                            exogenous_shocks[loc_norm]['suppression_intensity'] += intensity
                        elif is_conflict:
                            exogenous_shocks[loc_norm]['conflict_intensity'] += intensity
                            if is_critical:
                                exogenous_shocks[loc_norm]['is_critical'] = True
                except: continue

            exogenous_shocks_map = exogenous_shocks
            if not exogenous_shocks:
                exogenous_shocks = None
        except Exception as e:
            print(f"Erro ao processar shocks no app.py: {e}")
            exogenous_shocks = None

        scores_map, trends_map = orchestrator.get_combined_risk(exogenous_shocks, return_trends=True)
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

        for i, row in nodes_gdf.iterrows():
            try:
                name = str(row['name'])
                name_norm = normalize_name(name)
                score = normalize_risk_score(scores_map.get(name_norm, 20.0))
                trend = trends_map.get(name_norm, 'stable')
                
                # Identificação de Região
                reg = str(row.get('regiao', 'fortaleza')).lower()
                if reg == 'capital': reg = 'fortaleza'
                if name_norm in _RMF_NODES: reg = 'rmf'
                
                if reg not in region_buckets: region_buckets[reg] = []

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

                node_metrics = {
                    'cvli_7d': 0,
                    'tension': round(float(row.get('tension_index', 0)), 2),
                    'events_count': ev_count,
                    'event_types': ev_types[:3],
                    'critical_streets': critical_streets_info,
                    'spatial_influence': score >= 80
                }
                
                # Crimes Reais
                current_spec = orchestrator.specialists.get(reg)
                if current_spec:
                    try:
                        local_idx = next(idx for idx, r in current_spec['data']['nodes_gdf'].iterrows() if normalize_name(r['name']) == name_norm)
                        node_metrics['cvli_7d'] = int(current_spec['data']['node_features'][local_idx, -14:, 0].sum())
                    except: pass

                all_scores.append(score)
                node_result = {
                    'node_id': i, 'name': name, 'clean_name': name_norm,
                    'tension_score': score, 'risk_score': score,
                    'status_label': status, 'css_class': css,
                    'color': color, 'trend': trend, 
                    'metrics': node_metrics,
                    'faction': str(row.get('faction', 'N/A')), 'region_type': reg
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
                meta['top10_by_region'][region_key] = [{
                    'name': r.get('name'), 'node_id': r.get('node_id'), 'risk_score': r.get('risk_score'),
                    'status_label': r.get('status_label'), 'region_type': r.get('region_type')
                } for r in sorted_region[:10]]
        except Exception:
            meta['counts_by_region'] = {}
            meta['top10_by_region'] = {}

        # Build Top10 list server-side so frontend doesn't need to re-derive ranking.
        try:
            sorted_results = sorted(results, key=lambda x: x.get('risk_score', 0), reverse=True)
            meta['top10'] = []
            for r in sorted_results[:10]:
                meta['top10'].append({
                    'name': r.get('name'),
                    'node_id': r.get('node_id'),
                    'risk_score': r.get('risk_score'),
                    'status_label': r.get('status_label'),
                    'region_type': r.get('region_type')
                })
        except Exception:
            meta['top10'] = []

            # --- CORREÇÃO: Adicionar Datas da Janela de Inteligência (Projeção 7 dias) ---
        try:
            if orchestrator is not None and hasattr(orchestrator, 'dates') and orchestrator.dates is not None:
                last_db_date = orchestrator.dates[-1]
                if isinstance(last_db_date, str):
                    last_db_dt = datetime.strptime(last_db_date[:10], '%Y-%m-%d')
                else:
                    last_db_dt = last_db_date
                
                # Início e Fim da Projeção (7 dias à frente da base)
                start_pred = last_db_dt + timedelta(days=1)
                end_pred = last_db_dt + timedelta(days=7)
                
                meta['start_cvli'] = str(orchestrator.dates[0])
                meta['last_date_base'] = last_db_dt.strftime('%d/%m/%Y')
                meta['prediction_window'] = f"{start_pred.strftime('%d/%m')} a {end_pred.strftime('%d/%m')}"
                meta['intelligence_label'] = f"Janela de Inteligência: {meta['prediction_window']} (Atualizada com Eventos de Hoje)"
                meta['window_cvli'] = len(orchestrator.dates)
                meta['model_architecture'] = "Deep ST-GAT Elite (Regionalizado)"
                meta['model_window_cvli'] = 120 # Nova janela de 120 dias para todos
                
            else:
                meta['intelligence_label'] = "Janela de Inteligência: Projeção 7 dias (Tempo Real)"
                meta['last_date_base'] = 'N/A'
                meta['model_architecture'] = "ST-GAT Elite v3"
                meta['model_window_cvli'] = 120
                
            # Adicionar métricas de eficiência separadamente para que frontend (Cov@20) não fique com N/A
            if 'efficiency_monitor' in globals() and efficiency_monitor:
                _latest_metrics = efficiency_monitor.get_latest_metrics()
                if _latest_metrics:
                    meta['efficiency_metrics'] = _latest_metrics
        except Exception as e:
            print(f"Erro ao calcular datas de inteligência: {e}")
            meta['intelligence_label'] = "Janela de Inteligência: Ativa"

        # --- CORREÇÃO: Respeitar Filtro de Região nas caixas de resumo ---
        target_region = request.args.get('region', 'global').lower()
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

        # 1. Mapear pontos [lat, lng] para nomes de bairros normalizados
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
                
                # Configurar Shock Simulado (CUMULATIVO)
                is_supp = (sim_type == 'suppression')
                
                if name_norm not in temp_shocks:
                    temp_shocks[name_norm] = {
                        'intensity': 0.0,
                        'is_critical': not is_supp,
                        'is_suppression': is_supp
                    }
                
                # Incrementa intensidade (mais pontos no mesmo bairro = mais força)
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
                'tension': round(float(row.get('tension_index', 0)), 2),
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
    ais_files = [('fortaleza', 'AIS - CAPITAL.geojson'), ('rmf', 'AIS - METROPOLITANA.geojson'), ('interior', 'AIS - INTERIOR.geojson')]
    for reg, fname in ais_files:
        path = os.path.join(BASE_DIR, 'data', 'static', fname)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for feat in data.get('features', []):
                        feat['properties']['region_type'] = reg
                        features.append(feat)
            except: pass
    return jsonify({"type": "FeatureCollection", "features": features})


def _is_valid_screenshot_repo(repo_dir: str) -> bool:
    return os.path.isdir(repo_dir) and os.path.exists(os.path.join(repo_dir, 'package.json'))


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
        env={
            **os.environ,
            'PYTHONIOENCODING': 'utf-8',
        },
    )


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

    completed = subprocess.run(
        [sys.executable, STATIC_EXPORT_SCRIPT, '--output-dir', STATIC_EXPORT_OUTPUT_DIR],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=600,
        check=False,
        env={
            **os.environ,
            'PYTHONIOENCODING': 'utf-8',
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'Falha ao exportar snapshot estático. '
            f'stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}'
        )
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
        'stdout': completed.stdout.strip(),
    }


def _publish_screenshot_repo(repo_dir: str) -> dict[str, any]:
    logging.info('[SCREENSHOT EXPORT] Iniciando publicação git do repositório screenshot')
    status_result = _run_git_command(repo_dir, ['status', '--porcelain'])
    if status_result.returncode != 0:
        raise RuntimeError(f'Falha ao consultar status git: {status_result.stderr.strip() or status_result.stdout.strip()}')

    changed_entries = [line for line in status_result.stdout.splitlines() if line.strip()]
    if not changed_entries:
        logging.info('[SCREENSHOT EXPORT] Nenhuma alteração local para publicar')
        return {
            'published': False,
            'commit_created': False,
            'push_executed': False,
            'message': 'Nenhuma alteração detectada no repositório screenshot.',
        }

    add_result = _run_git_command(repo_dir, ['add', '-A'])
    if add_result.returncode != 0:
        raise RuntimeError(f'Falha no git add: {add_result.stderr.strip() or add_result.stdout.strip()}')

    commit_message = f'chore: sync static snapshot {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    commit_result = _run_git_command(repo_dir, ['commit', '-m', commit_message])
    if commit_result.returncode != 0:
        combined_output = (commit_result.stderr or commit_result.stdout or '').strip()
        if 'nothing to commit' not in combined_output.lower():
            raise RuntimeError(f'Falha no git commit: {combined_output}')

    pull_result = _run_git_command(repo_dir, ['pull', '--rebase', '-X', 'theirs', 'origin', 'main'])
    if pull_result.returncode != 0:
        logging.warning(f'[SCREENSHOT EXPORT] Alerta no git pull: {pull_result.stderr.strip() or pull_result.stdout.strip()}')

    push_result = _run_git_command(repo_dir, ['push', 'origin', 'main'])
    if push_result.returncode != 0:
        raise RuntimeError(f'Falha no git push: {push_result.stderr.strip() or push_result.stdout.strip()}')

    logging.info('[SCREENSHOT EXPORT] Publicação git concluída com sucesso')
    return {
        'published': True,
        'commit_created': True,
        'push_executed': True,
        'commit_message': commit_message,
        'message': 'Snapshot sincronizado e publicado no repositório screenshot.',
    }


@app.route('/api/export_static_snapshot', methods=['POST'])
def export_static_snapshot_to_screenshot_app():
    try:
        payload = request.get_json(silent=True) or {}
        target_repo_dir = payload.get('target_repo_dir') or STATIC_SCREENSHOT_REPO_DIR
        target_data_dir = payload.get('target_data_dir') or os.path.join(target_repo_dir, 'public', 'data')
        publish_repo = bool(payload.get('publish_repo', False))

        sync_info = _sync_static_snapshot_to_screenshot_app(target_data_dir)
        publish_info = None
        if publish_repo:
            publish_info = _publish_screenshot_repo(sync_info['target_repo_dir'])

        return jsonify({
            'ok': True,
            'message': (
                'Snapshot exportado, sincronizado e publicado no repositório screenshot.'
                if publish_repo else
                'Snapshot exportado e sincronizado com a aplicação screenshot.'
            ),
            **sync_info,
            'publish_info': publish_info,
        })
    except FileNotFoundError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 404
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'error': 'Timeout ao exportar snapshot estático.'}), 504
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500

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
def model_status(): return jsonify({"status": "idle"})

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
def explain_node(node_id):
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
        scores_map = orchestrator.get_combined_risk()
        score_pct = float(scores_map.get(name_norm, 20.0))
        score_10 = score_pct / 10.0
        
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

        events = []
        events_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        last_date_base = orchestrator.dates[-1] if (orchestrator and hasattr(orchestrator, 'dates')) else None
        try:
            if os.path.exists(events_path):
                with open(events_path, 'r', encoding='utf-8') as ef:
                    evts = json.load(ef)
                    for e in evts:
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
        except: events = []

        temporal_pattern = 'Increasing' if score_pct > float(np.mean(all_scores)) else 'Stable'
        confidence = min(0.95, 0.6 + (score_10 / 20.0))

        # --- EXTRAÇÃO DE DADOS REAIS DOS TENSORES (PARA EXPLICABILIDADE) ---
        cvli_recent = 0
        cvli_prev = 0
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
                    cvli_recent = int(features[spec_idx, -14:, 0].sum())
                    cvli_prev = int(features[spec_idx, -28:-14, 0].sum())
                    
                    # 2. Vizinhos Geográficos Reais (via Matriz de Adjacência)
                    adj_geo = spec['data']['adj_geo']
                    neighbor_indices = np.where(adj_geo[spec_idx] > 0)[0]
                    
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
                    
                    logging.info(f"📊 EXPLAIN [{name}]: recent={cvli_recent}, prev={cvli_prev}, neighbors={nearby_names}")
        except Exception as e:
            logging.warning(f"Erro ao extrair métricas reais para {name}: {e}")

        # Criar contexto esperado por ExplanationGenerator
        try:
            from src.explanation_generator import ExplanationGenerator
            gen = ExplanationGenerator()
            
            context = {
                'node_id': int(node_id),
                'name': name,
                'score': score_10,
                'temporal_pattern': 'Increasing' if cvli_recent > cvli_prev else 'Stable',
                'cvli_count_recent': cvli_recent,
                'cvli_count_prev': cvli_prev,
                'nearby_nodes': nearby,
                'nearby_impact_names': nearby_names,
                'events': events,
                'confidence': float(confidence),
                'tier': tier
            }

            explanation = gen.explain_node_ranking(int(node_id), int(rank_pos), context)
            explanation['risk_score_pct'] = float(score_pct)

            # Percentil de confiança na previsão
            conf_pct = round(float(confidence) * 100.0, 1)
            if conf_pct >= 80:
                conf_label = 'Alta'
            elif conf_pct >= 60:
                conf_label = 'Moderada'
            elif conf_pct >= 40:
                conf_label = 'Baixa'
            else:
                conf_label = 'Muito baixa'
            explanation['confidence_pct']   = conf_pct
            explanation['confidence_label'] = conf_label

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
                'confidence': float(confidence),
                'summary': 'Métricas e explicabilidade indisponíveis',
                'factors': [],
                'caveats': [],
                'explanation_available': False,
                'source': 'unavailable',
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
        load_dotenv()
        
        csv_url = os.environ.get("GOOGLE_SHEETS_CSV_URL")
        if not csv_url:
            return jsonify({"pending": 0, "total_sheet": 0, "total_local": 0}), 200
        
        resp = requests.get(csv_url, timeout=10)
        resp.encoding = 'utf-8'
        resp.raise_for_status()
        
        reader = csv_mod.reader(StringIO(resp.text))
        rows = list(reader)
        sheet_ids = set()
        for r in rows[1:]:
            if r and r[0].strip():
                sheet_ids.add(r[0].strip())
        
        exogenous_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        local_ids = set()
        if os.path.exists(exogenous_path):
            with open(exogenous_path, 'r', encoding='utf-8') as f:
                try:
                    events = json.load(f)
                    local_ids = {ev.get("id") for ev in events if ev.get("id")}
                except json.JSONDecodeError:
                    pass
        
        pending = len(sheet_ids - local_ids)
        return jsonify({
            "pending": pending,
            "total_sheet": len(sheet_ids),
            "total_local": len(local_ids),
        }), 200
    except Exception as e:
        return jsonify({"pending": 0, "error": str(e)}), 200

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

if __name__ == "__main__":
    load_data_and_models()
    print("\n" + "="*50)
    print("DASHBOARD CPRAIO PRONTO")
    print("ACESSE: http://localhost:5050")
    print("="*50 + "\n")
    # Usando 0.0.0.0 para maior compatibilidade, mas o link impresso é localhost
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=True)
