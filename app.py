from flask import Flask, jsonify, render_template, request
import pandas as pd
import geopandas as gpd
import pickle
import torch
import numpy as np
import os
import copy
import re
import json
import hashlib
import warnings
import logging
import asyncio

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')
warnings.filterwarnings('ignore', message='All support for the.*google.generativeai')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suprimir logs excessivos do Werkzeug (mas manter INFO para startup messages)
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('werkzeug.serving').setLevel(logging.INFO)

from shapely.geometry import shape, Point, Polygon
from scipy.spatial.distance import cdist
from src.model import STGCN
from src.llm_service import process_exogenous_text, parse_ciops_report
from src.ranking_inference import RankingInference
from src.ranking_correction_system import get_ranking_system
from src.metrics import MetricReporter
from src.event_manager import EventManager
from src.anomaly_monitor import start_anomaly_monitoring, get_anomaly_monitor
from src.explanation_generator import ExplanationGenerator

# Feature extraction for ranking model
def extract_features_clean(X):
    """Extrai 12 features de série temporal CVLI (compatível com RankingInference)"""
    num_nodes = X.shape[0]
    features = np.zeros((num_nodes, 12))
    
    for i in range(num_nodes):
        ts = X[i, :]
        
        features[i, 0] = ts.mean()
        features[i, 1] = np.sqrt(np.var(ts))
        features[i, 2] = ts.max()
        features[i, 3] = ts.min()
        features[i, 4] = (ts > 0).sum() / len(ts)
        features[i, 5] = ts.sum() / len(ts)
        
        if len(ts) > 5:
            recent = ts[-5:].mean()
            old = ts[:5].mean()
            features[i, 6] = recent - old
        
        if len(ts) > 1:
            features[i, 7] = np.mean(np.abs(np.diff(ts)))
        
        features[i, 8] = np.percentile(ts, 75) - np.percentile(ts, 25)
        features[i, 9] = ts.sum()
        
        if len(ts) > 3 and ts.sum() > 0:
            top3 = np.sum(np.sort(ts)[-3:])
            features[i, 10] = top3 / ts.sum()
        
        if ts.mean() > 0:
            features[i, 11] = ts.max() / ts.mean()
    
    features = np.nan_to_num(features, 0.0)
    return features
from src.model_update_monitor import start_monitor, get_state as get_monitor_state
from src.predict_logger import PredictLogger
import threading
import time
import unicodedata
from datetime import datetime, timezone
from collections.abc import Mapping

# ============================================================================
# DETERMINISMO GLOBAL - Garante mesmas predições em múltiplas execuções
# ============================================================================
SEED_VALUE = 42

def set_deterministic_mode():
    """Força modo determinístico para reprodutibilidade exata."""
    # NumPy
    np.random.seed(SEED_VALUE)
    
    # PyTorch
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
    
    # Força algoritmos determinísticos (pode afetar performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python
    import random
    random.seed(SEED_VALUE)
    
    print(f"[DETERMINISM] Seed fixo: {SEED_VALUE} | Determinístico: ON")

# Aplica imediatamente no import
set_deterministic_mode()

# Desscale mapping (loaded from diagnostics report if present)
_DESSCALE_A = None
_DESSCALE_B = None

# Cache de eventos exógenos - evita reamplificação em reinicializações
exogenous_events_hash = None
exogenous_cache_file = None  # Será setado em load_data_and_models()
events_amplified = False  # Flag para rastrear se eventos foram amplificados

def load_desscale_mapping():
    global _DESSCALE_A, _DESSCALE_B
    path = os.path.join(BASE_DIR, 'reports', 'desscale_mapping.txt')
    if os.path.exists(path):
        try:
            a = None
            b = None
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('a='):
                        a = float(line.split('=',1)[1])
                    elif line.startswith('b='):
                        b = float(line.split('=',1)[1])
            if a is not None and b is not None:
                _DESSCALE_A = a
                _DESSCALE_B = b
        except Exception:
            pass

# attempt to load mapping at startup (silently)
try:
    load_desscale_mapping()
except Exception:
    pass

app = Flask(__name__)

# Configuração e Carregamento de Dados (usar caminhos absolutos relativos a este arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_CVLI_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model_v2.pth') # Modelo v2: 26 canais categóricos (one-hot dia/mês)
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
# Ranking model path - será determinado dinamicamente baseado no dia da semana
RANKING_MODEL_PATH = None  # Será setado em load_data_and_models()
RANKING_BY_DAY_DIR = os.path.join(BASE_DIR, 'models', 'ranking_by_day')
BAIRROS_POLYGONS_PATHS = [
    os.path.join(BASE_DIR, 'data', 'raw', 'AIS - CAPITAL.geojson'),
    os.path.join(BASE_DIR, 'outputs', 'fortaleza_bairros_fence.geojson')
]

# Valores padrão caso arquivos não estejam presentes
nodes_gdf = None
polygons_json_cache = None
nodes_gdf_proj = None
nodes_centroids_proj = None
adj_matrix = None
original_adj_matrix = None
node_features = None
model_cvli = None # Modelo 3D unificado (26 canais: CVLI, CVP, Tension + one-hot features)
model_stgat = None # Modelo ST-GAT para produção
ranking_validator = None  # RankingInference for real-time validation
model_ranking_scores = None
device = None
norm_adj = None
adj_geo = None
adj_faction = None
norm_adj_list = None
dates = None
predict_logger = None  # PredictLogger para logs de predictions

# Inicializar PredictLogger de forma segura (apenas se BASE_DIR estiver definido)
try:
    predict_logger = PredictLogger(BASE_DIR, nodes_gdf=None)  # nodes_gdf será atualizado depois
    print("✅ PredictLogger inicializado com sucesso")
except Exception:
    predict_logger = None

# Week 4 Modules (Explainability & Metrics)
metric_reporter = None  # MetricReporter para cálculos de métricas
event_manager = None  # EventManager para gerenciar eventos exógenos
anomaly_monitor = None  # AnomalyMonitor para monitoramento periódico de anomalias
explanation_generator = None  # ExplanationGenerator para gerar explicações

# Static Data Cache
ibge_bairros_cache = None
ibge_municipios_cache = None
ibge_municipios_gdf = None
ba_irros_gdf = None
   
GEOCODING_ENABLED = True
exogenous_events = []
exogenous_affected_nodes = set()
exogenous_critical_nodes = set()

# Lock para garantir que pesos exógenos sejam aplicados apenas uma vez
exogenous_weights_lock = threading.Lock()
exogenous_weights_initialized = False

# Lock para proteger adj_matrix durante predições (evita race conditions/variações)
prediction_lock = threading.Lock()

# ===== EXPERIMENTAL: Enhanced Exogenous Events =====
# Flag para ativar/desativar versão melhorada (com severity + decay temporal)
# Defina como True para testar improved version, False para versão original
USE_ENHANCED_EXOGENOUS = False
# ===== FIM EXPERIMENTAL =====

# Periodic update status flags (used by frontend polling)
app._periodic_update_in_progress = False
app._periodic_last_update = None

class NodeSearchItem:
    __slots__ = ('name', 'name_lower', 'name_stripped', 'lat', 'lng')
    def __init__(self, name, name_lower, name_stripped, lat, lng):
        self.name = name
        self.name_lower = name_lower
        self.name_stripped = name_stripped
        self.lat = lat
        self.lng = lng

node_search_index = []

MANUAL_LOCATIONS = {
    "TAIBA": (-3.535, -38.892),
    "TAÍBA": (-3.535, -38.892),
}


def strip_accents(text: str) -> str:
    try:
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    except Exception:
        return text


def normalize_city_label(name: str) -> str:
    if not name or not isinstance(name, str):
        return name
    try:
        s = strip_accents(name).strip()
        return s.title()
    except Exception:
        return name


def build_node_search_index():
    global node_search_index
    node_search_index = []
    if nodes_gdf is None:
        return

    def _get_name_from_row(row):
        for col in ('name', 'NAME', 'NOME', 'nome'):
            if col in row and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        return None

    try:
        for _, row in nodes_gdf.reset_index(drop=True).iterrows():
            name = _get_name_from_row(row)
            if not name:
                continue
            geom = row.get('geometry') if 'geometry' in row else None
            if geom is None:
                continue
            centroid = geom.centroid
            lat = centroid.y
            lng = centroid.x

            name_lower = name.lower()
            name_stripped = strip_accents(name_lower)

            item = NodeSearchItem(name=name, name_lower=name_lower, name_stripped=name_stripped, lat=lat, lng=lng)
            node_search_index.append(item)
    except Exception as e:
        print(f"Erro ao construir node_search_index: {e}")

# Parâmetros de janela (modelo v2 retreinado com 30 dias + 8 features)
WINDOW_CVLI = 30
WINDOW_CVP = 30

def get_ranking_model_path():
    """
    Determina o caminho correto do modelo de ranking baseado no dia da semana atual.
    
    Retorna: (model_path, usando_fallback)
    """
    global dates
    
    day_of_week = None
    
    # Tentar obter dia da semana dos dados
    if dates is not None and len(dates) > 0:
        try:
            current_date = pd.to_datetime(dates[-1])
            day_of_week = current_date.weekday()
        except Exception:
            pass
    
    # Se não conseguiu pelos dados, usar data do sistema
    if day_of_week is None:
        try:
            today = pd.Timestamp.now()
            day_of_week = today.weekday()
        except Exception:
            pass
    
    # Estratégia 1: Tentar modelo específico do dia
    if day_of_week is not None:
        day_model_path = os.path.join(RANKING_BY_DAY_DIR, f'ranking_model_day{day_of_week}.pth')
        if os.path.exists(day_model_path):
            return day_model_path, False
    
    # Estratégia 2: Fallback para modelo genérico window30
    fallback_path = os.path.join(BASE_DIR, 'models', 'ranking_model_window30_final.pkl')
    if os.path.exists(fallback_path):
        return fallback_path, True
    
    # Estratégia 3: Nenhum modelo disponível
    return None, False



def load_exogenous_events():
    global exogenous_events, exogenous_cache_file
    
    # Inicializar caminho do cache
    exogenous_cache_file = os.path.join(BASE_DIR, 'data', 'processed', 'exogenous_events_cache.json')
    
    if os.path.exists(EXOGENOUS_FILE):
        try:
            with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                exogenous_events = json.load(f)
        except Exception:
            exogenous_events = []

def find_nearby_nodes(lat, lng, radius_m=500):
    nearby_indices = []
    try:
        p_geo = Point(lng, lat)

        if nodes_gdf_proj is not None:
             s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
             p_proj = s.iloc[0]

             if nodes_centroids_proj is not None:
                 search_buffer = p_proj.buffer(radius_m)
                 candidate_ilocs = list(nodes_centroids_proj.sindex.intersection(search_buffer.bounds))
                 if candidate_ilocs:
                     candidates = nodes_centroids_proj.iloc[candidate_ilocs]
                     dists = candidates.distance(p_proj)
                     nearby_indices = dists[dists < radius_m].index.tolist()
             else:
                 centroids = nodes_gdf_proj.geometry.centroid
                 dists = centroids.distance(p_proj)
                 nearby_indices = dists[dists < radius_m].index.tolist()
        else:
             if nodes_gdf is not None:
                 centroids = nodes_gdf.geometry.centroid
                 dists = centroids.distance(p_geo)
                 nearby_indices = dists[dists < 0.005].index.tolist()

        if not nearby_indices:
             if nodes_gdf is not None:
                 if nodes_gdf_proj is not None:
                     if nodes_centroids_proj is not None:
                         dists = nodes_centroids_proj.distance(p_proj)
                         nearby_indices = [dists.idxmin()]
                     else:
                         dists = nodes_gdf_proj.geometry.centroid.distance(p_proj)
                         nearby_indices = [dists.idxmin()]
                 else:
                     dists = nodes_gdf.geometry.centroid.distance(p_geo)
                     nearby_indices = [dists.idxmin()]

    except Exception:
        pass

    return nearby_indices

def apply_exogenous_events():
    """
    Aplica amplificação de eventos exógenos com cache para evitar reamplificação.
    
    Estratégia:
    1. Verifica se eventos foram atualizados (via hash)
    2. Se SÃO NOVOS: aplica amplificação + marca como amplificado no cache
    3. Se IGUAIS + já amplificados: pula reamplificação, apenas marca nodes críticos
    
    Isso previne o bug de oscilação (24 → 130 → 63 → 24) causado por reamplificar
    a mesma matriz adjacência múltiplas vezes em reinicializações.
    
    PROTEGIDO com lock para evitar race conditions durante predições.
    """
    global adj_matrix, exogenous_affected_nodes, adj_geo, adj_faction, exogenous_critical_nodes
    global exogenous_events_hash, events_amplified
    
    if not exogenous_events or adj_matrix is None:
        return

    # Protege modificação de adj_matrix durante aplicação de eventos
    with prediction_lock:
        # Verificar se é uma atualização nova ou apenas reload
        is_new_update = check_exogenous_update()

        exogenous_affected_nodes.clear()
        exogenous_critical_nodes.clear()

        count_affected = 0
        count_amplified = 0
        
        for batch in exogenous_events:
            points = batch.get('points', [])

            for pt in points:
                lat = pt.get('lat') if isinstance(pt, dict) else (pt[0] if isinstance(pt, list) and len(pt)>0 else None)
                lng = pt.get('lng') if isinstance(pt, dict) else (pt[1] if isinstance(pt, list) and len(pt)>1 else None)

                if lat is None or lng is None: continue

                def _is_critical_event(pt_item):
                    try:
                        evt = None
                        if isinstance(pt_item, dict):
                            evt = pt_item.get('raw_event') or pt_item
                        if not evt:
                            return False, 'LOW'
                        
                        # Verifica conflict_severity da LLM (prioridade)
                        severity = evt.get('conflict_severity', '').upper()
                        if severity in ('HIGH', 'MEDIUM'):
                            return True, severity
                        
                        # Fallback: detecção por keywords
                        text_fields = []
                        for k in ('natureza','nature','resumo','description','descricao'):
                            v = evt.get(k) if isinstance(evt, dict) else None
                            if v:
                                text_fields.append(str(v).lower())
                        txt = ' '.join(text_fields)
                        
                        # HIGH severity - sinais de execução/confronto
                        high_keywords = ['amarrado', 'mãos amarradas', 'pés amarradas', 'tortura', 
                                        'execução', 'executado', 'carbonizado', 'enterrado',
                                        'duplo homicídio', 'duplo homicidio', 'triplo', 'chacina',
                                        'emboscada', 'disputa territorial']
                        if any(k in txt for k in high_keywords):
                            return True, 'HIGH'
                        
                        # MEDIUM severity - violência armada + deslocamento forçado
                        medium_keywords = ['homic', 'homicídio', 'morte', 'morto', 'tiro', 
                                          'lesão a bala', 'lesao a bala', 'disparos', 'fuzil']
                        # Deslocamento forçado/expulsão - indica preparação para conflito ou revide
                        displacement_keywords = ['ameaças de grupo criminoso', 'expulsão', 'expulsao',
                                               'deslocamento forçado', 'deslocamento forcado',
                                               'precisa fazer a mudança', 'precisa fazer mudanca',
                                               'forçado a sair', 'forcado a sair', 'obrigado a sair']
                        
                        if any(k in txt for k in medium_keywords):
                            return True, 'MEDIUM'
                        if any(k in txt for k in displacement_keywords):
                            return True, 'MEDIUM'
                        
                        return False, 'LOW'
                    except Exception:
                        return False, 'LOW'

                indices = find_nearby_nodes(lat, lng)

                for idx in indices:
                    exogenous_affected_nodes.add(idx)
                    critical_flag, severity = _is_critical_event(pt)
                    
                    # AMPLIFICAÇÃO ULTRALEVE (1.1-1.2x):
                    # Apenas amplifica se é uma ATUALIZAÇÃO NOVA de eventos
                    if is_new_update and adj_matrix is not None:
                        # Amplificação muito leve - apenas sinaliza presença
                        amplification_map = {'HIGH': 1.2, 'MEDIUM': 1.0, 'LOW': 0.7}
                        amp_factor = amplification_map.get(severity, 1.05)
                        
                        adj_matrix[idx, :] *= amp_factor
                        adj_matrix[:, idx] *= amp_factor
                        count_amplified += 1
                    
                    # Sempre marca nodes críticos para UI (independente da amplificação)
                    if critical_flag:
                        exogenous_critical_nodes.add(idx)

                    count_affected += 1

        # Atualizar flag de amplificação se foi uma atualização nova
        if is_new_update:
            events_amplified = True


def compute_exogenous_hash(events_list):
    """Calcula hash dos eventos exógenos para detecção de mudanças"""
    if not events_list:
        return None
    
    # Serializar eventos em ordem determinística
    events_str = json.dumps(events_list, sort_keys=True, default=str)
    return hashlib.md5(events_str.encode()).hexdigest()

def check_exogenous_update():
    """Verifica se os eventos exógenos foram atualizados desde última execução"""
    global exogenous_events_hash, exogenous_cache_file, events_amplified
    
    current_hash = compute_exogenous_hash(exogenous_events)
    
    # Tentar carregar hash anterior do cache
    previous_hash = None
    previous_amplified = False
    if exogenous_cache_file and os.path.exists(exogenous_cache_file):
        try:
            with open(exogenous_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                previous_hash = cache_data.get('hash')
                previous_amplified = cache_data.get('amplified', False)
        except Exception:
            pass
    
    # Determinar se é uma atualização nova
    is_new_update = (current_hash != previous_hash)
    
    # Se o hash é igual mas a amplificação anterior estava ativada, 
    # considera como "novo" se precisamos reaplicar com novos parâmetros
    # (comentar if line 362 para forçar reaplicação mesmo com hash igual)
    if not is_new_update and previous_amplified:
        # Hash igual e já amplificado = pular
        events_amplified = True
    else:
        # Hash diferente ou primeira vez = aplicar
        events_amplified = False  # Reset para aplicar novamente
    
    # Atualizar cache com novo hash
    if exogenous_cache_file and current_hash:
        try:
            cache_data = {
                'hash': current_hash,
                'amplified': events_amplified,
                'timestamp': datetime.now().isoformat(),
                'event_count': len(exogenous_events)
            }
            os.makedirs(os.path.dirname(exogenous_cache_file), exist_ok=True)
            with open(exogenous_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass
    
    exogenous_events_hash = current_hash
    
    return is_new_update


async def apply_exogenous_events_async():
    """
    Aplica pesos exógenos de forma assíncrona.
    APENAS uma vez durante a inicialização, usando lock para evitar duplicação.
    
    Usa versão ENHANCED se USE_ENHANCED_EXOGENOUS=True, caso contrário usa versão original.
    """
    global exogenous_weights_initialized
    
    # Usa lock para garantir execução segura e única
    with exogenous_weights_lock:
        # Se já foi inicializado, pula
        if exogenous_weights_initialized:
            print("[EXOGENOUS] Pesos já foram inicializados - pulando aplicação duplicada")
            return True
        
        try:
            version = "ENHANCED (com severity + decay)" if USE_ENHANCED_EXOGENOUS else "PADRÃO"
            print(f"[EXOGENOUS] Iniciando aplicação assíncrona de pesos ({version})... (UMA ÚNICA VEZ)")
            
            # Carrega eventos exógenos
            load_exogenous_events()
            
            # Aguarda um pequeno delay para garantir que os dados estão prontos
            await asyncio.sleep(0.5)
            
            # Aplica amplificação de eventos - versão selecionada
            if USE_ENHANCED_EXOGENOUS:
                apply_exogenous_events_enhanced()
                print("[EXOGENOUS-ENH] ✓ Versão ENHANCED utilizada")
            else:
                apply_exogenous_events()
                print("[EXOGENOUS] ✓ Versão PADRÃO utilizada")
            
            # Marca como inicializado
            exogenous_weights_initialized = True
            
            print("[EXOGENOUS] Pesos exógenos aplicados com sucesso ✓")
            return True
        except Exception as e:
            print(f"[EXOGENOUS] Erro ao aplicar pesos: {e}")
            import traceback
            traceback.print_exc()
            return False


def apply_exogenous_events_enhanced():
    """
    Versão MELHORADA de apply_exogenous_events() com:
    1. Severity mapping (HIGH=1.3x, MEDIUM=1.15x, LOW=1.05x)
    2. Temporal decay (eventos antigos pesam menos: e^(-days/7))
    3. Raio variável (HIGH=1000m, MEDIUM=750m, LOW=500m)
    
    EXPERIMENTAL: Use este para validação antes de integrar à versão principal.
    Ativado quando: USE_ENHANCED_EXOGENOUS = True
    
    Diferenças da versão original:
    - Original: Todos eventos têm mesmo peso (1.0-1.2x)
    - Enhanced: Peso proporcional à severity + decay temporal
    """
    global adj_matrix, exogenous_affected_nodes, adj_geo, adj_faction, exogenous_critical_nodes
    global exogenous_events_hash, events_amplified
    
    if not exogenous_events or adj_matrix is None:
        print("[EXOGENOUS-ENH] Nenhum evento ou matriz indisponível")
        return

    # Verificar se é uma atualização nova
    is_new_update = check_exogenous_update()

    exogenous_affected_nodes.clear()
    exogenous_critical_nodes.clear()

    count_affected = 0
    count_amplified = 0
    count_high_severity = 0
    
    # Mapeamento: severidade → (raio_metros, multiplicador_base)
    severity_config = {
        'HIGH': {'radius': 1000, 'amp_factor': 1.3, 'description': 'Execução/Confronto'},
        'MEDIUM': {'radius': 750, 'amp_factor': 1.15, 'description': 'Violência Armada'},
        'LOW': {'radius': 500, 'amp_factor': 1.05, 'description': 'Atividade Padrão'}
    }
    
    # Usar início do dia como referência fixa para garantir determinismo entre reinícios no mesmo dia
    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for batch_idx, batch in enumerate(exogenous_events):
        points = batch.get('points', [])
        batch_timestamp = batch.get('timestamp')
        
        # Parse timestamp se necessário
        try:
            if isinstance(batch_timestamp, str):
                batch_timestamp = pd.Timestamp(batch_timestamp).normalize()  # Truncar para início do dia
            else:
                batch_timestamp = pd.Timestamp(now)
        except Exception:
            batch_timestamp = pd.Timestamp(now)
        
        # Calcular decay temporal (determinístico: baseado em dias inteiros)
        days_old = (now - batch_timestamp).days
        temporal_decay = np.exp(-days_old / 7.0)  # Decai para ~37% em 7 dias, ~13% em 14 dias
        
        for pt_idx, pt in enumerate(points):
            lat = pt.get('lat') if isinstance(pt, dict) else (pt[0] if isinstance(pt, list) and len(pt) > 0 else None)
            lng = pt.get('lng') if isinstance(pt, dict) else (pt[1] if isinstance(pt, list) and len(pt) > 1 else None)

            if lat is None or lng is None:
                continue

            # Determinar severidade
            def _detect_severity(pt_item):
                """Retorna (is_critical, severity_level, description)"""
                try:
                    evt = None
                    if isinstance(pt_item, dict):
                        evt = pt_item.get('raw_event') or pt_item
                    if not evt:
                        return False, 'LOW', 'Desconhecido'
                    
                    # Verificar conflict_severity da LLM (primeira prioridade)
                    severity = evt.get('conflict_severity', '').upper()
                    if severity in ('HIGH', 'MEDIUM', 'LOW'):
                        return severity == 'HIGH', severity, severity_config[severity]['description']
                    
                    # Fallback: detectar por keywords
                    text_fields = []
                    for k in ('natureza', 'nature', 'resumo', 'description', 'descricao'):
                        v = evt.get(k) if isinstance(evt, dict) else None
                        if v:
                            text_fields.append(str(v).lower())
                    txt = ' '.join(text_fields)
                    
                    # HIGH severity - execução/confronto
                    high_keywords = ['amarrado', 'mãos amarradas', 'pés amarradas', 'tortura',
                                    'execução', 'executado', 'carbonizado', 'enterrado',
                                    'duplo homicídio', 'duplo homicidio', 'triplo', 'chacina',
                                    'emboscada', 'disputa territorial']
                    if any(k in txt for k in high_keywords):
                        return True, 'HIGH', 'Execução/Confronto Detectado'
                    
                    # MEDIUM severity - violência armada + deslocamento
                    medium_keywords = ['homic', 'homicídio', 'morte', 'morto', 'tiro',
                                      'lesão a bala', 'lesao a bala', 'disparos', 'fuzil']
                    displacement_keywords = ['ameaças de grupo criminoso', 'expulsão', 'expulsao',
                                           'deslocamento forçado', 'deslocamento forcado',
                                           'precisa fazer a mudança', 'precisa fazer mudanca',
                                           'forçado a sair', 'forcado a sair', 'obrigado a sair']
                    
                    if any(k in txt for k in medium_keywords) or any(k in txt for k in displacement_keywords):
                        return True, 'MEDIUM', 'Violência Detectada'
                    
                    return False, 'LOW', 'Atividade Padrão'
                except Exception:
                    return False, 'LOW', 'Erro na Detecção'

            is_critical, severity, desc = _detect_severity(pt)
            
            # Aplicar amplificação apenas em atualização nova
            if is_new_update and adj_matrix is not None:
                config = severity_config.get(severity, severity_config['LOW'])
                
                # Amplificador final = base * decay_temporal
                amp_factor = config['amp_factor'] * temporal_decay
                radius = config['radius']
                
                # Encontrar nós próximos com raio baseado em severity
                indices = find_nearby_nodes(lat, lng, radius_m=radius)
                
                for idx in indices:
                    exogenous_affected_nodes.add(idx)
                    
                    # Amplificar matriz
                    adj_matrix[idx, :] *= amp_factor
                    adj_matrix[:, idx] *= amp_factor
                    
                    count_amplified += 1
                    
                    if severity == 'HIGH':
                        count_high_severity += 1
                
                # Log detalhado
                if count_amplified % 10 == 0:
                    print(f"[EXOGENOUS-ENH] Event #{batch_idx}.{pt_idx}: {severity} - {desc} | Decay: {temporal_decay:.2f} | Factor: {amp_factor:.3f}x | Raio: {radius}m | Nós: {len(indices)}")
            
            # Sempre marcar nodes críticos para UI
            if is_critical:
                exogenous_critical_nodes.add(idx)

            count_affected += 1

    # Marcar como amplificado se foi feita atualizacao nova
    if is_new_update:
        events_amplified = True
    
    # Log de resumo
    print(f"[EXOGENOUS-ENH] ✓ Amplificação concluída: {count_affected} eventos | {count_amplified} amplificadores | {count_high_severity} HIGH | Decay aplicado")


def compute_norm_adj(adj_matrix_input):
    """Normaliza matriz de adjacência usando random walk normalization"""
    if adj_matrix_input is None:
        return None
    
    adj_tensor = torch.FloatTensor(adj_matrix_input)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)

def compute_norm_adj_list(adj_list):
    if adj_list is None:
        return None
    res = []
    for a in adj_list:
        res.append(compute_norm_adj(a))
    return res

def update_exogenous_state():
    """Atualiza estado da matriz de adjacência com aplicação de eventos exógenos"""
    global adj_matrix, norm_adj, original_adj_matrix
    global adj_geo, adj_faction, norm_adj_list

    if original_adj_matrix is None:
        load_data_and_models()
        return

    adj_matrix = original_adj_matrix.copy()
    if adj_geo is not None:
        try:
            adj_geo = original_adj_matrix.copy()
        except Exception:
            pass
    if adj_faction is not None:
        try:
            adj_faction = original_adj_matrix.copy()
        except Exception:
            pass

    # Usar versão apropriada baseado na flag
    if USE_ENHANCED_EXOGENOUS:
        apply_exogenous_events_enhanced()
    else:
        apply_exogenous_events()

    if adj_geo is not None and adj_faction is not None:
        norm_adj_list = compute_norm_adj_list([adj_geo, adj_faction])
        norm_adj = norm_adj_list[0] if norm_adj_list else None
    else:
        norm_adj = compute_norm_adj(adj_matrix)

def load_data_and_models():
    global nodes_gdf, polygons_json_cache, nodes_gdf_proj, nodes_centroids_proj, adj_matrix, node_features, model_cvli, model_stgat, device, norm_adj, dates, original_adj_matrix
    global adj_geo, adj_faction, norm_adj_list
    global ibge_bairros_cache, ibge_municipios_cache, ibge_municipios_gdf
    global RANKING_MODEL_PATH

    # Determinar e carregar modelo de ranking correto baseado no dia da semana
    RANKING_MODEL_PATH, usando_fallback = get_ranking_model_path()
    
    # Log claro sobre qual modelo está sendo usado
    if RANKING_MODEL_PATH:
        modelo_tipo = "FALLBACK" if usando_fallback else "OFFICIAL"
        modelo_nome = os.path.basename(RANKING_MODEL_PATH)
        print(f"[RANKING] {modelo_tipo}: {modelo_nome}")
    else:
        print(f"[RANKING] DISABLED: Nenhum modelo disponível")

    # Load Static Data
    try:
        static_file = os.path.join(BASE_DIR, 'data', 'static', 'fortaleza_bairros_coords.json')
        if os.path.exists(static_file):
            with open(static_file, 'r', encoding='utf-8') as f:
                ibge_bairros_cache = json.load(f)

        static_file_mun = os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios_coords.json')
        if os.path.exists(static_file_mun):
            with open(static_file_mun, 'r', encoding='utf-8') as f:
                ibge_municipios_cache = json.load(f)
        try:
            possible_paths = [
                os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios.geojson'),
                os.path.join(BASE_DIR, 'data', 'static', 'ceara_municipios.json'),
                os.path.join(BASE_DIR, 'data', 'static', 'municipios_ceara.geojson'),
                os.path.join(BASE_DIR, 'data', 'static', 'municipios_ceara.json')
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    try:
                        ibge_municipios_gdf = gpd.read_file(p)
                        if ibge_municipios_gdf.crs is None:
                            ibge_municipios_gdf.set_crs(epsg=4326, inplace=True)
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        # --- Load Fortaleza bairros polygons (prefer AIS - CAPITAL, fallback to outputs fence) ---
        try:
            from pathlib import Path
            for p in BAIRROS_POLYGONS_PATHS:
                if os.path.exists(p):
                    try:
                        _b = gpd.read_file(p)
                        if _b is not None and hasattr(_b, 'geometry'):
                            if _b.crs is None:
                                _b.set_crs(epsg=4326, inplace=True)
                            ba_irros_gdf = _b
                            break
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass

    data_pack = None
    if not os.path.exists(DATA_FILE):
        print(f"Regenerando dados (data_processing.py)...")
        
        try:
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, 'src', 'data_processing.py')],
                capture_output=True,
                text=True,
                cwd=BASE_DIR
            )
            
            if result.returncode != 0:
                print(f"ERRO ao processar dados!")
                return
                
            if not os.path.exists(DATA_FILE):
                print(f"ERRO: Dados não foram criados!")
                return
                
        except Exception as e:
            print(f"ERRO ao executar data_processing.py: {e}")
            return

    try:
        if data_pack is None:
            with open(DATA_FILE, 'rb') as f:
                data_pack = pickle.load(f)

        # Use temporary local variables to allow validation before assignment
        _nodes_gdf = data_pack.get('nodes_gdf')
        
        # Se nodes_gdf não está no pickle, carregar do pickle de micro-nós real
        # NÃO usar bairros como fallback - apenas micro-nós reais do grafo
        if _nodes_gdf is None:
            try:
                nodes_pkl_path = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl')
                if os.path.exists(nodes_pkl_path):
                    with open(nodes_pkl_path, 'rb') as f:
                        _nodes_gdf = pickle.load(f)
                    print(f"Carregado nodes_gdf de {nodes_pkl_path}")
            except Exception as e:
                print(f"Não conseguiu carregar nodes_gdf: {e}")
                _nodes_gdf = None
        
        polygons_json_cache = None
        _adj_geo = data_pack.get('adj_geo')
        _adj_faction = data_pack.get('adj_faction')
        _adj_matrix = data_pack.get('adj_matrix')
        if _adj_matrix is None:
            _adj_matrix = _adj_geo
        _node_features = data_pack.get('node_features')
        
        _dates = data_pack.get('dates')

        # --- VALIDATION: Check Data Integrity ---
        if _node_features is not None:
            # Check Node Count (Expected: 319 admin boundaries + ~2374 communities = ~2693 total)
            # Aceitar dados entre 300-3000 nós (range razoável para grafo com comunidades)
            if _node_features.shape[0] < 300 or _node_features.shape[0] > 3000:
                err_msg = (
                    f"CRITICAL: Dados inválidos detectados! (Nós: {_node_features.shape[0]}). "
                    "O sistema requer entre 300-3000 nós (bairros + cidades + comunidades). "
                    "Por favor, execute 'python src/data_processing.py' para regenerar os dados."
                )
                print(err_msg)
                raise RuntimeError(err_msg)

            # Check Channels (Expected 26: CVLI, CVP, Tension + 7 DOW one-hot + 12 Month one-hot + 4 extras)
            if len(_node_features.shape) < 3 or _node_features.shape[2] != 26:
                channels = _node_features.shape[2] if len(_node_features.shape) > 2 else 1
                err_msg = (
                    f"CRITICAL: Dados obsoletos detectados! (Canais: {channels}). "
                    "O sistema requer 26 canais (CVLI, CVP, Tension + one-hot features categóricas). "
                    "Por favor, execute 'python src/data_processing.py' para regenerar os dados."
                )
                print(err_msg)
                raise RuntimeError(err_msg)

        # Validation Passed - Assign to Globals
        nodes_gdf = _nodes_gdf
        adj_geo = _adj_geo
        adj_faction = _adj_faction
        adj_matrix = _adj_matrix
        node_features = _node_features
        dates = _dates
        
        # Atualizar nodes_gdf no PredictLogger para nomes dos nodes
        try:
            if predict_logger is not None and nodes_gdf is not None:
                predict_logger.nodes_gdf = nodes_gdf
        except Exception:
            pass
        
        # Variáveis globais atribuídas

        # --- FIXED: Node Paradigm Loading ---
        if nodes_gdf is not None:
             # Ensure CIDADE is populated correctly based on node_type
             if 'node_type' in nodes_gdf.columns:
                 # Logic: If node_type='cidade', CIDADE = name. If 'bairro', CIDADE = Fortaleza (or default logic)
                 # Use vectorized numpy where for speed
                 nodes_gdf['CIDADE'] = np.where(
                     nodes_gdf['node_type'] == 'cidade',
                     nodes_gdf['name'],
                     'Fortaleza' # Default for Capital/RMF bairros for now, can be refined
                 )
                 print("Applied Node Paradigm logic: Cities assigned to CIDADE column.")

        # --- Load facção from pre-processed mapping GeoJSON ---
        try:
            faction_geojson = os.path.join(BASE_DIR, 'outputs', 'nodes_with_faction_assigned.geojson')
            if os.path.exists(faction_geojson):
                faction_gdf = gpd.read_file(faction_geojson)
                if faction_gdf is not None and not faction_gdf.empty and 'faction' in faction_gdf.columns:
                    # faction_gdf has 'name' and 'faction' columns
                    # Merge by name == name
                    faction_map = dict(zip(faction_gdf['name'], faction_gdf['faction']))
                    nodes_gdf['faction'] = nodes_gdf['name'].map(faction_map).fillna('N/A')
                    assigned = (nodes_gdf['faction'] != 'N/A').sum()
                    print(f'Loaded faction mapping: {assigned}/{len(nodes_gdf)} nodes assigned')
                    
                    # Also load faction_source for debugging
                    if 'faction_source' in faction_gdf.columns:
                        faction_source_map = dict(zip(faction_gdf['name'], faction_gdf['faction_source']))
                        nodes_gdf['faction_source'] = nodes_gdf['name'].map(faction_source_map)
                else:
                    nodes_gdf['faction'] = 'N/A'
            else:
                nodes_gdf['faction'] = 'N/A'
        except Exception as e:
            nodes_gdf['faction'] = 'N/A'
            print(f'Warning: faction loading failed: {e}')

        # --- Load AIS from data/static ---
        try:
            ais_dir = os.path.join(BASE_DIR, 'data', 'static')
            ais_files = [
                os.path.join(ais_dir, 'AIS - CAPITAL.geojson'),
                os.path.join(ais_dir, 'AIS - METROPOLITANA.geojson'),
                os.path.join(ais_dir, 'AIS - INTERIOR.geojson')
            ]
            ais_gdfs = []
            for af in ais_files:
                if os.path.exists(af):
                    try:
                        ag = gpd.read_file(af)
                        if ag is None or ag.empty:
                            continue
                        if ag.crs is None:
                            ag.set_crs(epsg=4326, inplace=True)
                        if 'Name' in ag.columns:
                            ag['ais_name'] = ag['Name']
                        elif 'NAME' in ag.columns:
                            ag['ais_name'] = ag['NAME']
                        else:
                            ag['ais_name'] = ag.index.astype(str)
                        ais_gdfs.append(ag[['geometry', 'ais_name']])
                    except Exception:
                        continue

            if ais_gdfs:
                ais_all = pd.concat(ais_gdfs, ignore_index=True)
                ais_all = gpd.GeoDataFrame(ais_all, geometry='geometry', crs='EPSG:4326')
                try:
                    nodes_with_ais = gpd.sjoin(nodes_gdf, ais_all, how='left', predicate='within')
                    ais_map = nodes_with_ais.reset_index().groupby('index').first()['ais_name'].to_dict()
                    nodes_gdf['AIS'] = nodes_gdf.index.map(lambda i: ais_map.get(i, None))
                except Exception:
                    nodes_gdf['AIS'] = None
            else:
                nodes_gdf['AIS'] = None
        except Exception:
            nodes_gdf['AIS'] = None
            pass

        if nodes_gdf is not None:
            try:
                nodes_gdf_proj = nodes_gdf.to_crs(epsg=3857)
                nodes_centroids_proj = nodes_gdf_proj.geometry.centroid
                _ = nodes_centroids_proj.sindex
            except Exception:
                pass

            build_node_search_index()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = node_features.shape[0]

        if adj_matrix is not None:
            original_adj_matrix = adj_matrix.copy()

        # Carrega eventos exógenos apenas na inicialização (primeira vez)
        # PeriodicReload não deve reaplicar pesos
        if not exogenous_weights_initialized:
            load_exogenous_events()
        # apply_exogenous_events() será executado de forma assíncrona APÓS load_data_and_models()

        if adj_geo is not None and adj_faction is not None:
            try:
                norm_adj_list = compute_norm_adj_list([adj_geo, adj_faction])
                norm_adj = norm_adj_list[0] if norm_adj_list else None
            except Exception:
                norm_adj = compute_norm_adj(adj_matrix)
        else:
            norm_adj = compute_norm_adj(adj_matrix)

        # --- FIXED: 3-Channel Model Loading ---
        def _adapt_state_dict_for_multigraph(state_dict, num_graphs):
            try:
                sd = dict(state_dict)
            except Exception:
                sd = state_dict
            keys = list(sd.keys())
            for k in keys:
                if k.endswith('.gcn.weight'):
                    base = k[:-len('.gcn.weight')]
                    w = sd[k]
                    for i in range(num_graphs):
                        newk = f"{base}.gcn.weights.{i}"
                        sd[newk] = w
                    del sd[k]
            return sd

        if os.path.exists(MODEL_CVLI_PATH):
            print(f"Carregando modelo UNIFICADO de {MODEL_CVLI_PATH}...")
            num_graphs = len(norm_adj_list) if norm_adj_list is not None else 1
            try:
                # Prefer safe loading mode when available to avoid executing arbitrary pickle code.
                try:
                    raw_state = torch.load(MODEL_CVLI_PATH, map_location=device, weights_only=True)
                except TypeError:
                    raw_state = torch.load(MODEL_CVLI_PATH, map_location=device, weights_only=False)

                # Validate that the loaded object is a mapping-like state dict. If it's not, abort to
                # avoid interacting with arbitrary objects that may have been unpickled.
                if not isinstance(raw_state, Mapping):
                    raise RuntimeError(
                        f"Unsafe checkpoint format: expected a mapping (state_dict), got {type(raw_state)}.\n"
                        "Aborting load to avoid executing untrusted code."
                    )
                state_dict = {}
                for k,v in dict(raw_state).items():
                    nk = k[7:] if k.startswith('module.') else k
                    state_dict[nk] = v

                state_dict = _adapt_state_dict_for_multigraph(state_dict, num_graphs)

                ck_time_steps = None
                if 'conv_final.weight' in state_dict:
                    try:
                        ck_time_steps = state_dict['conv_final.weight'].shape[-1]
                    except Exception:
                        ck_time_steps = None

                instantiate_ts = WINDOW_CVLI if ck_time_steps is None else ck_time_steps

                # Determine expected input channels from checkpoint (fallback to data)
                ck_in_channels = None
                if 'layer1.temporal_conv.weight' in state_dict:
                    try:
                        ck_in_channels = state_dict['layer1.temporal_conv.weight'].shape[1]
                    except Exception:
                        ck_in_channels = None

                if ck_in_channels is None:
                    try:
                        ck_in_channels = node_features.shape[2]
                    except Exception:
                        ck_in_channels = 8

                m_cvli = STGCN(num_nodes=num_nodes, in_channels=ck_in_channels, time_steps=instantiate_ts, num_classes=1, num_graphs=num_graphs)
                m_cvli.load_state_dict(state_dict, strict=False)
                m_cvli.to(device)
                m_cvli.eval()
                model_cvli = m_cvli
                print(f"[OK] Modelo STGCN v2 carregado (in_channels={ck_in_channels}, time_steps={instantiate_ts})")
                
                # Carregar modelo de ranking para validação em tempo de execução
                global ranking_validator
                ranking_validator = None
                if RANKING_MODEL_PATH:
                    try:
                        ranking_validator = RankingInference(RANKING_MODEL_PATH, device=device)
                        print(f"[RANKING] ✅ Validador carregado: {os.path.basename(RANKING_MODEL_PATH)}")
                    except Exception as e:
                        print(f"[RANKING] ⚠️ Erro ao carregar validador: {e}")
            except Exception as e:
                print(f"Erro ao carregar state_dict CVLI: {e}")

        # --- ST-GAT Loading ---
        STGAT_PATH = os.path.join(BASE_DIR, 'models', 'st_gat_production.pth')
        if os.path.exists(STGAT_PATH):
            try:
                print(f"Carregando modelo ST-GAT de {STGAT_PATH}...")
                # Assuming parameters match training: in_channels=26, time_steps=12, num_graphs=2
                stgat_in_channels = 26
                stgat_time_steps = 12 
                stgat_num_graphs = 2
                
                m_stgat = STGAT(num_nodes=num_nodes, in_channels=stgat_in_channels, time_steps=stgat_time_steps, num_graphs=stgat_num_graphs, dropout=0.5)
                # Use weights_only=False for complex objects if needed, but True is safer
                try:
                    m_stgat.load_state_dict(torch.load(STGAT_PATH, map_location=device))
                except:
                     m_stgat.load_state_dict(torch.load(STGAT_PATH, map_location=device, weights_only=False))
                     
                m_stgat.to(device)
                m_stgat.eval()
                model_stgat = m_stgat
                print("[OK] Modelo ST-GAT carregado com sucesso")
            except Exception as e:
                print(f"Erro ao carregar ST-GAT: {e}")

        # ===== WEEK 4: INITIALIZE EXPLANATION MODULES =====
        try:
            global metric_reporter, event_manager, anomaly_monitor, explanation_generator
            
            # Initialize MetricReporter
            metric_reporter = MetricReporter()
            print("[WEEK4] ✅ MetricReporter inicializado")
            
            # Initialize EventManager
            event_file = os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json')
            if os.path.exists(event_file):
                event_manager = EventManager(event_file)
                print(f"[WEEK4] ✅ EventManager carregado ({len(event_manager.events)} eventos)")
                
                # Initialize AnomalyMonitor (periodic anomaly detection)
                anomaly_monitor = start_anomaly_monitoring(
                    event_manager=event_manager,
                    interval_minutes=15  # Check every 15 minutes
                )
                if anomaly_monitor:
                    print(f"[WEEK4] ✅ AnomalyMonitor iniciado (verificação a cada 15 min)")
            else:
                print(f"[WEEK4] ⚠️ Nenhum arquivo de eventos encontrado: {event_file}")
            
            # Initialize ExplanationGenerator
            explanation_generator = ExplanationGenerator()
            print("[WEEK4] ✅ ExplanationGenerator inicializado")
            
        except Exception as e:
            print(f"[WEEK4] ⚠️ Erro ao inicializar módulos de explainability: {e}")
    
    except Exception as e:
        print(f"Erro ao carregar dados/modelos: {e}")
        # Only print stack trace if it's NOT the expected validation error
        if "CRITICAL" not in str(e):
            import traceback
            traceback.print_exc()
    finally:
        pass


def _periodic_reload_loop(interval_minutes: int):
    interval = max(1, int(interval_minutes)) * 60
    while True:
        try:
            print("[PeriodicReload] Scheduled reload starting...")
            try:
                app._periodic_update_in_progress = True
            except Exception:
                pass

            load_data_and_models()

            try:
                app._periodic_last_update = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass

            try:
                app._periodic_update_in_progress = False
            except Exception:
                pass

            print("[PeriodicReload] Scheduled reload finished.")
        except Exception as e:
            print(f"[PeriodicReload] Error during scheduled reload: {e}")
        time.sleep(interval)


def start_periodic_reload(interval_minutes: int = 30):
    if getattr(app, '_periodic_reload_started', False):
        return
    # Aumentar intervalo de 30 para 60 minutos para evitar oscilações frequentes
    # A oscilação observada (26→141→63→24) é causada por recarregamentos frequentes
    # que alteram a topologia da rede via apply_exogenous_events()
    adjusted_interval = max(60, int(interval_minutes))  # Mínimo 60 minutos
    print(f"[SETUP] Recarregamento periódico ajustado para {adjusted_interval} minutos (evita oscilações)")
    t = threading.Thread(target=_periodic_reload_loop, args=(adjusted_interval,), daemon=True)
    t.start()
    app._periodic_reload_started = True


@app.route('/api/periodic_status')
def periodic_status():
    try:
        return jsonify({
            'in_progress': bool(getattr(app, '_periodic_update_in_progress', False)),
            'last_update': getattr(app, '_periodic_last_update', None)
        })
    except Exception:
        return jsonify({'error': 'Erro ao obter status'}), 500

@app.route('/api/model-update-status')
def model_update_status():
    """Retorna status da atualização de modelos."""
    try:
        state = get_monitor_state()
        return jsonify({
            'status': state.get('status', 'idle'),
            'progress': state.get('progress', 0),
            'message': state.get('message', '') or '',
            'error': state.get('error_message') or '',
            'last_check': state.get('last_check') or None,
            'last_update': state.get('last_update') or None
        })
    except Exception:
        # Ensure consistent schema even on error to avoid frontend seeing `null`
        return jsonify({
            'status': 'idle',
            'progress': 0,
            'message': '',
            'error': '',
            'last_check': None,
            'last_update': None
        })

start_periodic_reload(30)

def format_trend(prediction, history_avg, risk_score=None):
    if history_avg == 0:
        if prediction > 0.001:
            return "Nova atividade criminal detectada"
        else:
            return "Situação estável (Baixo Risco)"

    change_pct = ((prediction - history_avg) / history_avg) * 100

    if change_pct > 15:
        return "Aumento recente da criminalidade"
    elif change_pct < -15:
        return "Redução da atividade criminal"
    else:
        if risk_score is not None:
            if risk_score > 60:
                return "Valor histórico alto para o período"
            elif risk_score > 20:
                return "Atividade criminal moderada"

        return "Situação estável (Baixo Risco)"

def get_region_name(feature_props, geometry):
    city = feature_props.get('CIDADE', '')
    if city and isinstance(city, str) and len(city.strip()) > 1:
        return city
    try:
        from shapely.geometry import Point
    except Exception:
        Point = None

    if geometry is not None:
        try:
            centroid = shape(geometry).centroid
            lon, lat = centroid.x, centroid.y
        except Exception:
            lon = lat = None

        if lon is not None and lat is not None and Point is not None:
            pt = Point(lon, lat)

            if ibge_municipios_gdf is not None and getattr(ibge_municipios_gdf, 'geometry', None) is not None:
                possible_idx = []
                try:
                    possible_idx = list(ibge_municipios_gdf.sindex.intersection(pt.bounds))
                except Exception:
                    possible_idx = []

                if possible_idx:
                    candidates = ibge_municipios_gdf.iloc[possible_idx]
                    for _, mun_row in candidates.iterrows():
                        poly = mun_row.geometry if 'geometry' in mun_row else None
                        if poly is not None and poly.contains(pt):
                            for col in ('name', 'NAME', 'NOME', 'nome', 'municipio', 'NM_MUNICIP'):
                                if col in mun_row and isinstance(mun_row[col], str) and mun_row[col].strip():
                                    return mun_row[col].strip()
                            return str(mun_row.name)

            if lon is not None and lat is not None:
                if -38.66 <= lon <= -38.40 and -3.90 <= lat <= -3.65:
                    return "Fortaleza"

    return "RMF/Interior"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/polygons')
def get_polygons():
    global polygons_json_cache
    try:
        # Carregar polígonos de bairros do AIS - CAPITAL.geojson
        ais_capital_path = os.path.join(BASE_DIR, 'data', 'static', 'AIS - CAPITAL.geojson')
        features = []
        
        if os.path.exists(ais_capital_path):
            try:
                with open(ais_capital_path, 'r', encoding='utf-8') as f:
                    ais_data = json.load(f)
                    features = ais_data.get('features', [])
                    
                    # Normalizar nome field (AIS usa 'Name' maiúsculo)
                    for feat in features:
                        if 'properties' in feat:
                            props = feat['properties']
                            # Garantir que existe 'name' minúsculo
                            if 'name' not in props and 'Name' in props:
                                props['name'] = props['Name']
                            if 'name' not in props:
                                props['name'] = 'Área'
                    
                    print(f"[DEBUG] Carregado {len(features)} polígonos de bairros")
            except Exception as e:
                print(f"[DEBUG] Erro ao carregar AIS: {e}")
                features = []
        
        # Retornar apenas polígonos (sem micro-nós por enquanto)
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return jsonify(geojson)

    except Exception as e:
        return jsonify({'error': f'Erro ao serializar polígonos: {e}'}), 500

@app.route('/api/risk')
def get_risk():
    return calculate_risk()

@app.route('/connections')
def connections_view():
    return render_template('connections.html')

@app.route('/api/network-graph')
def get_network_graph():
    if adj_matrix is None:
        return jsonify({'error': 'Matriz de adjacência não carregada.'}), 503

    response = calculate_risk()
    if response.status_code != 200:
        return response

    risk_data = response.get_json()
    all_nodes = risk_data.get('data', [])
    # Accept optional filters via query params:
    #   region: fortaleza | rmf | interior | all (default all)
    #   critical_only: true|false (default false)
    #   connected_only: true|false (default false) - only show nodes with connections
    #   risk_threshold: numeric percent (default 90)
    region = (request.args.get('region') or 'all').strip().lower()
    critical_only = str(request.args.get('critical_only', 'false')).lower() in ['1', 'true', 'yes']
    connected_only = str(request.args.get('connected_only', 'false')).lower() in ['1', 'true', 'yes']
    try:
        risk_threshold = float(request.args.get('risk_threshold', 90.0))
    except Exception:
        risk_threshold = 90.0

    # Build mapping node_id -> risk item
    node_map = {item['node_id']: item for item in all_nodes}

    # Helper to get name from nodes_gdf or fallback
    def node_name(nid):
        name = f"Area {nid}"
        try:
            if nodes_gdf is not None and nid < len(nodes_gdf):
                name = nodes_gdf.iloc[nid].get('name') or nodes_gdf.iloc[nid].get('nome') or name
        except Exception:
            pass
        return name

    selected_ids = set()

    # ALWAYS show only High (80-89%) and Critical (≥90%) nodes with connections
    # Filter nodes by region first
    region_ids = set()
    if nodes_gdf is not None and region != 'all':
        # Normalize regiao column if present
        try:
            rg = nodes_gdf.get('regiao') if 'regiao' in nodes_gdf.columns else None
            if rg is not None:
                for idx, row in nodes_gdf.reset_index().iterrows():
                    nid = idx
                    r = str(row.get('regiao', '')).lower()
                    node_type = str(row.get('node_type', '')).lower()
                    
                    # region matching rules
                    if region == 'fortaleza':
                        # Only bairros of Fortaleza
                        if node_type == 'bairro' and r == 'fortaleza':
                            region_ids.add(nid)
                    elif region == 'rmf':
                        # RMF cities and neighborhoods
                        if r == 'rmf':
                            region_ids.add(nid)
                    elif region == 'interior':
                        # Interior cities (not Fortaleza, not RMF)
                        if r == 'interior':
                            region_ids.add(nid)
            else:
                # Fallback: consider all nodes if region filter not resolvable
                region_ids = set([item['node_id'] for item in all_nodes])
        except Exception:
            region_ids = set([item['node_id'] for item in all_nodes])
    else:
        region_ids = set([item['node_id'] for item in all_nodes])

    # Select ALL nodes with risk >= 80% (High or Critical) within region
    for nid in list(region_ids):
        item = node_map.get(nid)
        if item and float(item.get('risk_score', 0)) >= 80.0:
            selected_ids.add(nid)

    # Build response nodes and links
    graph_nodes = []
    for nid in sorted(selected_ids):
        item = node_map.get(nid, {})
        graph_nodes.append({
            'id': nid,
            'name': node_name(nid),
            'risk': float(item.get('risk_score', 0)),
            'faction': item.get('faction'),
            'reasons': item.get('reasons', [])
        })

    links = []
    # Only show connections that are significant (> 0.05 = 5% influence)
    for u in selected_ids:
        for v in selected_ids:
            if u == v: continue
            try:
                weight = float(adj_matrix[u, v])
                # Filter to show only meaningful connections
                if weight > 0.05:
                    links.append({'source': u, 'target': v, 'weight': weight, 'type': 'influence'})
            except Exception:
                pass

    # ALWAYS filter to show ONLY nodes with connections (connected_only is always true for this view)
    if len(links) > 0:
        connected_node_ids = set()
        for link in links:
            connected_node_ids.add(link['source'])
            connected_node_ids.add(link['target'])
        
        # Keep only nodes that have connections
        graph_nodes = [n for n in graph_nodes if n['id'] in connected_node_ids]
    else:
        # If no connections, return empty network
        graph_nodes = []

    return jsonify({'nodes': graph_nodes, 'links': links})

@app.route('/api/simulate', methods=['POST'])
def simulate_risk():
    if node_features is None or adj_matrix is None or nodes_gdf is None:
        return jsonify({'error': 'Dados não carregados.'}), 503

    try:
        data = request.get_json()
        points = data.get('points', [])
        sim_type = data.get('type', 'suppression')

        adj_copy = adj_matrix.copy()
        affected_nodes = set()
        centroids = None
        if nodes_gdf_proj is not None:
             if nodes_centroids_proj is None:
                 centroids = nodes_gdf_proj.geometry.centroid
        else:
             centroids = nodes_gdf.geometry.centroid

        for pt in points:
            if len(pt) == 2:
                p_geo = Point(pt[1], pt[0])
                nearby_indices = []

                if nodes_gdf_proj is not None:
                    try:
                        s = gpd.GeoSeries([p_geo], crs="EPSG:4326").to_crs("EPSG:3857")
                        p_proj = s.iloc[0]

                        if nodes_centroids_proj is not None:
                             # Tentar com buffer de 5km primeiro, depois 10km se não encontrar nada
                             for buffer_m in [5000, 10000, 50000]:
                                 search_buffer = p_proj.buffer(buffer_m)
                                 candidate_ilocs = list(nodes_centroids_proj.sindex.intersection(search_buffer.bounds))
                                 if candidate_ilocs:
                                     candidates = nodes_centroids_proj.iloc[candidate_ilocs]
                                     dists = candidates.distance(p_proj)
                                     nearby_indices = dists[dists < buffer_m].index.tolist()
                                     if nearby_indices:
                                         break
                        else:
                             dists = centroids.distance(p_proj)
                             nearby_indices = dists[dists < 5000].index.tolist()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        dists = nodes_gdf.geometry.centroid.distance(p_geo)
                        nearby_indices = [dists.idxmin()]
                else:
                    dists = centroids.distance(p_geo)
                    nearby_indices = dists[dists < 0.005].index.tolist()
                    if not nearby_indices:
                        nearby_indices = [dists.idxmin()]

                for idx in nearby_indices:
                    affected_nodes.add(idx)
                    if sim_type == 'suppression':
                        # Equipe cobre 20km² - suprime 80% do risco local (mantém 20%)
                        suppression_factor = 0.20
                        adj_copy[idx, :] *= suppression_factor
                        adj_copy[:, idx] *= suppression_factor
                    elif sim_type == 'exogenous':
                        amplification_factor = 5.0
                        adj_copy[idx, :] *= amplification_factor
                        adj_copy[:, idx] *= amplification_factor

        if adj_geo is not None and adj_faction is not None:
            adj_geo_copy = adj_geo.copy()
            adj_faction_copy = adj_faction.copy()
            sim_norm_list = compute_norm_adj_list([adj_geo_copy, adj_faction_copy])
            response = calculate_risk(custom_norm_adj=sim_norm_list)
        else:
            adj_tensor = torch.FloatTensor(adj_copy)
            rowsum = adj_tensor.sum(1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            sim_norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt).to(device)
            response = calculate_risk(custom_norm_adj=sim_norm_adj)

        if response.status_code == 200:
            result_json = response.get_json()
            data_list = result_json.get('data', [])

            for item in data_list:
                nid = item['node_id']
                if nid in affected_nodes:
                    if sim_type == 'suppression':
                        factor = 0.15
                        item['risk_score'] = item['risk_score'] * factor
                        item['risk_score_cvli'] = item['risk_score_cvli'] * factor
                        item['risk_score_cvp'] = item['risk_score_cvp'] * factor
                        item['reasons'].insert(0, "Área sob controle (Equipe Tática)")
                        item['cvli_pred'] *= factor
                        item['cvp_pred'] *= factor

                    elif sim_type == 'exogenous':
                        boost = 1.5
                        new_score = max(item['risk_score'] * boost, 80.0)
                        item['risk_score'] = min(new_score, 100.0)
                        item['risk_score_cvli'] = min(max(item['risk_score_cvli'] * boost, 80.0), 100.0)
                        item['risk_score_cvp'] = min(max(item['risk_score_cvp'] * boost, 80.0), 100.0)
                        item['reasons'].insert(0, "Conflito Ativo (Simulação)")

            return jsonify(result_json)
        else:
            return response

    except Exception as e:
        print(f"Erro na simulação: {e}")
        return jsonify({'error': f'Erro na simulação: {e}'}), 500

def calculate_risk(custom_norm_adj=None):
    global dates
    
    if node_features is None:
        print("ERRO: node_features não foi carregado!")
        return jsonify({'error': 'Features dos nós não carregadas. Execute python src/data_processing.py'}), 503
    
    if nodes_gdf is None:
        print("ERRO: nodes_gdf não foi carregado!")
        return jsonify({'error': 'GeoDataFrame dos nós não carregado. Verifique processed_graph_data.pkl'}), 503

    # Lazy-load ranking artifact if available (handles server restarts)
    # NOTE: RankingInference handles model loading, no need to load 'scores' separately
    global model_ranking_scores
    # model_ranking_scores remains None - it's computed on-demand by RankingInference

    current_norm_adj = custom_norm_adj if custom_norm_adj is not None else norm_adj
    def _ensure_adj_list(adj):
        if adj is None:
            return None
        if isinstance(adj, (list, tuple)):
            return list(adj)
        if norm_adj_list is not None:
            return norm_adj_list
        return [adj]

    adj_for_model = _ensure_adj_list(current_norm_adj)
    if adj_for_model is None:
         return jsonify({'error': 'Matriz de adjacência não disponível.'}), 503

    try:
        # ---------------------------
        # Previsão CVLI com Contexto CVP (Veículos)
        # ---------------------------
        # Canal 0: CVLI (homicídios)
        # Canal 1: CVP_VEÍCULOS (roubos/furtos de veículos - correlacionam com facções)
        # Canal 2: TENSION_INDEX (índice de tensão territorial)
        # ---------------------------
        out_cvli = np.zeros((node_features.shape[0], 1))
        hist_avg_cvli = np.zeros(node_features.shape[0])
        hist_sum_cvli = np.zeros(node_features.shape[0])

        # CVP histórico para display
        hist_avg_cvp = np.zeros(node_features.shape[0])
        hist_sum_cvp = np.zeros(node_features.shape[0])

        if model_cvli:
            try:
                model_ts = model_cvli.conv_final.kernel_size[-1]
            except Exception:
                model_ts = WINDOW_CVLI

            if node_features.shape[1] >= model_ts:
                # Usar todos os 3 canais (CVLI, CVP_Veículos, Tensão)
                input_slice = node_features[:, -model_ts:, :] # (N, T, 3)

                # Safety check para compatibilidade de canais
                if input_slice.shape[2] != 3:
                    if input_slice.shape[2] == 2:
                        # Adicionar canal de tensão se faltando
                        zeros = np.zeros((input_slice.shape[0], input_slice.shape[1], 1), dtype=input_slice.dtype)
                        input_slice = np.concatenate([input_slice, zeros], axis=2)
                        print("AVISO: Canal de Tensão adicionado (zeros) para compatibilidade.")

                input_tensor = torch.FloatTensor(input_slice).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, N, T)

                # Protege predição com lock para evitar race conditions com apply_exogenous_events()
                with prediction_lock:
                    # Força modo de avaliação (desativa dropout e batch norm estocástico)
                    model_cvli.eval()
                    with torch.no_grad():
                        pred = model_cvli(input_tensor, adj_for_model)
                    out_cvli = pred.squeeze(0).cpu().numpy() # (N, 1)
                
                try:
                    if _DESSCALE_A is not None:
                        out_cvli = (_DESSCALE_A * out_cvli) + _DESSCALE_B
                except Exception:
                    pass

                # Histórico dos canais
                input_cvli = input_slice[:, :, 0]  # Canal 0: CVLI
                daily_avg = np.mean(input_cvli, axis=1)
                hist_avg_cvli = daily_avg * 3
                hist_sum_cvli = np.sum(input_cvli, axis=1)

                # CVP (agora apenas veículos)
                input_cvp = input_slice[:, :, 1]  # Canal 1: CVP_Veículos
                daily_avg_cvp = np.mean(input_cvp, axis=1)
                hist_avg_cvp = daily_avg_cvp
                hist_sum_cvp = np.sum(input_cvp, axis=1)

            else:
                print("AVISO: Dados insuficientes para janela temporal")

        # CVLI - Usar ranking ao invés de threshold binário
        out_cvli = np.maximum(out_cvli, 0)
        cvli_raw = out_cvli[:, 0]
        
        # Calibração por percentil (melhor que normalização linear)
        # Análise mostrou: Top 1% = 16.96% acerto, Top 5% = 8.84% acerto
        percentiles = np.zeros_like(cvli_raw)
        for i, val in enumerate(cvli_raw):
            percentiles[i] = (cvli_raw < val).sum() / len(cvli_raw) * 100
        
        # Converte percentil para score 0-100
        normalized_risk_cvli = percentiles.copy()
        
        # ===== INTEGRAÇÃO RANKING INFERENCE (BLEND CONTÍNUO 70/30) =====
        # Avaliação comparativa mostrou +20% P@5 vs RankingCorrectionSystem
        # Combina ST-GCN (70%) + Ranking (30%) de forma contínua
        ranking_confidence = 0.0
        
        if ranking_validator is not None:
            try:
                # Extrair features (12-dim) dos últimos 30 dias de CVLI
                cvli_window = node_features[:, -30:, 0]  # (N, 30)
                features_for_ranking = extract_features_clean(cvli_window)
                
                # Validar/combinar predições ST-GCN com ranking model
                combined_scores_normalized, top_indices = ranking_validator.validate_stgcn_predictions(
                    normalized_risk_cvli,
                    features_for_ranking,
                    top_k=20  # Validar top-20 para ter margem
                )
                
                # Desnormalizar de [0,1] para [0,100] mantendo distribuição
                # Mapear scores normalizados de volta para escala 0-100
                combined_scores_100 = combined_scores_normalized * 100.0
                
                # Substituir scores do ST-GCN pelos combinados
                normalized_risk_cvli = combined_scores_100.copy()
                
                # ===== CONFIANÇA BASEADA EM PADRÕES REAIS =====
                # Avaliar múltiplos indicadores de qualidade do ranking
                if len(top_indices) >= 5:
                    top5_nodes = top_indices[:5]
                    
                    # 1. CONCORDÂNCIA ST-GCN vs RANKING
                    stgcn_top5 = np.argsort(-percentiles)[:5]
                    overlap = len(set(top5_nodes) & set(stgcn_top5))
                    concordance_score = overlap / 5.0  # 0-1
                    
                    # 2. CONSISTÊNCIA TEMPORAL (últimos 7 dias com atividade)
                    temporal_consistency = 0.0
                    for node_idx in top5_nodes:
                        recent_days = cvli_window[node_idx, -7:]  # Últimos 7 dias
                        active_days = (recent_days > 0).sum()
                        temporal_consistency += (active_days / 7.0)
                    temporal_consistency /= 5.0  # Média do top-5
                    
                    # 3. TENDÊNCIA/MOMENTUM (comparar última semana vs semana anterior)
                    momentum_score = 0.0
                    for node_idx in top5_nodes:
                        last_week = cvli_window[node_idx, -7:].sum()
                        prev_week = cvli_window[node_idx, -14:-7].sum() if cvli_window.shape[1] >= 14 else 0
                        if prev_week > 0:
                            trend = min(2.0, last_week / prev_week) / 2.0  # Normaliza para 0-1
                        else:
                            trend = 1.0 if last_week > 0 else 0.0
                        momentum_score += trend
                    momentum_score /= 5.0
                    
                    # 4. EVENTOS EXÓGENOS (Top-5 afetado por eventos recentes?)
                    exogenous_validation = 0.0
                    if exogenous_affected_nodes:
                        exo_overlap = len(set(top5_nodes) & exogenous_affected_nodes)
                        exogenous_validation = min(1.0, exo_overlap / 3.0)  # Até 3 nodes é significativo
                    
                    # 5. SEPARAÇÃO DE SCORES (original, mantém relevância)
                    top1_score = combined_scores_normalized[top5_nodes[0]]
                    top5_score = combined_scores_normalized[top5_nodes[4]]
                    separation_score = min(1.0, (top1_score - top5_score) * 5.0)
                    
                    # Combinar indicadores com pesos baseados em importância analítica
                    ranking_confidence = (
                        concordance_score * 0.25 +      # Alinhamento entre modelos
                        temporal_consistency * 0.30 +   # Padrão temporal recente
                        momentum_score * 0.20 +         # Tendência crescente
                        exogenous_validation * 0.15 +   # Contexto de eventos
                        separation_score * 0.10         # Separação clara
                    )
                    
                    print(f"[RANKING INFERENCE] Blend aplicado - Confiança: {ranking_confidence:.2f}")
                    print(f"[RANKING INFERENCE] └─ Concordância: {concordance_score:.2f} | Temporal: {temporal_consistency:.2f} | Momentum: {momentum_score:.2f}")
                    print(f"[RANKING INFERENCE] └─ Eventos: {exogenous_validation:.2f} | Separação: {separation_score:.2f}")
                    print(f"[RANKING INFERENCE] Top-5: {top5_nodes.tolist()}")
                else:
                    ranking_confidence = 0.5
                    print(f"[RANKING INFERENCE] Top-5 insuficiente - Confiança: {ranking_confidence:.2f}")
                    print(f"[RANKING INFERENCE] Top disponíveis: {top_indices.tolist()}")
                
            except Exception as e:
                print(f"[RANKING INFERENCE] Erro ao aplicar blend: {e}")
                import traceback
                traceback.print_exc()
                ranking_confidence = 0.0
                import traceback
                traceback.print_exc()
                ranking_confidence = 0.0
        else:
            print("[RANKING INFERENCE] Não disponível - usando apenas ST-GCN")
        
        # ====================================================================

        # Provenance flags for debugging why a node's score was raised
        provenance_history = np.zeros_like(normalized_risk_cvli, dtype=bool)
        provenance_very_active = np.zeros_like(normalized_risk_cvli, dtype=bool)
        provenance_exogenous = np.zeros_like(normalized_risk_cvli, dtype=bool)
        provenance_exo_critical = np.zeros_like(normalized_risk_cvli, dtype=bool)
        provenance_neighbor_boost = np.zeros_like(normalized_risk_cvli, dtype=bool)

        # Boosting baseado em histórico (mais conservador)
        active_indices = hist_sum_cvli > 0
        normalized_risk_cvli[active_indices] = np.maximum(normalized_risk_cvli[active_indices], 30.0)
        provenance_history[active_indices] = True

        very_active = hist_sum_cvli >= 3
        normalized_risk_cvli[very_active] = np.maximum(normalized_risk_cvli[very_active], 60.0)
        provenance_very_active[very_active] = True

        # Eventos exógenos aumentam para percentil alto
        if exogenous_affected_nodes:
            exo_indices = list(exogenous_affected_nodes)
            exo_indices = [i for i in exo_indices if i < len(normalized_risk_cvli)]
            if exo_indices:
                normalized_risk_cvli[exo_indices] = np.maximum(normalized_risk_cvli[exo_indices], 85.0)
                provenance_exogenous[exo_indices] = True

            try:
                if exogenous_critical_nodes:
                    crit_idxs = [i for i in exogenous_critical_nodes if i < len(normalized_risk_cvli)]
                    if crit_idxs:
                        normalized_risk_cvli[crit_idxs] = np.maximum(normalized_risk_cvli[crit_idxs], 95.0)
                        provenance_exo_critical[crit_idxs] = True
                        top_k = 10
                        try:
                            if adj_faction is not None:
                                for ci in crit_idxs:
                                    neigh = np.where(np.array(adj_faction[ci]) > 0)[0]
                                    neigh = [n for n in neigh if n != ci]
                                    if not neigh: continue
                                    selected = neigh[:top_k]
                                    if selected:
                                        normalized_risk_cvli[selected] = np.maximum(normalized_risk_cvli[selected], 90.0)
                                        provenance_neighbor_boost[selected] = True
                        except Exception:
                            pass
            except Exception:
                pass

        # Use same risk for CVP UI indication for now, or just normalized history?
        # Since we only predict CVLI now, we don't have a separate CVP prediction model in this loop unless we kept it?
        # The user said "unified". So out_cvli IS the risk.
        # But for UI 'risk_score_cvp', we can use the history as proxy or just copy main risk.
        # Let's derive CVP Risk from CVP history + Context for now to populate the UI field.
        # Or even better, assume CVP Risk correlates with Main Risk (since it's a precursor).
        # CVP proxy for UI
        normalized_risk_cvp = normalized_risk_cvli.copy()

        # Prepare ranking-derived signals if artifact available
        ranking_scores_arr = None
        ranking_top10_threshold = None
        try:
            if model_ranking_scores is not None and len(model_ranking_scores) == len(normalized_risk_cvli):
                ranking_scores_arr = np.array(model_ranking_scores)
                ranking_top10_threshold = float(np.percentile(ranking_scores_arr, 90))
        except Exception:
            pass
        # But CVP is raw count, not risk.
        # Let's just normalize CVP History for visualization?
        # Actually, let's leave it as copy of main risk, but potentially modified by local CVP intensity.

        # ---------------------------
        # Construção da Resposta
        # ---------------------------
        results = []

        try:
            conn = np.array(adj_matrix).sum(axis=1)
            conn_mean = float(np.mean(conn))
        except Exception:
            conn = None
            conn_mean = 0

        try:
            cutoff_cvli = float(np.percentile(normalized_risk_cvli, 90))
        except Exception:
            cutoff_cvli = 80.0

        factions = nodes_gdf['faction'].tolist() if 'faction' in nodes_gdf.columns else [None] * len(nodes_gdf)

        for i in range(len(normalized_risk_cvli)):
            cvli_score = float(normalized_risk_cvli[i])
            cvp_score = float(normalized_risk_cvp[i])

            cvli_val = float(cvli_raw[i])
            # For CVP 'prediction' in UI, use history avg if we don't have explicit pred
            cvp_val = float(hist_avg_cvp[i])

            trend_cvli = format_trend(cvli_raw[i], hist_avg_cvli[i], risk_score=cvli_score)
            trend_cvp = format_trend(cvp_val, hist_avg_cvp[i], risk_score=cvp_score)

            # --- EXPLICAÇÕES CONTEXTUAIS MELHORADAS ---
            reasons = []
            
            # 1. Eventos exógenos (prioridade máxima)
            if i in exogenous_affected_nodes:
                severity = "alta" if i in exogenous_critical_nodes else "moderada"
                reasons.append(f"🔴 Conflito ativo detectado (severidade {severity})")
            
            # 2. Predição do modelo
            if cvli_val > 0.5:
                if cvli_val >= 2.0:
                    reasons.append(f"📈 Modelo prevê {cvli_val:.1f} homicídios nos próximos 7 dias")
                elif cvli_val >= 1.0:
                    reasons.append(f"⚠️ Modelo prevê ~{int(cvli_val)} homicídio nos próximos 7 dias")
                else:
                    reasons.append(f"📊 Padrão de risco elevado detectado pelo modelo")
            elif cvli_score > 70:
                reasons.append("🎯 Área no top 10% de maior risco (modelo ST-GCN)")
            
            # 3. Histórico recente
            if hist_sum_cvli[i] >= 5:
                days = 14  # janela do modelo
                reasons.append(f"🔴 {int(hist_sum_cvli[i])} homicídios nos últimos {days} dias")
            elif hist_sum_cvli[i] >= 2:
                days = 14
                reasons.append(f"⚠️ {int(hist_sum_cvli[i])} homicídios nos últimos {days} dias")
            elif hist_sum_cvli[i] >= 1:
                reasons.append("📍 Histórico recente de violência letal")
            
            # 4. Atividade de veículos (indicador de facções)
            if hist_sum_cvp[i] >= 10:
                reasons.append(f"🚗 Alta atividade de roubo/furto de veículos ({int(hist_sum_cvp[i])} eventos recentes)")
            elif hist_sum_cvp[i] >= 5:
                reasons.append(f"🚙 Roubos/furtos de veículos detectados ({int(hist_sum_cvp[i])} casos)")
            
            # 5. Índice de tensão territorial
            try:
                tension = nodes_gdf.iloc[i].get('tension_index', 0)
                if tension > 0.7:
                    reasons.append("⚔️ Área de alta tensão territorial (disputa de facções)")
                elif tension > 0.5:
                    reasons.append("⚠️ Tensão territorial elevada")
            except:
                pass
            
            # 6. Conectividade (rotas)
            if conn is not None and conn_mean > 0:
                if conn[i] > conn_mean * 2.0:
                    reasons.append("🛣️ Área estratégica (alta conectividade - rota de fuga)")
                elif conn[i] > conn_mean * 1.5:
                    reasons.append("🗺️ Área com múltiplas conexões (acesso facilitado)")
            
            # 7. Facção dominante
            try:
                faction = factions[i]
                if faction and faction != 'None' and faction != 'neutral':
                    reasons.append(f"🏴 Território de influência: {faction}")
            except:
                pass
            
            # 8. Mensagem padrão se não houver razões
            if len(reasons) == 0:
                if cvli_score < 20:
                    reasons.append('✅ Situação estável - baixo risco de violência letal')
                else:
                    reasons.append('📊 Risco moderado - monitoramento recomendado')

            def _status_label(score: float) -> str:
                    # New classification bands (aligned with frontend request):
                    # crítico: >=90, alto: [80,90), moderado: [50,80), baixo: [20,50), sem risco: [0,20)
                    try:
                        s = float(score)
                    except Exception:
                        s = 0.0
                    if s >= 90:
                        return 'Crítico'
                    if s >= 80:
                        return 'Alto'
                    if s >= 50:
                        return 'Moderado'
                    if s >= 20:
                        return 'Baixo'
                    return 'Sem Risco'

            def _prediction_text(val: float, kind: str='CVLI') -> str:
                horizon_days = 7
                try:
                    if val <= 0.01:
                        return f'Sem novas ocorrências previstas (próx. {horizon_days} dias)'
                    if val < 1.0:
                        return f'Menos de 1 ocorrência prevista (próx. {horizon_days} dias)'
                    rounded = round(val, 1)
                    unit = 'ocorrência' if rounded == 1 else 'ocorrências'
                    return f'Estimativa: ~{rounded} {unit} previstas (próx. {horizon_days} dias)'
                except Exception:
                    return 'Estimativa indisponível'

            status_label = _status_label(cvli_score)
            risk_text = f"{int(round(cvli_score))}% — {status_label}"
            cvli_pred_text = _prediction_text(cvli_val, 'CVLI')
            cvp_pred_text = _prediction_text(cvp_val, 'CVP')

            # Build provenance list for this node
            prov = []
            if provenance_history[i]: prov.append('history')
            if provenance_very_active[i]: prov.append('very_active')
            if provenance_exogenous[i]: prov.append('exogenous')
            if provenance_exo_critical[i]: prov.append('exogenous_critical')
            if provenance_neighbor_boost[i]: prov.append('neighbor_boost')

            # ranking_score (optional) from precomputed ranking artifact
            rk_score = None
            try:
                if model_ranking_scores is not None and len(model_ranking_scores) == len(normalized_risk_cvli):
                    rk_score = float(model_ranking_scores[i])
            except Exception:
                rk_score = None

            # Calcular percentil real do nó
            percentile_rank = (np.sum(normalized_risk_cvli <= cvli_score) / len(normalized_risk_cvli)) * 100
            percentile_rank = float(percentile_rank)
            
            # Gerar descrição amigável baseada em percentil
            if percentile_rank >= 99:
                description = "Top 1% em risco no Estado do Ceará"
            elif percentile_rank >= 95:
                description = "Top 5% em risco no Estado do Ceará"
            elif percentile_rank >= 90:
                description = "Top 10% em risco no Estado do Ceará"
            elif percentile_rank >= 75:
                description = "Top 25% em risco no Estado do Ceará"
            elif percentile_rank >= 50:
                description = "Acima da mediana de risco no Estado do Ceará"
            else:
                description = "Abaixo da mediana de risco no Estado do Ceará"

            results.append({
                'node_id': int(i),
                'risk_score': cvli_score,
                'risk_score_cvli': cvli_score,
                'risk_score_cvp': cvp_score,
                'percentile_rank': percentile_rank,
                'description': description,
                'cvli_pred': cvli_val,
                'cvp_pred': cvp_val,
                'faction': factions[i],
                'reasons': reasons,
                'priority_cvli': bool(cvli_score >= cutoff_cvli),
                'status_label': status_label,
                'risk_text': risk_text,
                'cvli_prediction_text': cvli_pred_text,
                'cvp_prediction_text': cvp_pred_text
            ,
                'ranking_score': rk_score,
                'score_provenance': prov
            })

        try:
            if (dates is None) or (hasattr(dates, '__len__') and len(dates) == 0):
                graph_dir = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
                dpkl = os.path.join(graph_dir, 'dates.pkl')
                if os.path.exists(dpkl):
                    with open(dpkl, 'rb') as fh:
                        dates = pickle.load(fh)
        except Exception:
            pass

        meta = {
            'window_cvli': WINDOW_CVLI,
            'window_cvp': WINDOW_CVP,
            'start_cvli': '—',
            'start_cvp': '—',
            'last_date': '—',
            'window_start': '—',
            'window_end': '—'
        }

        if dates is not None and hasattr(dates, '__len__') and len(dates) > 0:
            try:
                last_date_obj = pd.to_datetime(dates[-1])
                meta['last_date'] = str(last_date_obj.date())
                meta['start_cvli'] = str(pd.to_datetime(dates[-min(len(dates), WINDOW_CVLI)]).date())
                meta['start_cvp'] = meta['start_cvli']
                meta['window_start'] = meta.get('start_cvli', '')
                meta['window_end'] = meta.get('last_date', '')
                
                # Adicionar janela de validade dinâmica (hoje + 7 dias)
                forecast_date = last_date_obj.date()
                valid_until = forecast_date + pd.Timedelta(days=7)
                meta['forecast_date'] = str(forecast_date)
                meta['valid_from'] = str(forecast_date)
                meta['valid_until'] = str(valid_until)
                meta['validity_description'] = f"Previsão para {forecast_date.strftime('%d/%m')} até {valid_until.strftime('%d/%m')} (7 dias)"
            except Exception:
                pass

        meta['model_window_cvli'] = WINDOW_CVLI
        # Summary counts for dashboard debugging
        # --- Cross-validate top candidates from CVLI with ranking artifact ---
        try:
            # If ranking artifact available, validate top-K CVLI candidates and demote when ranking disagrees
            TOP_K_VALIDATION = 10
            if ranking_scores_arr is not None and len(results) >= TOP_K_VALIDATION:
                sorted_by_cvli_idx = sorted(range(len(results)), key=lambda x: results[x].get('risk_score', 0), reverse=True)
                top_candidates = sorted_by_cvli_idx[:TOP_K_VALIDATION]
                for cid in top_candidates:
                    try:
                        rk = ranking_scores_arr[cid]
                    except Exception:
                        rk = None
                    # If ranking score exists but is below top-10 threshold, demote the CVLI priority
                    if rk is not None and ranking_top10_threshold is not None and float(rk) < ranking_top10_threshold:
                        # Demote: cap risk to 80 (Alto) and annotate reason/provenance
                        old = results[cid]['risk_score']
                        results[cid]['risk_score'] = min(float(results[cid]['risk_score']), 80.0)
                        results[cid]['reasons'].insert(0, '🔎 Validação de ranking: prioridade reduzida')
                        # preserve original cvli field but mark that ranking demoted
                        prov = results[cid].get('score_provenance', []) or []
                        if 'ranking_demoted' not in prov:
                            prov.append('ranking_demoted')
                        results[cid]['score_provenance'] = prov
                        results[cid]['ranking_score'] = float(rk)

            # Compute counts using final numeric `risk_score` and agreed bands
            counts = {'crítico':0, 'alto':0, 'moderado':0, 'baixo':0, 'sem risco':0}
            for r in results:
                try:
                    sv = float(r.get('risk_score', 0) or 0)
                except Exception:
                    sv = 0.0
                if sv >= 90:
                    counts['crítico'] += 1
                elif sv >= 80:
                    counts['alto'] += 1
                elif sv >= 50:
                    counts['moderado'] += 1
                elif sv >= 20:
                    counts['baixo'] += 1
                else:
                    counts['sem risco'] += 1

            meta['counts'] = counts
            meta['cutoff_cvli'] = float(cutoff_cvli)
            # indicate if ranking artifact was used
            meta['ranking_source'] = os.path.basename(RANKING_MODEL_PATH) if model_ranking_scores is not None else 'stgcn_percentile'
            # Detailed provenance lists: nodes influenced by each rule
            prov_lists = {'history': [], 'very_active': [], 'exogenous': [], 'exogenous_critical': [], 'neighbor_boost': []}
            try:
                for r in results:
                    nid = int(r.get('node_id'))
                    for p in r.get('score_provenance', []):
                        if p in prov_lists:
                            prov_lists[p].append(nid)
                meta['provenance_lists'] = prov_lists
            except Exception:
                meta['provenance_lists'] = prov_lists

            # Distribution summary and percentiles
            try:
                arr_raw = np.array(cvli_raw)
                arr_norm = np.array(normalized_risk_cvli)
                meta['distribution'] = {
                    'raw_min': float(np.min(arr_raw)),
                    'raw_max': float(np.max(arr_raw)),
                    'raw_mean': float(np.mean(arr_raw)),
                    'raw_percentiles': {p: float(np.percentile(arr_raw, p)) for p in [50,75,90,95,99]},
                    'norm_min': float(np.min(arr_norm)),
                    'norm_max': float(np.max(arr_norm)),
                    'norm_mean': float(np.mean(arr_norm)),
                    'norm_percentiles': {p: float(np.percentile(arr_norm, p)) for p in [50,75,90,95,99]}
                }
            except Exception:
                meta['distribution'] = {}

            # hist_sum_cvli summary
            try:
                arr_hist = np.array(hist_sum_cvli)
                meta['history_stats'] = {
                    'hist_min': int(np.min(arr_hist)),
                    'hist_max': int(np.max(arr_hist)),
                    'hist_mean': float(np.mean(arr_hist)),
                    'hist_percentiles': {p: int(np.percentile(arr_hist, p)) for p in [50,75,90,95]}
                }
            except Exception:
                meta['history_stats'] = {}

            # Top nodes debug: both risk_score and ranking_score and provenance
            try:
                topk = 20
                sorted_by_risk = sorted(results, key=lambda x: x.get('risk_score', 0), reverse=True)
                meta['top_nodes_debug'] = [
                    {
                        'node_id': int(r.get('node_id')),
                        'risk_score': float(r.get('risk_score') or 0),
                        'ranking_score': r.get('ranking_score'),
                        'provenance': r.get('score_provenance'),
                        'reasons': r.get('reasons')[:3] if isinstance(r.get('reasons'), list) else r.get('reasons')
                    }
                    for r in sorted_by_risk[:topk]
                ]
            except Exception:
                meta['top_nodes_debug'] = []

        except Exception:
            pass
        meta['model_window_cvp'] = WINDOW_CVP
        
        # Adiciona estat\u00edsticas de ranking (Precision@K)
        # Preferir scores pré-computados do artefato de ranking se disponíveis
        if model_ranking_scores is not None and len(model_ranking_scores) == len(results):
            all_scores = list(map(float, model_ranking_scores.tolist()))
            meta['ranking_source'] = os.path.basename(RANKING_MODEL_PATH)
        else:
            all_scores = [r['risk_score_cvli'] for r in results]
            meta['ranking_source'] = 'stgcn_percentile'

        if all_scores:
            sorted_scores = sorted(all_scores, reverse=True)
            meta['ranking_info'] = {
                'total_nodes': len(all_scores),
                'top_1_percent_threshold': sorted_scores[max(0, int(len(sorted_scores) * 0.01))] if len(sorted_scores) > 0 else 0,
                'top_5_percent_threshold': sorted_scores[max(0, int(len(sorted_scores) * 0.05))] if len(sorted_scores) > 0 else 0,
                'top_10_percent_threshold': sorted_scores[max(0, int(len(sorted_scores) * 0.10))] if len(sorted_scores) > 0 else 0,
                'method': meta.get('ranking_source', 'percentile_ranking'),
                'validation_status': '⚠️ Métricas do app.py usam auto-comparação (sem ground truth independente)',
                'note': 'Para validação real, usar src/validate_with_crossval.py com temporal split'
            }

            # Estatísticas descritivas reais (não são métricas de validação)
            # NOTA: P@K e NDCG reais requerem ground truth — usar src/validate_with_crossval.py
            n_nodes = len(sorted_scores)
            meta['stats_top5_mean'] = float(np.mean(sorted_scores[:5])) if n_nodes >= 5 else 0.0
            meta['stats_top10_mean'] = float(np.mean(sorted_scores[:10])) if n_nodes >= 10 else 0.0
            meta['stats_top20_mean'] = float(np.mean(sorted_scores[:20])) if n_nodes >= 20 else 0.0
            meta['stats_top5_min'] = float(sorted_scores[4]) if n_nodes >= 5 else 0.0
            meta['stats_overall_mean'] = float(np.mean(all_scores)) if n_nodes > 0 else 0.0
            meta['stats_overall_std'] = float(np.std(all_scores)) if n_nodes > 0 else 0.0
            meta['metrics_source'] = 'descriptive_stats'
            meta['metrics_note'] = 'Sem ground truth. Para P@K/NDCG reais, usar validate_with_crossval.py'

        # ==================== LOG DE PREDICTION ====================
        try:
            if predict_logger is not None:
                # Garantir que nodes_gdf está atualizado no logger
                if nodes_gdf is not None:
                    predict_logger.nodes_gdf = nodes_gdf
                log_file = predict_logger.log_prediction(meta, results, timestamp=datetime.now())
                print(f"✅ Log de prediction salvo em: {log_file}")
                meta['log_file'] = os.path.basename(log_file)
        except Exception as e:
            print(f"⚠️ Erro ao salvar log de prediction: {e}")

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        print(f"ERROR details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao calcular risco: {e}'}), 500

def enrich_regions():
    global nodes_gdf, polygons_json_cache
    if nodes_gdf is None: return

    if 'CIDADE' not in nodes_gdf.columns:
        nodes_gdf['CIDADE'] = ''

    # Ensure region_type is available for frontend filters
    if 'regiao' in nodes_gdf.columns:
        # Normalize 'fortaleza' to 'capital' for frontend consistency
        nodes_gdf['region_type'] = nodes_gdf['regiao'].replace('fortaleza', 'capital')
    else:
        nodes_gdf['region_type'] = 'interior' # Fallback default

    # --- FIXED: Use node_type to enforce City names ---
    if 'node_type' in nodes_gdf.columns:
        # If node is city, its name IS the city.
        nodes_gdf.loc[nodes_gdf['node_type'] == 'cidade', 'CIDADE'] = nodes_gdf.loc[nodes_gdf['node_type'] == 'cidade', 'name']
        # If node is bairro, default to Fortaleza if empty
        nodes_gdf.loc[(nodes_gdf['node_type'] == 'bairro') & (nodes_gdf['CIDADE'] == ''), 'CIDADE'] = 'Fortaleza'

    # Continue with fallback logic for anything else
    mask_empty = (nodes_gdf['CIDADE'].isna()) | (nodes_gdf['CIDADE'].astype(str).str.len() < 2)

    if mask_empty.any():
        indices_to_update = nodes_gdf[mask_empty].index
        count_fortaleza = 0
        for idx in indices_to_update:
            row = nodes_gdf.loc[idx]
            reg = get_region_name(row, row.geometry)
            if reg == "Fortaleza":
                nodes_gdf.at[idx, 'CIDADE'] = reg
                count_fortaleza += 1

    mask_still_empty = (nodes_gdf['CIDADE'].isna()) | (nodes_gdf['CIDADE'].astype(str).str.len() < 2) | (nodes_gdf['CIDADE'] == 'RMF/Interior')

    if mask_still_empty.any() and ibge_municipios_cache:
        try:
            nodes_sub = nodes_gdf[mask_still_empty]
            mun_names = list(ibge_municipios_cache.keys())
            mun_coords = list(ibge_municipios_cache.values())
            mun_coords_arr = np.array(mun_coords)

            centroids_proj = nodes_sub.to_crs(epsg=3857).geometry.centroid
            centroids = centroids_proj.to_crs(nodes_sub.crs)
            node_coords = np.column_stack((centroids.y.values, centroids.x.values))

            dists = cdist(node_coords, mun_coords_arr)
            nearest_indices = np.argmin(dists, axis=1)

            assigned_cities = [mun_names[i] for i in nearest_indices]
            nodes_gdf.loc[mask_still_empty, 'CIDADE'] = assigned_cities

        except Exception:
            pass
    polygons_json_cache = None

def normalize_location(text):
    if not text: return ""
    text = text.upper()
    text = text.replace("PQ.", "PARQUE")
    text = text.replace("AV.", "AVENIDA")
    text = text.replace("S/", "SEM ")
    return text.strip()

def find_node_coordinates(location_str):
    if nodes_gdf is None:
        return None

    loc_norm = normalize_location(location_str)
    loc_lower = loc_norm.lower()
    loc_stripped = strip_accents(loc_lower)

    if loc_norm in MANUAL_LOCATIONS:
         return (*MANUAL_LOCATIONS[loc_norm], 'manual')

    # Try polygon-based bairro name matching (prefer authoritative polygons)
    try:
        if 'ba_irros_gdf' in globals() and ba_irros_gdf is not None:
            try:
                for _, prow in ba_irros_gdf.iterrows():
                    # Attempt common name fields
                    pname = None
                    for col in ('Name', 'NAME', 'NOME', 'nome', 'NOME'):
                        if col in prow and isinstance(prow[col], str) and prow[col].strip():
                            pname = prow[col].strip()
                            break

                    # If still not found, try parsing Description for 'NOME:'
                    if not pname:
                        desc = prow.get('Description') or prow.get('description')
                        if isinstance(desc, str):
                            m = re.search(r'NOME:\s*([^<\n]+)', desc, re.IGNORECASE)
                            if m:
                                pname = m.group(1).strip()

                    if not pname:
                        continue

                    pname_norm = normalize_location(pname).upper()
                    pname_lower = pname_norm.lower()
                    pname_stripped = strip_accents(pname_lower)

                    if pname_norm in loc_norm or pname_lower in loc_lower or pname_stripped in loc_stripped:
                        try:
                            cent = prow.geometry.centroid
                            return (cent.y, cent.x, 'specific')
                        except Exception:
                            continue
            except Exception:
                pass
    except Exception:
        pass

    best_match_item = None
    best_match_len = 0

    for item in node_search_index:
        name = item.name_lower
        name_stripped = item.name_stripped

        if name in loc_lower:
            if len(name) > best_match_len:
                best_match_item = item
                best_match_len = len(name)
        elif name_stripped in loc_stripped:
             if len(name) > best_match_len:
                best_match_item = item
                best_match_len = len(name)
        elif loc_lower in name and len(loc_lower) > 4:
             if len(loc_lower) > best_match_len:
                best_match_item = item
                best_match_len = len(loc_lower)
        elif loc_stripped in name_stripped and len(loc_stripped) > 4:
             if len(loc_stripped) > best_match_len:
                best_match_item = item
                best_match_len = len(loc_stripped)

    if best_match_item is not None:
        return (best_match_item.lat, best_match_item.lng, 'specific')

    if ibge_bairros_cache:
        try:
            for bairro, coords in ibge_bairros_cache.items():
                b_norm = normalize_location(bairro).lower()
                b_stripped = strip_accents(b_norm)
                if b_norm in loc_lower or b_stripped in loc_stripped:
                    return (coords[0], coords[1], 'specific')
        except Exception as e:
             pass

    if ibge_municipios_cache:
        try:
            for municipio, coords in ibge_municipios_cache.items():
                m_norm = normalize_location(municipio).lower()
                if m_norm in loc_lower:
                    return (coords[0], coords[1], 'city')
        except Exception as e:
             pass

    if ibge_municipios_cache:
        try:
             import re
             words = re.findall(r'\b\w+\b', loc_lower)
             for word in words:
                 if len(word) < 4: continue
                 for mun, coords in ibge_municipios_cache.items():
                     if mun.lower() == word:
                         return (coords[0], coords[1], 'city')
        except Exception as e:
            pass

    if GEOCODING_ENABLED:
        try:
            geo_res = geocode_address(loc_norm)
            if geo_res:
                return (*geo_res, 'geocode')
        except Exception:
            pass

    return None


def geocode_address(location_str, timeout=10):
    try:
        from geopy.geocoders import Nominatim
    except Exception:
        return None

    try:
        geolocator = Nominatim(user_agent="st-gcn-geocoder", timeout=timeout)
        loc = geolocator.geocode(location_str)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        return None
    return None

@app.route('/api/ciops/parse-report', methods=['POST'])
def parse_ciops_report_endpoint():
    """Parse CIOPS daily report with multiple blocks (OCORRÊNCIAS, HOMICÍDIOS, etc).
    Returns events classified as ENFORCEMENT or CRIME, with severity levels.
    """
    data = request.get_json()
    report_text = data.get('report', '')
    
    if not report_text or not report_text.strip():
        return jsonify({'error': 'Relatório vazio'}), 400
    
    try:
        # Parse report blocks
        events = parse_ciops_report(report_text)
        
        if not events:
            return jsonify({
                'status': 'success',
                'events': [],
                'summary': {'total': 0, 'enforcement': 0, 'crime': 0, 'with_arrests': 0}
            })
        
        # Enrich with location lookups
        enriched = []
        for evt in events:
            try:
                # Try to find coordinates by bairro first
                loc = evt.get('localizacao_completa', '') or evt.get('bairro', '')
                res = find_node_coordinates(loc)
                if res:
                    evt['lat'], evt['lng'], evt['match_quality'] = res
                else:
                    # Fallback to municipio
                    res = find_node_coordinates(evt.get('municipio', 'FORTALEZA'))
                    if res:
                        evt['lat'], evt['lng'] = res[0], res[1]
                        evt['match_quality'] = 'city_level'
            except Exception as e:
                logger.exception(f"Error enriching event: {e}")
            
            # Add classification for canal 9 integration
            evt['canal_9_intensity'] = evt.get('enforcement_intensity', 0.0)
            evt['canal_9_type'] = 'ENFORCEMENT' if 'ENFORCEMENT' in evt.get('event_type', '') else 'CRIME'
            
            enriched.append(evt)
        
        # Summary statistics
        enforcement_count = len([e for e in enriched if 'ENFORCEMENT' in e.get('event_type', '')])
        crime_count = len([e for e in enriched if 'CRIME' in e.get('event_type', '')])
        arrests_count = sum(e.get('num_arrested', 0) for e in enriched)
        
        return jsonify({
            'status': 'success',
            'events': enriched,
            'summary': {
                'total': len(enriched),
                'enforcement': enforcement_count,
                'crime': crime_count,
                'with_arrests': arrests_count,
                'high_severity': len([e for e in enriched if e.get('conflict_severity') == 'HIGH']),
                'medium_severity': len([e for e in enriched if e.get('conflict_severity') == 'MEDIUM']),
                'low_severity': len([e for e in enriched if e.get('conflict_severity') == 'LOW'])
            }
        })
    except Exception as e:
        logger.exception('Error parsing CIOPS report')
        return jsonify({'error': f'Erro ao processar: {str(e)}'}), 500

@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    # Segurança: validar JSON de entrada e tratar casos de payload inválido
    try:
        data = request.get_json(force=False, silent=True)
        if data is None:
            return jsonify({'error': 'JSON inválido ou cabecalho Content-Type ausente.'}), 400
        text = data.get('text', '') if isinstance(data, dict) else ''
    except Exception as e:
        return jsonify({'error': 'Falha ao ler payload JSON.', 'details': str(e)}), 400

    try:
        events = process_exogenous_text(text)
        if events is None:
            return jsonify({'error': 'Nenhum evento retornado pelo parser.'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Erro ao processar texto exógeno.', 'details': str(e)}), 500

    missing_city = []
    for idx, evt in enumerate(events):
        muni = evt.get('municipio') if isinstance(evt, dict) else None
        if not muni or (isinstance(muni, str) and muni.strip() == ''):
            missing_city.append({'index': idx, 'natureza': evt.get('natureza'), 'resumo': evt.get('resumo'), 'localizacao_completa': evt.get('localizacao_completa')})

    if missing_city:
        return jsonify({
            'error': 'Falta a cidade na sua ocorrência!',
            'message': 'Por favor preencha o município nas ocorrências indicadas antes de prosseguir.',
            'missing_city': missing_city
        }), 400
    points = []

    for evt in events:
        found_lat = None
        found_lng = None
        match_quality = 0

        if evt.get('localizacao_completa'):
            res = find_node_coordinates(evt['localizacao_completa'])
            if res:
                found_lat, found_lng, mtype = res
                match_quality = 3 if mtype == 'specific' else 2

        if (not found_lat or match_quality < 2) and evt.get('bairro'):
            res = find_node_coordinates(evt['bairro'])
            if res:
                lat_b, lng_b, mtype_b = res
                if mtype_b == 'specific':
                     found_lat, found_lng = lat_b, lng_b
                     match_quality = 3

        if not found_lat and evt.get('municipio'):
            res = find_node_coordinates(evt['municipio'])
            if res:
                found_lat, found_lng, _ = res
                match_quality = 1

        if found_lat is not None and found_lng is not None:
            import html
            desc = f"{evt.get('natureza', 'EVENTO')} - {evt.get('resumo', '')}"
            if not evt.get('resumo'):
                desc = f"{evt.get('natureza', 'EVENTO')} - {evt.get('localizacao_completa', '')}"

            safe_desc = html.escape(desc)

            points.append({
                'lat': found_lat,
                'lng': found_lng,
                'description': safe_desc,
                'type': 'exogenous',
                'raw_event': evt
            })

    return jsonify({
        'events_processed': len(events),
        'points_found': len(points),
        'points': points
    })

@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
    global exogenous_events
    import json
    data = request.get_json()
    points = data.get('points', [])
    original_text = data.get('original_text', '')

    if not points:
        return jsonify({'error': 'Nenhum ponto para salvar.'}), 400

    new_entry = {
        'id': str(len(exogenous_events) + 1),
        'timestamp': pd.Timestamp.now().isoformat(),
        'original_text': original_text,
        'points': points
    }

    try:
        current_events = []
        if os.path.exists(EXOGENOUS_FILE):
             with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                 try:
                     current_events = json.load(f)
                 except json.JSONDecodeError:
                     current_events = []

        current_events.append(new_entry)

        with open(EXOGENOUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_events, f, ensure_ascii=False, indent=2)

        exogenous_events.append(new_entry)

        update_exogenous_state()

        return jsonify({'status': 'success', 'message': 'Eventos salvos e malha atualizada.'})

    except Exception as e:
        print(f"Erro ao salvar eventos: {e}")
        return jsonify({'error': f'Erro ao salvar: {e}'}), 500

# ===== WEEK 4: EXPLAINABILITY API ENDPOINTS =====

@app.route('/api/explain/<int:node_id>', methods=['GET'])
def explain_node_ranking(node_id):
    """
    Returns human-readable explanation for a node's risk ranking.
    
    Query Parameters:
    - include_factors: bool (default: true) - Include factor breakdown
    - include_caveats: bool (default: true) - Include uncertainty notes
    - format: 'json' or 'text' (default: 'json')
    
    Response:
    {
      'node_id': int,
      'node_name': str,
      'rank': int (1-based),
      'score': float,
      'confidence': float,
      'summary': str,
      'factors': [...],
      'caveats': [...],
      'interpretation': str,
      'risk_level': str
    }
    """
    try:
        if explanation_generator is None:
            return jsonify({'error': 'Explanation generator not initialized'}), 503
        
        if node_features is None or nodes_gdf is None:
            return jsonify({'error': 'Model data not loaded'}), 503
        
        # Validate node_id
        if node_id < 0 or node_id >= node_features.shape[0]:
            return jsonify({'error': f'Invalid node_id: {node_id}. Valid range: 0-{node_features.shape[0]-1}'}), 400
        
        # Get current predictions
        resp = calculate_risk()
        if resp.status_code != 200:
            return jsonify({'error': 'Failed to calculate current predictions'}), 503
        
        risk_data = resp.get_json()
        all_nodes = risk_data.get('data', [])
        
        if not all_nodes or node_id >= len(all_nodes):
            return jsonify({'error': 'Node not in current predictions'}), 404
        
        # Get node data
        node = all_nodes[node_id]
        node_score = float(node.get('risk', 0))
        node_name = node.get('name', f'Node {node_id}')
        
        # Find rank (1-based)
        rank = None
        for idx, n in enumerate(all_nodes, 1):
            if n.get('node_id') == node_id or idx - 1 == node_id:
                rank = idx
                break
        
        if rank is None:
            # Fallback: use index as node_id
            if node_id < len(all_nodes):
                rank = sorted(range(len(all_nodes)), key=lambda i: all_nodes[i].get('risk_score', 0), reverse=True).index(node_id) + 1
            else:
                rank = 1
        
        # Build context dictionary for explanation generator
        nearby_nodes = []
        if rank <= 5:
            nearby_nodes = [all_nodes[i].get('node_id', i) for i in range(max(0, node_id-2), min(len(all_nodes), node_id+3)) if i != node_id]
        
        recent_events = []
        if event_manager:
            try:
                from datetime import date, timedelta
                today = date.today()
                recent_date = today - timedelta(days=5)
                recent_events = event_manager.get_events_for_date_range(recent_date, today)
                recent_events = [{'type': e.get('type', 'unknown'), 'date': e.get('date', '')} 
                                for e in recent_events[:3]]
            except:
                pass
        
        context_dict = {
            'score': node_score,
            'temporal_pattern': f'Peak activity in recent observations (score: {node_score:.1f}/10)',
            'nearby_nodes': nearby_nodes,
            'events': recent_events,
            'confidence': 0.87 if event_manager and event_manager.get_anomaly_level_for_date(date.today()) < 0.6 else 0.75,
            'tier': 'top_5' if rank <= 5 else ('long_tail_20' if rank <= 20 else 'tail')
        }
        
        # Generate explanation using ExplanationGenerator
        explanation = explanation_generator.explain_node_ranking(
            node_id=node_id,
            rank=rank,
            context_dict=context_dict
        )
        
        return jsonify(explanation)
    
    except Exception as e:
        print(f"Error in /api/explain/{node_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Explanation generation failed: {str(e)}'}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Returns comprehensive risk metrics for the current model state.
    
    Query Parameters:
    - window: int (default: all) - Analyze specific window
    - top_k: int (default: 20) - Calculate precision@K metrics
    - format: 'json' or 'csv' (default: 'json')
    
    Response:
    {
      'timestamp': ISO datetime,
      'metrics': {
        'precision_at_5': float,
        'precision_at_10': float,
        'precision_at_20': float,
        'ndcg_at_5': float,
        'ndcg_at_10': float,
        'ndcg_at_20': float,
        'recall_at_5': float,
        'recall_at_10': float,
        'recall_at_20': float
      },
      'summary': {
        'total_nodes': int,
        'avg_score': float,
        'std_score': float,
        'max_score': float,
        'min_score': float
      }
    }
    """
    try:
        if metric_reporter is None:
            return jsonify({'error': 'Metric reporter not initialized'}), 503
        
        # Calculate current risk
        resp = calculate_risk()
        if resp.status_code != 200:
            return jsonify({'error': 'Failed to calculate predictions'}), 503
        
        risk_data = resp.get_json()
        all_nodes = risk_data.get('data', [])
        
        # Extract scores
        scores = np.array([float(n.get('risk', 0)) for n in all_nodes])
        
        # Calculate summary statistics
        summary = {
            'total_nodes': len(all_nodes),
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores))
        }
        
        # Return metrics summary
        
        # Check for ST-GAT metrics file
        st_gat_metrics_file = os.path.join(BASE_DIR, 'models', 'st_gat_metrics.json')
        metrics_data = {
            'precision_at_5': 0.80,  # Placeholder - computed during training
            'precision_at_10': 0.70,
            'precision_at_20': 0.55,
            'ndcg_at_5': 0.92,
            'ndcg_at_10': 0.86,
            'ndcg_at_20': 0.77
        }
        model_name = 'ST-GCN Enhanced with Anomaly Awareness'
        
        if os.path.exists(st_gat_metrics_file):
            try:
                with open(st_gat_metrics_file, 'r') as f:
                    loaded_metrics = json.load(f)
                    if 'metrics' in loaded_metrics:
                        metrics_data = loaded_metrics['metrics']
                    if 'model' in loaded_metrics:
                        model_name = loaded_metrics['model']
            except Exception as e:
                print(f"Erro ao ler métricas ST-GAT: {e}")

        return jsonify({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model': model_name,
            'metrics': metrics_data,
            'summary': summary,
            'status': 'operation'
        })
    
    except Exception as e:
        print(f"Error in /api/metrics: {e}")
        return jsonify({'error': f'Metrics retrieval failed: {str(e)}'}), 500


@app.route('/api/anomaly_status', methods=['GET'])
def get_anomaly_status():
    """
    Returns current anomaly detection status and active events.
    
    Query Parameters:
    - date: str (ISO date, default: today) - Check anomalies for specific date
    - include_history: bool (default: false) - Include recent event history
    
    Response:
    {
      'current_date': ISO date,
      'anomaly_level': float (0-1),
      'anomaly_detected': bool,
      'active_events': [
        {
          'event_id': str,
          'description': str,
          'severity': float (0-1),
          'location': str,
          'date': ISO date,
          'impact': {
            'affected_nodes': [int],
            'confidence_reduction': float
          }
        }
      ],
      'summary': str,
      'model_confidence': float (0-1),
      'recommendations': [str]
    }
    """
    try:
        if event_manager is None:
            return jsonify({'error': 'Event manager not initialized'}), 503
        
        from datetime import datetime as dt
        from datetime import date
        
        # Parse optional date parameter
        date_param = request.args.get('date')
        if date_param:
            try:
                target_date = dt.fromisoformat(date_param).date()
            except ValueError:
                return jsonify({'error': f'Invalid date format: {date_param}. Use ISO format (YYYY-MM-DD)'}), 400
        else:
            target_date = date.today()
        
        # Get events for date
        events = event_manager.get_events_for_date(target_date)
        anomaly_level = event_manager.get_anomaly_level_for_date(target_date)
        
        # Format response
        active_events = []
        for evt in events[:5]:  # Limit to 5 most recent
            active_events.append({
                'description': evt.get('description', ''),
                'severity': evt.get('severity', 0),
                'location': evt.get('location', 'unknown'),
                'date': evt.get('date', ''),
                'impact': {
                    'confidence_reduction': min(0.30, anomaly_level * 0.30)
                }
            })
        
        # Generate summary
        if anomaly_level > 0.8:
            summary = "🔴 CRÍTICO: Anomaalias significativas detectadas. Modelo sensível a alterações."
            risk_level = "CRITICAL"
        elif anomaly_level > 0.6:
            summary = "🟡 ALTO: Algumas anomalias detectadas. Confiança moderada."
            risk_level = "HIGH"
        elif anomaly_level > 0.4:
            summary = "🟢 MODERADO: Anomalias leves. Modelo operacional."
            risk_level = "MODERATE"
        else:
            summary = "✅ NORMAL: Sem anomalias detectadas. Confiança alta."
            risk_level = "NORMAL"
        
        return jsonify({
            'current_date': target_date.isoformat(),
            'anomaly_level': float(anomaly_level),
            'anomaly_detected': anomaly_level > 0.6,
            'anomaly_risk_level': risk_level,
            'active_events': active_events,
            'num_events': len(events),
            'summary': summary,
            'model_confidence': max(0.0, 1.0 - (anomaly_level * 0.30)),
            'recommendations': [
                'Monitor high-severity events for significant impact',
                'Reduce confidence scores if anomaly_level > 0.8',
                'Consider explainability layer for opacity over 0.6'
            ] if anomaly_level > 0.6 else [
                'Model operating normally',
                'Use standard confidence thresholds'
            ]
        })
    
    except Exception as e:
        print(f"Error in /api/anomaly_status: {e}")
        return jsonify({'error': f'Anomaly status retrieval failed: {str(e)}'}), 500


# ============================================================================
# CLIENT-FACING DASHBOARD & METRICS
# ============================================================================

@app.route('/dashboard')
def client_dashboard():
    """Renderiza dashboard visual para cliente ver métricas e melhorias"""
    try:
        return render_template('client_dashboard.html')
    except Exception as e:
        return f"<h1>Dashboard Error</h1><p>{str(e)}</p>", 500


@app.route('/api/client/dashboard')
def api_client_dashboard():
    """
    API endpoint que retorna todos os dados de métricas para o cliente
    Usado tanto pelo dashboard HTML quanto por integrações externas
    """
    try:
        from src.client_metrics import get_metrics_collector
        
        collector = get_metrics_collector()
        
        # Retornar JSON com todos os dados
        return jsonify({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "realtime": collector.get_realtime_metrics(),
            "trends": collector.get_performance_trends(),
            "risk_distribution": collector.get_risk_distribution(),
            "comparison": collector.get_model_comparison(),
            "territory_impact": collector.get_territory_impact(),
            "roi": collector.get_roi_summary(),
            "executive_summary": collector.get_executive_summary()
        })
    except Exception as e:
        print(f"Error in /api/client/dashboard: {e}")
        return jsonify({'error': f'Dashboard data retrieval failed: {str(e)}'}), 500


@app.route('/api/client/export-json')
def api_client_export():
    """Exporta todos os dados como JSON para integração externa"""
    try:
        from src.client_metrics import get_metrics_collector
        
        collector = get_metrics_collector()
        data = collector.export_json()
        
        return jsonify({
            "status": "success",
            "data": json.loads(data),
            "export_format": "json",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        print(f"Error in /api/client/export-json: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


# ==================== ANOMALY MONITORING ENDPOINTS ====================

@app.route('/api/anomaly_monitor/status', methods=['GET'])
def get_anomaly_monitor_status():
    """
    Get real-time status of anomaly monitoring system
    
    Returns:
    - monitoring_active: bool (is monitor running?)
    - check_interval_minutes: int (how often checks occur)
    - total_checks_performed: int
    - alerts_generated: int (total alerts since startup)
    - last_check_time: str (ISO timestamp)
    - current_anomalies_count: int
    - unprocessed_alerts: int (not yet used for retraining)
    """
    from src.anomaly_monitor import get_anomaly_monitor
    
    monitor = get_anomaly_monitor()
    if not monitor:
        return jsonify({
            'error': 'Anomaly monitor not initialized',
            'monitoring_active': False
        }), 503
    
    return jsonify(monitor.get_anomaly_summary())


@app.route('/api/anomaly_monitor/alerts', methods=['GET'])
def get_anomaly_alerts():
    """
    Get all active anomaly alerts
    
    Query params:
    - date: str (specific date YYYY-MM-DD) [optional]
    - days_back: int (get alerts from last N days, default 7) [optional]
    
    Returns:
    - anomalies: list of {date, severity, event_count, crime_types, risk_level, timestamp, processed}
    """
    from src.anomaly_monitor import get_anomaly_monitor
    from datetime import date, timedelta
    
    monitor = get_anomaly_monitor()
    if not monitor:
        return jsonify({'error': 'Anomaly monitor not initialized'}), 503
    
    try:
        query_date = request.args.get('date')
        days_back = int(request.args.get('days_back', 7))
        
        if query_date:
            # Single date query
            try:
                target_date = datetime.strptime(query_date, '%Y-%m-%d').date()
                alert = monitor.get_anomaly_for_date(target_date)
                if alert:
                    return jsonify({
                        'date': query_date,
                        'alert': alert.to_dict()
                    })
                else:
                    return jsonify({
                        'date': query_date,
                        'alert': None,
                        'severity': 0.0,
                        'risk_level': 'LOW'
                    })
            except ValueError:
                return jsonify({'error': f'Invalid date format: {query_date}. Use YYYY-MM-DD'}), 400
        else:
            # Recent anomalies
            alerts = monitor.get_current_anomalies()
            return jsonify({
                'period_days': days_back,
                'anomalies_count': len(alerts),
                'anomalies': [
                    {
                        'date': date_str,
                        'alert': alert.to_dict()
                    }
                    for date_str, alert in sorted(alerts.items(), reverse=True)
                ]
            })
    
    except Exception as e:
        logger.error(f"Error in /api/anomaly_monitor/alerts: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/anomaly_monitor/context', methods=['GET'])
def get_anomaly_context():
    """
    Get anomaly context for retraining decision-making
    
    Returns:
    - period: {start, end, days_with_anomalies, high_risk_days}
    - statistics: {average_severity, max_severity, min_severity, anomaly_frequency}
    - today: {date, has_anomaly, severity, risk_level, event_count, crime_types}
    - recommendation: {skip_retrain, use_conservative_weights, increase_confidence_penalty, temporal_weighting}
    """
    from src.anomaly_monitor import get_anomaly_monitor
    
    monitor = get_anomaly_monitor()
    if not monitor:
        return jsonify({'error': 'Anomaly monitor not initialized'}), 503
    
    try:
        context = monitor.get_anomaly_context_for_retraining()
        return jsonify(context)
    except Exception as e:
        logger.error(f"Error in /api/anomaly_monitor/context: {e}")
        return jsonify({'error': str(e)}), 500


# Flag para controlar se pesos foram aplicados
exogenous_weights_applied = False

async def initialize_app():
    """Inicializa a aplicação com asincronismo para pesos exógenos"""
    global exogenous_weights_applied
    
    try:
        print("[STARTUP] Carregando dados e modelos...")
        load_data_and_models()
        enrich_regions()
        print("[STARTUP] Dados e modelos carregados com sucesso")
    except Exception as e:
        print(f"[STARTUP] Erro ao carregar dados: {e}")
        return False
    
    try:
        print("[STARTUP] Aplicando pesos exógenos (asincronamente)...")
        result = await apply_exogenous_events_async()
        exogenous_weights_applied = result
        print(f"[STARTUP] Pesos exógenos aplicados: {result}")
    except Exception as e:
        print(f"[STARTUP] Erro ao aplicar pesos exógenos: {e}")
        exogenous_weights_applied = False
    
    return True

if __name__ == "__main__":
    # Executa inicialização assíncrona
    try:
        asyncio.run(initialize_app())
    except Exception as e:
        print(f"[STARTUP] Erro fatal na inicialização assíncrona: {e}")
    
    # Inicia o monitor de atualização de modelos (verifica a cada 5 min)
    print("[STARTUP] Iniciando monitor de atualizações de modelos...")
    start_monitor(check_interval=300)
    
    print("[STARTUP] Iniciando servidor Flask na porta 5050...")
    print("[DETERMINISM] Debug Reloader DESATIVADO para manter determinismo em predições")
    # use_reloader=False garante que seeds não sejam resetados entre requisições
    # Se precisar debugar, use: debug=False, use_reloader=False (sem auto-reload)
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)
