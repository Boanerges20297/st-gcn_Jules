import sys
import numpy as np

import torch
import pickle
import os
import pandas as pd
import unicodedata
import re
from datetime import datetime

# ============================================================================
# ARQUITETURA REGIONAL ST-GAT - ORQUESTRADOR DE ELITE
# ============================================================================

try:
    from .architectures import DeepSTGAT_64, DeepSTGAT_32
except ImportError:
    from architectures import DeepSTGAT_64, DeepSTGAT_32

# --- DEBUG: FUNÇÃO DE NORMALIZAÇÃO DE NOMES ---
def normalize_name(text):
    """
    Remove acentos e sufixos de AIS (regex) para garantir match perfeito 
    entre o GeoJSON do Leaflet e os nomes no modelo.
    """
    if not isinstance(text, str): return ""
    import re
    # 1. Decodifica acentos (ex: 'Ã' vira 'A')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
    # 2. Deleta sufixos operacionais (ex: ' - AIS 12, 26')
    text = re.sub(r'\s*-\s*AIS.*$', '', text)
    return text.strip()

class StateOrchestrator:
    """
    CÉREBRO CENTRAL: Roteia requisições para os 3 especialistas regionalizados.
    Lida com janelas de tempo diferentes (30 vs 45 dias) e arquiteturas polimórficas.
    """
    def __init__(self, project_root):
        self.root = project_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- CONFIGURACAO CONSOLIDADA (ISM Production Standard) ---
        self.configs = {
            'fortaleza': {
                'model_path': os.path.join(self.root, 'models', 'active', 'fortaleza_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_fortaleza.pkl'),
                'class': DeepSTGAT_64,
                'window': 120
            },
            'rmf': {
                'model_path': os.path.join(self.root, 'models', 'active', 'rmf_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_rmf.pkl'),
                'class': DeepSTGAT_64,
                'window': 120
            },
            'interior': {
                'model_path': os.path.join(self.root, 'models', 'active', 'interior_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_interior.pkl'),
                'class': DeepSTGAT_64,
                'window': 120
            }
        }
        
        self.specialists = {}
        
        # Parâmetros de calibração por região (ajustáveis em runtime)
        # dampening removido: comprimir outliers é irracional para tensão territorial
        self.calib_params = {
            region: {
                'tension_factor':     0.50,   # peso do tension_index no logit
                'min_risk':          30.0,    # piso apenas para territórios com CVLI recente
                'tag_bias_direct':    1.50,   # boost INTEL_TRIGGER no nó
                'tag_bias_neighbor':  0.50,   # vazamento de tensão para vizinhos
            }
            for region in ['fortaleza', 'rmf', 'interior']
        }
        
        self._initialize_models()

    # --- DEBUG: CARREGAMENTO DE CHECKPOINTS ---
    def _initialize_models(self):
        """Inicializa os modelos regionais apenas se os arquivos existirem."""
        for region, cfg in self.configs.items():
            if os.path.exists(cfg['model_path']) and os.path.exists(cfg['data_path']):
                try:
                    with open(cfg['data_path'], 'rb') as f:
                        data = pickle.load(f)
                    
                    # Cria instância da classe correta (64 ou 32 canais)
                    model = cfg['class'](num_nodes=len(data['nodes_gdf']), in_channels=29, time_steps=cfg['window']).to(self.device)
                    ckpt = torch.load(cfg['model_path'], map_location=self.device, weights_only=False)
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.eval()
                    
                    self.specialists[region] = {'model': model, 'data': data, 'window': cfg['window']}
                    
                    # Salva as datas do primeiro especialista para metadados globais
                    if not hasattr(self, 'dates'):
                        self.dates = data['dates']
                        
                    print(f"✅ Orquestrador: Especialista {region.upper()} carregado com sucesso.")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"❌ Erro ao carregar {region}: {e}")

    # --- DEBUG: MOTOR DE INFERÊNCIA COMBINADA ---
    def get_combined_risk(self, exogenous_shocks=None, return_trends=False):
        """Calcula o risco para as 299 localidades em uma única chamada."""
        combined_scores = {}
        trends = {}
        
        for region, spec in self.specialists.items():
            model, data, window = spec['model'], spec['data'], spec['window']
            cp = self.calib_params.get(region, self.calib_params['fortaleza'])
            
            # Prepara tensor de entrada (Últimos dias da base)
            x_raw = data['node_features'][:, -window:, :].copy()
            
            # --- NOVO: CÁLCULO DE TAG-BIAS (Trigger Alert Graph Bias) ---
            # Identifica gatilhos (L.B. em Facção) nos últimos 3 dias
            spatial_bias = self._compute_spatial_bias(x_raw, data['adj_geo'])
            
            # Cálculo de Tendência Real (Se solicitado)
            if return_trends:
                # CVLI é o canal 0. Comparamos última semana vs penúltima
                last_7 = data['node_features'][:, -7:, 0].sum(axis=1)
                prev_7 = data['node_features'][:, -14:-7, 0].sum(axis=1)
                for i, row in data['nodes_gdf'].iterrows():
                    name_norm = normalize_name(row['name'])
                    diff = last_7[i] - prev_7[i]
                    # Valor da tendência: positive (subindo), negative (descendo), neutral (igual)
                    if diff > 0: trends[name_norm] = 'up'
                    elif diff < 0: trends[name_norm] = 'down'
                    else: trends[name_norm] = 'stable'

            # --- DEBUG: INJEÇÃO DE EVENTOS DINÂMICOS (Canais 23, 24 e 25) ---
            sim_impact = np.zeros(len(data['nodes_gdf']))
            if exogenous_shocks:
                # Esperamos exogenous_shocks = {'NOME': {'intensity': 1.0, 'is_critical': True, 'is_suppression': False}}
                for loc_name, info in exogenous_shocks.items():
                    norm_target = normalize_name(loc_name)
                    if isinstance(info, dict):
                        conf_int = float(info.get('conflict_intensity', 0.0))
                        supp_int = float(info.get('suppression_intensity', 0.0))
                        is_crit = info.get('is_critical', False)

                        # Fallback to old format
                        if 'intensity' in info:
                            if info.get('is_suppression', False):
                                supp_int += float(info.get('intensity', 0.0))
                            else:
                                conf_int += float(info.get('intensity', 0.0))

                        # Limitar intensidade individual máxima para não explodir
                        conf_int = min(conf_int, 3.0)
                        supp_int = min(supp_int, 3.0)

                        # Calcula impacto líquido: Conflito aumenta (+), Supressão reduz (-)
                        impact_value = (2.5 * conf_int) - (2.5 * supp_int)

                        # Escolher um canal predominante apenas para fins visuais no logit (opcional)
                        channel_idx = 25 if is_crit else (24 if conf_int >= supp_int else 23)
                        intensity = max(conf_int, supp_int)
                    else:
                        intensity = float(info)
                        is_crit = (intensity >= 0.8)
                        channel_idx = 25 if is_crit else 24
                        impact_value = 2.5 * intensity

                    for i, row in data['nodes_gdf'].iterrows():
                        if normalize_name(row['name']) == norm_target:
                            x_raw[i, :, channel_idx] = intensity
                            sim_impact[i] = impact_value
            x = torch.from_numpy(x_raw).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            adj = self._norm_adj(data['adj_geo'], data['adj_conflict'])
            
            with torch.no_grad():
                # Inferência Pura (Sem bias interno)
                out = model(x, adj).squeeze().cpu().numpy()
            
            # --- CORREÇÃO: NORMALIZAÇÃO ROBUSTA E AMORTECIMENTO (DAMPENING) ---
            # --- NORMALIZAÇÃO REFINADA (SENSÍVEL A CONSISTÊNCIA) ---
            if len(out) > 1:
                # INJEÇÃO PÓS-MODELO (ADITIVA) DO TAG-BIAS E SIMULAÇÃO
                # O bias agora é somado diretamente aos logits, garantindo impacto linear
                if spatial_bias is not None:
                     out = out + spatial_bias
                
                # Somar Impacto Direto da Simulação (Equipes/Conflitos)
                out = out + sim_impact

                # --- PESO DINÂMICO DE FACÇÃO (context-sensitive) ---
                # Princípio: território de facção sem conflito ativo = domínio consolidado,
                # não criticidade extrema. O peso sobe apenas quando há evidência real:
                #   - CVLI recente (14d): atividade violenta confirmada
                #   - Rival adjacente:    risco latente de disputa territorial
                recent_14d_cvli = data['node_features'][:, -14:, 0].sum(axis=1)
                rival_adj = ((data['adj_conflict'] - np.eye(len(data['nodes_gdf']))).sum(axis=1) > 0)
                has_cvli   = recent_14d_cvli > 0

                # Fatores: calmo=0.10 | rival sem CVLI=0.30 | CVLI sem rival=0.50 | CVLI+rival=0.70
                dynamic_tension_factor = np.where(
                    has_cvli,
                    np.where(rival_adj, 0.70, cp['tension_factor']),   # 0.70 ou 0.50
                    np.where(rival_adj, 0.30, 0.10)                    # 0.30 ou 0.10
                )
                tension_weight = data['nodes_gdf']['tension_index'].values.astype(float) * dynamic_tension_factor

                # Logit Final: Modelo + Tags Intel + Simulação + Facções (dinâmico)
                final_logits = out + spatial_bias + sim_impact + tension_weight

                # Mapeamento Sigmoidal com Âncoras Fixas:
                # Pivot -1.0: Define o ponto de transição para risco Moderado.
                # Scale 0.7: Define a inclinação da curva (sensibilidade ao aumento de tensão).
                pivot = -1.0
                sensitivity = 0.7

                s = 1 / (1 + np.exp(-sensitivity * (final_logits - pivot)))
                out_norm = s * 100

                # Piso mínimo: apenas se houver CVLI recente (não por mera filiação a facção)
                # Facção calma = domínio territorial; não justifica floor artificial.
                recent_cvli = data['node_features'][:, -7:, 0].sum(axis=1)
                for i in range(len(out_norm)):
                    if recent_cvli[i] > 0 and sim_impact[i] >= 0:
                        out_norm[i] = max(out_norm[i], cp['min_risk'])
                
                # Clipping final de segurança
                out_norm = np.clip(out_norm, 5.0, 100.0)
                
                # DAMPENING REMOVIDO: comprimir outliers altos é irracional para tensão territorial.
                # Um município isolado com domínio total de facção pode ter score 95%+ legitimamente.
            else:
                out_norm = np.zeros_like(out) + 30.0
            
            # Mapeia para o dicionário global pelo NOME NORMALIZADO
            for i, row in data['nodes_gdf'].iterrows():
                name_key = normalize_name(row['name'])
                # Fusão: Se já existir (fronteira), tira a média (Mode-like)
                if name_key in combined_scores:
                    combined_scores[name_key] = (combined_scores[name_key] + float(out_norm[i])) / 2
                else:
                    combined_scores[name_key] = float(out_norm[i])
                
        if return_trends:
            return combined_scores, trends
        return combined_scores

    def _compute_spatial_bias(self, x_raw, adj_geo):
        """
        Calcula o TAG-Bias (Trigger Alert Graph Bias) como um VETOR ADITIVO.
        Retorna: array de shape (N,) com valores a serem somados aos logits.
        """
        num_nodes = x_raw.shape[0]
        bias_vector = np.zeros(num_nodes)
        
        # Canal 27: INTEL_TRIGGER (LB, Disparos) | Canal 2: TENSION (Facções)
        # Analisamos os últimos 7 dias (semana ativa de risco) para disparar o alerta
        recent_intel = x_raw[:, -7:, 27].sum(axis=1)
        tension = x_raw[:, -1, 2]
        
        for i in range(num_nodes):
            if recent_intel[i] > 0 and tension[i] > 0.5:
                # GATILHO: Incidente crítico em zona de facção
                # Boost Direto no Logit (+1.5 move 50% -> ~80%)
                bias_vector[i] += cp['tag_bias_direct']
                
                # Vazamento para vizinhos
                neighbors = np.where(adj_geo[i] > 0)[0]
                bias_vector[neighbors] += cp['tag_bias_neighbor']
                    
        return bias_vector

    # --- DEBUG: NORMALIZAÇÃO DE MATRIZES DE ADJACÊNCIA ---
    def _norm_adj(self, geo, conf):
        """Prepara os grafos para o processamento via GAT."""
        def n(a):
            s = np.array(a.sum(1)); d = np.power(s, -0.5).flatten(); d[np.isinf(d)]=0.; m=np.diag(d)
            return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(self.device)
        return [n(geo), n(conf)]

# --- PONTE DE COMPATIBILIDADE PARA O DASHBOARD ---
class Phase5Bridge:
    """Interface legada para o app.py (Mapeamento Global Leaflet)."""
    def __init__(self, project_root):
        self.orchestrator = StateOrchestrator(project_root)
        global_path = os.path.join(project_root, 'data', 'processed', 'processed_graph_data_global.pkl')
        with open(global_path, 'rb') as f:
            self.global_data = pickle.load(f)
        self.dates = getattr(self.orchestrator, 'dates', None)

    def get_risk_scores(self, exogenous_shocks=None):
        """Roteia o risco e mapeia de volta para o array global do Dashboard."""
        scores_map = self.orchestrator.get_combined_risk(exogenous_shocks)
        nodes_gdf = self.global_data['nodes_gdf']
        risk_array = np.zeros(len(nodes_gdf))
        
        for i, row in nodes_gdf.iterrows():
            name_norm = normalize_name(row['name'])
            # Se o local for desconhecido, risco base de 20%
            risk_array[i] = scores_map.get(name_norm, 20.0) 
            
        return risk_array, risk_array
