import torch
import pickle
import numpy as np
import os
import sys
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
        
        # --- CONFIGURACAO CONSOLIDADA (Phase 5/6/7 Active) ---
        self.configs = {
            'fortaleza': {
                'model_path': os.path.join(self.root, 'models', 'active', 'fortaleza_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_fortaleza.pkl'),
                'class': DeepSTGAT_64,
                'window': 120
            },
            'rmf': {
                'model_path': os.path.join(self.root, 'models', 'active', 'rmf_model_elite.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_rmf.pkl'),
                'class': DeepSTGAT_32,
                'window': 30
            },
            'interior': {
                'model_path': os.path.join(self.root, 'models', 'active', 'interior_model_elite.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_interior.pkl'),
                'class': DeepSTGAT_32,
                'window': 30
            }
        }
        
        self.specialists = {}
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
                    ckpt = torch.load(cfg['model_path'], map_location=self.device)
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.eval()
                    
                    self.specialists[region] = {'model': model, 'data': data, 'window': cfg['window']}
                    
                    # Salva as datas do primeiro especialista para metadados globais
                    if not hasattr(self, 'dates'):
                        self.dates = data['dates']
                        
                    print(f"✅ Orquestrador: Especialista {region.upper()} carregado com sucesso.")
                except Exception as e:
                    print(f"❌ Erro ao carregar {region}: {e}")

    # --- DEBUG: MOTOR DE INFERÊNCIA COMBINADA ---
    def get_combined_risk(self, exogenous_shocks=None, return_trends=False):
        """Calcula o risco para as 299 localidades em uma única chamada."""
        combined_scores = {}
        trends = {}
        
        for region, spec in self.specialists.items():
            model, data, window = spec['model'], spec['data'], spec['window']
            
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
            if exogenous_shocks:
                # Esperamos exogenous_shocks = {'NOME': {'intensity': 1.0, 'is_critical': True, 'is_suppression': False}}
                for loc_name, info in exogenous_shocks.items():
                    norm_target = normalize_name(loc_name)
                    if isinstance(info, dict):
                        intensity = float(info.get('intensity', 0.5))
                        is_crit = info.get('is_critical', False)
                        is_supp = info.get('is_suppression', False)
                    else:
                        intensity = float(info)
                        is_crit = (intensity >= 0.8)
                        is_supp = False
                    
                    # Definição de Canal:
                    # 25 = Crítico (Ameaça Alta)
                    # 24 = Padrão (Tensão/Evento Exógeno)
                    # 23 = Supressão (Ação Policial/Alívio)
                    if is_supp:
                        channel_idx = 23
                    else:
                        channel_idx = 25 if is_crit else 24
                    
                    for i, row in data['nodes_gdf'].iterrows():
                        if normalize_name(row['name']) == norm_target:
                            x_raw[i, :, channel_idx] = intensity

            x = torch.from_numpy(x_raw).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            adj = self._norm_adj(data['adj_geo'], data['adj_conflict'])
            
            with torch.no_grad():
                # Inferência Pura (Sem bias interno)
                out = model(x, adj).squeeze().cpu().numpy()
            
            # --- CORREÇÃO: NORMALIZAÇÃO ROBUSTA E AMORTECIMENTO (DAMPENING) ---
            # --- NORMALIZAÇÃO REFINADA (SENSÍVEL A CONSISTÊNCIA) ---
            if len(out) > 1:
                # INJEÇÃO PÓS-MODELO (ADITIVA) DO TAG-BIAS
                # O bias agora é somado diretamente aos logits, garantindo impacto linear
                if spatial_bias is not None:
                     out = out + spatial_bias

                # Em vez de Z-Score puro (que depende de vizinhos), usamos uma escala logística
                # Onde o valor 0.5 (50%) é ancorado em uma atividade moderada constante.
                # Isso impede que 9 mortes/mês virem "Risco 7".
                
                # Boost por Tensão Geográfica (Facções) direto no logit
                tension_weight = data['nodes_gdf']['tension_index'].values.astype(float) * 0.5
                adjusted_logits = out + tension_weight
                
                # Sigmoide de escala fixa (não relativa ao desvio padrão da região)
                # Isso torna o risco mais "Absoluto"
                s = 1 / (1 + np.exp(-(adjusted_logits - adjusted_logits.mean()) / (adjusted_logits.std() + 1e-6)))
                out_norm = s * 100
                
                # Garantia de Consistência: Se houve CVLI recente, o risco mínimo é 45%
                recent_cvli = data['node_features'][:, -7:, 0].sum(axis=1)
                for i in range(len(out_norm)):
                    if recent_cvli[i] > 0:
                        out_norm[i] = max(out_norm[i], 45.0)
                
                # Amortecimento para evitar 100% constante
                mask_high = out_norm > 50
                out_norm[mask_high] = 50 + (out_norm[mask_high] - 50) * 0.82
            else:
                out_norm = np.zeros_like(out) + 30.0
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
                bias_vector[i] += 1.5
                
                # Vazamento para vizinhos (+0.5 move 50% -> ~62%)
                neighbors = np.where(adj_geo[i] > 0)[0]
                bias_vector[neighbors] += 0.5
                    
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
