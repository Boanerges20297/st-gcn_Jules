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

def normalize_name(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
    text = re.sub(r'\s*-\s*AIS.*$', '', text)
    return text.strip()

class StateOrchestrator:
    def __init__(self, project_root):
        self.root = project_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ATUALIZAÇÃO: Roteamento dinâmico para suportar a arquitetura com 33 canais Multi-Scale Momentum + Cold Streak
        fortaleza_model_file = 'fortaleza_super_elite.pth' if os.path.exists(os.path.join(self.root, 'models', 'active', 'fortaleza_super_elite.pth')) else 'fortaleza_model_active.pth'
        retrain_file = 'fortaleza_retrain_64.pth'
        active_file = 'fortaleza_model_active.pth'
        
        if os.path.exists(os.path.join(self.root, 'models', 'active', retrain_file)):
            fortaleza_model_file = retrain_file
            has_momentum = True
        else:
            fortaleza_model_file = active_file
            has_momentum = False
        
        self.configs = {
            'fortaleza': {
                'model_path': os.path.join(self.root, 'models', 'active', fortaleza_model_file),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_fortaleza.pkl'),
                'class': DeepSTGAT_64,
                'in_channels': 33 if has_momentum else 29, 
                'window': 120 if has_momentum else 90 
            },
            'rmf': {
                'model_path': os.path.join(self.root, 'models', 'active', 'rmf_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_rmf.pkl'),
                'class': DeepSTGAT_64,
                'in_channels': 29,
                'window': 90
            },
            'interior': {
                'model_path': os.path.join(self.root, 'models', 'active', 'interior_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_interior.pkl'),
                'class': DeepSTGAT_64,
                'in_channels': 29,
                'window': 90
            }
        }
        
        self.specialists = {}
        self.calib_params = {
            reg: {'tension_factor': 0.50, 'min_risk': 30.0, 'tag_bias_direct': 1.50, 'tag_bias_neighbor': 0.50, 'dynamic_window': None}
            for reg in ['fortaleza', 'rmf', 'interior']
        }
        
        self._initialize_models()

    def adjust_temporal_focus(self, region, efficiency_score):
        """
        Auto-Ajuste de Janela (Temporal Shrinkage) baseado no feedback do Monitor.
        Reduz a janela gradativamente se a eficiência cair, cortando ruído histórico.
        """
        if region not in self.specialists: return
        
        cp = self.calib_params.setdefault(region, self.calib_params.get('fortaleza', {}).copy())
        base_window = self.specialists[region]['window']
        current_window = cp.get('dynamic_window') or base_window
        
        if efficiency_score < 0.50:
            # Encolhe a janela em blocos de 30 dias até o mínimo de 30 dias
            new_window = max(30, current_window - 30)
            if new_window != current_window:
                print(f"📉 [Auto-Tune] Eficiência baixa ({efficiency_score*100:.1f}%) em {region.upper()}. Reduzindo janela de {current_window}d para {new_window}d.")
                cp['dynamic_window'] = new_window
        elif efficiency_score >= 0.50:
            # Expande novamente se a performance se consolidar alta
            new_window = min(base_window, current_window + 30)
            if new_window != current_window:
                print(f"📈 [Auto-Tune] Eficiência excelente ({efficiency_score*100:.1f}%) em {region.upper()}. Expandindo janela para {new_window}d.")
                cp['dynamic_window'] = new_window

    def _initialize_models(self):
        for region, cfg in self.configs.items():
            if os.path.exists(cfg['model_path']) and os.path.exists(cfg['data_path']):
                try:
                    data = self._load_pickle_safe(cfg['data_path'])
                    if not data or 'nodes_gdf' not in data: continue

                    num_nodes = len(data['nodes_gdf'])
                    model = cfg['class'](num_nodes=num_nodes, in_channels=cfg['in_channels'], time_steps=cfg['window']).to(self.device)
                    ckpt = torch.load(cfg['model_path'], map_location=self.device, weights_only=False)
                    
                    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    
                    self.specialists[region] = {'model': model, 'data': data, 'window': cfg['window'], 'channels': cfg['in_channels']}
                    if not hasattr(self, 'dates') and 'dates' in data:
                        self.dates = data['dates']
                        
                    print(f"✅ Orquestrador: Especialista {region.upper()} ({num_nodes} nós) carregado com {cfg['class'].__name__} ({cfg['in_channels']} Canais).")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"❌ Erro ao carregar {region}: {e}")

    def _load_pickle_safe(self, path):
        try:
            return pd.read_pickle(path)
        except Exception as e:
            print(f"❌ Erro crítico ao ler {path}: {e}")
            return None

    def get_combined_risk(self, exogenous_shocks=None, return_trends=False):
        combined_scores = {}
        trends = {}
        
        for region, spec in self.specialists.items():
            model, data, window = spec['model'], spec['data'], spec['window']
            channels = spec.get('channels', 29)
            cp = self.calib_params.get(region, self.calib_params['fortaleza'])
            num_nodes = len(data['nodes_gdf'])
            
            # --- ATUALIZAÇÃO: CÁLCULO DINÂMICO DE MOMENTUM MULTI-SCALE ---
            # Se a rede exigir 32 canais, recuamos 60 dias extras no passado para a base de cálculo (Janela macro de 30 dias)
            extra_history = 60 if channels == 32 else 0
            total_window = window + extra_history
            
            # Prevenção: garantir que a janela total não seja maior que o histórico de dados disponível
            total_window = min(total_window, data['node_features'].shape[1])
            
            x_raw_extended = data['node_features'][:, -total_window:, :].copy()
            sim_impact = np.zeros(num_nodes)
            
            if exogenous_shocks:
                for loc_name, info in exogenous_shocks.items():
                    norm_target = normalize_name(loc_name)
                    if isinstance(info, dict):
                        intensity = float(info.get('conflict_intensity', info.get('intensity', 0.0)))
                        impact_value = (3.0 if region == 'rmf' else 1.8) * intensity
                        channel_idx = 25 if info.get('is_critical') else 24
                        for i, row in data['nodes_gdf'].iterrows():
                            if normalize_name(row['name']) == norm_target:
                                x_raw_extended[i, :, channel_idx] = min(intensity, 3.0)
                                sim_impact[i] = impact_value

            # INJEÇÃO DOS CANAIS DE MOMENTUM (ESCALA MÚLTIPLA E FRIO)
            if channels >= 32:
                momentum_feat = np.zeros((num_nodes, total_window, channels - 29))
                cold_streak = np.zeros(num_nodes)
                
                for t in range(60, total_window):
                    # Escala 1 (7 dias - Micro conflito)
                    recent_7 = x_raw_extended[:, t-7:t, 0].sum(axis=1)
                    past_7 = x_raw_extended[:, t-14:t-7, 0].sum(axis=1)
                    momentum_feat[:, t, 0] = recent_7 - past_7
                    
                    # Escala 2 (14 dias - Meso conflito)
                    recent_14 = x_raw_extended[:, t-14:t, 0].sum(axis=1)
                    past_14 = x_raw_extended[:, t-28:t-14, 0].sum(axis=1)
                    momentum_feat[:, t, 1] = recent_14 - past_14
                    
                    # Escala 3 (30 dias - Macro tendência)
                    recent_30 = x_raw_extended[:, t-30:t, 0].sum(axis=1)
                    past_30 = x_raw_extended[:, t-60:t-30, 0].sum(axis=1)
                    momentum_feat[:, t, 2] = recent_30 - past_30
                    
                    # Escala Fria (33º Canal: Cold Streak) - Se existir na arquitetura
                    if channels == 33:
                        crimes_today = x_raw_extended[:, t, 0]
                        cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)
                        momentum_feat[:, t, 3] = np.clip(cold_streak, 0, 30)
                
                # Anexa os novos canais ao tensor original de 29 canais
                x_raw_extended = np.concatenate([x_raw_extended, momentum_feat], axis=2)
                
                # Normalização adaptativa para cada canal novo
                for c_idx in range(29, channels):
                    m_mean = x_raw_extended[:, :, c_idx].mean()
                    m_std = x_raw_extended[:, :, c_idx].std() + 1e-6
                    x_raw_extended[:, :, c_idx] = (x_raw_extended[:, :, c_idx] - m_mean) / m_std

            # Recorta a janela exata esperada pela rede neural (os 90 ou 120 dias finais após cálculo)
            x_final = x_raw_extended[:, -window:, :].copy()
            
            # --- ATUALIZAÇÃO: TEMPORAL SHRINKAGE (MÁSCARA DE ATENÇÃO DINÂMICA) ---
            # Se o Monitor de Eficiência reduziu a janela (ex: de 120 para 60), 
            # nós zeramos o passado distante no tensor para forçar a rede a focar apenas no presente,
            # sem quebrar a dimensão exigida pela camada de convolução neural.
            active_window = cp.get('dynamic_window', window)
            if active_window and active_window < window:
                x_final[:, :window - active_window, :] = 0.0

            x = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            adj = self._norm_adj(data['adj_geo'], data['adj_conflict'])
            
            with torch.no_grad():
                out = model(x, adj).squeeze().cpu().numpy()
            
            # Normalização Sigmoidal
            final_logits = out + sim_impact + (data['nodes_gdf']['tension_index'].values.astype(float) * 0.5)
            s = 1 / (1 + np.exp(-0.7 * (final_logits - (-1.0))))
            out_norm = np.clip(s * 100, 5.0, 100.0)
            
            for i, row in data['nodes_gdf'].iterrows():
                name_key = normalize_name(row['name'])
                combined_scores[name_key] = float(out_norm[i])
                if return_trends: trends[name_key] = 'stable'
                
        return (combined_scores, trends) if return_trends else combined_scores

    def _norm_adj(self, geo, conf):
        def n(a):
            s = np.array(a.sum(1)); d = np.power(s, -0.5).flatten(); d[np.isinf(d)]=0.; m=np.diag(d)
            return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(self.device)
        return [n(geo), n(conf)]
