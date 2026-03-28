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
        
        # ⭐ ATUALIZAÇÃO (2026-03-18): Modelo oficial de Fortaleza agora é retreinado com Blindagem Temporal
        # Paradigma Tentativa 49: Gradiente Agressivo + Z-Score Local.
        fortaleza_model_file = 'fortaleza_model_active.pth'
        has_momentum_fortaleza = True

        interior_model_file   = 'interior_model.pth'
        interior_has_momentum = True
        
        self.configs = {
            'fortaleza': {
                'model_path': os.path.join(self.root, 'models', 'active', fortaleza_model_file),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_fortaleza.pkl'),
                'class': DeepSTGAT_64,
                'in_channels': 33 if has_momentum_fortaleza else 29, 
                'window': 120 if has_momentum_fortaleza else 90 
            },
            'rmf': {
                'model_path': os.path.join(self.root, 'models', 'active', 'rmf_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_rmf.pkl'),
                'class': DeepSTGAT_64,
                'in_channels': 29,
                'window': 90
            },
            'interior': {
                'model_path': os.path.join(self.root, 'models', 'active', interior_model_file),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_interior.pkl'),
                'class': DeepSTGAT_64,
                'in_channels': 33 if interior_has_momentum else 29,
                'window': 120 if interior_has_momentum else 90
            }
        }
        
        self.specialists = {}
        self.calib_params = {
            reg: {
                'tension_factor': 0.80, 'min_risk': 30.0,
                'tag_bias_direct': 2.00, 'tag_bias_neighbor': 0.60,
                'norm_neural_weight': 0.20, 'dynamic_window': None,
                'use_historical_fallback': False,
            }
            for reg in self.configs
        }
        
        self._window_state_path = os.path.join(self.root, 'data', 'window_state.json')
        self.dates = None
        self._initialize_models()
        self._restore_window_state()

    def _restore_window_state(self):
        try:
            if os.path.exists(self._window_state_path):
                import json
                with open(self._window_state_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                for region, state in saved.items():
                    if region in self.calib_params:
                        dw = state.get('dynamic_window')
                        hf = state.get('use_historical_fallback', False)
                        self.calib_params[region]['dynamic_window'] = dw
                        self.calib_params[region]['use_historical_fallback'] = hf
                        if hf:
                            self._load_historical_fallback(region)
        except Exception as e:
            print(f"⚠️ [Window State] Erro ao restaurar: {e}")

    def _save_window_state(self):
        try:
            import json
            state = {}
            for region, cp in self.calib_params.items():
                state[region] = {
                    'dynamic_window': cp.get('dynamic_window'),
                    'use_historical_fallback': cp.get('use_historical_fallback', False),
                    'historical_top10': cp.get('historical_top10', []),
                    'updated_at': datetime.now().isoformat(),
                }
            os.makedirs(os.path.dirname(self._window_state_path), exist_ok=True)
            with open(self._window_state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ [Window State] Erro ao salvar: {e}")

    _WINDOW_LADDER = [120, 90, 60, 30]

    def adjust_temporal_focus(self, region, efficiency_score):
        """
        Auto-Ajuste de Janela (Temporal Shrinkage) baseado no feedback do Monitor.
        Thresholds atualizados para Baseline Cego (Tentativa 49): 0.40 / 0.55
        """
        if region not in self.specialists: return

        cp = self.calib_params.setdefault(region, next(iter(self.calib_params.values()), {}).copy())
        base_window = self.specialists[region]['window']
        current_window = cp.get('dynamic_window') or base_window
        current_window = min(current_window, base_window)

        if efficiency_score < 0.40:
            ladder = [w for w in self._WINDOW_LADDER if w <= base_window]
            if not ladder: ladder = [30]
            current_rung = max((w for w in ladder if w <= current_window), default=ladder[0])
            current_idx = ladder.index(current_rung)

            if current_rung > ladder[-1]:
                next_rung = ladder[current_idx + 1] if current_idx + 1 < len(ladder) else ladder[-1]
                if next_rung != current_window:
                    print(f"📉 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Reduzindo janela {current_window}d → {next_rung}d.")
                    cp['dynamic_window'] = next_rung
                    cp['use_historical_fallback'] = False
                    self._save_window_state()
            else:
                if not cp.get('use_historical_fallback', False):
                    print(f"📉 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. ATIVANDO fallback histórico.")
                    cp['use_historical_fallback'] = True
                    self._load_historical_fallback(region)
                    self._save_window_state()

        elif efficiency_score >= 0.55:
            ladder = [w for w in self._WINDOW_LADDER if w <= base_window]
            if not ladder: ladder = [base_window]
            current_rung = min((w for w in ladder if w >= current_window), default=base_window)
            current_idx = ladder.index(current_rung)

            if current_rung < base_window:
                next_rung = ladder[current_idx - 1] if current_idx > 0 else base_window
                print(f"📈 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Expandindo janela {current_window}d → {next_rung}d.")
                cp['dynamic_window'] = next_rung
            else:
                cp['dynamic_window'] = None
                print(f"✅ [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Janela base restaurada.")

            if cp.get('use_historical_fallback'):
                cp['use_historical_fallback'] = False
            self._save_window_state()

    def _load_historical_fallback(self, region):
        if region not in self.specialists: return
        gdf = self.specialists[region]['data']['nodes_gdf']
        sort_col = 'total_cvli' if 'total_cvli' in gdf.columns else 'recent_cvli'
        ranked = gdf.sort_values(sort_col, ascending=False)
        self.calib_params[region]['historical_top10'] = list(ranked['name'].head(10))
        self.calib_params[region]['tag_bias_direct'] = 5.00
        self.calib_params[region]['tension_factor']  = 3.00

    def _initialize_models(self):
        for region, cfg in self.configs.items():
            if os.path.exists(cfg['model_path']) and os.path.exists(cfg['data_path']):
                try:
                    data = self._load_pickle_safe(cfg['data_path'])
                    if not data: continue
                    num_nodes = len(data['nodes_gdf'])
                    model = cfg['class'](num_nodes=num_nodes, in_channels=cfg['in_channels'], time_steps=cfg['window']).to(self.device)
                    ckpt = torch.load(cfg['model_path'], map_location=self.device, weights_only=False)
                    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.specialists[region] = {'model': model, 'data': data, 'window': cfg['window'], 'channels': cfg['in_channels']}
                    if self.dates is None:
                        self.dates = data.get('dates')
                    print(f"✅ Orquestrador: Especialista {region.upper()} carregado ({cfg['in_channels']} Canais).")
                except Exception as e:
                    print(f"❌ Erro ao carregar {region}: {e}")
        self._node_owners = {normalize_name(str(r['name'])): reg for reg, spec in self.specialists.items() for _, r in spec['data']['nodes_gdf'].iterrows()}

    def _load_pickle_safe(self, path):
        try: return pd.read_pickle(path)
        except: return None

    def get_combined_risk(self, exogenous_shocks=None, return_trends=False):
        combined_scores = {}
        trends = {}
        
        for region, spec in self.specialists.items():
            model, data, window = spec['model'], spec['data'], spec['window']
            channels = spec.get('channels', 29)
            cp = self.calib_params.get(region, next(iter(self.calib_params.values())))
            num_nodes = len(data['nodes_gdf'])
            
            extra_history = 60
            total_window = window + extra_history
            x_raw_extended = data['node_features'][:, -total_window:, :].copy()
            sim_impact, sim_relief = np.zeros(num_nodes), np.zeros(num_nodes)
            
            if exogenous_shocks:
                for loc_name, info in exogenous_shocks.items():
                    norm_target = normalize_name(loc_name)
                    if isinstance(info, dict):
                        intensity = float(info.get('intensity', 0.0))
                        suppression = float(info.get('suppression_intensity', 0.0))
                        impact_value = (3.0 if region != 'fortaleza' else 2.5) * intensity
                        relief_value = (1.6 if region != 'fortaleza' else 1.3) * suppression
                        for i, row in data['nodes_gdf'].iterrows():
                            if normalize_name(row['name']) == norm_target:
                                if intensity > 0:
                                    x_raw_extended[i, :, 24] = min(intensity, 3.0)
                                    sim_impact[i] = impact_value
                                if suppression > 0:
                                    x_raw_extended[i, :, 23] = min(suppression, 3.0)
                                    sim_relief[i] = relief_value

            momentum_feat = np.zeros((num_nodes, total_window, 4))
            cold_streak = np.zeros(num_nodes)
            for t in range(60, total_window):
                r7 = x_raw_extended[:, t-7:t, 0].sum(axis=1)
                p7 = x_raw_extended[:, t-14:t-7, 0].sum(axis=1)
                momentum_feat[:, t, 0] = r7 - p7
                momentum_feat[:, t, 1] = x_raw_extended[:, t-14:t, 0].sum(axis=1) - x_raw_extended[:, t-28:t-14, 0].sum(axis=1)
                momentum_feat[:, t, 2] = x_raw_extended[:, t-30:t, 0].sum(axis=1) - x_raw_extended[:, t-60:t-30, 0].sum(axis=1)
                cold_streak = np.where(x_raw_extended[:, t, 0] > 0, 0, cold_streak + 1)
                momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
            
            if channels >= 32:
                x_raw_extended = np.concatenate([x_raw_extended, momentum_feat[:, :, :channels-29]], axis=2)

            x_final = x_raw_extended[:, -window:, :channels].copy()
            
            # ⭐ ATUALIZAÇÃO (2026-03-18): NORMALIZAÇÃO Z-SCORE LOCAL EM RUNTIME
            # Garante consistência com a nova metodologia de treino Tentativa 49
            for c in range(channels):
                m_c = x_final[:, :, c].mean()
                s_c = x_final[:, :, c].std() + 1e-6
                x_final[:, :, c] = (x_final[:, :, c] - m_c) / s_c

            active_window = cp.get('dynamic_window', window)
            if active_window and active_window < window:
                x_final[:, :window - active_window, :29] = 0.0

            x = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            adj = self._norm_adj(data['adj_geo'], data['adj_conflict'])
            
            with torch.no_grad():
                out = model(x, adj).squeeze().cpu().numpy()
            
            norm_neural = (out - out.min()) / (out.max() - out.min() + 1e-6)
            tension_vec = data['nodes_gdf']['tension_index'].values.astype(float)
            norm_tension = (tension_vec - tension_vec.min()) / (tension_vec.max() - tension_vec.min() + 1e-6)

            inclusion_horizon = 28 if region == 'interior' else 14
            recent_crime_signal = np.clip(x_raw_extended[:, -inclusion_horizon:, 0].sum(axis=1), 0, 2) / 2.0
            neighbor_signal = np.clip((data['adj_geo'].dot((sim_impact > 0).astype(float))) * (cp.get('tag_bias_neighbor', 0.6)/0.6), 0, 1)
            inclusion_signal = np.clip(np.maximum.reduce([recent_crime_signal, neighbor_signal]), 0, 1)
            calm_signal = np.clip(-momentum_feat[:, -1, 3], 0, 30) / 30.0

            if cp.get('use_historical_fallback') and cp.get('historical_top10'):
                hist_norms = set(normalize_name(n) for n in cp['historical_top10'])
                hist_signal = np.array([1.0 if normalize_name(str(row['name'])) in hist_norms else 0.0 for _, row in data['nodes_gdf'].iterrows()])
                final_logic = (0.50 * norm_neural) + (0.20 * norm_tension) + (0.10 * inclusion_signal) + (0.20 * hist_signal) - (0.10 * calm_signal)
            else:
                # ⭐ Peso neural aumentado para 0.60 (mais confiança no novo modelo honesto)
                n_w = 0.60 if region != 'interior' else 0.45
                t_w, i_w = 0.20, 0.20
                final_logic = (n_w * norm_neural) + (t_w * norm_tension) + (i_w * inclusion_signal) - (0.10 * calm_signal)
            
            out_norm = np.clip(final_logic, 0, 1) * 100.0
            for i, row in data['nodes_gdf'].iterrows():
                name_key = normalize_name(str(row['name']))
                if self._node_owners.get(name_key, region) == region:
                    combined_scores[name_key] = float(out_norm[i])
                    if return_trends: trends[name_key] = 'stable'
        return (combined_scores, trends) if return_trends else combined_scores

    def _norm_adj(self, geo, conf):
        def n(a):
            s = np.array(a.sum(1)); d = np.power(s, -0.5).flatten(); d[np.isinf(d)]=0.; m=np.diag(d)
            return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(self.device)
        return [n(geo), n(conf)]
