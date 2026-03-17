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
        
        # ⭐ ATUALIZAÇÃO (2026-03-16): Modelo oficial de Fortaleza agora é retreinado com 33 canais (Multi-Scale Momentum + Cold Streak)
        # P@10: 87.84% — Promoção: fortaleza_retrain_64.pth → fortaleza_model_active.pth
        fortaleza_model_file = 'fortaleza_super_elite.pth' if os.path.exists(os.path.join(self.root, 'models', 'active', 'fortaleza_super_elite.pth')) else 'fortaleza_model_active.pth'
        has_momentum_fortaleza = True  # Modelo oficial agora com 33 canais (momentum)

        # ⭐ ATUALIZAÇÃO (2026-03-16): Modelo oficial Interior agora é retreinado com 33 canais (Multi-Scale Momentum + Cold Streak)
        # P@10: 81.54% — Promoção: interior_retrain_64.pth → interior_model.pth
        interior_model_file   = 'interior_model.pth'
        interior_has_momentum = True  # Modelo oficial agora com 33 canais (momentum)
        
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
        self._initialize_models()
        self._restore_window_state()  # restaura janelas persistidas após modelos carregados

    def _restore_window_state(self):
        """Restaura dynamic_window e use_historical_fallback do disco ao reiniciar."""
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
                        label = f"{dw}d" if dw else "base"
                        flag = " + fallback histórico ATIVO" if hf else ""
                        print(f"🔄 [Window State] {region.upper()} restaurado: janela={label}{flag}")
                        if hf:
                            self._load_historical_fallback(region)
        except Exception as e:
            print(f"⚠️ [Window State] Erro ao restaurar: {e}")

    def _save_window_state(self):
        """Persiste dynamic_window e use_historical_fallback no disco."""
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

    # Escada de redução de janela: base (120 ou 90) → 90 → 60 → 30 → fallback histórico
    _WINDOW_LADDER = [120, 90, 60, 30]

    def adjust_temporal_focus(self, region, efficiency_score):
        """
        Auto-Ajuste de Janela (Temporal Shrinkage) baseado no feedback do Monitor.

        Escada de redução ao detectar eficiência baixa (P10 < 0.50):
          base_window → 90 → 60 → 30 → fallback top10 histórico do último ano

        Recuperação gradual (P10 >= 0.60):
          sobe um degrau por ciclo até restaurar a janela base.
        """
        if region not in self.specialists: return

        cp = self.calib_params.setdefault(region, next(iter(self.calib_params.values()), {}).copy())
        base_window = self.specialists[region]['window']
        current_window = cp.get('dynamic_window') or base_window
        current_window = min(current_window, base_window)

        if efficiency_score < 0.50:
            # Encontra o degrau atual e desce um nível
            ladder = [w for w in self._WINDOW_LADDER if w <= base_window]
            if not ladder:
                ladder = [30]
            # Degrau atual: maior valor da escada <= current_window
            current_rung = max((w for w in ladder if w <= current_window), default=ladder[0])
            current_idx = ladder.index(current_rung)

            if current_rung > ladder[-1]:  # ainda há degrau abaixo
                next_rung = ladder[current_idx + 1] if current_idx + 1 < len(ladder) else ladder[-1]
                if next_rung != current_window:
                    print(f"📉 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. "
                          f"Reduzindo janela {current_window}d → {next_rung}d.")
                    cp['dynamic_window'] = next_rung
                    cp['use_historical_fallback'] = False
                    self._save_window_state()
            else:
                # Já no menor degrau — ativa fallback top10 histórico
                if not cp.get('use_historical_fallback', False):
                    print(f"📉 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. "
                          f"Janela mínima (30d) mantida — ATIVANDO fallback top10 histórico.")
                    cp['use_historical_fallback'] = True
                    self._load_historical_fallback(region)
                    self._save_window_state()
                else:
                    print(f"📚 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. "
                          f"Fallback histórico já ativo (30d). Top10: {cp.get('historical_top10', [])[:5]}")

        elif efficiency_score >= 0.60:
            # Sobe um degrau por ciclo
            ladder = [w for w in self._WINDOW_LADDER if w <= base_window]
            if not ladder:
                ladder = [base_window]
            current_rung = min((w for w in ladder if w >= current_window), default=base_window)
            current_idx = ladder.index(current_rung)

            if current_rung < base_window:
                next_rung = ladder[current_idx - 1] if current_idx > 0 else base_window
                print(f"📈 [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. "
                      f"Expandindo janela {current_window}d → {next_rung}d.")
                cp['dynamic_window'] = next_rung
            else:
                cp['dynamic_window'] = None  # restaura base
                print(f"✅ [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Janela base restaurada.")

            if cp.get('use_historical_fallback'):
                cp['use_historical_fallback'] = False
                print(f"✅ [Auto-Tune] Fallback histórico desativado em {region.upper()}.")
            
            self._save_window_state()
        else:
            print(f"⏸️ [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Janela mantida em {current_window}d.")

    def _load_historical_fallback(self, region):
        """
        Deriva o top10 histórico diretamente do nodes_gdf já em memória,
        ordenando por total_cvli decrescente — sem hardcode, sem JSON externo.
        """
        if region not in self.specialists:
            print(f"⚠️ [Fallback] Região {region.upper()} não inicializada.")
            return

        gdf = self.specialists[region]['data']['nodes_gdf']
        sort_col = 'total_cvli' if 'total_cvli' in gdf.columns else \
                   'recent_cvli' if 'recent_cvli' in gdf.columns else None

        if sort_col is None:
            print(f"⚠️ [Fallback] nodes_gdf de {region.upper()} sem coluna de CVLI para ranking.")
            return

        ranked = gdf.sort_values(sort_col, ascending=False)
        regional_top = list(ranked['name'].head(10))

        self.calib_params[region]['historical_top10'] = regional_top
        print(f"📊 [Fallback] Top10 derivado de {sort_col} para {region.upper()}: {regional_top}")
        self.calib_params[region]['tag_bias_direct'] = 5.00
        self.calib_params[region]['tension_factor']  = 3.00

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
        # Índice de propriedade: cada nome normalizado → região dona
        # Impede que um especialista emita scores de nós pertencentes a outra região
        self._node_owners = {
            normalize_name(str(r['name'])): reg
            for reg, spec in self.specialists.items()
            for _, r in spec['data']['nodes_gdf'].iterrows()
        }

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
            cp = self.calib_params.get(region, next(iter(self.calib_params.values())))
            num_nodes = len(data['nodes_gdf'])
            
            # --- CÁLCULO DE MOMENTUM MULTI-SCALE (AGORA PARA TODAS AS REGIÕES) ---
            # Mesmo que o modelo use 29 canais, calculamos o momentum para o Blend de Inclusão (Peso 0.4)
            extra_history = 60
            total_window = window + extra_history
            total_window = min(total_window, data['node_features'].shape[1])
            
            x_raw_extended = data['node_features'][:, -total_window:, :].copy()
            sim_impact = np.zeros(num_nodes)
            
            if exogenous_shocks:
                for loc_name, info in exogenous_shocks.items():
                    norm_target = normalize_name(loc_name)
                    if isinstance(info, dict):
                        intensity = float(info.get('conflict_intensity', info.get('intensity', 0.0)))
                        # Peso exógeno adaptativo: Interior e RMF são mais sensíveis (3.0)
                        impact_value = (3.0 if region != 'fortaleza' else 2.5) * intensity
                        channel_idx = 25 if info.get('is_critical') else 24
                        for i, row in data['nodes_gdf'].iterrows():
                            if normalize_name(row['name']) == norm_target:
                                x_raw_extended[i, :, channel_idx] = min(intensity, 3.0)
                                sim_impact[i] = impact_value

            # CÁLCULO DO MOMENTUM (Para uso no Blend de Inclusão e Modelos 33ch)
            momentum_feat = np.zeros((num_nodes, total_window, 4))
            cold_streak = np.zeros(num_nodes)
            
            for t in range(60, total_window):
                # Escala 1 (7 dias)
                recent_7 = x_raw_extended[:, t-7:t, 0].sum(axis=1)
                past_7 = x_raw_extended[:, t-14:t-7, 0].sum(axis=1)
                momentum_feat[:, t, 0] = recent_7 - past_7
                # Escala 2 (14 dias)
                momentum_feat[:, t, 1] = x_raw_extended[:, t-14:t, 0].sum(axis=1) - x_raw_extended[:, t-28:t-14, 0].sum(axis=1)
                # Escala 3 (30 dias)
                momentum_feat[:, t, 2] = x_raw_extended[:, t-30:t, 0].sum(axis=1) - x_raw_extended[:, t-60:t-30, 0].sum(axis=1)
                # Cold Streak (Invertido)
                crimes_today = x_raw_extended[:, t, 0]
                cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)
                momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
            
            # Se o modelo exigir canais extras, injetamos no tensor de entrada
            if channels >= 32:
                # Canais de momentum concatenados com valores brutos (sem boost)
                # O modelo foi treinado com momentum bruto — manter consistência treino/inferência
                x_raw_extended = np.concatenate([x_raw_extended, momentum_feat[:, :, :channels-29]], axis=2)
                
                # AJUSTE PARA REGIÕES ESPARSAS (ex: Interior com muitos dias frios)
                # Se a maioria dos nós tem cold_streak alto (>7 dias), isso indica uma região segura, não ameaçada
                # Invertemos a interpretação do canal 32 (cold_streak) como fator de segurança
                if region in ['interior']:
                    # Para Interior: alto cold_streak = zona segura = reduz urgência
                    # Clipamos para evitar dominância total: max +5 em vez de -30
                    momentum_feat_adj = momentum_feat.copy()
                    momentum_feat_adj[:, :, 3] = np.clip(momentum_feat[:, :, 3] / 6.0, -5, 5)  # Suaviza: [-5, 5] em vez de [-30, 0]
                    x_raw_extended[:, :, -1] = momentum_feat_adj[:, :, 3]

            x_final = x_raw_extended[:, -window:, :channels].copy()
            
            # --- ATUALIZAÇÃO: TEMPORAL SHRINKAGE (MÁSCARA DE ATENÇÃO DINÂMICA) ---
            # Se o Monitor de Eficiência reduziu a janela (ex: de 120 para 60), 
            # nós zeramos o passado distante nos canais brutos (0-28) para focar no presente,
            # mas PRESERVAMOS os canais de Momentum (29-32) que já embutem o histórico macro.
            active_window = cp.get('dynamic_window', window)
            if active_window and active_window < window:
                # Aplica a máscara apenas nos canais originais (0 a 28)
                x_final[:, :window - active_window, :29] = 0.0
                # Os canais 29, 30, 31 e 32 permanecem intactos para manter a 'inércia' da inteligência

            x = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            adj = self._norm_adj(data['adj_geo'], data['adj_conflict'])
            
            with torch.no_grad():
                out = model(x, adj).squeeze().cpu().numpy()
            
            # --- ESTRATÉGIA DE TRANSIÇÃO (BLEND 20/40/40) ---
            # Ideal para meses com pouco histórico (como início de março)
            
            # 1. Componente Neural (Ranking Dinâmico) - Peso 0.2
            r_min, r_max = out.min(), out.max()
            norm_neural = (out - r_min) / (r_max - r_min + 1e-6)
            
            # 2. Componente de Tensão (Estabilidade Territorial) - Peso 0.4
            # Fornece a 'âncora' histórica para o ranking não oscilar demais
            tension_vec = data['nodes_gdf']['tension_index'].values.astype(float)
            t_min, t_max = tension_vec.min(), tension_vec.max()
            norm_tension = (tension_vec - t_min) / (t_max - t_min + 1e-6)
            
            # 3. Componente de Inclusão (Eventos Recentes) - Peso 0.4
            # Garante que quem agiu nos últimos 3 dias suba, mas sem dominar 80% do peso
            recent_crime = x_raw_extended[:, -3:, 0].sum(axis=1) > 0
            inclusion_signal = (recent_crime | (sim_impact > 0)).astype(float)

            # 4. Fallback Histórico Top10 — ativado quando P10 persiste baixo mesmo em 30d
            # Blend com fallback: 50% neural + 20% tensão + 10% recente + 20% histórico
            if cp.get('use_historical_fallback') and cp.get('historical_top10'):
                historical_top_norms = set(normalize_name(n) for n in cp['historical_top10'])
                historical_signal = np.array([
                    1.0 if normalize_name(str(row['name'])) in historical_top_norms else 0.0
                    for _, row in data['nodes_gdf'].iterrows()
                ])
                final_logic = (0.50 * norm_neural) + (0.20 * norm_tension) + (0.10 * inclusion_signal) + (0.20 * historical_signal)
                print(f"📚 [Blend] {region.upper()} usando fallback histórico top10 ({len(historical_top_norms)} nós).")
            else:
                # Blend padrão: ajustado por região
                # Interior com alta sparsidade em crime → aumentar confiança no neural (40%)
                # Regiões com crime regular → manter neural em 60%
                neural_weight = 0.40 if region == 'interior' else 0.60
                tension_weight = 0.25 if region == 'interior' else 0.20
                inclusion_weight = 0.35 if region == 'interior' else 0.20  # Inclusão importante quando há crime
                
                # Modelo treinado com dados brutos — inferência deve refletir saída neural
                final_logic = (neural_weight * norm_neural) + (tension_weight * norm_tension) + (inclusion_weight * inclusion_signal)
            
            # Escalonamento Dashboard (5% a 100%)
            out_norm = 5.0 + (final_logic * 95.0)
            
            # BLOCO DE SEGURANÇA: cada especialista emite scores apenas para seus próprios nós
            for i, row in data['nodes_gdf'].iterrows():
                name_raw = str(row['name'])
                name_key = normalize_name(name_raw)
                # Anti-poluição dinâmica: ignora nós que pertencem a outro especialista
                if self._node_owners.get(name_key, region) != region:
                    continue
                combined_scores[name_key] = float(out_norm[i])
                if return_trends: trends[name_key] = 'stable'
                
        return (combined_scores, trends) if return_trends else combined_scores

    def _norm_adj(self, geo, conf):
        def n(a):
            s = np.array(a.sum(1)); d = np.power(s, -0.5).flatten(); d[np.isinf(d)]=0.; m=np.diag(d)
            return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(self.device)
        return [n(geo), n(conf)]
