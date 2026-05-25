import sys
import json
import numpy as np
import torch
import pickle
import os
import pandas as pd
import unicodedata
import re
import threading
import time
from datetime import datetime
from pathlib import Path

# --- Champion/Challenger LGBM Lean (Sentinela V3) ---
try:
    from .champion_challenger import ChampionChallenger
except ImportError:
    try:
        from champion_challenger import ChampionChallenger
    except ImportError:
        ChampionChallenger = None

# ============================================================================
# ARQUITETURA REGIONAL ST-GAT - ORQUESTRADOR DE ELITE
# ============================================================================

try:
    from .architectures import DeepSTGAT_64, DeepSTGAT_32, ShallowGAT
except ImportError:
    from architectures import DeepSTGAT_64, DeepSTGAT_32, ShallowGAT

def normalize_name(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
    text = re.sub(r'\s*-\s*AIS.*$', '', text)
    return text.strip()

class StateOrchestrator:
    def __init__(self, project_root):
        self.root = project_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._hostinger_sync_manager = None
        self._hostinger_sync_lock = threading.Lock()
        self._hostinger_sync_worker_started = False
        self._hostinger_sync_pending_risk_artifact = None
        self._hostinger_sync_pending_startup_merge = False
        self._hostinger_sync_retry_interval_seconds = 60
        
        # â­ ATUALIZAÃ‡ÃƒO (2026-03-18): Modelo oficial de Fortaleza agora Ã© retreinado com Blindagem Temporal
        # Paradigma Tentativa 49: Gradiente Agressivo + Z-Score Local.
        fortaleza_model_file = 'fortaleza_model_active.pth'
        has_momentum_fortaleza = True

        interior_model_file   = 'interior_model.pth'
        interior_has_momentum = True
        
        self.configs = {
            'fortaleza': {
                'model_path': os.path.join(self.root, 'models', 'active', fortaleza_model_file),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_fortaleza.pkl'),
                'class': ShallowGAT,
                'in_channels': 41, 
                'window': 14 
            },
            'rmf': {
                'model_path': os.path.join(self.root, 'models', 'active', 'rmf_model.pth'),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_rmf.pkl'),
                'class': ShallowGAT,
                'in_channels': 41,
                'window': 14
            },
            'interior': {
                'model_path': os.path.join(self.root, 'models', 'active', interior_model_file),
                'data_path': os.path.join(self.root, 'data', 'processed', 'processed_interior.pkl'),
                'class': ShallowGAT,
                'in_channels': 41,
                'window': 14
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
        
        # --- Champion/Challenger (Sentinela V3) ---
        self.champion_challenger = None
        if ChampionChallenger is not None:
            try:
                self.champion_challenger = ChampionChallenger(self.root)
                print("âœ… [Sentinela V3] Refinamento LGBM integrado ao Orquestrador.")
            except Exception as cc_err:
                print(f"âš ï¸ [Sentinela V3] Falha ao integrar: {cc_err}")

    def _ensure_hostinger_sync_worker(self):
        with self._hostinger_sync_lock:
            if self._hostinger_sync_worker_started:
                return
            self._hostinger_sync_worker_started = True
        worker = threading.Thread(target=self._hostinger_sync_worker_loop, daemon=True)
        worker.start()
        print("[Hostinger Sync] worker iniciado (assincrono, retry a cada 60s).")

    def _hostinger_sync_worker_loop(self):
        while True:
            pending_risk = None
            pending_merge = False
            with self._hostinger_sync_lock:
                pending_risk = self._hostinger_sync_pending_risk_artifact
                pending_merge = self._hostinger_sync_pending_startup_merge

            if pending_merge:
                ok = self._try_sync_startup_data_merge_once()
                if ok:
                    with self._hostinger_sync_lock:
                        self._hostinger_sync_pending_startup_merge = False

            if pending_risk is not None:
                ok = self._try_sync_risk_artifacts_once(pending_risk, 'risk_update')
                if ok:
                    with self._hostinger_sync_lock:
                        self._hostinger_sync_pending_risk_artifact = None

            time.sleep(self._hostinger_sync_retry_interval_seconds)

    def _get_hostinger_sync_manager(self):
        if self._hostinger_sync_manager is not None:
            return self._hostinger_sync_manager
        try:
            from src.hostinger_sync import HostingerSyncManager
            self._hostinger_sync_manager = HostingerSyncManager(self.root)
            return self._hostinger_sync_manager
        except Exception as e:
            print(f"[Hostinger Sync] falha ao inicializar manager: {e}")
            return None

    @staticmethod
    def _log_hostinger_sync_result(context, result):
        if not isinstance(result, dict):
            print(f"[Hostinger Sync] {context}: resultado invalido ({result}).")
            return
        status = str(result.get('status', 'unknown')).strip().lower()
        reason = result.get('reason')
        fingerprint = result.get('fingerprint')
        uploaded = result.get('uploaded_files') or []
        if status == 'synced':
            print(f"[Hostinger Sync] {context}: synced | arquivos={len(uploaded)} | fingerprint={fingerprint or '-'}")
            return
        if status == 'skipped':
            print(f"[Hostinger Sync] {context}: skipped | motivo={reason or 'unchanged'} | fingerprint={fingerprint or '-'}")
            return
        if status == 'disabled':
            print(f"[Hostinger Sync] {context}: disabled | motivo={reason or 'not configured'}")
            return
        print(f"[Hostinger Sync] {context}: status={status or 'unknown'} | motivo={reason or '-'} | fingerprint={fingerprint or '-'}")

    def _log_hostinger_sync_config_snapshot(self, context):
        manager = self._get_hostinger_sync_manager()
        if manager is None:
            return
        cfg = manager.config
        masked_user = f"{cfg.user[:2]}***" if cfg.user else "-"
        print(f"[Hostinger Sync] {context}: config enabled={cfg.enabled} host={cfg.host or '-'} port={cfg.port} user={masked_user} timeout={cfg.timeout_seconds}s configured={cfg.is_configured}")

    def _try_sync_risk_artifacts_once(self, artifact, context):
        manager = self._get_hostinger_sync_manager()
        if manager is None:
            return False
        self._log_hostinger_sync_config_snapshot(context)
        try:
            result = manager.sync_risk_artifacts(artifact)
            self._log_hostinger_sync_result(context, result)
            status = str((result or {}).get('status', '')).lower()
            return status in {'synced', 'skipped', 'disabled'}
        except Exception as e:
            print(f"[Hostinger Sync] {context}: erro ao sincronizar risk artifacts: {e}")
            return False

    def _try_sync_startup_data_merge_once(self):
        manager = self._get_hostinger_sync_manager()
        if manager is None:
            return False
        self._log_hostinger_sync_config_snapshot('startup:data_merge')
        try:
            merge_result = manager.sync_data_merge_artifacts()
            self._log_hostinger_sync_result('startup:data_merge', merge_result)
            status = str((merge_result or {}).get('status', '')).lower()
            return status in {'synced', 'skipped', 'disabled'}
        except Exception as e:
            print(f"[Hostinger Sync] startup:data_merge: erro ao sincronizar data merge: {e}")
            return False

    def _enqueue_hostinger_risk_sync(self, artifact):
        self._ensure_hostinger_sync_worker()
        with self._hostinger_sync_lock:
            self._hostinger_sync_pending_risk_artifact = artifact
        print("[Hostinger Sync] risk_update: sync enfileirado (assincrono).")

    def sync_hostinger_on_startup(self):
        self._ensure_hostinger_sync_worker()
        with self._hostinger_sync_lock:
            self._hostinger_sync_pending_startup_merge = True
        print("[Hostinger Sync] startup:data_merge: sync enfileirado (assincrono).")

        latest_json_path = Path(self.root) / 'outputs' / 'hermes' / 'risk_snapshot_latest.json'
        if not latest_json_path.exists():
            print("[Hostinger Sync] startup:risk_outputs: arquivo latest inexistente; aguardando proxima atualizacao de risco.")
            return
        try:
            artifact = json.loads(latest_json_path.read_text(encoding='utf-8'))
            with self._hostinger_sync_lock:
                self._hostinger_sync_pending_risk_artifact = artifact
            print("[Hostinger Sync] startup:risk_outputs: sync enfileirado (assincrono).")
        except Exception as e:
            print(f"[Hostinger Sync] startup:risk_outputs: erro ao processar snapshot latest: {e}")

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
            print(f"âš ï¸ [Window State] Erro ao restaurar: {e}")

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
            print(f"âš ï¸ [Window State] Erro ao salvar: {e}")

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
                    print(f"ðŸ“‰ [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Reduzindo janela {current_window}d â†’ {next_rung}d.")
                    cp['dynamic_window'] = next_rung
                    cp['use_historical_fallback'] = False
                    self._save_window_state()
            else:
                if not cp.get('use_historical_fallback', False):
                    print(f"ðŸ“‰ [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. ATIVANDO fallback histÃ³rico.")
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
                print(f"ðŸ“ˆ [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Expandindo janela {current_window}d â†’ {next_rung}d.")
                cp['dynamic_window'] = next_rung
            else:
                cp['dynamic_window'] = None
                print(f"âœ… [Auto-Tune] P10={efficiency_score*100:.1f}% em {region.upper()}. Janela base restaurada.")

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
                    print(f"âœ… Orquestrador: Especialista {region.upper()} carregado ({cfg['in_channels']} Canais).")
                except Exception as e:
                    print(f"âŒ Erro ao carregar {region}: {e}")
        self._node_owners = {normalize_name(str(r['name'])): reg for reg, spec in self.specialists.items() for _, r in spec['data']['nodes_gdf'].iterrows()}

    def _load_pickle_safe(self, path):
        """Carregamento robusto para evitar falhas de StringDtype (NotImplementedError)."""
        import pickle
        import pandas as pd
        
        # 1. Tenta o caminho padrÃ£o (mais rÃ¡pido)
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
            
        # 2. Fallback: Unpickler customizado para interceptar StringDtype problemÃ¡ticos
        class RobustUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Se for StringDtype do pandas (python ou arrow), redirecionamos para algo seguro
                if 'pandas' in module and 'StringDtype' in name:
                    try:
                        from pandas import StringDtype
                        return StringDtype
                    except ImportError:
                        return object
                return super().find_class(module, name)

        try:
            with open(path, 'rb') as f:
                return RobustUnpickler(f).load()
        except Exception as e:
            # 3. Ãšltimo recurso: Carregar via pickle puro e converter se necessÃ¡rio
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'nodes_gdf' in data:
                    # Tenta converter o GDF para algo legÃ­vel forÃ§ando dtypes
                    gdf = data['nodes_gdf']
                    if hasattr(gdf, 'astype'):
                        for col in gdf.columns:
                            try:
                                if gdf[col].dtype == 'string':
                                    gdf[col] = gdf[col].astype(object)
                            except: pass
                    data['nodes_gdf'] = gdf
                return data
            except Exception as final_e:
                print(f"âŒ Falha total ao carregar {path}: {final_e}")
                return None

    def _risk_level(self, score):
        score = max(0.0, min(100.0, float(score)))
        if score >= 71.0:
            return 'crÃ­tico'
        if score >= 51.0:
            return 'alto'
        if score >= 31.0:
            return 'moderado'
        return 'baixo'

    def _label_band(self, value, bands):
        for threshold, label in bands:
            if value >= threshold:
                return label
        return bands[-1][1]

    def _score_percentile(self, rank, total_items):
        if total_items <= 1:
            return 100.0
        return round(100.0 * (1.0 - ((rank - 1) / (total_items - 1))), 1)

    def _compute_confidence_pct(self, item):
        neural = float(item.get('neural_score', 0.0)) / 100.0
        tension = float(item.get('tension_score', 0.0)) / 100.0
        inclusion = float(item.get('inclusion_score', 0.0)) / 100.0
        support = float(item.get('territorial_support_pct', 0.0)) / 100.0
        recent_30d = min(float(item.get('recent_cvli_30d', 0.0)) / 4.0, 1.0)
        coherence = 1.0 - min(float(np.std([neural, tension, inclusion])), 1.0)
        base = (0.30 * coherence) + (0.25 * max(neural, tension, inclusion)) + (0.25 * support) + (0.20 * recent_30d)
        if item.get('historical_fallback'):
            base -= 0.12
        return round(max(5.0, min(99.0, base * 100.0)), 1)

    def _build_driver_list(self, item):
        drivers = [
            ('sinal_neural', float(item.get('neural_score', 0.0)), 'Sinal neural do ST-GAT'),
            ('tensao_territorial', float(item.get('tension_score', 0.0)), 'TensÃ£o territorial'),
            ('inclusao_recente', float(item.get('inclusion_score', 0.0)), 'Atividade recente e vizinhanÃ§a'),
        ]

        if float(item.get('recent_cvli_30d', 0.0)) > 0:
            drivers.append(('cvli_recente', min(float(item.get('recent_cvli_30d', 0.0)) * 20.0, 100.0), 'CVLI recente na janela de 30 dias'))
        if item.get('historical_fallback'):
            drivers.append(('fallback_historico', 55.0, 'Fallback histÃ³rico ativado'))

        ordered = sorted(drivers, key=lambda entry: entry[1], reverse=True)
        return [
            {
                'key': key,
                'label': label,
                'strength_pct': round(strength, 1),
            }
            for key, strength, label in ordered[:3]
        ]

    def _build_manager_feedback(self, item, rank, total_items, score, confidence_pct, expressiveness_pct, drivers):
        primary_driver = drivers[0]['label'] if drivers else 'Sinal combinado do modelo'
        recent_cvli = int(item.get('recent_cvli_30d', 0))
        support_pct = float(item.get('territorial_support_pct', 0.0))
        risk_level = self._risk_level(score)

        if rank <= 5:
            priority_text = 'prioridade imediata de acompanhamento'
        elif rank <= 10:
            priority_text = 'prioridade alta de acompanhamento'
        elif risk_level in ('crÃ­tico', 'alto'):
            priority_text = 'territÃ³rio relevante para monitoramento'
        else:
            priority_text = 'territÃ³rio de atenÃ§Ã£o tÃ¡tica'

        leitura_rapida = (
            f"{item['name']} aparece na posiÃ§Ã£o {rank} de {total_items}, com risco {score:.1f} e nÃ­vel {risk_level}; "
            f"Ã© {priority_text}."
        )
        por_que_importa = (
            f"O peso principal vem de {primary_driver.lower()}, com suporte territorial de {support_pct:.1f}%"
            f" e {recent_cvli} registros recentes na janela de 30 dias."
        )

        if confidence_pct >= 85:
            confidence_note = 'Leitura com boa sustentaÃ§Ã£o dos sinais atuais.'
        elif confidence_pct >= 70:
            confidence_note = 'Leitura consistente, mas ainda pede validaÃ§Ã£o operacional pontual.'
        elif item.get('historical_fallback'):
            confidence_note = 'Leitura mais fraca; parte do peso vem de fallback histÃ³rico e deve ser confirmada em campo.'
        else:
            confidence_note = 'Leitura sensÃ­vel a ruÃ­do; requer conferÃªncia adicional antes de decisÃ£o forte.'

        if expressiveness_pct >= 85:
            expressiveness_note = 'O territÃ³rio estÃ¡ bem destacado no ranking frente aos pares.'
        elif expressiveness_pct >= 70:
            expressiveness_note = 'O territÃ³rio se diferencia do bloco intermediÃ¡rio, mas sem isolamento absoluto.'
        else:
            expressiveness_note = 'O territÃ³rio estÃ¡ prÃ³ximo de pares vizinhos e pode oscilar no prÃ³ximo ciclo.'

        proxima_acao = (
            f"Verificar eventos recentes, pressÃ£o territorial e coerÃªncia com inteligÃªncia local antes da prÃ³xima atualizaÃ§Ã£o do ranking."
        )

        return {
            'leitura_rapida': leitura_rapida,
            'por_que_importa': por_que_importa,
            'confianca_limites': confidence_note,
            'expressividade_previsao': expressiveness_note,
            'proxima_acao': proxima_acao,
        }

    def _build_ranking_entry(self, item, rank, peer_scores, cc_status, data_limit):
        score = round(float(item.get('score_final', item.get('score_raw', 0.0))), 2)
        total_items = max(1, len(peer_scores))
        mean_score = float(np.mean(peer_scores)) if peer_scores else score
        std_score = float(np.std(peer_scores)) if len(peer_scores) > 1 else 0.0
        z_score = (score - mean_score) / std_score if std_score > 1e-6 else 0.0
        percentile = self._score_percentile(rank, total_items)
        separation_pct = max(0.0, min(100.0, 50.0 + (12.0 * z_score)))
        expressiveness_pct = round((0.6 * percentile) + (0.4 * separation_pct), 1)
        confidence_pct = self._compute_confidence_pct(item)
        drivers = self._build_driver_list(item)
        manager_feedback = self._build_manager_feedback(item, rank, total_items, score, confidence_pct, expressiveness_pct, drivers)

        entry = {
            'rank': rank,
            'name': item['name'],
            'name_normalized': item['name_normalized'],
            'region': item['region'],
            'territorial_level': item['territorial_level'],
            'risk_score': score,
            'risk_level': self._risk_level(score),
            'confidence_pct': confidence_pct,
            'confidence_label': self._label_band(confidence_pct, [(85, 'alta'), (70, 'moderada'), (55, 'baixa'), (0, 'muito baixa')]),
            'expressiveness_pct': expressiveness_pct,
            'expressiveness_label': self._label_band(expressiveness_pct, [(85, 'muito alta'), (70, 'alta'), (50, 'moderada'), (0, 'baixa')]),
            'data_limit': data_limit,
            'metrics': {
                'recent_cvli_14d': int(item.get('recent_cvli_14d', 0)),
                'recent_cvli_30d': int(item.get('recent_cvli_30d', 0)),
                'historical_cvli': round(float(item.get('historical_cvli', 0.0)), 1),
                'neural_score': round(float(item.get('neural_score', 0.0)), 1),
                'tension_score': round(float(item.get('tension_score', 0.0)), 1),
                'inclusion_score': round(float(item.get('inclusion_score', 0.0)), 1),
                'calm_penalty': round(float(item.get('calm_penalty', 0.0)), 1),
                'territorial_support_pct': round(float(item.get('territorial_support_pct', 0.0)), 1),
                'historical_support_pct': round(float(item.get('historical_support_pct', 0.0)), 1),
                'live_support_pct': round(float(item.get('live_support_pct', 0.0)), 1),
                'score_z': round(z_score, 2),
                'peer_percentile': percentile,
                'dynamic_window_days': int(item.get('dynamic_window', 0) or 0),
                'base_window_days': int(item.get('base_window', 0) or 0),
            },
            'explainability': {
                'top_drivers': drivers,
                'summary': ' | '.join(f"{driver['label']}: {driver['strength_pct']:.1f}%" for driver in drivers),
            },
            'manager_feedback': manager_feedback,
            'prediction_details': {
                'historical_fallback': bool(item.get('historical_fallback', False)),
                'trend': item.get('trend', 'stable'),
            },
        }

        if item['region'] == 'fortaleza' and cc_status:
            entry['prediction_details']['champion_challenger'] = {
                'champion_pct': float(cc_status.get('champion_pct', 100.0)),
                'challenger_pct': float(cc_status.get('challenger_pct', 0.0)),
                'cc_weight': float(cc_status.get('cc_weight', 0.0)),
                'last_eval': cc_status.get('last_eval'),
            }

        return entry

    def _build_hermes_rankings(self, scores_map, component_details):
        cc_status = self.champion_challenger.status() if self.champion_challenger is not None else None
        data_limit = None
        if self.dates is not None and len(self.dates) > 0:
            try:
                data_limit = pd.Timestamp(self.dates[-1]).strftime('%Y-%m-%d')
            except Exception:
                data_limit = str(self.dates[-1])

        rows = []
        for name_key, score in scores_map.items():
            meta = component_details.get(name_key)
            if not meta:
                continue
            row = dict(meta)
            row['score_final'] = float(score)
            rows.append(row)

        rows.sort(key=lambda item: item['score_final'], reverse=True)

        general_cities = [item for item in rows if item['territorial_level'] == 'cidade']
        rmf_cities = [item for item in general_cities if item['region'] == 'rmf']
        interior_cities = [item for item in general_cities if item['region'] == 'interior']
        fortaleza_bairros = [item for item in rows if item['region'] == 'fortaleza']

        def build_slice(items, limit):
            peer_scores = [float(item['score_final']) for item in items]
            return [
                self._build_ranking_entry(item, idx + 1, peer_scores, cc_status, data_limit)
                for idx, item in enumerate(items[:limit])
            ]

        return {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'data_limit': data_limit,
            'source': 'src/core/orchestrator.py:get_combined_risk',
            'model': {
                'type': 'ST-GAT + Sentinela V3',
                'challenger': cc_status,
            },
            'rankings': {
                'general_cities_top30': build_slice(general_cities, 30),
                'rmf_cities_top20': build_slice(rmf_cities, 20),
                'interior_cities_top30': build_slice(interior_cities, 30),
                'fortaleza_bairros_top30': build_slice(fortaleza_bairros, 30),
            },
        }

    def _export_recent_enriched_status(self, output_dir, history_dir, timestamp_slug, data_limit):
        source_path = os.path.join(self.root, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
        latest_path = os.path.join(output_dir, 'dados_status_enriquecido_14d_latest.csv')
        history_path = os.path.join(history_dir, f'dados_status_enriquecido_14d_{timestamp_slug}.csv')

        if not os.path.exists(source_path):
            raise FileNotFoundError(f'Fonte enriquecida nao encontrada: {source_path}')

        status_df = pd.read_csv(source_path, encoding='utf-8', low_memory=False)
        if 'data' not in status_df.columns:
            raise KeyError("Coluna 'data' nao encontrada em dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")

        status_df['data'] = pd.to_datetime(status_df['data'], errors='coerce')
        status_df = status_df.dropna(subset=['data']).copy()

        if status_df.empty:
            filtered_df = status_df
            reference_date = pd.Timestamp(data_limit) if data_limit else pd.Timestamp.today().normalize()
        else:
            csv_max_date = status_df['data'].max().normalize()
            if data_limit:
                reference_date = min(pd.Timestamp(data_limit).normalize(), csv_max_date)
            else:
                reference_date = csv_max_date
            window_start = reference_date - pd.Timedelta(days=13)
            filtered_df = status_df[(status_df['data'] >= window_start) & (status_df['data'] <= reference_date)].copy()

        filtered_df['data'] = filtered_df['data'].dt.strftime('%Y-%m-%d')
        filtered_df = filtered_df.sort_values(['data', 'hora', 'cidade', 'bairro'], ascending=[False, False, True, True], na_position='last')

        for path in (latest_path, history_path):
            filtered_df.to_csv(path, index=False, encoding='utf-8-sig')

        tactical_summary = self._build_tactical_14d_summary(filtered_df, reference_date)
        tactical_latest_json_path = os.path.join(output_dir, 'dados_status_enriquecido_14d_summary_latest.json')
        tactical_history_json_path = os.path.join(history_dir, f'dados_status_enriquecido_14d_summary_{timestamp_slug}.json')
        tactical_latest_md_path = os.path.join(output_dir, 'dados_status_enriquecido_14d_summary_latest.md')
        tactical_history_md_path = os.path.join(history_dir, f'dados_status_enriquecido_14d_summary_{timestamp_slug}.md')

        for path in (tactical_latest_json_path, tactical_history_json_path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(tactical_summary, f, indent=2, ensure_ascii=False)

        tactical_md = self._render_tactical_14d_summary_markdown(tactical_summary)
        for path in (tactical_latest_md_path, tactical_history_md_path):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(tactical_md)

        return {
            'source': 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv',
            'latest_csv': 'outputs/hermes/dados_status_enriquecido_14d_latest.csv',
            'history_csv': f'outputs/hermes/history/dados_status_enriquecido_14d_{timestamp_slug}.csv',
            'latest_summary_json': 'outputs/hermes/dados_status_enriquecido_14d_summary_latest.json',
            'history_summary_json': f'outputs/hermes/history/dados_status_enriquecido_14d_summary_{timestamp_slug}.json',
            'latest_summary_md': 'outputs/hermes/dados_status_enriquecido_14d_summary_latest.md',
            'history_summary_md': f'outputs/hermes/history/dados_status_enriquecido_14d_summary_{timestamp_slug}.md',
            'reference_date': reference_date.strftime('%Y-%m-%d'),
            'window_days': 14,
            'row_count': int(len(filtered_df)),
        }

    def _build_tactical_14d_summary(self, status_df, reference_date):
        summary = {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'reference_date': reference_date.strftime('%Y-%m-%d'),
            'window_days': 14,
            'total_rows': int(len(status_df)),
            'date_min': None,
            'date_max': None,
            'top_cities': [],
            'top_bairros_fortaleza': [],
            'top_tipos_evento': [],
            'fortaleza': {
                'row_count': 0,
                'top_bairros': [],
                'top_tipos_evento': [],
            },
            'rmf': {
                'row_count': 0,
                'top_cidades': [],
                'top_tipos_evento': [],
            },
            'interior': {
                'row_count': 0,
                'top_cidades': [],
                'top_tipos_evento': [],
            },
        }

        if status_df.empty:
            return summary

        working_df = status_df.copy()
        working_df['data'] = pd.to_datetime(working_df['data'], errors='coerce')
        working_df['cidade_norm'] = working_df['cidade'].fillna('').astype(str).str.strip().str.upper()
        working_df['bairro_norm'] = working_df['bairro'].fillna('').astype(str).str.strip().str.upper()
        working_df['tipo_evento_norm'] = working_df['tipo_evento'].fillna('').astype(str).str.strip().str.upper()

        summary['date_min'] = working_df['data'].min().strftime('%Y-%m-%d') if working_df['data'].notna().any() else None
        summary['date_max'] = working_df['data'].max().strftime('%Y-%m-%d') if working_df['data'].notna().any() else None

        def _group_counts(df, column, limit, field_name):
            if column not in df.columns:
                return []
            grouped = (
                df[df[column].fillna('').astype(str).str.strip() != '']
                .groupby(column)
                .size()
                .sort_values(ascending=False)
                .head(limit)
            )
            return [
                {field_name: str(index), 'count': int(value)}
                for index, value in grouped.items()
            ]

        summary['top_cities'] = _group_counts(working_df, 'cidade_norm', 10, 'cidade')
        summary['top_bairros_fortaleza'] = _group_counts(working_df[working_df['cidade_norm'] == 'FORTALEZA'], 'bairro_norm', 15, 'bairro')
        summary['top_tipos_evento'] = _group_counts(working_df, 'tipo_evento_norm', 10, 'tipo_evento')

        fortaleza_df = working_df[working_df['cidade_norm'] == 'FORTALEZA'].copy()
        summary['fortaleza'] = {
            'row_count': int(len(fortaleza_df)),
            'top_bairros': _group_counts(fortaleza_df, 'bairro_norm', 15, 'bairro'),
            'top_tipos_evento': _group_counts(fortaleza_df, 'tipo_evento_norm', 10, 'tipo_evento'),
        }

        rmf_names = {
            'CAUCAIA', 'MARACANAU', 'MARANGUAPE', 'PACATUBA', 'HORIZONTE', 'EUSEBIO', 'AQUIRAZ', 'PARACURU',
            'PARAIPABA', 'ITAITINGA', 'GUAIUBA', 'SAO GONCALO DO AMARANTE', 'CASCAVEL', 'PACAJUS', 'CHOROZINHO',
            'PINDORETAMA', 'BEBERIBE', 'TRAIRI', 'SAO LUIS DO CURU'
        }
        rmf_df = working_df[working_df['cidade_norm'].isin(rmf_names)].copy()
        interior_df = working_df[(working_df['cidade_norm'] != 'FORTALEZA') & (~working_df['cidade_norm'].isin(rmf_names))].copy()

        summary['rmf'] = {
            'row_count': int(len(rmf_df)),
            'top_cidades': _group_counts(rmf_df, 'cidade_norm', 10, 'cidade'),
            'top_tipos_evento': _group_counts(rmf_df, 'tipo_evento_norm', 10, 'tipo_evento'),
        }
        summary['interior'] = {
            'row_count': int(len(interior_df)),
            'top_cidades': _group_counts(interior_df, 'cidade_norm', 10, 'cidade'),
            'top_tipos_evento': _group_counts(interior_df, 'tipo_evento_norm', 10, 'tipo_evento'),
        }

        return summary

    def _render_tactical_14d_summary_markdown(self, summary):
        def _render_list(title, items, key_name):
            lines = [f'## {title}', '']
            if not items:
                lines.append('- Sem registros relevantes nesta janela.')
                lines.append('')
                return lines
            for idx, item in enumerate(items, start=1):
                lines.append(f"{idx}. {item[key_name]} - {item['count']} registros")
            lines.append('')
            return lines

        lines = [
            '# Resumo Tatico Independente - Ultimos 14 Dias',
            '',
            f"- Gerado em: {summary['generated_at']}",
            f"- Janela: {summary.get('date_min') or '-'} ate {summary.get('date_max') or '-'}",
            f"- Data de referencia: {summary['reference_date']}",
            f"- Total de registros: {summary['total_rows']}",
            '- Fonte: outputs/hermes/dados_status_enriquecido_14d_latest.csv',
            '- Este resumo e tatico e independente; nao incorpora necessariamente os artefatos Hermes mais recentes.',
            '',
        ]
        lines.extend(_render_list('Top cidades por registros', summary.get('top_cities', []), 'cidade'))
        lines.extend(_render_list('Top bairros de Fortaleza por registros', summary.get('top_bairros_fortaleza', []), 'bairro'))
        lines.extend(_render_list('Top tipos de evento por registros', summary.get('top_tipos_evento', []), 'tipo_evento'))
        lines.extend(_render_list('RMF - Top cidades', summary.get('rmf', {}).get('top_cidades', []), 'cidade'))
        lines.extend(_render_list('Interior - Top cidades', summary.get('interior', {}).get('top_cidades', []), 'cidade'))
        return '\n'.join(lines).strip() + '\n'

    def _write_hermes_outputs(self, scores_map, component_details):
        try:
            artifact = self._build_hermes_rankings(scores_map, component_details)
            output_dir = os.path.join(self.root, 'outputs', 'hermes')
            history_dir = os.path.join(output_dir, 'history')
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(history_dir, exist_ok=True)

            json_path = os.path.join(output_dir, 'risk_snapshot_latest.json')
            md_path = os.path.join(output_dir, 'risk_snapshot_latest.md')
            brief_path = os.path.join(output_dir, 'risk_brief_latest.md')
            csv_path = os.path.join(output_dir, 'risk_snapshot_latest.csv')
            timestamp_slug = datetime.fromisoformat(artifact['generated_at']).strftime('%Y%m%d_%H%M%S')
            history_json_path = os.path.join(history_dir, f'risk_snapshot_{timestamp_slug}.json')
            history_md_path = os.path.join(history_dir, f'risk_snapshot_{timestamp_slug}.md')
            history_brief_path = os.path.join(history_dir, f'risk_brief_{timestamp_slug}.md')
            history_csv_path = os.path.join(history_dir, f'risk_snapshot_{timestamp_slug}.csv')

            artifact['artifacts'] = {
                'latest_json': 'outputs/hermes/risk_snapshot_latest.json',
                'latest_md': 'outputs/hermes/risk_snapshot_latest.md',
                'latest_brief': 'outputs/hermes/risk_brief_latest.md',
                'latest_csv': 'outputs/hermes/risk_snapshot_latest.csv',
                'history_json': f"outputs/hermes/history/risk_snapshot_{timestamp_slug}.json",
                'history_md': f"outputs/hermes/history/risk_snapshot_{timestamp_slug}.md",
                'history_brief': f"outputs/hermes/history/risk_brief_{timestamp_slug}.md",
                'history_csv': f"outputs/hermes/history/risk_snapshot_{timestamp_slug}.csv",
            }

            recent_status_artifact = self._export_recent_enriched_status(
                output_dir=output_dir,
                history_dir=history_dir,
                timestamp_slug=timestamp_slug,
                data_limit=artifact.get('data_limit'),
            )
            artifact['artifacts']['latest_status_enriquecido_14d_csv'] = recent_status_artifact['latest_csv']
            artifact['artifacts']['history_status_enriquecido_14d_csv'] = recent_status_artifact['history_csv']
            artifact['artifacts']['status_enriquecido_14d_source'] = recent_status_artifact['source']
            artifact['status_enriquecido_14d'] = {
                'source': recent_status_artifact['source'],
                'reference_date': recent_status_artifact['reference_date'],
                'window_days': recent_status_artifact['window_days'],
                'row_count': recent_status_artifact['row_count'],
            }

            for path in (json_path, history_json_path):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(artifact, f, indent=2, ensure_ascii=False)

            title_map = {
                'general_cities_top30': 'Ranking das cidades - Top 30 (Geral)',
                'rmf_cities_top20': 'Ranking das cidades - Top 20 (RMF)',
                'interior_cities_top30': 'Ranking das cidades - Top 30 (Interior)',
                'fortaleza_bairros_top30': 'Ranking dos bairros - Top 30 (Fortaleza)',
            }

            lines = [
                '# Artefato Hermes de Risco',
                '',
                f"- Gerado em: {artifact['generated_at']}",
                f"- Base de dados ate: {artifact['data_limit'] or 'indisponivel'}",
                f"- Base de dados formatada: {pd.Timestamp(artifact['data_limit']).strftime('%d/%m/%Y') if artifact['data_limit'] else 'indisponivel'}",
                f"- Origem: {artifact['source']}",
                '- Fonte oficial para o Hermes: outputs/hermes/',
                f"- Snapshot historico JSON: outputs/hermes/history/risk_snapshot_{timestamp_slug}.json",
                f"- Snapshot historico Markdown: outputs/hermes/history/risk_snapshot_{timestamp_slug}.md",
                f"- CSV enriquecido (ultimos 14 dias): outputs/hermes/dados_status_enriquecido_14d_latest.csv",
                f"- Snapshot historico do CSV enriquecido: outputs/hermes/history/dados_status_enriquecido_14d_{timestamp_slug}.csv",
                '',
                '## Leitura operacional',
                '',
                '- Este artefato deve ser a fonte primaria do Hermes para rankings e leitura de risco.',
                '- As metricas de confianca e expressividade sao heuristicas operacionais calculadas a partir do score, separacao no ranking e coerencia dos sinais.',
                f"- O CSV enriquecido complementar cobre {artifact['status_enriquecido_14d']['row_count']} registros ate {artifact['status_enriquecido_14d']['reference_date']} para analise independente de convergencia.",
                '',
            ]

            for key, title in title_map.items():
                lines.append(f'## {title}')
                lines.append('')
                lines.append('| Rank | Localidade | Risco | Nivel | Confianca | Expressividade | Drivers | Base ate |')
                lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
                for entry in artifact['rankings'][key]:
                    drivers = '; '.join(driver['label'] for driver in entry['explainability']['top_drivers'])
                    lines.append(
                        f"| {entry['rank']} | {entry['name']} | {entry['risk_score']:.1f} | {entry['risk_level']} | "
                        f"{entry['confidence_pct']:.1f}% ({entry['confidence_label']}) | "
                        f"{entry['expressiveness_pct']:.1f}% ({entry['expressiveness_label']}) | {drivers} | {entry['data_limit'] or '-'} |"
                    )
                lines.append('')

            markdown_payload = '\n'.join(lines).strip() + '\n'
            for path in (md_path, history_md_path):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(markdown_payload)

            data_limit_br = pd.Timestamp(artifact['data_limit']).strftime('%d/%m/%Y') if artifact['data_limit'] else 'indisponivel'

            def _brief_block(title, entries):
                brief_lines = [f'## {title}']
                if not entries:
                    brief_lines.append('- Ranking indisponivel neste snapshot.')
                    brief_lines.append('')
                    return brief_lines

                lead = entries[0]
                brief_lines.append(f"- Leitura rapida: {lead['manager_feedback']['leitura_rapida']}")
                brief_lines.append(f"- Por que importa: {lead['manager_feedback']['por_que_importa']}")
                brief_lines.append(f"- Proxima acao: {lead['manager_feedback']['proxima_acao']}")
                brief_lines.append('')

                for entry in entries:
                    brief_lines.append(
                        f"{entry['rank']}. {entry['name']} â€” risco {entry['risk_score']:.1f} | {entry['risk_level']} | confianca {entry['confidence_pct']:.1f}%"
                    )
                brief_lines.append('')
                return brief_lines

            fortaleza_top10 = artifact['rankings']['fortaleza_bairros_top30'][:10]
            rmf_top10 = artifact['rankings']['rmf_cities_top20'][:10]
            geral_top10 = artifact['rankings']['general_cities_top30'][:10]
            drivers_fortaleza = ', '.join(driver['label'] for driver in fortaleza_top10[0]['explainability']['top_drivers']) if fortaleza_top10 else 'indisponivel'

            brief_lines = [
                '# Hermes Brief de Risco',
                '',
                f'Dados ate DD/MM/AAAA: {data_limit_br}',
                'Fonte: outputs/hermes',
                f'Snapshot consultado: outputs/hermes/history/risk_brief_{timestamp_slug}.md',
                '',
                'Use este arquivo como resposta pronta para chat e Telegram.',
                'Responder com leitura util para gestor: o que aparece no topo, por que importa e o que validar em seguida.',
                'Nao dizer que o ranking esta indisponivel se ele estiver listado abaixo.',
                '',
                '## Leitura rapida',
                '',
                f"- Bairros mais criticos em Fortaleza neste snapshot: {', '.join(entry['name'] for entry in fortaleza_top10[:5])}." if fortaleza_top10 else '- Bairros mais criticos em Fortaleza indisponiveis neste snapshot.',
                f"- Principal driver do topo de Fortaleza: {drivers_fortaleza}." if fortaleza_top10 else '- Principal driver indisponivel.',
                f"- Para gestor: {fortaleza_top10[0]['manager_feedback']['por_que_importa']}" if fortaleza_top10 else '- Leitura gerencial indisponivel.',
                '',
            ]
            brief_lines.extend(_brief_block('Top 10 bairros de Fortaleza', fortaleza_top10))
            brief_lines.extend(_brief_block('Top 10 cidades - Geral', geral_top10))
            brief_lines.extend(_brief_block('Top 10 cidades - RMF', rmf_top10))
            brief_lines.extend(_brief_block('Top 10 cidades - Interior', artifact['rankings']['interior_cities_top30'][:10]))
            brief_lines.extend([
                '## Regra de resposta',
                '',
                '- Quando o usuario pedir bairros mais perigosos de Fortaleza, responder a partir da secao `Top 10 bairros de Fortaleza`.',
                '- Quando o usuario pedir cidades mais criticas, responder a partir das secoes de cidades.',
                '- Sempre citar `Dados ate DD/MM/AAAA` e `Fonte: outputs/hermes`.',
                '- Quando houver pedido de convergencia, validacao independente ou discordancia do modelo, consultar tambem `dados_status_enriquecido_14d_latest.csv`.',
                '- Depois da lista, incluir pelo menos uma frase curta de leitura gerencial com impacto e proxima validacao.',
                '- Template preferido de resposta: `Dados ate`, `Fonte`, `Leitura rapida`, lista principal, `Por que importa`, `Proxima acao`.',
                '- Se precisar aprofundar, usar o arquivo risk_snapshot_latest.md ou risk_snapshot_latest.json.',
                '',
            ])

            brief_payload = '\n'.join(brief_lines).strip() + '\n'
            for path in (brief_path, history_brief_path):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(brief_payload)

            csv_rows = []
            scope_titles = {
                'general_cities_top30': 'cities_general_top30',
                'rmf_cities_top20': 'cities_rmf_top20',
                'interior_cities_top30': 'cities_interior_top30',
                'fortaleza_bairros_top30': 'fortaleza_bairros_top30',
            }
            for scope_key, scope_name in scope_titles.items():
                for entry in artifact['rankings'][scope_key]:
                    top_drivers = entry['explainability']['top_drivers']
                    csv_rows.append({
                        'snapshot_generated_at': artifact['generated_at'],
                        'data_limit': artifact['data_limit'],
                        'source': artifact['source'],
                        'scope': scope_name,
                        'rank': entry['rank'],
                        'name': entry['name'],
                        'name_normalized': entry['name_normalized'],
                        'region': entry['region'],
                        'territorial_level': entry['territorial_level'],
                        'risk_score': entry['risk_score'],
                        'risk_level': entry['risk_level'],
                        'confidence_pct': entry['confidence_pct'],
                        'confidence_label': entry['confidence_label'],
                        'expressiveness_pct': entry['expressiveness_pct'],
                        'expressiveness_label': entry['expressiveness_label'],
                        'recent_cvli_14d': entry['metrics']['recent_cvli_14d'],
                        'recent_cvli_30d': entry['metrics']['recent_cvli_30d'],
                        'historical_cvli': entry['metrics']['historical_cvli'],
                        'neural_score': entry['metrics']['neural_score'],
                        'tension_score': entry['metrics']['tension_score'],
                        'inclusion_score': entry['metrics']['inclusion_score'],
                        'calm_penalty': entry['metrics']['calm_penalty'],
                        'territorial_support_pct': entry['metrics']['territorial_support_pct'],
                        'historical_support_pct': entry['metrics']['historical_support_pct'],
                        'live_support_pct': entry['metrics']['live_support_pct'],
                        'score_z': entry['metrics']['score_z'],
                        'peer_percentile': entry['metrics']['peer_percentile'],
                        'dynamic_window_days': entry['metrics']['dynamic_window_days'],
                        'base_window_days': entry['metrics']['base_window_days'],
                        'historical_fallback': entry['prediction_details']['historical_fallback'],
                        'trend': entry['prediction_details']['trend'],
                        'top_driver_1': top_drivers[0]['label'] if len(top_drivers) > 0 else '',
                        'top_driver_1_strength_pct': top_drivers[0]['strength_pct'] if len(top_drivers) > 0 else '',
                        'top_driver_2': top_drivers[1]['label'] if len(top_drivers) > 1 else '',
                        'top_driver_2_strength_pct': top_drivers[1]['strength_pct'] if len(top_drivers) > 1 else '',
                        'top_driver_3': top_drivers[2]['label'] if len(top_drivers) > 2 else '',
                        'top_driver_3_strength_pct': top_drivers[2]['strength_pct'] if len(top_drivers) > 2 else '',
                        'explainability_summary': entry['explainability']['summary'],
                        'leitura_rapida_gestor': entry['manager_feedback']['leitura_rapida'],
                        'por_que_importa_gestor': entry['manager_feedback']['por_que_importa'],
                        'confianca_limites_gestor': entry['manager_feedback']['confianca_limites'],
                        'expressividade_previsao_gestor': entry['manager_feedback']['expressividade_previsao'],
                        'proxima_acao_gestor': entry['manager_feedback']['proxima_acao'],
                    })

            csv_df = pd.DataFrame(csv_rows)
            for path in (csv_path, history_csv_path):
                csv_df.to_csv(path, index=False, encoding='utf-8-sig')

            separate_csv_specs = {
                'fortaleza_bairros_top30': (
                    os.path.join(output_dir, 'risk_fortaleza_latest.csv'),
                    os.path.join(history_dir, f'risk_fortaleza_{timestamp_slug}.csv'),
                ),
                'rmf_cities_top20': (
                    os.path.join(output_dir, 'risk_rmf_latest.csv'),
                    os.path.join(history_dir, f'risk_rmf_{timestamp_slug}.csv'),
                ),
                'interior_cities_top30': (
                    os.path.join(output_dir, 'risk_interior_latest.csv'),
                    os.path.join(history_dir, f'risk_interior_{timestamp_slug}.csv'),
                ),
            }

            artifact['artifacts']['latest_csv_fortaleza'] = 'outputs/hermes/risk_fortaleza_latest.csv'
            artifact['artifacts']['latest_csv_rmf'] = 'outputs/hermes/risk_rmf_latest.csv'
            artifact['artifacts']['latest_csv_interior'] = 'outputs/hermes/risk_interior_latest.csv'
            artifact['artifacts']['history_csv_fortaleza'] = f'outputs/hermes/history/risk_fortaleza_{timestamp_slug}.csv'
            artifact['artifacts']['history_csv_rmf'] = f'outputs/hermes/history/risk_rmf_{timestamp_slug}.csv'
            artifact['artifacts']['history_csv_interior'] = f'outputs/hermes/history/risk_interior_{timestamp_slug}.csv'

            for scope_key, paths in separate_csv_specs.items():
                scope_name = scope_titles[scope_key]
                scope_df = csv_df[csv_df['scope'] == scope_name].copy()
                for path in paths:
                    scope_df.to_csv(path, index=False, encoding='utf-8-sig')

            for path in (json_path, history_json_path):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(artifact, f, indent=2, ensure_ascii=False)
            self._enqueue_hostinger_risk_sync(artifact)
        except Exception as e:
            print(f"âš ï¸ [Hermes Output] Erro ao gerar artefato: {e}")

    def get_combined_risk(self, exogenous_shocks=None, return_trends=False):
        combined_scores = {}
        trends = {}
        component_details = {}
        
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
                                    sim_impact[i] = impact_value
                                if suppression > 0:
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
            
            # --- INJEÃ‡ÃƒO DE MOMENTUM (V37 Elite) ---
            if channels >= 37:
                # Preenche os canais 33-36 (Momentum calculado on-the-fly)
                x_raw_extended[:, :, 33:37] = momentum_feat[:, :, :4]

            x_final = x_raw_extended[:, -window:, :channels].copy()
            
            # â­ NORMALIZAÃ‡ÃƒO Z-SCORE (V37 Elite)
            for c in range(channels):
                # Evitamos normalizar canais binÃ¡rios/sazonais fixos (3-22, 29, 30, 32)
                # Normalizamos apenas: Crime(0), VeÃ­culos(1), TensÃ£o(2), Intel(27), Global(28), Chuva(31), Momentum(33-36)
                if c in [0, 1, 2, 24, 27, 28, 31, 33, 34, 35, 36]:
                    m_c = x_final[:, :, c].mean()
                    s_c = x_final[:, :, c].std() + 1e-6
                    x_final[:, :, c] = (x_final[:, :, c] - m_c) / s_c

            active_window = cp.get('dynamic_window', window)
            if active_window and active_window < window:
                x_final[:, :window - active_window, :29] = 0.0

            # Pad se o modelo exigir mais canais do que o dataset possui (ex: MemPalace e CVLI Ratio de Fortaleza)
            if x_final.shape[2] < channels:
                pad_width = channels - x_final.shape[2]
                x_final = np.pad(x_final, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

            x = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            adj = self._norm_adj(data['adj_geo'], data['adj_conflict'])
            
            with torch.no_grad():
                out = model(x, adj).squeeze().cpu().numpy()
            
            norm_neural = (out - out.min()) / (out.max() - out.min() + 1e-6)
            tension_vec = data['nodes_gdf']['tension_index'].values.astype(float)
            norm_tension = (tension_vec - tension_vec.min()) / (tension_vec.max() - tension_vec.min() + 1e-6)

            inclusion_horizon = 28 if region == 'interior' else 14
            current_cvli_recent = x_raw_extended[:, -inclusion_horizon:, 0].sum(axis=1)
            current_cvli_30d = x_raw_extended[:, -30:, 0].sum(axis=1)

            historical_col = 'total_cvli' if 'total_cvli' in data['nodes_gdf'].columns else 'recent_cvli'
            historical_cvli = data['nodes_gdf'][historical_col].fillna(0).values.astype(float)

            # TensÃ£o territorial nÃ£o deve sozinha promover bairros frios.
            # Exigimos atividade CVLI recente real ou lastro histÃ³rico relevante.
            historical_support = np.clip((historical_cvli - 20.0) / 40.0, 0, 1)
            live_support = np.maximum(
                np.clip(current_cvli_recent / 1.0, 0, 1),
                np.clip(current_cvli_30d / 2.0, 0, 1)
            )
            territorial_support = np.maximum(
                historical_support,
                live_support
            )
            norm_tension = norm_tension * territorial_support

            recent_crime_signal = np.clip(x_raw_extended[:, -inclusion_horizon:, 0].sum(axis=1), 0, 2) / 2.0
            neighbor_signal = np.clip((data['adj_geo'].dot((sim_impact > 0).astype(float))) * (cp.get('tag_bias_neighbor', 0.6)/0.6), 0, 1)
            # NÃ³ com evento direto tambÃ©m recebe inclusÃ£o mÃ¡xima (nÃ£o apenas seus vizinhos)
            own_event_signal = (sim_impact > 0).astype(float)
            inclusion_signal = np.clip(np.maximum.reduce([recent_crime_signal, neighbor_signal, own_event_signal]), 0, 1)
            calm_signal = np.clip(-momentum_feat[:, -1, 3], 0, 30) / 30.0

            if cp.get('use_historical_fallback') and cp.get('historical_top10'):
                hist_norms = set(normalize_name(n) for n in cp['historical_top10'])
                hist_signal = np.array([1.0 if normalize_name(str(row['name'])) in hist_norms else 0.0 for _, row in data['nodes_gdf'].iterrows()])
                final_logic = (0.50 * norm_neural) + (0.20 * norm_tension) + (0.10 * inclusion_signal) + (0.20 * hist_signal) - (0.10 * calm_signal)
            else:
                # Foco Total em CVLI (AtualizaÃ§Ã£o ProduÃ§Ã£o 2026-05-21)
                neural_weight = 0.50 if region != 'interior' else 0.35
                tension_weight = 0.10
                inclusion_weight = 0.40 # O Gatilho de Conflito agora manda!
                
                # Fator Anti-AmnÃ©sia: 
                # Se a Ã¡rea estÃ¡ em calmaria severa, a rede neural perde a "certeza absoluta" baseada no passado.
                # calm_signal vai de 0.0 (violÃªncia) a 1.0 (calma de 30 dias).
                decay_factor = 1.0 - (calm_signal * 0.5) 
                
                final_logic = (
                    (neural_weight * norm_neural * decay_factor)
                    + (tension_weight * norm_tension)
                    + (inclusion_weight * inclusion_signal)
                )
            
            out_norm = np.clip(final_logic, 0, 1) * 100.0
            for i, row in data['nodes_gdf'].iterrows():
                name_key = normalize_name(str(row['name']))
                if self._node_owners.get(name_key, region) == region:
                    score_value = float(out_norm[i])
                    combined_scores[name_key] = score_value
                    if return_trends:
                        trends[name_key] = 'stable'
                    component_details[name_key] = {
                        'name': str(row['name']),
                        'name_normalized': name_key,
                        'region': region,
                        'territorial_level': 'bairro' if region == 'fortaleza' else 'cidade',
                        'score_raw': score_value,
                        'neural_score': float(norm_neural[i] * 100.0),
                        'tension_score': float(norm_tension[i] * 100.0),
                        'inclusion_score': float(inclusion_signal[i] * 100.0),
                        'calm_penalty': float(calm_signal[i] * 100.0),
                        'recent_cvli_14d': int(current_cvli_recent[i]),
                        'recent_cvli_30d': int(current_cvli_30d[i]),
                        'historical_cvli': float(historical_cvli[i]),
                        'territorial_support_pct': float(territorial_support[i] * 100.0),
                        'historical_support_pct': float(historical_support[i] * 100.0),
                        'live_support_pct': float(live_support[i] * 100.0),
                        'dynamic_window': active_window if active_window else window,
                        'base_window': window,
                        'historical_fallback': bool(cp.get('use_historical_fallback')),
                        'trend': 'stable',
                    }
        
        # --- REFINAMENTO SENTINELA V3 (LGBM) ---
        if self.champion_challenger is not None:
            try:
                combined_scores = self.champion_challenger.apply(combined_scores)
            except Exception as e:
                print(f"âš ï¸ [Sentinela V3] Falha ao aplicar refinamento: {e}")
        # ---------------------------------------

        self._write_hermes_outputs(combined_scores, component_details)
        self._log_predict_p10(combined_scores)
        return (combined_scores, trends) if return_trends else combined_scores

    def _log_predict_p10(self, scores_map):
        """Registra o top-10 predito por regiÃ£o a cada cÃ¡lculo de risco para anÃ¡lise e validaÃ§Ã£o."""
        try:
            log_path = os.path.join(self.root, 'logs', 'predict_p10.jsonl')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            record = {'timestamp': datetime.now().isoformat(timespec='seconds'), 'regions': {}}

            for region, spec in self.specialists.items():
                region_nodes = {
                    normalize_name(str(row['name']))
                    for _, row in spec['data']['nodes_gdf'].iterrows()
                }
                region_scores = {n: s for n, s in scores_map.items() if n in region_nodes}
                top10 = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                record['regions'][region] = [{'name': n, 'score': round(s, 2)} for n, s in top10]

            global_top10 = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:10]
            record['global_top10'] = [{'name': n, 'score': round(s, 2)} for n, s in global_top10]

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"âš ï¸ [P10 Log] Erro ao registrar prediÃ§Ã£o: {e}")

    def _norm_adj(self, geo, conf):
        def n(a):
            s = np.array(a.sum(1)); d = np.power(s, -0.5).flatten(); d[np.isinf(d)]=0.; m=np.diag(d)
            return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(self.device)
        return [n(geo), n(conf)]
