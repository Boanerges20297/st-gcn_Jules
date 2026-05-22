"""
ModelCalibrator — Auto-ajuste de parâmetros de inferência do StateOrchestrator.

Quando P20/P10 fica abaixo do limite configurado em _CONFIDENCE_THRESHOLDS,
o calibrador aplica ajustes graduais nos parâmetros do orchestrator sem retraining.

Estratégia:
- Cada degradação detectada aplica 1 passo de ajuste
- Máximo de 5 passos acumulados por região (aumentado de 3)
- Após 5 passos sem melhora: alerta CRITICAL pedindo intervenção manual
- Quando a métrica volta ao normal: rollback completo para defaults

Parâmetros ajustados:
  tension_factor  0.80 → max 3.00  — tensão de facção pesa mais no ranking (base mais alta)
  tag_bias_direct 2.00 → max 5.00  — gatilho INTEL_TRIGGER empurra o nó para o topo
  tag_bias_neighbor 0.60 → max 1.50 — influência de vizinhos
  norm_neural_weight 0.20 → max 0.50 — peso do componente neural no blend final
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Limites absolutos para não degradar a inferência
_PARAM_LIMITS = {
    'tension_factor':       (0.10, 3.00),  # Permite redução drástica se houver ruído
    'min_risk':             (15.0, 30.0),
    'tag_bias_direct':      (1.00, 5.00),  
    'tag_bias_neighbor':    (0.30, 1.50),
    'norm_neural_weight':   (0.20, 0.90),  # Permite que a rede neural domine quase totalmente
}

# Passos calibrados: agora priorizamos o peso neural em cenários de incerteza
_STEPS = {
    'p20': {
        'tension_factor':       -0.10,  # Reduz peso estático se a precisão cair
        'tag_bias_direct':      +0.20,
        'tag_bias_neighbor':    +0.10,
        'min_risk':              0.0,
        'norm_neural_weight':   +0.15,  # Aumenta peso da rede neural
    },
    'p10': {
        'tension_factor':       -0.15,
        'tag_bias_direct':      +0.30,
        'tag_bias_neighbor':    +0.15,
        'min_risk':              0.0,
        'norm_neural_weight':   +0.20,
    },
    'faction_coverage': {
        'tension_factor':       +0.20,
        'tag_bias_direct':      +0.50,
        'tag_bias_neighbor':    +0.10,
        'min_risk':              0.0,
        'norm_neural_weight':   +0.05,
    },
}


class ModelCalibrator:
    """
    Gerencia calibração automática dos parâmetros de inferência por região.
    Estado persiste em data/calibration_state.json.
    """

    MAX_STEPS = 5

    def __init__(self, base_dir: str, health_monitor=None):
        self.base_dir = base_dir
        self.health_monitor = health_monitor
        self.state_path = os.path.join(base_dir, 'data', 'calibration_state.json')
        # state: { region: { 'steps': int, 'history': [...] } }
        self.state = self._load_state()

    # ------------------------------------------------------------------ #
    # Persistência                                                         #
    # ------------------------------------------------------------------ #

    def _load_state(self) -> Dict:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Calibrator: erro ao carregar estado: {e}")
        return {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Calibrator: erro ao salvar estado: {e}")

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def on_degradation(self, orchestrator, region: str, metric: str,
                       current_value: float, threshold: float):
        """
        Chamado quando uma métrica fica abaixo do limite.
        Aplica 1 passo de ajuste e loga no health_monitor.
        """
        reg_state = self.state.setdefault(region, {'steps': 0, 'history': []})
        current_steps = reg_state['steps']

        if current_steps >= self.MAX_STEPS:
            # Já no máximo — dispara CRITICAL se ainda não foi disparado hoje
            self._alert_critical(region, metric, current_value, current_steps)
            return

        # Calcula novos parâmetros
        step = _STEPS.get(metric, _STEPS['p20'])
        old_cp = dict(orchestrator.calib_params.get(region, {}))
        new_cp = {}

        for param, delta in step.items():
            old_val = old_cp.get(param, self._default(param))
            lo, hi = _PARAM_LIMITS[param]
            new_val = round(max(lo, min(hi, old_val + delta)), 4)
            new_cp[param] = new_val

        # Aplica ao orchestrator
        orchestrator.calib_params[region].update(new_cp)

        reg_state['steps'] += 1
        reg_state['history'].append({
            'timestamp': datetime.now().isoformat(),
            'trigger': f"{region}.{metric}={current_value*100:.1f}% < {threshold*100:.0f}%",
            'step': reg_state['steps'],
            'old_params': old_cp,
            'new_params': new_cp,
        })
        self._save_state()

        msg = (
            f"[Auto-calibração {region.upper()}] Passo {reg_state['steps']}/{self.MAX_STEPS} — "
            f"{metric.upper()}={current_value*100:.1f}% abaixo de {threshold*100:.0f}%. "
            f"tension_factor={new_cp.get('tension_factor','?')}, "
            f"tag_bias={new_cp.get('tag_bias_direct','?')}"
        )
        print(f"🔧 {msg}")
        logger.info(msg)

        if self.health_monitor:
            self.health_monitor.add_alert(
                alert_type=f'auto_calibration_{region}',
                severity='MEDIUM',
                message=msg,
                details={
                    'region': region,
                    'metric': metric,
                    'step': reg_state['steps'],
                    'new_params': new_cp,
                }
            )

    def on_recovery(self, orchestrator, region: str, metric: str,
                    current_value: float):
        """
        Chamado quando a métrica volta ao normal.
        Faz rollback COMPLETO para os valores padrão originais.
        """
        reg_state = self.state.get(region, {})
        steps_applied = reg_state.get('steps', 0)
        if steps_applied <= 0:
            return

        # Captura parâmetros atuais (antes do rollback) para o log
        params_before = dict(orchestrator.calib_params.get(region, {}))

        defaults = {
            'tension_factor':     0.80,
            'min_risk':          30.0,
            'tag_bias_direct':    2.00,
            'tag_bias_neighbor':  0.60,
            'norm_neural_weight': 0.20,
        }
        orchestrator.calib_params[region].update(defaults)

        # Zera o contador de passos
        reg_state['steps'] = 0
        reg_state['history'].append({
            'timestamp': datetime.now().isoformat(),
            'event': 'full_rollback',
            'trigger': f'{region}.{metric}={current_value*100:.1f}% recuperado acima do limite',
            'steps_reverted': steps_applied,
            'params_before': params_before,
            'params_after': defaults,
        })
        self._save_state()

        msg = (
            f"[Rollback {region.upper()}] {metric.upper()}={current_value*100:.1f}% — "
            f"recuperado. {steps_applied} passo(s) revertidos, "
            f"parâmetros restaurados ao estado original."
        )
        print(f"✅ {msg}")
        logger.info(msg)

        if self.health_monitor:
            self.health_monitor.add_alert(
                alert_type=f'calibration_rollback_{region}',
                severity='LOW',
                message=msg,
                details={
                    'region': region,
                    'metric': metric,
                    'steps_reverted': steps_applied,
                    'params_restored': defaults,
                }
            )

    def get_status(self) -> Dict:
        """Retorna estado atual de calibração de todas as regiões."""
        return {
            region: {
                'steps': info.get('steps', 0),
                'max_steps': self.MAX_STEPS,
                'is_degraded': info.get('steps', 0) > 0,
                'is_critical': info.get('steps', 0) >= self.MAX_STEPS,
                'last_event': info['history'][-1] if info.get('history') else None,
            }
            for region, info in self.state.items()
        }

    # ------------------------------------------------------------------ #
    # Helpers internos                                                     #
    # ------------------------------------------------------------------ #

    def reapply_on_startup(self, orchestrator):
        """
        Reaplica os parâmetros salvos ao orchestrator após reinício do servidor.
        Lê calibration_state.json e recalcula os parâmetros acumulados por região.
        """
        for region, info in self.state.items():
            steps = info.get('steps', 0)
            if steps <= 0:
                continue
            defaults = {
                'tension_factor': 0.80,
                'min_risk': 30.0,
                'tag_bias_direct': 2.00,
                'tag_bias_neighbor': 0.60,
                'norm_neural_weight': 0.20,
            }
            # Recalcula parâmetros acumulados (p20 steps por simplicidade)
            step = _STEPS['p20']
            new_cp = {}
            for param, delta in step.items():
                if param not in _PARAM_LIMITS:
                    continue
                lo, hi = _PARAM_LIMITS[param]
                base = defaults.get(param, lo)
                new_val = round(max(lo, min(hi, base + delta * steps)), 4)
                new_cp[param] = new_val
            if region in orchestrator.calib_params:
                orchestrator.calib_params[region].update(new_cp)
                print(f"🔄 [Calibrator] Reapplied {steps} step(s) for {region}: {new_cp}")

    def _default(self, param: str) -> float:  # noqa: E301
        defaults = {
            'tension_factor':     0.80,
            'min_risk':           30.0,
            'tag_bias_direct':    2.00,
            'tag_bias_neighbor':  0.60,
            'norm_neural_weight': 0.20,
        }
        return defaults.get(param, 0.0)

    def _alert_critical(self, region: str, metric: str,
                        current_value: float, steps: int):
        """Dispara CRITICAL quando o máximo de passos foi atingido sem melhora."""
        if not self.health_monitor:
            return
        from datetime import timedelta
        now = datetime.now()
        # Supressão de 12h para não repetir
        cutoff = (now - timedelta(hours=12)).isoformat()
        alert_type = f'calibration_maxed_{region}'
        already = any(
            a['type'] == alert_type and a['timestamp'] >= cutoff
            for a in self.health_monitor.alerts_history
        )
        if not already:
            msg = (
                f"INTERVENÇÃO MANUAL NECESSÁRIA — {region.upper()}: "
                f"{metric.upper()}={current_value*100:.1f}% após {steps} ajustes automáticos. "
                f"O modelo pode precisar de retreinamento."
            )
            print(f"🚨 {msg}")
            self.health_monitor.add_alert(
                alert_type=alert_type,
                severity='CRITICAL',
                message=msg,
                details={
                    'region': region,
                    'metric': metric,
                    'current_value': current_value,
                    'steps_applied': steps,
                    'recommendation': 'Retreinar o modelo regional ou atualizar os dados de treinamento.'
                }
            )
