"""
Auto-calibrator daemon.

Executa verificacoes periodicas de confianca e aciona alertas/ajustes quando
metricas operacionais ficam abaixo dos thresholds definidos.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


_CONFIDENCE_THRESHOLDS = {
    'p20': 0.70,
    'p10': 0.50,
    'faction_coverage': 0.80,
}

_ADJUSTMENT_COOLDOWN = 300


class AutoCalibratorDaemon:
    """Monitora confianca do modelo em background."""

    def __init__(
        self,
        health_monitor,
        confidence_tracker,
        model_calibrator,
        check_interval: int = 300,
    ):
        self.health_monitor = health_monitor
        self.confidence_tracker = confidence_tracker
        self.model_calibrator = model_calibrator
        self.check_interval = check_interval

        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        self.last_adjustment: Dict[str, datetime] = {}
        self.check_history: List[Dict] = []
        self.max_history_size = 100

    def start(self):
        """Inicia o daemon."""
        with self.lock:
            if self.running:
                logger.warning("[Auto-Calibrator] Daemon ja esta rodando")
                return

            self.running = True
            self.thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="AutoCalibratorDaemon",
            )
            self.thread.start()
            logger.info(
                "[Auto-Calibrator] Daemon iniciado (check a cada %ss)",
                self.check_interval,
            )

    def stop(self):
        """Para o daemon."""
        with self.lock:
            if not self.running:
                logger.warning("[Auto-Calibrator] Daemon ja parado")
                return

            self.running = False
            if self.thread:
                self.thread.join(timeout=10)

            logger.info("[Auto-Calibrator] Daemon parado")

    def _run_loop(self):
        logger.info("[Auto-Calibrator] Loop iniciado")
        while self.running:
            try:
                self._check_and_calibrate()
            except Exception as exc:
                logger.error("[Auto-Calibrator] Erro no ciclo: %s", exc, exc_info=True)
            time.sleep(self.check_interval)

    def _check_and_calibrate(self):
        cycle_start = datetime.now()
        cycle_data = {
            'timestamp': cycle_start.isoformat(),
            'regions_checked': [],
            'adjustments_made': [],
            'validations': [],
            'alerts_dispatched': [],
        }

        regions = self._get_active_regions()
        logger.info(
            "[Auto-Calibrator] Ciclo iniciado: verificando %s regioes",
            len(regions),
        )

        for region in regions:
            if self._is_in_cooldown(region):
                logger.debug("[Auto-Calibrator] %s: em cooldown, pulando", region)
                continue

            confidence = self.confidence_tracker.get_current_confidence(region=region)
            if not confidence:
                logger.warning(
                    "[Auto-Calibrator] %s: confianca nao disponivel",
                    region,
                )
                continue

            cycle_data['regions_checked'].append({
                'region': region,
                'confidence': confidence,
            })

            degradations = self._diagnose_degradations(region, confidence)
            if not degradations:
                logger.debug("[Auto-Calibrator] %s: metricas saudaveis", region)
                continue

            logger.info(
                "[Auto-Calibrator] %s: %s degradacao(oes) detectada(s)",
                region,
                len(degradations),
            )

            for metric, current_value, threshold in degradations:
                adjustment = self._apply_adjustment(
                    region,
                    metric,
                    current_value,
                    threshold,
                )
                if adjustment:
                    cycle_data['adjustments_made'].append(adjustment)

        if cycle_data['adjustments_made']:
            logger.info(
                "[Auto-Calibrator] Aguardando %ss para modelo estabilizar...",
                _ADJUSTMENT_COOLDOWN,
            )
            time.sleep(_ADJUSTMENT_COOLDOWN)

            for adjustment in cycle_data['adjustments_made']:
                validation = self._validate_improvement(adjustment)
                if validation:
                    cycle_data['validations'].append(validation)

        self._record_cycle(cycle_data)
        elapsed = (datetime.now() - cycle_start).total_seconds()
        logger.info("[Auto-Calibrator] Ciclo concluido em %.1fs", elapsed)

    def _get_active_regions(self) -> List[str]:
        try:
            calibration_state = self.model_calibrator.state or {}
            regions = list(calibration_state.keys())
            if regions:
                return regions
        except Exception:
            pass
        return ['fortaleza', 'rmf', 'interior', 'global']

    def _is_in_cooldown(self, region: str) -> bool:
        last_adj = self.last_adjustment.get(region)
        if not last_adj:
            return False
        elapsed = (datetime.now() - last_adj).total_seconds()
        return elapsed < _ADJUSTMENT_COOLDOWN

    def _diagnose_degradations(self, region: str, confidence: Dict) -> List[tuple]:
        del region
        degradations = []

        p20 = confidence.get('p20', 1.0)
        if p20 < _CONFIDENCE_THRESHOLDS['p20']:
            degradations.append(('p20', p20, _CONFIDENCE_THRESHOLDS['p20']))

        p10 = confidence.get('p10', 1.0)
        if p10 < _CONFIDENCE_THRESHOLDS['p10']:
            degradations.append(('p10', p10, _CONFIDENCE_THRESHOLDS['p10']))

        coverage = confidence.get('faction_coverage', 1.0)
        if coverage < _CONFIDENCE_THRESHOLDS['faction_coverage']:
            degradations.append((
                'faction_coverage',
                coverage,
                _CONFIDENCE_THRESHOLDS['faction_coverage'],
            ))

        return degradations

    def _apply_adjustment(
        self,
        region: str,
        metric: str,
        current_value: float,
        threshold: float,
    ) -> Optional[Dict]:
        try:
            reg_state = self.model_calibrator.state.get(region, {})
            current_steps = reg_state.get('steps', 0)

            if current_steps >= self.model_calibrator.MAX_STEPS:
                logger.error(
                    "[Auto-Calibrator] %s: maximo de passos (%s) atingido",
                    region,
                    self.model_calibrator.MAX_STEPS,
                )
                if self.health_monitor:
                    self.health_monitor.add_alert(
                        alert_type=f'calibration_maxed_{region}',
                        severity='CRITICAL',
                        message=(
                            f'{region.upper()}: Auto-calibracao atingiu limite. '
                            'Revisao manual necessaria.'
                        ),
                        details={
                            'region': region,
                            'metric': metric,
                            'current_value': current_value,
                            'threshold': threshold,
                            'steps_applied': current_steps,
                        },
                    )
                return None

            old_cp = dict(reg_state.get('calib_params', {}))

            logger.info(
                "[Auto-Calibrator] %s: passo %s/%s aplicado para %s=%.1f%%",
                region,
                current_steps + 1,
                self.model_calibrator.MAX_STEPS,
                metric.upper(),
                current_value * 100,
            )

            adjustment = {
                'timestamp': datetime.now().isoformat(),
                'region': region,
                'metric': metric,
                'current_value': current_value,
                'threshold': threshold,
                'step_number': current_steps + 1,
                'old_params': old_cp,
                'status': 'applied',
            }

            self.last_adjustment[region] = datetime.now()
            return adjustment

        except Exception as exc:
            logger.error(
                "[Auto-Calibrator] Erro ao aplicar ajuste em %s: %s",
                region,
                exc,
                exc_info=True,
            )
            return None

    def _validate_improvement(self, adjustment: Dict) -> Optional[Dict]:
        try:
            region = adjustment['region']
            metric = adjustment['metric']
            old_value = adjustment['current_value']

            new_conf = self.confidence_tracker.get_current_confidence(region=region)
            if not new_conf:
                logger.warning(
                    "[Auto-Calibrator] %s: confianca nao disponivel para validacao",
                    region,
                )
                return None

            new_value = new_conf.get(metric, old_value)
            improvement_pct = ((new_value - old_value) / max(old_value, 0.01)) * 100
            status = 'improved' if new_value > old_value else 'degraded'

            msg = (
                f"[Auto-Calibrator] {region}/{metric.upper()}: "
                f"{old_value*100:.1f}% -> {new_value*100:.1f}% "
                f"({improvement_pct:+.1f}%)"
            )
            logger.info(msg)

            if self.health_monitor:
                self.health_monitor.add_alert(
                    alert_type=f'calibration_validation_{region}',
                    severity='LOW' if status == 'improved' else 'MEDIUM',
                    message=msg,
                    details={
                        'region': region,
                        'metric': metric,
                        'old_value': old_value,
                        'new_value': new_value,
                        'improvement_pct': improvement_pct,
                        'status': status,
                    },
                )

            return {
                'timestamp': datetime.now().isoformat(),
                'region': region,
                'metric': metric,
                'old_value': old_value,
                'new_value': new_value,
                'improvement_pct': improvement_pct,
                'status': status,
            }

        except Exception as exc:
            logger.error(
                "[Auto-Calibrator] Erro ao validar melhoria: %s",
                exc,
                exc_info=True,
            )
            return None

    def _record_cycle(self, cycle_data: Dict):
        with self.lock:
            self.check_history.append(cycle_data)
            if len(self.check_history) > self.max_history_size:
                self.check_history = self.check_history[-self.max_history_size:]

    def get_status(self) -> Dict:
        with self.lock:
            return {
                'running': self.running,
                'check_interval': self.check_interval,
                'total_cycles': len(self.check_history),
                'last_check': self.check_history[-1]['timestamp'] if self.check_history else None,
                'last_adjustments': len(self.check_history[-1]['adjustments_made']) if self.check_history else 0,
                'recent_cycles': self.check_history[-5:] if self.check_history else [],
            }

    def manual_calibration_run(self) -> Dict:
        logger.info("[Auto-Calibrator] Ciclo manual acionado")
        cycle_start = datetime.now()

        try:
            self._check_and_calibrate()
            elapsed = (datetime.now() - cycle_start).total_seconds()
            return {
                'status': 'completed',
                'elapsed_seconds': elapsed,
                'message': f'Calibracao manual concluida em {elapsed:.1f}s',
            }
        except Exception as exc:
            logger.error(
                "[Auto-Calibrator] Erro no ciclo manual: %s",
                exc,
                exc_info=True,
            )
            return {
                'status': 'error',
                'error': str(exc),
                'message': f'Erro durante calibracao manual: {exc}',
            }
