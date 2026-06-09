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

from src.agent.calibration_memory import record_validation_outcome


logger = logging.getLogger(__name__)


_CONFIDENCE_THRESHOLDS = {
    'p20': 0.70,
    'p10': 0.50,
    'faction_coverage': 0.80,
}

_ADJUSTMENT_COOLDOWN = 300
_SEMANTIC_REVIEW_THRESHOLDS = {
    'p20': 0.78,
    'p10': 0.58,
    'faction_coverage': 0.90,
}


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
        self.base_dir = getattr(model_calibrator, 'base_dir', None)
        self.check_interval = check_interval
        self.orchestrator_getter = None
        self.intervention_callback = None
        self._agent_manager = None

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
            semantic_review = None
            if degradations or self._should_run_semantic_review(confidence):
                try:
                    semantic_review = self._run_agent_review(region, confidence)
                    if semantic_review:
                        cycle_data['alerts_dispatched'].append({
                            'region': region,
                            'type': 'agent_review',
                            'timestamp': datetime.now().isoformat(),
                            'should_intervene': semantic_review.get('should_intervene', False),
                            'analysis': semantic_review.get('data_analysis', {}),
                        })
                except Exception as exc:
                    logger.error("[Auto-Calibrator] %s: revisão do agente falhou: %s", region, exc)
                    if self.health_monitor:
                        self.health_monitor.add_alert(
                            alert_type=f'agent_review_failed_{region}',
                            severity='HIGH',
                            message=f'{region.upper()}: revisão semântica do agente falhou: {exc}',
                            details={'region': region},
                        )

            if semantic_review and semantic_review.get('should_intervene') and not degradations:
                p20_value = confidence.get('p20', 1.0)
                degradations.append(('p20', min(p20_value, _CONFIDENCE_THRESHOLDS['p20'] - 0.01), _CONFIDENCE_THRESHOLDS['p20']))

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
                    semantic_review=semantic_review,
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

    def _get_orchestrator(self):
        if callable(self.orchestrator_getter):
            try:
                return self.orchestrator_getter()
            except Exception as exc:
                logger.error("[Auto-Calibrator] Falha ao obter orchestrator: %s", exc)
        return None

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

    def _should_run_semantic_review(self, confidence: Dict) -> bool:
        p20 = confidence.get('p20', 1.0)
        p10 = confidence.get('p10', 1.0)
        coverage = confidence.get('faction_coverage', 1.0)
        convergence_gap = abs(p10 - p20)
        return (
            p20 < _SEMANTIC_REVIEW_THRESHOLDS['p20']
            or p10 < _SEMANTIC_REVIEW_THRESHOLDS['p10']
            or coverage < _SEMANTIC_REVIEW_THRESHOLDS['faction_coverage']
            or convergence_gap > 0.22
        )

    def _build_agent_payload(self, region: str, confidence: Dict) -> Optional[Dict]:
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            return None

        scores_map = orchestrator.get_combined_risk()
        top_predictions = [
            {'name': name, 'score': round(float(score), 4)}
            for name, score in sorted(scores_map.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
        return {
            'raw_stgcn_data': {
                'timestamp': datetime.now().isoformat(),
                'region': region,
                'confidence_scores': {
                    'p10': confidence.get('p10'),
                    'p20': confidence.get('p20'),
                    'faction_coverage': confidence.get('faction_coverage'),
                },
                'top_predictions': top_predictions,
                'combined_risk_sample_size': len(scores_map),
            },
            'user_profile': {
                'region': region.upper(),
                'focus': 'CVLI',
                'historical_alerts': len(self.health_monitor.get_active_alerts()) if self.health_monitor else 0,
                'trigger': 'semantic_deviation_or_convergence_error',
            },
        }

    def _run_agent_review(self, region: str, confidence: Dict) -> Optional[Dict]:
        payload = self._build_agent_payload(region, confidence)
        if payload is None:
            return None

        if self._agent_manager is None:
            from src.agent.multi_agent_system import GeneralManagerAgent
            self._agent_manager = GeneralManagerAgent(base_dir=self.base_dir)

        return self._agent_manager.process_and_calibrate(
            payload['raw_stgcn_data'],
            payload['user_profile'],
        )

    def _notify_intervention(self, event: Dict):
        if callable(self.intervention_callback):
            try:
                self.intervention_callback(event)
            except Exception as exc:
                logger.error("[Auto-Calibrator] Falha no callback de intervenção: %s", exc)

    def _apply_adjustment(
        self,
        region: str,
        metric: str,
        current_value: float,
        threshold: float,
        semantic_review: Optional[Dict] = None,
    ) -> Optional[Dict]:
        try:
            orchestrator = self._get_orchestrator()
            if orchestrator is None:
                raise RuntimeError('Orchestrator indisponível para aplicar calibração')

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

            old_cp = dict(orchestrator.calib_params.get(region, {}))

            self.model_calibrator.on_degradation(
                orchestrator,
                region,
                metric,
                current_value,
                threshold,
            )
            new_cp = dict(orchestrator.calib_params.get(region, {}))

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
                'new_params': new_cp,
                'semantic_review': semantic_review,
                'status': 'applied',
            }

            self._notify_intervention({
                'status': 'success',
                'region': region,
                'metric': metric,
                'current_value': current_value,
                'threshold': threshold,
                'old_params': old_cp,
                'new_params': new_cp,
                'agent_review': semantic_review,
                'timestamp': adjustment['timestamp'],
            })

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

            validation = {
                'timestamp': datetime.now().isoformat(),
                'region': region,
                'metric': metric,
                'old_value': old_value,
                'new_value': new_value,
                'improvement_pct': improvement_pct,
                'status': status,
            }
            if self.base_dir:
                try:
                    record_validation_outcome(self.base_dir, adjustment, validation)
                except Exception as exc:
                    logger.warning("[Auto-Calibrator] Falha ao registrar memoria semantica: %s", exc)
            return validation

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
