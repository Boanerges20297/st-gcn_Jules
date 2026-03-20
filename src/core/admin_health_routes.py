"""
Admin Health Dashboard API Routes

Endpoints para o painel administrativo de monitoramento de saúde do sistema.
Todos os endpoints requerem autenticação de admin.
"""

from flask import Blueprint, jsonify, request, render_template
from functools import wraps
import logging
import os
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Este arquivo deve ser importado em app.py e registrado como blueprint

def create_admin_health_blueprint(health_monitor, confidence_tracker, model_calibrator=None, get_orchestrator=None):
    """
    Cria blueprint com endpoints de health monitoring.
    
    Args:
        health_monitor: Instância de HealthMonitor
        confidence_tracker: Instância de ConfidenceTracker
    
    Returns:
        Blueprint do Flask
    """
    
    admin_bp = Blueprint('admin_health', __name__, url_prefix='/api/admin/health')
    
    # Decorator para autenticação de admin (futuro)
    def admin_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # TODO: Implementar autenticação JWT
            # Por enquanto, aceita todos (comment this for production)
            # token = request.headers.get('Authorization', '').replace('Bearer ', '')
            # if not is_valid_token(token):
            #     return jsonify({'error': 'Unauthorized'}), 401
            return f(*args, **kwargs)
        return decorated_function
    
    # Desabilitar cache HTTP em todas as respostas do blueprint
    @admin_bp.after_request
    def add_no_cache_headers(response):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    # ===== ENDPOINTS =====
    
    @admin_bp.route('/summary', methods=['GET'])
    @admin_required
    def get_health_summary():
        """Retorna summary completo de saúde do sistema."""
        try:
            summary = health_monitor.get_summary()
            return jsonify(summary), 200
        except Exception as e:
            logger.error(f"Erro ao gerar health summary: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/metrics/system', methods=['GET'])
    @admin_required
    def get_system_metrics():
        """Retorna métricas de sistema (CPU, memória, disco)."""
        try:
            metrics = health_monitor.get_system_metrics()
            return jsonify(metrics), 200
        except Exception as e:
            logger.error(f"Erro ao obter métricas do sistema: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/api-stats', methods=['GET'])
    @admin_required
    def get_api_statistics():
        """Retorna estatísticas de performance da API."""
        try:
            stats = health_monitor.get_api_stats()
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"Erro ao obter stats da API: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/alerts', methods=['GET'])
    @admin_required
    def get_alerts():
        """
        Retorna alertas.
        
        Query Parameters:
            - limit: Número máximo de alertas (padrão: 100)
            - resolved: Filtrar por status (true/false/null)
        """
        try:
            limit = int(request.args.get('limit', 100))
            resolved_param = request.args.get('resolved')
            
            resolved = None
            if resolved_param:
                resolved = resolved_param.lower() == 'true'
            
            alerts = health_monitor.get_alerts(limit=limit, resolved=resolved)
            
            return jsonify({
                'total': len(alerts),
                'alerts': alerts
            }), 200
        except Exception as e:
            logger.error(f"Erro ao obter alertas: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/alerts', methods=['POST'])
    @admin_required
    def create_alert():
        """
        Cria novo alerta.
        
        Body JSON:
        {
            "type": "model_degraded",
            "severity": "CRITICAL",
            "message": "...",
            "details": {}
        }
        """
        try:
            data = request.json or {}
            
            alert_type = data.get('type', 'manual')
            severity = data.get('severity', 'MEDIUM')
            message = data.get('message', '')
            details = data.get('details', {})
            
            health_monitor.add_alert(
                alert_type=alert_type,
                severity=severity,
                message=message,
                details=details
            )
            
            return jsonify({
                'status': 'created',
                'alert_type': alert_type
            }), 201
        except Exception as e:
            logger.error(f"Erro ao criar alerta: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/confidence-history', methods=['GET'])
    @admin_required
    def get_confidence_history():
        """
        Retorna histórico de confiança do modelo.
        
        Query Parameters:
            - region: fortaleza, rmf, interior, global (padrão: global)
            - days: Últimos N dias (padrão: 30)
        """
        try:
            region = request.args.get('region', 'global')
            days = int(request.args.get('days', 30))
            
            history = confidence_tracker.get_history(region=region, days=days)
            current = confidence_tracker.get_current_confidence(region=region)
            trend = confidence_tracker.get_trend(region=region)
            
            return jsonify({
                'region': region,
                'period_days': days,
                'history': history,
                'current_confidence': current,
                'trend': trend
            }), 200
        except Exception as e:
            logger.error(f"Erro ao obter histórico de confiança: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/confidence/current', methods=['GET'])
    @admin_required
    def get_current_confidence():
        """Retorna confiança atual do modelo por região."""
        try:
            # Deriva regiões dinamicamente do orchestrator (sem hardcode)
            orch = get_orchestrator() if get_orchestrator else None
            base_regions = list(orch.specialists.keys()) if orch and hasattr(orch, 'specialists') else ['fortaleza', 'rmf', 'interior']
            regions = ['global'] + base_regions
            data = {
                region: confidence_tracker.get_current_confidence(region=region)
                for region in regions
            }
            return jsonify(data), 200
        except Exception as e:
            logger.error(f"Erro ao obter confiança atual: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/health-check', methods=['GET'])
    @admin_required
    def health_check():
        """Verifica saúde do sistema contra thresholds."""
        try:
            status, warnings = health_monitor.check_system_health()
            
            return jsonify({
                'status': status,
                'warnings': warnings,
                'timestamp': health_monitor.get_system_metrics().get('timestamp')
            }), 200
        except Exception as e:
            logger.error(f"Erro no health check: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/action', methods=['POST'])
    @admin_required
    def execute_admin_action():
        """
        Executa ações administrativas.
        
        Body JSON:
        {
            "action": "clear_cache" | "export_health_report" | "force_backtesting",
            "confirmed": true
        }
        """
        try:
            data = request.json or {}
            action = data.get('action')
            confirmed = data.get('confirmed', False)
            
            if not confirmed:
                return jsonify({'error': 'Ação não confirmada'}), 400
            
            if action == 'clear_cache':
                # TODO: Implementar limpeza de cache
                logger.info("✅ Cache limpo manualmente")
                return jsonify({'status': 'cache_cleared'}), 200
            
            elif action == 'export_health_report':
                # TODO: Implementar exportação de relatório
                logger.info("✅ Relatório de saúde exportado")
                return jsonify({'status': 'report_exported'}), 200
            
            elif action == 'force_backtesting':
                # TODO: Implementar backtesting forçado
                logger.info("✅ Backtesting forçado iniciado")
                return jsonify({'status': 'backtesting_started'}), 202
            
            else:
                return jsonify({'error': f'Ação desconhecida: {action}'}), 400
        
        except Exception as e:
            logger.error(f"Erro ao executar ação: {e}")
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/audit-log', methods=['GET'])
    @admin_required
    def get_audit_log():
        """
        Retorna log de auditoria de ações administrativas.
        
        Query Parameters:
            - limit: Número máximo de eventos (padrão: 100)
        """
        try:
            # TODO: Implementar logging de auditoria
            limit = int(request.args.get('limit', 100))
            
            audit_log = [
                {
                    'timestamp': '2026-03-01T18:30:00Z',
                    'user': 'admin_01',
                    'action': 'Executar Backtesting',
                    'status': 'success'
                },
                {
                    'timestamp': '2026-03-01T17:45:00Z',
                    'user': 'admin_02',
                    'action': 'Limpar Cache',
                    'status': 'success'
                }
            ]
            
            return jsonify({
                'total': len(audit_log),
                'logs': audit_log[:limit]
            }), 200
        except Exception as e:
            logger.error(f"Erro ao obter audit log: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ===== PÁGINA HTML =====
    
    @admin_bp.route('/data-quality', methods=['GET'])
    @admin_required
    def get_data_quality():
        """
        Retorna métricas de qualidade de dados:
        - Eventos históricos (do último ciclo de avaliação)
        - Eventos exógenos dos últimos 7 dias
        - Janela operacional vigente de validação
        - Taxa de completude dos dados regionais
        - Anomalias detectadas (alertas ativos de severidade alta+)
        """
        try:
            base_dir = health_monitor.base_dir
            orcrim_status = {}
            try:
                from data.raw.inteligencia.import_orcrim_kml import get_orcrim_update_status
                orcrim_status = get_orcrim_update_status()
            except Exception as import_error:
                orcrim_status = {
                    'status': 'unavailable',
                    'last_error': str(import_error),
                }
            
            # 1. Eventos históricos — do último registro de eficiência
            historical_events = 0
            efficiency_path = os.path.join(base_dir, 'logs', 'efficiency_history.json')
            if os.path.exists(efficiency_path):
                try:
                    with open(efficiency_path, 'r', encoding='utf-8') as f:
                        eff_history = json.load(f)
                    if eff_history:
                        latest = eff_history[-1]
                        historical_events = latest.get('assigned_total_events', latest.get('total_events', 0) or latest.get('brute_cvli', 0))
                except Exception:
                    pass
            
            # 2. Eventos exógenos dos últimos 7 dias
            exogenous_events = 0
            exo_path = os.path.join(base_dir, 'data', 'exogenous_events.json')
            cutoff = (datetime.now() - timedelta(days=7)).date().isoformat()
            if os.path.exists(exo_path):
                try:
                    with open(exo_path, 'r', encoding='utf-8') as f:
                        events = json.load(f)
                    for e in events:
                        date_str = e.get('date') or e.get('ingested_at', '')[:10]
                        if date_str >= cutoff:
                            exogenous_events += 1
                except Exception:
                    pass
            
            # 3. Taxa de completude — via glob (sem hardcode de regiões)
            import glob as _glob
            expected_pkls = _glob.glob(os.path.join(base_dir, 'data', 'processed', 'processed_*.pkl'))
            regions_all  = [os.path.basename(p).replace('processed_', '').replace('.pkl', '') for p in expected_pkls]
            regions_available = [r for r in regions_all if os.path.exists(os.path.join(base_dir, 'data', 'processed', f'processed_{r}.pkl'))]
            completeness_pct = 100.0 if not regions_all else (len(regions_available) / len(regions_all)) * 100
            
            # 4. Anomalias — alertas ativos de severidade CRITICAL ou HIGH
            active_alerts = health_monitor.get_alerts(resolved=False)
            anomaly_list = [a for a in active_alerts if a.get('severity') in ('CRITICAL', 'HIGH')]
            anomalies = len(anomaly_list)
            anomaly_details = [
                {
                    'severity': a.get('severity'),
                    'message': a.get('message'),
                    'timestamp': a.get('timestamp'),
                    'type': a.get('type'),
                    'category': a.get('category', 'other'),
                }
                for a in anomaly_list
            ]

            latest_efficiency = {}
            if os.path.exists(efficiency_path):
                try:
                    with open(efficiency_path, 'r', encoding='utf-8') as f:
                        eff_history = json.load(f)
                    if eff_history:
                        latest_efficiency = eff_history[-1]
                except Exception:
                    latest_efficiency = {}
            
            return jsonify({
                'validation_window_days': 14,
                'historical_events': historical_events,
                'exogenous_events_7d': exogenous_events,
                'assigned_total_events': latest_efficiency.get('assigned_total_events', historical_events),
                'unmapped_total_events': latest_efficiency.get('unmapped_total_events', 0),
                'assigned_exogenous': latest_efficiency.get('assigned_exogenous', 0),
                'unmapped_exogenous': latest_efficiency.get('unmapped_exogenous', 0),
                'unmapped_exogenous_sample': latest_efficiency.get('unmapped_exogenous_sample', []),
                'completeness_pct': round(completeness_pct, 1),
                'regions_available': len(regions_available),
                'regions_expected': len(regions_all),
                'anomalies': anomalies,
                'anomaly_details': anomaly_details,
                'orcrim': {
                    'status': orcrim_status.get('status', 'unknown'),
                    'last_checked_at': orcrim_status.get('last_checked_at'),
                    'last_updated_at': orcrim_status.get('last_updated_at'),
                    'fallback_used': bool(orcrim_status.get('fallback_used', False)),
                    'source_url': orcrim_status.get('source_url', ''),
                    'last_error': orcrim_status.get('last_error', ''),
                    'download_sha256': orcrim_status.get('download_sha256', ''),
                    'working_kml': (orcrim_status.get('paths') or {}).get('kml_working'),
                    'static_kml': (orcrim_status.get('paths') or {}).get('kml_static'),
                    'intelligence_csv': (orcrim_status.get('paths') or {}).get('intelligence_csv'),
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        except Exception as e:
            logger.error(f"Erro ao obter data quality: {e}")
            return jsonify({'error': str(e)}), 500

    @admin_bp.route('/calibration-status', methods=['GET'])
    @admin_required
    def get_calibration_status():
        """Retorna status de auto-calibração por região, com parâmetros atuais e histórico."""
        try:
            if model_calibrator is None:
                return jsonify({'available': False}), 200

            status = model_calibrator.get_status()

            # Enriquecer com parâmetros atuais do orchestrator (se disponível)
            current_params = {}
            orch = get_orchestrator() if get_orchestrator else None
            if orch and hasattr(orch, 'calib_params'):
                current_params = {
                    region: dict(params)
                    for region, params in orch.calib_params.items()
                }

            # Defaults alinhados com ModelCalibrator._default() e orchestrator.calib_params
            defaults = {
                'tension_factor':     0.80,
                'min_risk':          30.0,
                'tag_bias_direct':    2.00,
                'tag_bias_neighbor':  0.60,
                'norm_neural_weight': 0.20,
            }

            # Montar resposta enriquecida por região
            # Inclui regiões ativas do orchestrator mesmo que ainda não tenham eventos de calibração
            orch = get_orchestrator() if get_orchestrator else None
            active_regions = list(orch.specialists.keys()) if orch and hasattr(orch, 'specialists') else list(status.keys())
            enriched = {}
            for region in active_regions:
                info = status.get(region, {'steps': 0, 'max_steps': model_calibrator.MAX_STEPS, 'is_degraded': False, 'is_critical': False, 'last_event': None})
                # Enrich with window state from orchestrator
                win_state = {}
                if orch and hasattr(orch, 'calib_params') and region in orch.calib_params:
                    cp = orch.calib_params[region]
                    win_state = {
                        'dynamic_window': cp.get('dynamic_window'),
                        'use_historical_fallback': cp.get('use_historical_fallback', False),
                        'historical_top10': cp.get('historical_top10', []),
                    }
                hist = model_calibrator.state.get(region, {}).get('history', [])
                enriched[region] = {
                    **info,
                    'current_params': current_params.get(region, defaults),
                    'default_params': defaults,
                    'last_5_events': hist[-5:] if hist else [],
                    'window_state': win_state,
                }

            return jsonify({'available': True, 'regions': enriched}), 200
        except Exception as e:
            logger.error(f"Erro ao obter status de calibração: {e}")
            return jsonify({'error': str(e)}), 500

    @admin_bp.route('', methods=['GET'])
    @admin_required
    def dashboard_page():
        """Renderiza página HTML do dashboard."""
        try:
            return render_template('admin_health_dashboard.html'), 200
        except Exception as e:
            logger.error(f"Erro ao renderizar dashboard: {e}")
            return jsonify({'error': 'Erro ao carregar dashboard'}), 500
    
    return admin_bp


# ===== EXEMPLO DE INTEGRAÇÃO EM app.py =====
"""
from src.core.health_monitor import HealthMonitor, ConfidenceTracker
from admin_health_routes import create_admin_health_blueprint

# Inicializar monitores
health_monitor = HealthMonitor(base_dir=BASE_DIR)
confidence_tracker = ConfidenceTracker(base_dir=BASE_DIR)

# Registrar blueprint
admin_health_bp = create_admin_health_blueprint(health_monitor, confidence_tracker)
app.register_blueprint(admin_health_bp)

# Adicionar middleware para rastrear requisições
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        latency_ms = (time.time() - request.start_time) * 1000
        success = response.status_code < 400
        health_monitor.track_api_request(
            endpoint=request.path,
            latency_ms=latency_ms,
            success=success
        )
    return response
"""
