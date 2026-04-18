"""
Health Monitor Module - REPORT PREVIEW

Módulo responsável por:
- Coletar métricas do sistema (CPU, memória, disco)
- Monitorar performance da API
- Rastrear confiança do modelo
- Gerar alertas automáticos
- Manter histórico de métricas
"""

import os
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    Monitora a saúde do sistema REPORT PREVIEW.
    """
    
    def __init__(self, base_dir: str = None, max_history_days: int = 30):
        """
        Args:
            base_dir: Diretório raiz do projeto
            max_history_days: Dias de histórico a manter em memória
        """
        self.base_dir = base_dir or os.getcwd()
        self.max_history_days = max_history_days
        
        # Histórico em memória (será persisted em arquivo JSON)
        self.metrics_history = defaultdict(list)
        self.alerts_history = []
        self.api_stats = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'total_latency_ms': 0,
            'latencies': []
        })
        
        # Arquivo de persistência
        self.metrics_file = os.path.join(self.base_dir, 'data', 'health_metrics.json')
        self.alerts_file = os.path.join(self.base_dir, 'data', 'health_alerts.json')
        
        # Criar diretório data se não existir
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        
        # Carregar histórico existente
        self._load_history()
        
        # Timestamps de início
        self.start_time = time.time()
        
        logger.info("✅ HealthMonitor inicializado")
    
    def _load_history(self):
        """Carrega histórico de métricas e alertas do disco."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metrics_history = defaultdict(list, data.get('metrics', {}))
            
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.alerts_history = data.get('alerts', [])
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar histórico: {e}")
    
    def _save_history(self):
        """Persiste histórico em arquivo JSON."""
        try:
            # Salvar métricas
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {'metrics': dict(self.metrics_history)},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            
            # Salvar alertas
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {'alerts': self.alerts_history},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"❌ Erro ao salvar histórico: {e}")
    
    def get_system_metrics(self) -> Dict:
        """
        Retorna métricas atuais do sistema.
        
        Returns:
            Dict com CPU, memória, disco, uptime
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            uptime_seconds = int(time.time() - self.start_time)
            uptime_str = self._format_uptime(uptime_seconds)
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory': {
                    'used_mb': memory.used / 1024 / 1024,
                    'total_mb': memory.total / 1024 / 1024,
                    'percent': memory.percent
                },
                'disk': {
                    'used_gb': disk.used / 1024 / 1024 / 1024,
                    'total_gb': disk.total / 1024 / 1024 / 1024,
                    'percent': disk.percent
                },
                'uptime_seconds': uptime_seconds,
                'uptime_str': uptime_str
            }
            
            # Armazenar no histórico
            for key in ['cpu_percent', 'memory', 'disk']:
                self.metrics_history[key].append({
                    'timestamp': metrics['timestamp'],
                    'value': metrics[key]
                })
            
            # Manter apenas últimos N dias
            self._cleanup_old_metrics()
            
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Erro ao coletar métricas do sistema: {e}")
            return {}
    
    def _format_uptime(self, seconds: int) -> str:
        """Formata segundos para string legível."""
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        return f"{days}d {hours}h {minutes}m"
    
    def _cleanup_old_metrics(self):
        """Remove métricas com mais de max_history_days."""
        cutoff_date = (datetime.now() - timedelta(days=self.max_history_days)).isoformat()
        
        for key in self.metrics_history:
            self.metrics_history[key] = [
                m for m in self.metrics_history[key]
                if m['timestamp'] > cutoff_date
            ]
    
    def track_api_request(self, endpoint: str, latency_ms: float, success: bool = True):
        """
        Rastreia requisição de API.
        
        Args:
            endpoint: Path do endpoint (ex: /api/risk)
            latency_ms: Latência em milissegundos
            success: Requisição bem-sucedida?
        """
        stats = self.api_stats[endpoint]
        stats['requests'] += 1
        stats['latencies'].append(latency_ms)
        stats['total_latency_ms'] += latency_ms
        
        if not success:
            stats['errors'] += 1
        
        # Manter apenas últimas 10k latências por endpoint
        if len(stats['latencies']) > 10000:
            stats['latencies'] = stats['latencies'][-5000:]
    
    def get_api_stats(self) -> Dict:
        """
        Retorna estatísticas de API.
        
        Returns:
            Dict com stats por endpoint e global
        """
        global_stats = {
            'total_requests': 0,
            'total_errors': 0,
            'error_rate_percent': 0.0,
            'avg_latency_ms': 0.0,
            'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0
        }
        
        endpoints = {}
        all_latencies = []
        
        for endpoint, stats in self.api_stats.items():
            if stats['requests'] == 0:
                continue
            
            latencies = sorted(stats['latencies'])
            
            endpoints[endpoint] = {
                'requests': stats['requests'],
                'errors': stats['errors'],
                'error_rate_percent': (stats['errors'] / stats['requests']) * 100,
                'avg_latency_ms': stats['total_latency_ms'] / stats['requests'],
                'p50_latency_ms': latencies[int(len(latencies) * 0.50)],
                'p95_latency_ms': latencies[int(len(latencies) * 0.95)],
                'p99_latency_ms': latencies[int(len(latencies) * 0.99)]
            }
            
            global_stats['total_requests'] += stats['requests']
            global_stats['total_errors'] += stats['errors']
            all_latencies.extend(latencies)
        
        if global_stats['total_requests'] > 0:
            global_stats['error_rate_percent'] = (
                global_stats['total_errors'] / global_stats['total_requests']
            ) * 100
            all_latencies = sorted(all_latencies)
            global_stats['avg_latency_ms'] = sum(all_latencies) / len(all_latencies)
            global_stats['p50_latency_ms'] = all_latencies[int(len(all_latencies) * 0.50)]
            global_stats['p95_latency_ms'] = all_latencies[int(len(all_latencies) * 0.95)]
            global_stats['p99_latency_ms'] = all_latencies[int(len(all_latencies) * 0.99)]
        
        return {
            'global': global_stats,
            'endpoints': endpoints
        }
    
    def add_alert(self, alert_type: str, severity: str, message: str, 
                  details: Dict = None, resolved: bool = False):
        """
        Adiciona alerta ao histórico.
        
        Args:
            alert_type: Tipo de alerta (model_degraded, data_stale, etc)
            severity: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
            message: Mensagem do alerta
            details: Detalhes adicionais (dict)
            resolved: Alerta resolvido?
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {},
            'resolved': resolved
        }
        
        self.alerts_history.append(alert)
        
        # Manter apenas últimos 1000 alertas
        if len(self.alerts_history) > 1000:
            self.alerts_history = self.alerts_history[-500:]
        
        self._save_history()
        
        logger.warning(f"🔔 [{severity}] {alert_type}: {message}")

    @staticmethod
    def classify_alert(alert_type: str) -> str:
        """Classifica alertas por domínio para separação no dashboard."""
        alert_type = (alert_type or '').lower()
        if 'faction_coverage' in alert_type:
            return 'territorial_coverage'
        if 'model_degraded' in alert_type:
            return 'predictive_degradation'
        if 'auto_calibration' in alert_type or 'calibration_maxed' in alert_type:
            return 'calibration'
        if alert_type.startswith('high_') or 'error_rate' in alert_type:
            return 'system'
        return 'other'

    def _enrich_alert(self, alert: Dict) -> Dict:
        enriched = dict(alert)
        enriched['category'] = self.classify_alert(alert.get('type', ''))
        return enriched
    
    def get_alerts(self, limit: int = 100, resolved: Optional[bool] = None) -> List[Dict]:
        """
        Retorna alertas.
        
        Args:
            limit: Número máximo de alertas
            resolved: Filtrar por status (True/False/None=todos)
        
        Returns:
            Lista de alertas (ordenada por timestamp descendente)
        """
        alerts = self.alerts_history
        
        if resolved is not None:
            alerts = [a for a in alerts if a['resolved'] == resolved]
        
        # Retornar mais recentes primeiro
        alerts = sorted(alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
        return [self._enrich_alert(a) for a in alerts]
    
    def check_system_health(self, thresholds: Dict = None) -> Tuple[str, List[str]]:
        """
        Verifica saúde geral do sistema contra thresholds.
        Gera alertas automáticos quando limites são excedidos (máx 1 por tipo a cada 30min).
        
        Returns:
            Tupla (status_string, list_of_warnings)
        """
        thresholds = thresholds or {
            'cpu_max_percent': 80,
            'memory_max_percent': 85,
            'disk_max_percent': 90,
            'error_rate_max_percent': 5.0
        }
        
        metrics = self.get_system_metrics()
        api_stats = self.get_api_stats()
        warnings = []
        status = 'OK'
        
        # Janela de supressão: não repetir o mesmo tipo de alerta em 30 min
        suppression_window = timedelta(minutes=30)
        now = datetime.now()
        
        def _recent_alert_exists(alert_type: str) -> bool:
            cutoff = (now - suppression_window).isoformat()
            return any(
                a['type'] == alert_type and a['timestamp'] >= cutoff and not a['resolved']
                for a in self.alerts_history
            )
        
        # Check CPU
        cpu = metrics.get('cpu_percent', 0)
        if cpu > thresholds['cpu_max_percent']:
            msg = f"CPU alta: {cpu:.1f}%"
            warnings.append(msg)
            status = 'WARNING'
            if not _recent_alert_exists('high_cpu'):
                self.add_alert('high_cpu', 'HIGH', msg, {'cpu_percent': cpu})
        
        # Check Memory
        mem_pct = metrics.get('memory', {}).get('percent', 0)
        if mem_pct > thresholds['memory_max_percent']:
            msg = f"Memória alta: {mem_pct:.1f}%"
            warnings.append(msg)
            status = 'WARNING'
            if not _recent_alert_exists('high_memory'):
                self.add_alert('high_memory', 'HIGH', msg, {'memory_percent': mem_pct})
        
        # Check Disk
        disk_pct = metrics.get('disk', {}).get('percent', 0)
        if disk_pct > thresholds['disk_max_percent']:
            msg = f"Disco cheio: {disk_pct:.1f}%"
            warnings.append(msg)
            status = 'CRITICAL' if disk_pct > 95 else 'WARNING'
            severity = 'CRITICAL' if disk_pct > 95 else 'HIGH'
            if not _recent_alert_exists('high_disk'):
                self.add_alert('high_disk', severity, msg, {'disk_percent': disk_pct})
        
        # Check Error Rate
        err_rate = api_stats.get('global', {}).get('error_rate_percent', 0)
        if err_rate > thresholds['error_rate_max_percent']:
            msg = f"Taxa de erro elevada: {err_rate:.2f}%"
            warnings.append(msg)
            status = 'WARNING'
            if not _recent_alert_exists('high_error_rate'):
                self.add_alert('high_error_rate', 'MEDIUM', msg, {'error_rate_percent': err_rate})
        
        return status, warnings
    
    def get_summary(self) -> Dict:
        """
        Retorna summary completo do sistema para o dashboard.
        
        Returns:
            Dict com todas as métricas principais
        """
        metrics = self.get_system_metrics()
        api_stats = self.get_api_stats()
        status, warnings = self.check_system_health()
        alerts_active = self.get_alerts(resolved=False)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'status': status,
                'warnings': warnings,
                'cpu_percent': metrics.get('cpu_percent', 0),
                'memory_mb': metrics.get('memory', {}).get('used_mb', 0),
                'memory_total_mb': metrics.get('memory', {}).get('total_mb', 0),
                'memory_percent': metrics.get('memory', {}).get('percent', 0),
                'disk_gb': metrics.get('disk', {}).get('used_gb', 0),
                'disk_total_gb': metrics.get('disk', {}).get('total_gb', 0),
                'disk_percent': metrics.get('disk', {}).get('percent', 0),
                'uptime_str': metrics.get('uptime_str', 'N/A')
            },
            'api': api_stats['global'],
            'alerts': {
                'total_active': len(alerts_active),
                'critical': len([a for a in alerts_active if a['severity'] == 'CRITICAL']),
                'high': len([a for a in alerts_active if a['severity'] == 'HIGH']),
                'medium': len([a for a in alerts_active if a['severity'] == 'MEDIUM']),
                'by_category': {
                    'territorial_coverage': len([a for a in alerts_active if a.get('category') == 'territorial_coverage']),
                    'predictive_degradation': len([a for a in alerts_active if a.get('category') == 'predictive_degradation']),
                    'calibration': len([a for a in alerts_active if a.get('category') == 'calibration']),
                    'system': len([a for a in alerts_active if a.get('category') == 'system']),
                    'other': len([a for a in alerts_active if a.get('category') == 'other']),
                },
                'recent': alerts_active[:10]
            }
        }


class ConfidenceTracker:
    """
    Rastreia confiança do modelo ao longo do tempo.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Args:
            base_dir: Diretório raiz do projeto
        """
        self.base_dir = base_dir or os.getcwd()
        self.history_file = os.path.join(self.base_dir, 'data', 'confidence_history.json')
        
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Carrega histórico de confiança."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar histórico de confiança: {e}")
        
        return []
    
    def _save_history(self):
        """Persiste histórico."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar histórico de confiança: {e}")
    
    def record_evaluation(self, evaluation_date: str, global_metrics: Dict, 
                         region_metrics: Dict):
        """
        Registra avaliação de confiança.
        
        Args:
            evaluation_date: Data no formato YYYY-MM-DD
            global_metrics: Métricas globais {p10, p20, precision, recall, f1}
            region_metrics: Métricas por região
        """
        record = {
            'date': evaluation_date,
            'timestamp': datetime.now().isoformat(),
            'global': global_metrics,
            'regions': region_metrics
        }
        
        # Substituir registro existente da mesma data (evita duplicatas)
        self.history = [h for h in self.history if h['date'] != evaluation_date]
        self.history.append(record)
        
        # Manter apenas últimos 365 dias
        cutoff = (datetime.now() - timedelta(days=365)).date().isoformat()
        self.history = [h for h in self.history if h['date'] >= cutoff]
        
        self._save_history()
    
    def get_history(self, region: str = None, days: int = 30) -> List[Dict]:
        """
        Retorna histórico de confiança.
        
        Args:
            region: Filtrar por região (fortaleza, rmf, interior, None=global)
            days: Últimos N dias
        
        Returns:
            Lista de registros
        """
        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        filtered = [h for h in self.history if h['date'] >= cutoff]
        
        if region and region != 'global':
            filtered = [
                h for h in filtered
                if region in h.get('regions', {})
            ]
        
        return sorted(filtered, key=lambda x: x['date'])
    
    def get_current_confidence(self, region: str = 'global') -> Dict:
        """
        Retorna confiança mais recente.
        
        Args:
            region: Região desejada
        
        Returns:
            Dict com métricas de confiança
        """
        if not self.history:
            return {}
        
        latest = self.history[-1]
        
        if region == 'global':
            return latest.get('global', {})
        else:
            return latest.get('regions', {}).get(region, {})
    
    def get_trend(self, region: str = 'global', metric: str = 'p10') -> str:
        """
        Retorna tendência de confiança.
        
        Args:
            region: Região desejada
            metric: Métrica (p10, p20, precision, etc)
        
        Returns:
            String com tendência: '↑', '→', '↓'
        """
        history = self.get_history(region=region, days=7)
        
        if len(history) < 2:
            return '→'
        
        old_val = history[0].get('global' if region == 'global' else 'regions', {})
        if region != 'global':
            old_val = old_val.get(region, {})
        
        new_val = history[-1].get('global' if region == 'global' else 'regions', {})
        if region != 'global':
            new_val = new_val.get(region, {})
        
        old_metric = old_val.get(metric, 0)
        new_metric = new_val.get(metric, 0)
        
        if new_metric > old_metric * 1.02:
            return '↑'
        elif new_metric < old_metric * 0.98:
            return '↓'
        else:
            return '→'
    
    def seed_from_efficiency_history(self, efficiency_history_path: str):
        """
        Popula o histórico de confiança a partir do arquivo do EfficiencyMonitor.
        Deve ser chamado na inicialização para aproveitar dados já existentes.
        
        Args:
            efficiency_history_path: Caminho para logs/efficiency_history.json
        """
        try:
            if not os.path.exists(efficiency_history_path):
                return
            
            with open(efficiency_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            existing_dates = {h['date'] for h in self.history}
            
            for entry in history:
                date = entry.get('date')
                if not date or date in existing_dates:
                    continue
                
                # Mapear métricas do efficiency_monitor → formato do confidence_tracker
                global_data = entry.get('global', {})
                global_metrics = {
                    'p10': global_data.get('p10', 0),
                    'p20': global_data.get('p20', 0),
                    'precision': global_data.get('p10', 0),  # p10 ≈ precision@10
                    'recall': global_data.get('recall20', global_data.get('p20', 0)),
                    'recall10': global_data.get('recall10', 0),
                    'recall20': global_data.get('recall20', 0),
                    'active_locations': global_data.get('active_locations', 0),
                    'total_nodes': global_data.get('total_nodes', 0),
                    'total_events': global_data.get('total_events', 0),
                    'assigned_total_events': entry.get('assigned_total_events', 0),
                    'unmapped_total_events': entry.get('unmapped_total_events', 0),
                    'f1_score': 0.0
                }
                # Calcular f1 se precision e recall disponíveis
                p = global_metrics['precision']
                r = global_metrics['recall']
                if p + r > 0:
                    global_metrics['f1_score'] = round(2 * p * r / (p + r), 4)
                
                # Regiões — dinamic a partir das chaves do entry (sem hardcode)
                region_metrics = {}
                skip_keys = {'global', 'date', 'timestamp', 'total_events', 'brute_cvli', 'exogenous'}
                for reg in entry:
                    if reg in skip_keys:
                        continue
                    reg_data = entry.get(reg, {})
                    if reg_data and isinstance(reg_data, dict) and reg_data.get('p10') is not None:
                        region_metrics[reg] = {
                            'p10': reg_data.get('p10', 0),
                            'p20': reg_data.get('p20', 0),
                            'precision': reg_data.get('p10', 0),
                            'recall': reg_data.get('recall20', reg_data.get('p20', 0)),
                            'recall10': reg_data.get('recall10', 0),
                            'recall20': reg_data.get('recall20', 0),
                            'active_locations': reg_data.get('active_locations', 0),
                            'total_nodes': reg_data.get('total_nodes', 0),
                            'total_events': reg_data.get('total_events', 0),
                            'f1_score': 0.0
                        }
                        rp = region_metrics[reg]['precision']
                        rr = region_metrics[reg]['recall']
                        if rp + rr > 0:
                            region_metrics[reg]['f1_score'] = round(2 * rp * rr / (rp + rr), 4)
                
                self.record_evaluation(date, global_metrics, region_metrics)
                existing_dates.add(date)
            
            logger.info(f"✅ ConfidenceTracker populado com {len(history)} registros do efficiency history")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao importar efficiency history: {e}")
