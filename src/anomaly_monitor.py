"""
Anomaly Monitor - Continuous Periodic Monitoring
Monitors exogenous events for anomalies in background, independent of retraining

Author: ST-GCN Enhanced System
Date: Feb 2026

Features:
- Periodic background monitoring (every 15 min, configurable)
- Tracks anomaly state changes
- Generates alerts on severity changes
- Feeds into model retraining adjustments
- Thread-safe with locks for concurrent access
"""

import threading
import time
import json
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyAlert:
    """Represents a single anomaly alert"""
    
    def __init__(self, alert_date: date, severity: float, event_count: int, 
                 crime_types: List[str], risk_level: str):
        self.alert_date = alert_date
        self.severity = severity  # [0-1]
        self.event_count = event_count
        self.crime_types = crime_types
        self.risk_level = risk_level  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        self.timestamp = datetime.now(timezone.utc)
        self.processed = False  # Whether model retraining has considered this
    
    def to_dict(self) -> Dict:
        return {
            'alert_date': str(self.alert_date),
            'severity': float(self.severity),
            'event_count': self.event_count,
            'crime_types': self.crime_types,
            'risk_level': self.risk_level,
            'timestamp': self.timestamp.isoformat(),
            'processed': self.processed
        }


class AnomalyMonitor:
    """
    Continuous anomaly monitoring system
    
    Runs independently from retraining to:
    1. Detect anomalies across events in real-time
    2. Track state changes (no anomaly → anomaly, severity increases, etc.)
    3. Generate alerts for operational awareness
    4. Provide anomaly context to model retraining
    """
    
    def __init__(self, event_manager=None, check_interval_minutes: int = 15):
        """
        Initialize AnomalyMonitor
        
        Args:
            event_manager: EventManager instance (required for anomaly detection)
            check_interval_minutes: How often to check for anomalies (default 15 min)
        """
        self.event_manager = event_manager
        self.check_interval_minutes = max(5, int(check_interval_minutes))  # Min 5 min
        
        # State tracking
        self.current_alerts: Dict[str, AnomalyAlert] = {}  # date_str -> AnomalyAlert
        self.alert_history: List[AnomalyAlert] = []  # Historical alerts
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Monitoring stats
        self.check_count = 0
        self.alerts_generated = 0
        self.last_check_time: Optional[datetime] = None
        self.anomaly_trends = {}  # date -> list of severity samples
        
        logger.info(f"[AnomalyMonitor] Initialized with {check_interval_minutes}min check interval")
    
    def start(self):
        """Start the anomaly monitoring background thread"""
        with self.lock:
            if self.monitoring_active:
                logger.warning("[AnomalyMonitor] Already monitoring. Ignoring start() call.")
                return
            
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="AnomalyMonitor"
            )
            self.monitor_thread.start()
            logger.info(f"[AnomalyMonitor] Started monitoring thread (interval: {self.check_interval_minutes} min)")
    
    def stop(self):
        """Stop the anomaly monitoring thread"""
        with self.lock:
            self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("[AnomalyMonitor] Stopped monitoring thread")
    
    def set_event_manager(self, event_manager):
        """Update event manager reference (after EventManager is initialized)"""
        with self.lock:
            self.event_manager = event_manager
            logger.info("[AnomalyMonitor] EventManager updated")
    
    def _monitoring_loop(self):
        """Main monitoring loop - runs in background thread"""
        logger.info("[AnomalyMonitor] Monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Perform check
                self._check_anomalies_for_recent_dates()
                self.check_count += 1
                self.last_check_time = datetime.now(timezone.utc)
                
                # Wait for next check
                for _ in range(self.check_interval_minutes * 60):
                    if not self.monitoring_active:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"[AnomalyMonitor] Error in monitoring loop: {e}", exc_info=True)
                time.sleep(5)  # Brief pause on error to prevent tight loops
    
    def _check_anomalies_for_recent_dates(self):
        """Check anomalies for recent dates (last 7 days) and today"""
        if not self.event_manager:
            logger.debug("[AnomalyMonitor] EventManager not set. Skipping check.")
            return
        
        try:
            today = date.today()
            start_date = today - timedelta(days=7)
            
            with self.lock:
                # Check each date in range
                current_date = start_date
                while current_date <= today:
                    self._check_single_date(current_date)
                    current_date += timedelta(days=1)
                    
        except Exception as e:
            logger.error(f"[AnomalyMonitor] Error checking recent dates: {e}")
    
    def _check_single_date(self, check_date: date):
        """Check anomalies for a single date and generate alert if needed"""
        try:
            # Get current anomaly level
            anomaly_level = self.event_manager.get_anomaly_level_for_date(check_date)
            
            # Get events for detailed analysis
            events = self.event_manager.get_events_for_date(check_date)
            
            # Extract crime types from events
            crime_types = self._extract_crime_types(events)
            
            # Determine risk level
            risk_level = self._severity_to_risk_level(anomaly_level)
            
            # Create/update alert
            date_str = str(check_date)
            old_alert = self.current_alerts.get(date_str)
            
            new_alert = AnomalyAlert(
                alert_date=check_date,
                severity=anomaly_level,
                event_count=len(events),
                crime_types=crime_types,
                risk_level=risk_level
            )
            
            # Log changes
            if old_alert is None and anomaly_level > 0.0:
                # New anomaly
                logger.warning(
                    f"[AnomalyMonitor] 🚨 NEW ANOMALY on {check_date}: "
                    f"severity={anomaly_level:.2f}, risk={risk_level}, events={len(events)}, "
                    f"crimes={crime_types}"
                )
                self.alerts_generated += 1
                
            elif old_alert and anomaly_level != old_alert.severity:
                # Severity change
                direction = "⬆️ INCREASED" if anomaly_level > old_alert.severity else "⬇️ DECREASED"
                logger.info(
                    f"[AnomalyMonitor] {direction} anomaly on {check_date}: "
                    f"{old_alert.severity:.2f} → {anomaly_level:.2f} (risk: {risk_level})"
                )
            
            # Store alert
            self.current_alerts[date_str] = new_alert
            
            # Track trend
            if date_str not in self.anomaly_trends:
                self.anomaly_trends[date_str] = []
            self.anomaly_trends[date_str].append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'severity': anomaly_level
            })
            
            # Keep trend history limited (last 100 samples per date)
            if len(self.anomaly_trends[date_str]) > 100:
                self.anomaly_trends[date_str] = self.anomaly_trends[date_str][-100:]
                
        except Exception as e:
            logger.error(f"[AnomalyMonitor] Error checking date {check_date}: {e}")
    
    def _extract_crime_types(self, events: List[Dict]) -> List[str]:
        """Extract unique crime types from events"""
        crime_types = set()
        
        for event in events:
            # Try multiple field names
            for field in ['crime_type', 'crime_types', 'type', 'event_type', 'classification']:
                if field in event:
                    types = event[field]
                    if isinstance(types, list):
                        crime_types.update(types)
                    elif isinstance(types, str):
                        crime_types.add(types)
        
        return sorted(list(crime_types))
    
    def _severity_to_risk_level(self, severity: float) -> str:
        """Convert severity score to risk level"""
        if severity >= 0.9:
            return 'CRITICAL'
        elif severity >= 0.7:
            return 'HIGH'
        elif severity >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    # ==================== Query Methods ====================
    
    def get_current_anomalies(self) -> Dict[str, AnomalyAlert]:
        """Get current active anomalies (detected in last 7 days)"""
        with self.lock:
            # Filter to last 7 days
            cutoff = date.today() - timedelta(days=7)
            return {
                date_str: alert for date_str, alert in self.current_alerts.items()
                if alert.alert_date >= cutoff
            }
    
    def get_anomaly_for_date(self, check_date: date) -> Optional[AnomalyAlert]:
        """Get anomaly alert for specific date"""
        date_str = str(check_date)
        with self.lock:
            return self.current_alerts.get(date_str)
    
    def get_unprocessed_alerts(self) -> List[AnomalyAlert]:
        """Get alerts that haven't been processed by retraining yet"""
        with self.lock:
            return [alert for alert in self.current_alerts.values() if not alert.processed]
    
    def mark_alerts_processed(self, date_list: List[date]):
        """Mark alerts as processed by retraining"""
        with self.lock:
            for d in date_list:
                date_str = str(d)
                if date_str in self.current_alerts:
                    self.current_alerts[date_str].processed = True
                    logger.debug(f"[AnomalyMonitor] Marked {d} alerts as processed")
    
    def get_severity_for_date(self, check_date: date) -> float:
        """Get anomaly severity for a date"""
        alert = self.get_anomaly_for_date(check_date)
        return alert.severity if alert else 0.0
    
    def get_high_risk_dates(self, days_back: int = 7) -> List[date]:
        """Get dates with HIGH or CRITICAL risk in recent days"""
        with self.lock:
            cutoff = date.today() - timedelta(days=days_back)
            return [
                alert.alert_date for alert in self.current_alerts.values()
                if alert.alert_date >= cutoff and alert.risk_level in ('HIGH', 'CRITICAL')
            ]
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of monitoring status"""
        with self.lock:
            active_anomalies = self.get_current_anomalies()
            unprocessed = self.get_unprocessed_alerts()
            high_risk = self.get_high_risk_dates()
            
            return {
                'monitoring_active': self.monitoring_active,
                'check_interval_minutes': self.check_interval_minutes,
                'total_checks_performed': self.check_count,
                'alerts_generated': self.alerts_generated,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'current_anomalies_count': len(active_anomalies),
                'unprocessed_alerts': len(unprocessed),
                'high_risk_dates': [str(d) for d in high_risk],
                'anomaly_details': {
                    date_str: alert.to_dict() 
                    for date_str, alert in active_anomalies.items()
                }
            }
    
    # ==================== Integration with Retraining ====================
    
    def get_anomaly_context_for_retraining(self, retrain_date: date = None) -> Dict:
        """
        Get anomaly context to feed into retraining
        
        Used to adjust model weights/confidence based on recent anomalies
        
        Args:
            retrain_date: Date being retrained (default: today)
        
        Returns:
            Context dict with anomaly adjustments for retraining
        """
        if retrain_date is None:
            retrain_date = date.today()
        
        with self.lock:
            # Check last 30 days for anomaly patterns
            period_start = retrain_date - timedelta(days=30)
            period_end = retrain_date
            
            high_risk_dates = []
            avg_severity = 0.0
            severity_samples = []
            
            current_date = period_start
            while current_date <= period_end:
                alert = self.get_anomaly_for_date(current_date)
                if alert and alert.severity > 0.0:
                    severity_samples.append(alert.severity)
                    if alert.risk_level in ('HIGH', 'CRITICAL'):
                        high_risk_dates.append(current_date)
                current_date += timedelta(days=1)
            
            if severity_samples:
                avg_severity = sum(severity_samples) / len(severity_samples)
            
            # Today's anomaly
            today_anomaly = self.get_anomaly_for_date(retrain_date)
            
            return {
                'period': {
                    'start': str(period_start),
                    'end': str(period_end),
                    'days_with_anomalies': len(severity_samples),
                    'high_risk_days': [str(d) for d in high_risk_dates]
                },
                'statistics': {
                    'average_severity': float(avg_severity),
                    'max_severity': float(max(severity_samples)) if severity_samples else 0.0,
                    'min_severity': float(min(severity_samples)) if severity_samples else 0.0,
                    'anomaly_frequency': len(severity_samples) / 30  # % of days with anomalies
                },
                'today': {
                    'date': str(retrain_date),
                    'has_anomaly': today_anomaly is not None and today_anomaly.severity > 0.0,
                    'severity': today_anomaly.severity if today_anomaly else 0.0,
                    'risk_level': today_anomaly.risk_level if today_anomaly else 'LOW',
                    'event_count': today_anomaly.event_count if today_anomaly else 0,
                    'crime_types': today_anomaly.crime_types if today_anomaly else []
                },
                'recommendation': self._get_retraining_recommendation(avg_severity, today_anomaly)
            }
    
    def _get_retraining_recommendation(self, avg_severity: float, today_alert: Optional[AnomalyAlert]) -> Dict:
        """Get recommendation for retraining based on anomaly patterns"""
        rec = {
            'skip_retrain': False,
            'use_conservative_weights': False,
            'increase_confidence_penalty': 0.0,
            'temporal_weighting': 'normal',
            'notes': []
        }
        
        if today_alert and today_alert.severity > 0.8:
            rec['use_conservative_weights'] = True
            rec['increase_confidence_penalty'] = 0.25
            rec['temporal_weighting'] = 'recent_events_emphasized'
            rec['notes'].append('High anomaly today: Using conservative model weights')
        
        elif avg_severity > 0.6:
            rec['increase_confidence_penalty'] = 0.15
            rec['temporal_weighting'] = 'recent_events_emphasized'
            rec['notes'].append('Elevated anomaly period: Emphasizing recent temporal patterns')
        
        else:
            rec['notes'].append('Normal anomaly levels: Standard retraining')
        
        return rec


# Global instance
anomaly_monitor: Optional[AnomalyMonitor] = None


def start_anomaly_monitoring(event_manager=None, interval_minutes: int = 15):
    """
    Initialize and start global anomaly monitoring
    
    Args:
        event_manager: EventManager instance
        interval_minutes: Check interval in minutes (default 15)
    """
    global anomaly_monitor
    
    if anomaly_monitor is not None:
        logger.warning("[AnomalyMonitor] Global monitor already running")
        return anomaly_monitor
    
    try:
        anomaly_monitor = AnomalyMonitor(
            event_manager=event_manager,
            check_interval_minutes=interval_minutes
        )
        anomaly_monitor.start()
        logger.info("[AnomalyMonitor] 🚀 Global anomaly monitoring started")
        
        return anomaly_monitor
    except Exception as e:
        logger.error(f"[AnomalyMonitor] Failed to start monitoring: {e}")
        return None


def get_anomaly_monitor() -> Optional[AnomalyMonitor]:
    """Get global anomaly monitor instance"""
    return anomaly_monitor
