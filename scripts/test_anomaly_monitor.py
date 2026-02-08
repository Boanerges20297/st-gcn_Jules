"""
Test Suite for Anomaly Monitoring System
Demonstrates periodic monitoring and integration with retraining

Author: ST-GCN Enhanced System
Date: Feb 2026
"""

import sys
import time
import json
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.anomaly_monitor import AnomalyMonitor, AnomalyAlert
from src.event_manager import EventManager


def test_anomaly_monitor_initialization():
    """Test 1: AnomalyMonitor initializes without errors"""
    print("\n" + "="*60)
    print("TEST 1: AnomalyMonitor Initialization")
    print("="*60)
    
    try:
        # Create monitor without EventManager (for initial test)
        monitor = AnomalyMonitor(event_manager=None, check_interval_minutes=1)
        print(f"✅ Monitor created: {monitor}")
        print(f"   - Check interval: {monitor.check_interval_minutes} min")
        print(f"   - Monitoring active: {monitor.monitoring_active}")
        print(f"   - Total checks: {monitor.check_count}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_anomaly_alert_creation():
    """Test 2: AnomalyAlert creation and serialization"""
    print("\n" + "="*60)
    print("TEST 2: AnomalyAlert Creation & Serialization")
    print("="*60)
    
    try:
        alert = AnomalyAlert(
            alert_date=date.today(),
            severity=0.75,
            event_count=5,
            crime_types=['robbery', 'assault'],
            risk_level='HIGH'
        )
        print(f"✅ Alert created:")
        print(f"   - Date: {alert.alert_date}")
        print(f"   - Severity: {alert.severity}")
        print(f"   - Risk Level: {alert.risk_level}")
        print(f"   - Crime Types: {alert.crime_types}")
        
        # Test serialization
        alert_dict = alert.to_dict()
        print(f"✅ Alert serialization:")
        print(f"   {json.dumps(alert_dict, indent=2, default=str)}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_event_manager_integration():
    """Test 3: EventManager integration with AnomalyMonitor"""
    print("\n" + "="*60)
    print("TEST 3: EventManager Integration")
    print("="*60)
    
    try:
        event_file = Path(__file__).parent / 'data' / 'exogenous_events_geocoded.json'
        
        if not event_file.exists():
            print(f"⚠️  Event file not found: {event_file}")
            print("   Creating with sample data...")
            
            # Create sample events
            event_file.parent.mkdir(parents=True, exist_ok=True)
            sample_events = [
                {
                    "date": str(date.today()),
                    "event": "robbery",
                    "text": "Armed robbery reported",
                    "lat": -3.7319,
                    "lng": -38.5267,
                    "severity": 0.8,
                    "conflict_severity": "HIGH"
                },
                {
                    "date": str(date.today()),
                    "event": "assault",
                    "text": "Assault in downtown area",
                    "lat": -3.7250,
                    "lng": -38.4900,
                    "severity": 0.6,
                    "conflict_severity": "MEDIUM"
                }
            ]
            with open(event_file, 'w', encoding='utf-8') as f:
                json.dump(sample_events, f, indent=2)
            print(f"   ✅ Sample events created")
        
        # Load EventManager
        event_manager = EventManager(str(event_file))
        print(f"✅ EventManager loaded: {len(event_manager.events)} events")
        
        # Create AnomalyMonitor with EventManager
        monitor = AnomalyMonitor(
            event_manager=event_manager,
            check_interval_minutes=1
        )
        print(f"✅ AnomalyMonitor integrated with EventManager")
        
        # Check anomaly for today
        anomaly = monitor.get_severity_for_date(date.today())
        print(f"✅ Anomaly check for today:")
        print(f"   - Severity: {anomaly}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anomaly_summary():
    """Test 4: Get anomaly monitoring summary"""
    print("\n" + "="*60)
    print("TEST 4: Anomaly Monitoring Summary")
    print("="*60)
    
    try:
        event_file = Path(__file__).parent / 'data' / 'exogenous_events_geocoded.json'
        
        if event_file.exists():
            event_manager = EventManager(str(event_file))
            monitor = AnomalyMonitor(event_manager=event_manager, check_interval_minutes=1)
            
            # Get summary before monitoring
            summary = monitor.get_anomaly_summary()
            print(f"✅ Anomaly Summary (before monitoring):")
            print(f"   - Monitoring Active: {summary['monitoring_active']}")
            print(f"   - Check Interval: {summary['check_interval_minutes']} min")
            print(f"   - Total Checks: {summary['total_checks_performed']}")
            print(f"   - Alerts Generated: {summary['alerts_generated']}")
            print(f"   - Current Anomalies: {summary['current_anomalies_count']}")
            
            return True
        else:
            print("⚠️  Skipping: Event file not found")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retraining_context():
    """Test 5: Get anomaly context for retraining"""
    print("\n" + "="*60)
    print("TEST 5: Retraining Context Generation")
    print("="*60)
    
    try:
        event_file = Path(__file__).parent / 'data' / 'exogenous_events_geocoded.json'
        
        if event_file.exists():
            event_manager = EventManager(str(event_file))
            monitor = AnomalyMonitor(event_manager=event_manager, check_interval_minutes=1)
            
            # Get retraining context
            context = monitor.get_anomaly_context_for_retraining(date.today())
            print(f"✅ Retraining Context:")
            print(f"   Period: {context['period']['start']} to {context['period']['end']}")
            print(f"   Days with Anomalies: {context['period']['days_with_anomalies']}")
            print(f"   Average Severity: {context['statistics']['average_severity']:.2f}")
            print(f"   Today's Severity: {context['today']['severity']:.2f}")
            print(f"   Today's Risk Level: {context['today']['risk_level']}")
            print(f"   Recommendation:")
            print(f"     - Use Conservative Weights: {context['recommendation']['use_conservative_weights']}")
            print(f"     - Confidence Penalty: {context['recommendation']['increase_confidence_penalty']:.2f}")
            print(f"     - Temporal Weighting: {context['recommendation']['temporal_weighting']}")
            print(f"     - Notes: {context['recommendation']['notes']}")
            
            return True
        else:
            print("⚠️  Skipping: Event file not found")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring_thread_lifecycle():
    """Test 6: Monitor start/stop lifecycle"""
    print("\n" + "="*60)
    print("TEST 6: Monitoring Thread Lifecycle")
    print("="*60)
    
    try:
        event_file = Path(__file__).parent / 'data' / 'exogenous_events_geocoded.json'
        
        if event_file.exists():
            event_manager = EventManager(str(event_file))
            monitor = AnomalyMonitor(event_manager=event_manager, check_interval_minutes=1)
            
            # Start monitoring
            print("   Starting monitor...")
            monitor.start()
            print(f"   ✅ Monitor started: {monitor.monitoring_active}")
            
            # Let it run for a few seconds
            print("   Waiting 3 seconds for background checks...")
            time.sleep(3)
            
            # Check status
            checks_performed = monitor.check_count
            print(f"   ✅ Checks performed in 3 seconds: {checks_performed}")
            
            # Stop monitoring
            print("   Stopping monitor...")
            monitor.stop()
            print(f"   ✅ Monitor stopped: {not monitor.monitoring_active}")
            
            return checks_performed > 0
        else:
            print("⚠️  Skipping: Event file not found")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("ANOMALY MONITORING SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Initialization", test_anomaly_monitor_initialization),
        ("Alert Creation", test_anomaly_alert_creation),
        ("EventManager Integration", test_event_manager_integration),
        ("Monitoring Summary", test_anomaly_summary),
        ("Retraining Context", test_retraining_context),
        ("Thread Lifecycle", test_monitoring_thread_lifecycle),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:15} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Anomaly monitoring system is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
