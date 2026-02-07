#!/usr/bin/env python3
"""
Test Week 4 API Endpoints
Tests the new explanation, metrics, and anomaly status endpoints
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("TEST 1: Checking imports")
    print("=" * 60)
    
    try:
        from src.metrics import MetricReporter
        print("✓ MetricReporter imported")
    except Exception as e:
        print(f"✗ MetricReporter import failed: {e}")
        return False
    
    try:
        from src.event_manager import EventManager
        print("✓ EventManager imported")
    except Exception as e:
        print(f"✗ EventManager import failed: {e}")
        return False
    
    try:
        from src.explanation_generator import ExplanationGenerator
        print("✓ ExplanationGenerator imported")
    except Exception as e:
        print(f"✗ ExplanationGenerator import failed: {e}")
        return False
    
    return True


def test_metric_reporter():
    """Test MetricReporter functionality"""
    print("\n" + "=" * 60)
    print("TEST 2: MetricReporter Initialization")
    print("=" * 60)
    
    try:
        from src.metrics import MetricReporter
        reporter = MetricReporter()
        print("✓ MetricReporter initialized successfully")
        print(f"  - Methods: {[m for m in dir(reporter) if not m.startswith('_')]}")
        return True
    except Exception as e:
        print(f"✗ MetricReporter initialization failed: {e}")
        return False


def test_event_manager():
    """Test EventManager functionality"""
    print("\n" + "=" * 60)
    print("TEST 3: EventManager Initialization")
    print("=" * 60)
    
    try:
        from src.event_manager import EventManager
        
        event_file = "data/exogenous_events_geocoded.json"
        if not os.path.exists(event_file):
            print(f"⚠️  Event file not found: {event_file}")
            print("   Creating EventManager with mock data...")
            em = EventManager(event_file)
            if em.events:
                print(f"✓ EventManager initialized with {len(em.events)} events")
            else:
                print("✓ EventManager initialized (no events loaded)")
        else:
            em = EventManager(event_file)
            print(f"✓ EventManager initialized with {len(em.events)} events")
            
            # Test getting events
            from datetime import date
            today = date.today()
            events_today = em.get_events_for_date(today)
            anomaly_level = em.get_anomaly_level_for_date(today)
            print(f"  - Events today: {len(events_today)}")
            print(f"  - Anomaly level: {anomaly_level:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ EventManager initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explanation_generator():
    """Test ExplanationGenerator functionality"""
    print("\n" + "=" * 60)
    print("TEST 4: ExplanationGenerator Initialization")
    print("=" * 60)
    
    try:
        from src.explanation_generator import ExplanationGenerator, create_sample_context
        
        gen = ExplanationGenerator()
        print("✓ ExplanationGenerator initialized successfully")
        
        # Create sample context
        sample_context = create_sample_context(146)
        print(f"✓ Sample context created:")
        print(f"  - Score: {sample_context['score']:.2f}")
        print(f"  - Confidence: {sample_context['confidence']:.2f}")
        print(f"  - Tier: {sample_context['tier']}")
        
        # Generate explanation
        explanation = gen.explain_node_ranking(
            node_id=146,
            rank=1,
            context_dict=sample_context
        )
        print(f"✓ Explanation generated:")
        if isinstance(explanation, dict):
            print(f"  - Summary: {explanation.get('summary', '')[:80]}...")
            print(f"  - Risk Level: {explanation.get('risk_level', 'N/A')}")
            print(f"  - Factors: {len(explanation.get('factors', []))} identified")
        else:
            print(f"  - Type: {type(explanation)}")
        
        return True
    except Exception as e:
        print(f"✗ ExplanationGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test Flask API endpoints"""
    print("\n" + "=" * 60)
    print("TEST 5: Flask API Endpoints (Syntax Check)")
    print("=" * 60)
    
    try:
        # Check if app.py has the new endpoints
        with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        endpoints = [
            '/api/explain',
            '/api/metrics',
            '/api/anomaly_status'
        ]
        
        for endpoint in endpoints:
            if endpoint in content:
                print(f"✓ Found endpoint: {endpoint}")
            else:
                print(f"✗ Missing endpoint: {endpoint}")
                return False
        
        # Check for imports
        required_imports = ['MetricReporter', 'EventManager', 'ExplanationGenerator']
        for imp in required_imports:
            if imp in content:
                print(f"✓ Found import reference: {imp}")
            else:
                print(f"✗ Missing import reference: {imp}")
        
        return True
    except Exception as e:
        print(f"✗ API endpoints check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           WEEK 4 API ENDPOINTS TEST SUITE                  ║")
    print("║              Testing Explainability Modules                ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    results = {
        "Imports": test_imports(),
        "MetricReporter": test_metric_reporter(),
        "EventManager": test_event_manager(),
        "ExplanationGenerator": test_explanation_generator(),
        "API Endpoints": test_api_endpoints(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Week 4 API is ready.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
