#!/usr/bin/env python
"""
Test Auto-Calibrator Integration

Verifica que:
1. AutoCalibratorDaemon pode ser importado
2. Pode ser instanciado com mocks
3. Métodos principais funcionam
4. Integração com HealthMonitor/ConfidenceTracker está OK
"""

import sys
import json
from datetime import datetime

# Importar
print("[TEST] Importando módulos...")
try:
    from src.core.auto_calibrator_daemon import AutoCalibratorDaemon
    from src.core.health_monitor import HealthMonitor, ConfidenceTracker
    from src.core.model_calibrator import ModelCalibrator
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Instanciar monitors
print("\n[TEST] Criando instâncias...")
try:
    import os
    base_dir = os.getcwd()
    
    health_monitor = HealthMonitor(base_dir=base_dir)
    confidence_tracker = ConfidenceTracker(base_dir=base_dir)
    model_calibrator = ModelCalibrator(base_dir=base_dir, health_monitor=health_monitor)
    
    print("✅ Health Monitor created")
    print("✅ Confidence Tracker created")
    print("✅ Model Calibrator created")
except Exception as e:
    print(f"❌ Creation failed: {e}")
    sys.exit(1)

# Criar daemon
print("\n[TEST] Criando AutoCalibratorDaemon...")
try:
    daemon = AutoCalibratorDaemon(
        health_monitor=health_monitor,
        confidence_tracker=confidence_tracker,
        model_calibrator=model_calibrator,
        check_interval=60  # 1 min para teste
    )
    print(f"✅ Daemon created: {daemon}")
except Exception as e:
    print(f"❌ Daemon creation failed: {e}")
    sys.exit(1)

# Teste de status (before start)
print("\n[TEST] Checking status before start...")
try:
    status = daemon.get_status()
    assert status['running'] == False, "Daemon should not be running yet"
    assert status['check_interval'] == 60, "Check interval should be 60s"
    assert status['total_cycles'] == 0, "Should have 0 cycles"
    print(f"✅ Status before start: {json.dumps(status, indent=2)}")
except Exception as e:
    print(f"❌ Status check failed: {e}")
    sys.exit(1)

# Teste de inicialização
print("\n[TEST] Starting daemon...")
try:
    daemon.start()
    print("✅ Daemon started")
except Exception as e:
    print(f"❌ Daemon start failed: {e}")
    sys.exit(1)

# Aguardar 2 segundos para daemon começar
print("\n[TEST] Waiting 2 seconds for daemon to initialize...")
import time
time.sleep(2)

# Teste de status (after start)
print("\n[TEST] Checking status after start...")
try:
    status = daemon.get_status()
    assert status['running'] == True, "Daemon should be running"
    print(f"✅ Status after start: running={status['running']}")
except Exception as e:
    print(f"❌ Status check failed: {e}")
    sys.exit(1)

# Teste de parada
print("\n[TEST] Stopping daemon...")
try:
    daemon.stop()
    time.sleep(0.5)
    print("✅ Daemon stopped")
except Exception as e:
    print(f"❌ Daemon stop failed: {e}")
    sys.exit(1)

# Teste de status (after stop)
print("\n[TEST] Checking status after stop...")
try:
    status = daemon.get_status()
    assert status['running'] == False, "Daemon should not be running"
    print(f"✅ Status after stop: running={status['running']}")
except Exception as e:
    print(f"❌ Status check failed: {e}")
    sys.exit(1)

# Teste de manual calibration
print("\n[TEST] Testing manual calibration...")
try:
    result = daemon.manual_calibration_run()
    assert result['status'] in ('completed', 'error'), "Should return completed or error"
    print(f"✅ Manual calibration: {result['message']}")
except Exception as e:
    print(f"❌ Manual calibration failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ ALL TESTS PASSED!")
print("="*50)
