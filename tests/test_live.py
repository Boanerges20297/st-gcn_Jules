"""Testes rápidos contra a API em produção (localhost:5050)"""
import requests
import json

BASE = "http://localhost:5050"

def test_health_summary():
    r = requests.get(f"{BASE}/api/admin/health/summary", timeout=10)
    d = r.json()
    print(f"[health/summary] status={r.status_code} | CPU={d.get('cpu_percent')}% | MEM={d.get('memory_percent')}% | uptime={d.get('uptime_seconds')}s")

def test_calibration_status():
    r = requests.get(f"{BASE}/api/admin/health/calibration-status", timeout=10)
    print(f"[calibration-status] status={r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

def test_alerts():
    r = requests.get(f"{BASE}/api/admin/health/alerts?resolved=false", timeout=10)
    alerts = r.json().get("alerts", [])
    print(f"[alerts] status={r.status_code} | total={len(alerts)}")
    for a in alerts[:5]:
        print(f"  [{a['severity']}] {a['message'][:80]}")

def test_risk_baseline():
    r = requests.get(f"{BASE}/api/risk", timeout=30)
    data = r.json().get("data", [])
    top = sorted(data, key=lambda x: -x.get("risk_score", 0))[:10]
    print(f"[/api/risk] status={r.status_code} | total={len(data)} nós")
    for i, t in enumerate(top):
        print(f"  {i+1}. {t['name']}: {t.get('risk_score', 0):.1f}%")

def test_data_quality():
    r = requests.get(f"{BASE}/api/admin/health/data-quality", timeout=10)
    print(f"[data-quality] status={r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    for fn in [test_health_summary, test_calibration_status, test_alerts, test_risk_baseline, test_data_quality]:
        print("\n" + "="*60)
        try:
            fn()
        except Exception as e:
            print(f"  ERRO: {e}")
