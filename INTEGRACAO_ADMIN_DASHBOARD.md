# 🔗 Integração do Admin Dashboard no app.py

## Como Integrar os Novos Componentes

### 1. Adicionar Imports em `app.py`

```python
# No topo do arquivo app.py, adicionar:
import time
from src.core.health_monitor import HealthMonitor, ConfidenceTracker
from src.core.admin_health_routes import create_admin_health_blueprint
```

### 2. Inicializar Monitores (após criar Flask app)

```python
# Após `app = Flask(__name__)`:

# Inicializar Health Monitor
health_monitor = HealthMonitor(base_dir=BASE_DIR)
confidence_tracker = ConfidenceTracker(base_dir=BASE_DIR)
```

### 3. Registrar Blueprint

```python
# Registrar rotas de admin health
admin_health_bp = create_admin_health_blueprint(health_monitor, confidence_tracker)
app.register_blueprint(admin_health_bp)
```

### 4. Adicionar Middleware para Rastreamento de Requisições

```python
# Adicionar este código no app.py:

@app.before_request
def track_request_start():
    """Marca o início de uma requisição."""
    request.start_time = time.time()

@app.after_request
def track_request_end(response):
    """Rastreia latência e status de cada requisição."""
    if hasattr(request, 'start_time'):
        latency_ms = (time.time() - request.start_time) * 1000
        success = response.status_code < 400
        health_monitor.track_api_request(
            endpoint=request.path,
            latency_ms=latency_ms,
            success=success
        )
    return response
```

### 5. Rota do Dashboard (opcional, já incluída no blueprint)

```python
# A rota principal do dashboard já está em admin_health_routes.py:
# GET /admin/health -> renderiza admin_health_dashboard.html
```

---

## Verificação Pós-Integração

### 1. Verificar Imports
```bash
python -c "from src.core.health_monitor import HealthMonitor; print('✅ Importação OK')"
```

### 2. Testar Health Check
```bash
# Com o app rodando:
curl http://localhost:5050/api/admin/health/summary | jq '.'
```

**Resposta esperada:**
```json
{
  "timestamp": "2026-03-01T19:08:36.955Z",
  "system": {
    "status": "OK",
    "cpu_percent": 38,
    "memory_mb": 2100,
    "uptime_str": "42d 15h 23m"
  },
  "api": {
    "total_requests": 87280,
    "error_rate_percent": 0.2
  }
}
```

### 3. Acessar Dashboard
Navegue para: **http://localhost:5050/admin/health**

---

## Arquivos Necessários

✅ Existem e podem ser integrados:
- `src/core/health_monitor.py` ← HealthMonitor + ConfidenceTracker
- `src/core/admin_health_routes.py` ← Rotas da API
- `templates/admin_health_dashboard.html` ← Frontend

---

## Dependências Necessárias

```
psutil==5.9.4  # Para coletar métricas do sistema (CPU, memória, disco)
```

**Adicionar a requirements.txt:**
```bash
pip install psutil
echo "psutil==5.9.4" >> requirements.txt
```

---

## Exemplo Completo de Integração (app.py)

```python
# === IMPORTS ===
from flask import Flask, jsonify, render_template, request
import sys, os, time
import logging

# Health Monitor
from src.core.health_monitor import HealthMonitor, ConfidenceTracker
from src.core.admin_health_routes import create_admin_health_blueprint

# === SETUP ===
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)

# === INICIALIZAR MONITORES ===
health_monitor = HealthMonitor(base_dir=BASE_DIR)
confidence_tracker = ConfidenceTracker(base_dir=BASE_DIR)

# === REGISTRAR BLUEPRINT ===
admin_health_bp = create_admin_health_blueprint(health_monitor, confidence_tracker)
app.register_blueprint(admin_health_bp)

# === MIDDLEWARE DE RASTREAMENTO ===
@app.before_request
def track_request_start():
    request.start_time = time.time()

@app.after_request
def track_request_end(response):
    if hasattr(request, 'start_time'):
        latency_ms = (time.time() - request.start_time) * 1000
        success = response.status_code < 400
        health_monitor.track_api_request(
            endpoint=request.path,
            latency_ms=latency_ms,
            success=success
        )
    return response

# === ROTAS EXISTENTES ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/risk')
def get_risk():
    # ... código existente ...
    pass

# === INICIAR ===
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050)
```

---

## Testes Pós-Integração

### Teste 1: Health Summary
```bash
curl http://localhost:5050/api/admin/health/summary | jq '.'
```

### Teste 2: API Stats
```bash
curl http://localhost:5050/api/admin/health/api-stats | jq '.global'
```

### Teste 3: Alertas
```bash
curl http://localhost:5050/api/admin/health/alerts?limit=5 | jq '.'
```

### Teste 4: Dashboard HTML
```bash
# Abrir no navegador:
# http://localhost:5050/admin/health
```

### Teste 5: Criar Alerta (POST)
```bash
curl -X POST http://localhost:5050/api/admin/health/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "type": "test_alert",
    "severity": "MEDIUM",
    "message": "Teste de alerta"
  }'
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'psutil'"
```bash
pip install psutil
```

### ❌ "HealthMonitor: Permission denied"
Certifique-se de que `/data` existe e tem permissão de escrita:
```bash
mkdir -p data/archives
chmod 755 data
```

### ❌ "TypeError: cannot pickle <function>"
Pode ocorrer ao salvar histórico. Solução: Usar `json` ao invés de `pickle`.

---

## Configuração Opcional: Alertas por Email (Futuro)

```python
# Adicionar em admin_health_routes.py:

import smtplib
from email.mime.text import MIMEText

def send_alert_email(alert):
    """Envia alerta por email."""
    if alert['severity'] not in ['CRITICAL', 'HIGH']:
        return
    
    msg = MIMEText(f"Alerta: {alert['message']}")
    msg['Subject'] = f"🔔 [{alert['severity']}] REPORT PREVIEW Alert"
    msg['From'] = "noreply@cpraio.ce.gov.br"
    msg['To'] = "admin@cpraio.ce.gov.br"
    
    # smtp_server.send_message(msg)
```

---

## Monitoramento Contínuo (Futuro)

Para monitoramento em tempo real com Prometheus/Grafana:

```python
# Adicionar em app.py:
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
request_latency = Histogram('api_request_duration_ms', 'API request latency')
```

---

## Checklist Final de Integração

- [ ] `psutil` instalado em requirements.txt
- [ ] Imports adicionados ao app.py
- [ ] HealthMonitor + ConfidenceTracker inicializados
- [ ] Blueprint registrado
- [ ] Middleware de rastreamento adicionado
- [ ] Testes de endpoints passando
- [ ] Dashboard acessível em `/admin/health`
- [ ] Histórico sendo salvo em `data/health_*.json`
- [ ] Alertas funcionando
- [ ] Logs sendo coletados

---

## Endpoints Disponíveis Após Integração

```
GET  /admin/health                              # Dashboard HTML
GET  /api/admin/health/summary                  # Summary completo
GET  /api/admin/health/metrics/system           # Métricas de sistema
GET  /api/admin/health/api-stats                # Stats de API
GET  /api/admin/health/alerts                   # Listar alertas
POST /api/admin/health/alerts                   # Criar alerta
GET  /api/admin/health/confidence-history       # Histórico de confiança
GET  /api/admin/health/confidence/current       # Confiança atual
POST /api/admin/health/action                   # Ações admin
```

---

**Versão:** 1.0  
**Data:** 01 de Março de 2026  
**Status:** ✅ Pronto para Integração
