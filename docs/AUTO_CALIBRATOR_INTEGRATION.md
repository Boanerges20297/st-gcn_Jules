# Auto-Calibrator Daemon - Guia de Integração

## ✅ Implementado

### 1. **AutoCalibratorDaemon** (`src/core/auto_calibrator_daemon.py`)
Roda continuamente em background monitorando:
- **P20** (percentil 20 de confiança) — deve estar > 70%
- **P10** (percentil 10 de confiança) — deve estar > 50%
- **Faction Coverage** — deve estar > 80%

**Comportamento:**
- ✅ Verifica a cada 5 minutos (configurável)
- ✅ Detecta degradações automaticamente
- ✅ Aplica ajustes via `ModelCalibrator`
- ✅ Valida melhorias após 5 min de espera
- ✅ Registra tudo em alerts + health_monitor
- ✅ Para correndo a cada ajuste (cooldown = 5min/região)

### 2. **Integração no App**
- **Inicialização:** Cria instância ao startup
- **Autoinício:** Daemon começa automaticamente
- **Shutdown Graceful:** Para o daemon ao desligar via `atexit`

### 3. **Novos Endpoints de Controle**

#### Start Daemon
```bash
POST /api/admin/health/action
Content-Type: application/json

{
  "action": "start_auto_calibration",
  "confirmed": true
}
```
**Response:** `{"status": "daemon_started"}`

#### Stop Daemon
```bash
POST /api/admin/health/action
Content-Type: application/json

{
  "action": "stop_auto_calibration",
  "confirmed": true
}
```
**Response:** `{"status": "daemon_stopped"}`

#### Calibrate Now (Manual)
```bash
POST /api/admin/health/action
Content-Type: application/json

{
  "action": "calibrate_now",
  "confirmed": true
}
```
**Response:**
```json
{
  "status": "completed",
  "elapsed_seconds": 12.5,
  "message": "Calibração manual concluída em 12.5s"
}
```

#### Get Daemon Status
```bash
GET /api/admin/health/calibration-status
```
**Response:**
```json
{
  "running": true,
  "check_interval": 300,
  "total_cycles": 42,
  "last_check": "2026-05-26T14:30:45.123456",
  "last_adjustments": 2,
  "recent_cycles": [
    {
      "timestamp": "2026-05-26T14:30:45.123456",
      "regions_checked": [
        {
          "region": "fortaleza",
          "confidence": {
            "p20": 0.72,
            "p10": 0.55,
            "faction_coverage": 0.85
          }
        }
      ],
      "adjustments_made": [
        {
          "timestamp": "2026-05-26T14:30:50.123456",
          "region": "fortaleza",
          "metric": "p20",
          "current_value": 0.68,
          "threshold": 0.70,
          "step_number": 1,
          "old_params": {...},
          "status": "applied"
        }
      ],
      "validations": [
        {
          "timestamp": "2026-05-26T14:35:50.123456",
          "region": "fortaleza",
          "metric": "p20",
          "old_value": 0.68,
          "new_value": 0.73,
          "improvement_pct": 7.4,
          "status": "improved"
        }
      ],
      "alerts_dispatched": []
    }
  ]
}
```

---

## 📊 Dashboard Integration

O daemon está totalmente integrado ao dashboard admin:

1. **Seção "🌡️ Termômetro - Auto-Ajuste Territorial"**
   - Mostra status atual do daemon
   - Lista regiões ajustadas
   - Exibe histórico de ciclos
   - Botão "↻ Atualizar" para refresh manual
   - Botões de Controle: Start | Stop | Calibrate Now

2. **Alertas Automáticos**
   - Cada ajuste gera alerta MEDIUM
   - Validações bem-sucedidas geram alerta LOW
   - Máximo de passos atingido gera alerta CRITICAL

3. **Status em Tempo Real**
   - Dashboard atualiza a cada 30 segundos
   - Mostra últimos 5 ciclos de calibração
   - Exibe % de melhoria por ajuste

---

## 🔄 Fluxo Operacional

```
[Every 5 minutes]
    ↓
[AutoCalibratorDaemon._check_and_calibrate()]
    ├─ Obter regiões ativas
    ├─ Verificar cooldown (5min/região)
    ├─ Obter confiança atual via ConfidenceTracker
    ├─ Diagnosticar degradações (P20 < 0.70, P10 < 0.50, etc)
    │
    ├─ [Se degradações encontradas]
    │  ├─ Aplicar ajuste via ModelCalibrator.on_degradation()
    │  ├─ Registrar em alerts (MEDIUM severity)
    │  ├─ Aguardar 5 minutos
    │  └─ Validar melhoria
    │
    ├─ [Validação de Melhoria]
    │  ├─ Comparar confiança antes vs depois
    │  ├─ Calcular % de melhoria
    │  └─ Registrar em alerts (LOW ou MEDIUM)
    │
    └─ Gravar ciclo no histórico

[atexit handler]
    └─ Para daemon gracefully
```

---

## ⚙️ Configuração

### Intervalo de Verificação
**Em `app.py` (linha ~295):**
```python
auto_calibrator_daemon = AutoCalibratorDaemon(
    health_monitor=health_monitor,
    confidence_tracker=confidence_tracker,
    model_calibrator=model_calibrator,
    check_interval=300  # ← Alterar aqui (em segundos)
)
```

Recomendações:
- `60` — Muito frequente (alto CPU, muitos logs)
- **`300`** — Padrão (5 min, bom equilíbrio)
- `600` — Conservador (10 min, menor overhead)
- `1800` — Muito conservador (30 min)

### Limites de Confiança
**Em `src/core/auto_calibrator_daemon.py` (linha ~25):**
```python
_CONFIDENCE_THRESHOLDS = {
    'p20': 0.70,           # ← P20 deve estar > 70%
    'p10': 0.50,           # ← P10 deve estar > 50%
    'faction_coverage': 0.80,  # ← Cobertura > 80%
}
```

### Cooldown entre Ajustes
**Em `src/core/auto_calibrator_daemon.py` (linha ~33):**
```python
_ADJUSTMENT_COOLDOWN = 300  # ← 5 minutos entre ajustes por região
```

---

## 📋 Campos de Confiança Rastreados

Cada região (fortaleza, rmf, interior, global) rastreia:

```python
confidence = {
    'p20': 0.72,              # Percentil 20 (baixa confiança)
    'p10': 0.55,              # Percentil 10 (muita baixa confiança)
    'faction_coverage': 0.85, # % de territórios cobertos
    'timestamp': '2026-05-26T14:30:45.123456',
    'mean': 0.85,             # Média geral
    'std': 0.12,              # Desvio padrão
}
```

---

## 🚨 Alertas Gerados

| Alert Type | Severity | Trigger | Ação |
|-----------|----------|---------|------|
| `auto_calibration_{region}` | MEDIUM | Degradação detectada | Ajuste aplicado |
| `calibration_validation_{region}` | LOW/MEDIUM | Após validação | Melhoria = LOW, Degradação = MEDIUM |
| `calibration_maxed_{region}` | CRITICAL | 5 passos atingidos | Revisão manual necessária |

---

## 📈 Histórico de Ciclos

Cada ciclo registra:
```json
{
  "timestamp": "2026-05-26T14:30:45.123456",
  "regions_checked": [...],           // Regiões verificadas
  "adjustments_made": [...],          // Ajustes aplicados
  "validations": [...],               // Resultados de validação
  "alerts_dispatched": [...]          // Alertas criados
}
```

Máximo de 100 ciclos em memória (FIFO).

---

## 🧪 Testing

### Import Check
```bash
python -c "from src.core.auto_calibrator_daemon import AutoCalibratorDaemon; print('✅ OK')"
```

### Syntax Check
```bash
python -m py_compile src/core/auto_calibrator_daemon.py
```

### API Test
```bash
# Check daemon status
curl http://localhost:5050/api/admin/health/calibration-status

# Trigger manual calibration
curl -X POST http://localhost:5050/api/admin/health/action \
  -H "Content-Type: application/json" \
  -d '{"action": "calibrate_now", "confirmed": true}'

# Start daemon
curl -X POST http://localhost:5050/api/admin/health/action \
  -H "Content-Type: application/json" \
  -d '{"action": "start_auto_calibration", "confirmed": true}'

# Stop daemon
curl -X POST http://localhost:5050/api/admin/health/action \
  -H "Content-Type: application/json" \
  -d '{"action": "stop_auto_calibration", "confirmed": true}'
```

---

## 🔗 Integração com Componentes Existentes

### Health Monitor
- Daemon **lê** métricas via `HealthMonitor.get_system_metrics()`
- Daemon **registra** alertas via `health_monitor.add_alert()`

### Confidence Tracker
- Daemon **lê** confiança atual via `confidence_tracker.get_current_confidence(region)`
- Usa histórico para detectar tendências

### Model Calibrator
- Daemon **chama** `model_calibrator.on_degradation()` para aplicar ajustes
- Daemon **chama** `model_calibrator.on_recovery()` para fazer rollback
- Persiste estado em `data/calibration_state.json`

### Admin Routes
- Daemon **expõe** status via `/api/admin/health/calibration-status`
- Daemon **recebe** comandos via `/api/admin/health/action`

---

## ⚠️ Limitações Conhecidas

1. **Sem Retraining**: O daemon ajusta apenas pesos, não retreina o modelo
   - Para retraining, o sistema existente já tem `load_data_and_models()` a cada 1 hora

2. **Sem Orchestrator Direto**: O daemon não tem acesso direto ao orchestrator
   - Solução: Integração com `orchestrator.calib_params` será feita na próxima fase

3. **Em Memória**: Histórico é mantido apenas em memória durante sessão
   - Dados se perdem ao restart (mas `calibration_state.json` persiste)
   - Solução: Futura persistência em banco de dados

---

## 📝 Logs Esperados

### Startup
```
🔧 Auto-Calibrator Daemon iniciado (verificação a cada 300s)
[Auto-Calibrator] ✅ Daemon iniciado (check a cada 300s)
```

### Ciclo Normal (sem ajustes)
```
[Auto-Calibrator] 🔍 Ciclo iniciado: verificando 4 regiões
[Auto-Calibrator] ✅ fortaleza: métricas saudáveis
[Auto-Calibrator] ✅ rmf: métricas saudáveis
[Auto-Calibrator] ✅ interior: métricas saudáveis
[Auto-Calibrator] ✅ global: métricas saudáveis
[Auto-Calibrator] ✅ Ciclo concluído em 2.3s
```

### Ciclo com Ajuste
```
[Auto-Calibrator] 🔍 Ciclo iniciado: verificando 4 regiões
[Auto-Calibrator] ⚠️ fortaleza: 1 degradação(ões) detectada(s)
[Auto-Calibrator] ⚙️ fortaleza: Passo 1/5 aplicado para P20=68.0%
[Auto-Calibrator] ⏳ Aguardando 300s para modelo estabilizar...
[Auto-Calibrator] ✅ fortaleza/P20: 68.0% → 73.0% (+7.4%)
[Auto-Calibrator] ✅ Ciclo concluído em 305.2s
```

### Crítico
```
[Auto-Calibrator] ❌ fortaleza: máximo de passos (5) atingido
🚨 CRITICAL ALERT: fortaleza: Auto-calibração atingiu limite. Revisão manual necessária.
```

### Shutdown
```
[SHUTDOWN] Parando daemons...
[SHUTDOWN] ✅ Auto-Calibrator parado
```

---

## ✅ Checklist de Integração

- [x] Arquivo `auto_calibrator_daemon.py` criado
- [x] Classe `AutoCalibratorDaemon` implementada
- [x] Métodos de ciclo (diagnóstico, ajuste, validação) prontos
- [x] Integração em `admin_health_routes.py` (3 ações + 1 endpoint)
- [x] Inicialização em `app.py` (instância + startup + shutdown)
- [x] Logging configurado com prefixo `[Auto-Calibrator]`
- [x] Histórico de ciclos mantido
- [x] Cooldown por região implementado
- [x] Alerts integrados com HealthMonitor
- [x] Syntax validation passed ✅
- [x] Import check passed ✅

---

**Pronto para uso!** 🚀

Você agora tem um sistema completamente autônomo de calibração que:
1. Monitora confiança continuamente
2. Detecta degradações automaticamente  
3. Aplica ajustes sem intervenção manual
4. Valida melhorias em tempo real
5. Se integra perfeitamente ao dashboard existente

Não precisa mais procurar manualmente! 🎯
