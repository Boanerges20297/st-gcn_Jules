# 📊 Dashboard Admin de Health - REPORT PREVIEW

## Visão Geral

O **Dashboard Admin de Health** é um painel executivo exclusivo para monitoramento de saúde do sistema, métricas de confiança e alertas automáticos. Acesso restrito a Administradores.

**URL:** `http://localhost:5050/admin/health`

---

## 1. Componentes Principais

### 1.1 Status Geral do Sistema

```
┌─────────────────────────────────────────────────┐
│ 🟢 SISTEMA OPERACIONAL                          │
│                                                 │
│ Disponibilidade: 99.7% ↑                       │
│ Uptime: 42 dias 15h 23m                        │
│ Última Falha: 18-02-2026 (13 dias atrás)       │
│ Próxima Manutenção: 15-03-2026                 │
└─────────────────────────────────────────────────┘
```

### 1.2 Indicadores em Tempo Real

```
┌─────────────┬──────────┬─────────┬──────────────┐
│ Métrica     │ Atual    │ Limite  │ Status       │
├─────────────┼──────────┼─────────┼──────────────┤
│ CPU         │ 38%      │ 80%     │ 🟢 OK        │
│ Memória     │ 2.1GB    │ 4GB     │ 🟢 OK        │
│ Disco       │ 23GB     │ 100GB   │ 🟢 OK        │
│ Latência P95│ 280ms    │ 500ms   │ 🟢 OK        │
│ Taxa Erro   │ 0.2%     │ 1%      │ 🟢 OK        │
│ Gemini API  │ Online   │ -       │ 🟢 Online    │
└─────────────┴──────────┴─────────┴──────────────┘
```

---

## 2. Métricas de Confiança (por Região e Período)

### 2.1 Série Temporal de Confiança

```
Confiança Global (últimos 30 dias):

100% ┤                                     ╱╲
     │                                 ╱╲╱  ╲╱╲
 90% ┤     ╱╲                      ╱╲╱
     │  ╱╱  ╲╱╲                 ╱╲╱
 80% ┤╱╱      ╲               ╱╱
     │         ╲╱╲───────────╱
 70% ├─────────────────────────────────────────
     └─────────────────────────────────────────
     01  05  10  15  20  25  30 (Dias de Março)

Métrica:
├── Pico: 92% (dia 22, após retraining)
├── Mínimo: 76% (dia 05, após evento crítico)
├── Média: 87% ✅
└── Tendência: ↑ Crescendo (+2% últimos 7 dias)
```

### 2.2 Confiança por Região

```
Região          | P10 Score | P20 Score | Tendência | Status
────────────────┼───────────┼───────────┼───────────┼─────────
FORTALEZA       | 88%       | 91%       | ↑ +3%     | 🟢 OK
RMF             | 82%       | 86%       | → 0%      | 🟡 Atenção
INTERIOR        | 75%       | 79%       | ↓ -2%     | 🟡 Atenção
────────────────┼───────────┼───────────┼───────────┼─────────
GLOBAL          | 87%       | 89%       | ↑ +2%     | 🟢 OK
```

### 2.3 Histórico Granular

**Filtros disponíveis:**
- Data/Período: Últimas 24h, 7 dias, 30 dias, customizado
- Região: Fortaleza, RMF, Interior, Global
- Métrica: P10, P20, Precision, Recall, F1-Score

**Exemplo de Série Histórica:**
```json
{
  "period": "2026-02-20 a 2026-03-01",
  "region": "fortaleza",
  "data": [
    {
      "date": "2026-02-20",
      "p10": 0.85,
      "p20": 0.89,
      "precision": 0.87,
      "recall": 0.84,
      "f1_score": 0.855,
      "events_evaluated": 145,
      "hits": ["BOM JARDIM", "PIRAMBU", "PARQUE MANIBURA"]
    },
    {
      "date": "2026-02-21",
      "p10": 0.84,
      "p20": 0.88,
      "precision": 0.86,
      "recall": 0.83,
      "f1_score": 0.845,
      "events_evaluated": 152,
      "hits": ["BOM JARDIM", "PARQUE MANIBURA", "BOM MEANDRO"]
    }
  ]
}
```

---

## 3. Alertas Automáticos

### 3.1 Tipos de Alertas

| Alerta | Critério | Severidade | Ação |
|--------|----------|-----------|------|
| **Modelo Degradado** | P10 < 70% ou P20 < 75% | 🔴 Crítico | Email + SMS admin |
| **Dados Atrasados** | Último evento > 24h | 🔴 Crítico | Email admin |
| **Gemini API Down** | Falha por > 10 min | 🟠 Alto | Email + Fallback |
| **CPU Elevada** | > 80% por > 5 min | 🟠 Alto | Log + Monitoring |
| **Memória Alta** | > 85% | 🟠 Alto | Log + Alertas |
| **Disco Baixo** | < 5GB | 🟠 Alto | Log + Alertas |
| **Taxa Erro Alta** | > 2% em 5 min | 🟡 Médio | Log |
| **Latência P95 Alta** | > 1s | 🟡 Médio | Log |
| **Confiança Flutuante** | Var > ±5% em 1h | 🟡 Médio | Log |
| **RMF/Interior Baixa** | P10 < 75% | 🟡 Médio | Log + Review |

### 3.2 Centro de Alertas

```
┌────────────────────────────────────────────┐
│ 🔔 ALERTAS ATIVOS (3)                      │
├────────────────────────────────────────────┤
│                                            │
│ 🟡 Confiança RMF em Queda                 │
│    Data: 2026-03-01 18:45                 │
│    P10: 82% → 79% (variação -3%)          │
│    Status: Monitorando                    │
│    Ação: Nenhuma (dentro de limites)      │
│                                            │
│ 🟡 Latência P95 Acelerada                 │
│    Data: 2026-03-01 19:02                 │
│    Latência: 280ms → 450ms                │
│    Status: Investigando                   │
│    Ação: Verificar carga de requisições   │
│                                            │
│ 🟡 Última Avaliação > 7 dias               │
│    Data: 2026-02-22 (9 dias atrás)        │
│    Status: Planejado para 2026-03-01      │
│    Ação: Executar backtesting manual      │
│                                            │
└────────────────────────────────────────────┘
```

### 3.3 Histórico de Alertas

```
Data/Hora           | Alerta              | Severidade | Resolvido | Ação
────────────────────┼─────────────────────┼────────────┼──────────┼──────────
2026-03-01 18:45    | Confiança RMF Baixa | 🟡 Médio   | Não      | Monitor
2026-02-28 14:22    | Gemini API Timeout  | 🟠 Alto    | Sim      | Retry OK
2026-02-27 09:15    | CPU Elevada         | 🟠 Alto    | Sim      | Restart
2026-02-20 16:40    | Disco Baixo         | 🟠 Alto    | Sim      | Limpeza
```

---

## 4. Painel de Saúde da API

### 4.1 Performance por Endpoint

```
Endpoint                      | Req/dia | P50   | P95   | Erro% | Status
──────────────────────────────┼─────────┼───────┼───────┼───────┼───────
GET /api/risk                 | 45,210  | 145ms | 280ms | 0.1%  | 🟢 OK
POST /api/exogenous/parse     | 3,420   | 2.1s  | 4.8s  | 0.8%  | 🟢 OK
POST /api/simulate            | 1,240   | 820ms | 1.5s  | 0.2%  | 🟢 OK
GET /api/explain/<id>         | 28,960  | 95ms  | 180ms | 0.0%  | 🟢 OK
GET /api/polygons             | 8,450   | 210ms | 420ms | 0.0%  | 🟢 OK
──────────────────────────────┴─────────┴───────┴───────┴───────┴───────
TOTAL                         | 87,280  | -     | -     | 0.2%  | 🟢 OK
```

### 4.2 Taxa de Erro Detalhada

```
Tipo de Erro            | Ocorrências | % Total | Última Ocorrência
────────────────────────┼─────────────┼─────────┼───────────────────
Validation Error        | 12          | 40%     | 2 horas atrás
LLM Timeout             | 9           | 30%     | 4 horas atrás
Model Error             | 6           | 20%     | 8 horas atrás
Database Error          | 3           | 10%     | 1 dia atrás
────────────────────────┼─────────────┼─────────┼───────────────────
TOTAL (últimas 24h)     | 30          | 100%    | 2 horas atrás
```

---

## 5. Monitoramento de Dados

### 5.1 Qualidade de Dados

```
┌──────────────────────────────────────────┐
│ 📊 QUALIDADE DE DADOS                    │
├──────────────────────────────────────────┤
│ Eventos Históricos (120 dias): 387,245   │
│ Eventos Exógenos (últimos 7d): 12,456    │
│ Taxa Completude: 98.7% ✅               │
│                                          │
│ Anomalias Detectadas: 23                 │
│ ├── Outliers (risco): 15                 │
│ ├── Missing Values: 5                    │
│ └── Duplicatas: 3                        │
│                                          │
│ Última Sincronização CIOPS: 45 min atrás │
│ Status: 🟢 Sincronizado                  │
└──────────────────────────────────────────┘
```

### 5.2 Arquivamento

```
Período             | Eventos | Tamanho | Localização
────────────────────┼─────────┼─────────┼─────────────────────────
2026-02 (Feb-22)    | 18,452  | 45MB    | data/archives/exogenous_..
2026-01             | 21,340  | 52MB    | data/archives/exogenous_..
2025-12             | 19,876  | 48MB    | data/archives/exogenous_..
────────────────────┼─────────┼─────────┼─────────────────────────
TOTAL ARQUIVADO     | 456,892 | 1.2GB   | data/archives/
ATIVO (< 7 dias)    | 12,456  | 28MB    | data/exogenous_events.json
```

---

## 6. Monitoramento de Modelos

### 6.1 Status de Treino

```
Modelo      | Versão | Data Treino | Epochs | Loss Final | Status
────────────┼────────┼─────────────┼────────┼────────────┼────────
Fortaleza   | 2.1    | 2026-02-28  | 150    | 0.0234     | ✅ OK
RMF         | 2.0    | 2026-02-15  | 150    | 0.0312     | ⚠️ Revisar
Interior    | 2.0    | 2026-02-01  | 150    | 0.0298     | ⚠️ Revisar
────────────┴────────┴─────────────┴────────┴────────────┴────────

⚠️ RECOMENDAÇÃO: Retreinar RMF e Interior (degradação esperada)
   Próximo Treino: 15-03-2026
```

### 6.2 Convergência de Treino (Gráfico)

```
Loss durante Treino do Fortaleza (v2.1):

0.5 ┤ ✕
    │ ✕ ✕
0.4 ┤   ✕ ✕
    │     ✕ ✕
0.3 ┤       ✕ ✕
    │         ✕ ✕ ✕
0.2 ┤           ✕ ✕ ✕
    │             ✕ ✕ ✕ ✕
0.1 ┤               ✕ ✕ ✕ ✕ ✕
    │                 ✕ ✕ ✕ ✕ ✕
0.0 ├──────────────────────────────
    0   20   40   60   80   100  150 (Epochs)

Status: 🟢 Convergência Normal
```

---

## 7. Integração com Alertas

### 7.1 Notificações

**Canais Configuráveis:**
- ✅ Email (admin@cpraio.ce.gov.br)
- ✅ Slack (#report-preview-alerts)
- 📋 SMS (futuro)
- 📋 Webhook (futuro)

**Exemplo de Email:**
```
Assunto: 🔴 ALERTA CRÍTICO: Dados Atrasados no REPORT PREVIEW

Olá Admin,

O Sistema REPORT PREVIEW detectou que não recebe eventos exógenos 
há mais de 24 horas.

Última Sincronização: 2026-03-01 10:30
Agora: 2026-03-02 11:15
Atraso: 24h 45m

AÇÕES RECOMENDADAS:
1. Verificar conectividade com CIOPS
2. Verificar logs em logs/app.log
3. Contactar time de DevOps

Status: CRÍTICO 🔴
Timestamp: 2026-03-02T11:15:00Z
```

### 7.2 Configurações de Alertas

```json
{
  "alerts": {
    "model_degraded": {
      "enabled": true,
      "threshold_p10": 0.70,
      "threshold_p20": 0.75,
      "channels": ["email", "slack"],
      "cooldown_minutes": 120
    },
    "data_stale": {
      "enabled": true,
      "max_age_hours": 24,
      "channels": ["email", "slack"],
      "cooldown_minutes": 60
    },
    "api_error_rate": {
      "enabled": true,
      "threshold_percent": 2.0,
      "window_minutes": 5,
      "channels": ["email", "slack"],
      "cooldown_minutes": 30
    }
  }
}
```

---

## 8. Ações Administrativas

### 8.1 Botões de Ação Disponíveis

```
┌─────────────────────────────────────┐
│ ⚙️ AÇÕES ADMINISTRATIVAS             │
├─────────────────────────────────────┤
│                                     │
│ [🔄] Forçar Recalcular Risco        │
│ [📊] Executar Backtesting Manual    │
│ [🧹] Limpar Cache de Explicações   │
│ [📦] Arquivar Eventos Manual        │
│ [🔧] Recarregar Modelos             │
│ [📋] Exportar Relatório de Saúde    │
│ [⚠️] Marcar Alerta como Resolvido   │
│ [🔌] Testar Gemini API              │
│                                     │
└─────────────────────────────────────┘
```

### 8.2 Logs de Auditoria

```
Timestamp           | Usuário  | Ação                      | Status
────────────────────┼──────────┼───────────────────────────┼────────
2026-03-01 18:30    | admin_01 | Executar Backtesting      | ✅ OK
2026-03-01 17:45    | admin_02 | Limpar Cache              | ✅ OK
2026-03-01 14:20    | admin_01 | Marcar Alerta Resolvido   | ✅ OK
2026-02-28 10:15    | admin_03 | Exportar Relatório        | ✅ OK
```

---

## 9. Endpoints do Admin Dashboard

### 9.1 Dados do Dashboard

```http
GET /api/admin/health/summary
```

Retorna snapshot completo do sistema.

```json
{
  "timestamp": "2026-03-01T19:08:36.955Z",
  "system": {
    "uptime_seconds": 3687023,
    "cpu_percent": 38,
    "memory_mb": 2100,
    "disk_gb": 23
  },
  "api": {
    "requests_today": 87280,
    "error_rate_percent": 0.2,
    "latency_p95_ms": 280
  },
  "model": {
    "confidence_global": 0.87,
    "p10_scores": {"fortaleza": 0.88, "rmf": 0.82, "interior": 0.75},
    "last_evaluation": "2026-02-22"
  },
  "alerts": {
    "total": 3,
    "critical": 0,
    "high": 0,
    "medium": 3
  }
}
```

### 9.2 Histórico de Confiança

```http
GET /api/admin/health/confidence-history?period=30&region=fortaleza
```

### 9.3 Executar Ação

```http
POST /api/admin/health/action
Content-Type: application/json

{
  "action": "clear_cache",
  "confirmed": true
}
```

---

## 10. Segurança do Dashboard

### 10.1 Acesso Restrito

- **Autenticação:** JWT Token obrigatório
- **Autorização:** Role `ADMIN` necessário
- **IP Whitelist:** (futuro) Apenas IPs corporativos
- **Auditoria:** Todas as ações logadas

### 10.2 Exemplo de Requisição Autenticada

```bash
curl -H "Authorization: Bearer <JWT_TOKEN>" \
     http://localhost:5050/api/admin/health/summary
```

---

## 11. Checklist de Implementação

### Backend
- [ ] Função `get_system_health()` em `src/core/health_monitor.py`
- [ ] Função `get_confidence_history()` com filtros
- [ ] Sistema de alertas com persistência
- [ ] Endpoints `/api/admin/health/*`
- [ ] Logs de auditoria
- [ ] Notificações por email/Slack

### Frontend
- [ ] Página `/admin/health` com layout responsivo
- [ ] Gráficos de série temporal (Chart.js)
- [ ] Tabelas com filtros e paginação
- [ ] Botões de ação administrativos
- [ ] Auto-refresh a cada 30s

### Testes
- [ ] Teste de carga para alertas
- [ ] Validação de acesso (autenticação)
- [ ] Teste de persistência de alertas

---

**Última atualização:** 01 de Março de 2026  
**Versão:** 2.0
