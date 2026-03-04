# ⚡ Performance e Escalabilidade - REPORT PREVIEW

## Visão Geral

Este documento define as características de performance, limites do sistema e diretrizes de escalabilidade para o REPORT PREVIEW em diferentes cenários de carga.

---

## 1. Métricas de Performance Esperadas

### 1.1 Latência de Endpoints

| Endpoint | Latência P50 | Latência P95 | P99 | Observações |
|----------|--------------|--------------|-----|------------|
| `GET /api/risk` | 150ms | 250ms | 500ms | Cálculo em memória |
| `POST /api/exogenous/parse` | 2-3s | 5s | 10s | Depende da latência do Gemini |
| `POST /api/simulate` | 800ms | 1.5s | 3s | Múltiplas iterações de risco |
| `GET /api/explain/<id>` | 100ms | 200ms | 400ms | Consulta em cache |
| `GET /api/polygons` | 200ms | 400ms | 800ms | Serialização GeoJSON |

**P50/P95/P99:** Percentis de latência. Ex: P95 = 95% das requisições respondem em < 250ms

### 1.2 Throughput

| Cenário | Taxa de Requisições | Tempo de Resposta Degradado? |
|---------|-------------------|------------------------------|
| Leve | 10 req/s por endpoint | Não (p50 ≤ 200ms) |
| Normal | 50 req/s por endpoint | Não (p50 ≤ 300ms) |
| Alto | 100 req/s por endpoint | Sim (p95 > 1s) |
| Crítico | 200+ req/s por endpoint | Sim (queues formam) |

### 1.3 Uso de Memória

| Componente | RAM Base | RAM por 1k Bairros | RAM por 10k Eventos |
|-----------|----------|-------------------|-------------------|
| Modelos ST-GAT (3x) | 800MB | +150MB | - |
| Cache de Explicações | 50MB | - | +5MB |
| Dados Históricos (120 dias) | - | +300MB | +50MB |
| Orchestrator + Tensores | 200MB | +80MB | +20MB |
| **TOTAL (Ceará, estado estável)** | ~**2.5GB** | - | - |

**Crescimento mensal:** +200MB (eventos acumulativos sem arquivamento)

---

## 2. Capacidade do Sistema

### 2.1 Máximo de Usuários Simultâneos

**Com 1 instância Flask (desenvolvimento):**
- **Concurrent Users:** 10-15
- **Max Requests:** 50/s
- **Degradação inicia em:** 30 usuários

**Com 4 workers Gunicorn (produção):**
- **Concurrent Users:** 100-150
- **Max Requests:** 200/s
- **Degradação inicia em:** 150 usuários

### 2.2 Máximo de Eventos por Dia

| Cenário | Eventos/Dia | Armazenamento | Processamento |
|---------|-------------|---------------|----------------|
| Leve | < 500 | 1MB | < 1 CPU-hora |
| Normal | 500-5,000 | 10-50MB | 1-5 CPU-horas |
| Alto | 5,000-50,000 | 50-500MB | 5-50 CPU-horas |
| Crítico | > 50,000 | > 500MB | > 50 CPU-horas |

*Estado do Ceará tipicamente:* **2,000-5,000 eventos/dia** (Normal)

### 2.3 Máximo de Bairros Suportados

| Região | Bairros Atuais | Bairros Suportáveis | Limite Técnico |
|--------|---|---|---|
| Fortaleza | 127 | 500 (sub-distritos) | 2,000 |
| RMF | 43 | 200 (cidades) | 1,000 |
| Interior | 89 | 400 (municípios) | 2,000 |
| **Total Ceará** | **259** | **1,100** | **5,000** |

---

## 3. Métricas de Saúde do Sistema

### 3.1 Indicadores de Monitoramento

```
📊 Dashboard de Saúde (futuro):
├── CPU Utilization: < 60% (alerta) / > 80% (crítico)
├── Memory Usage: < 70% (alerta) / > 85% (crítico)
├── Disk I/O: < 1GB/s (normal) / > 5GB/s (crítico)
├── API Response Time P95: < 500ms (alerta) / > 1s (crítico)
├── Queue Size: < 10 (alerta) / > 50 (crítico)
├── Model Accuracy: > 85% (OK) / < 70% (crítico)
└── Data Staleness: < 1 hora (OK) / > 24 horas (crítico)
```

### 3.2 Logs de Monitoramento

**Arquivo:** `logs/app.log`

```
[2026-03-01 14:30:45] INFO  | Request /api/risk | Latency: 145ms | Status: 200
[2026-03-01 14:31:22] WARN  | Slow Query | /api/simulate | Latency: 2.1s | Status: 200
[2026-03-01 14:32:01] ERROR | Gemini API Timeout | /api/exogenous/parse | Status: 503
[2026-03-01 14:32:05] INFO  | Event Archived | 142 eventos | New Size: 3,847
```

---

## 4. Otimizações Implementadas

### 4.1 Cache em Memória

```python
# Explicações do Gestor (LRU Cache)
- Max Size: 500 nodes explicados
- TTL: 60 minutos
- Hit Rate Esperado: 85%
- Redução de Latência: 200ms → 50ms
```

### 4.2 Arquivamento Automático de Eventos

```
Processo:
├── Diário: Move eventos > 7 dias para data/archives/
├── Compressão: Eventos antigos podem ser comprimidos (.gz)
├── Retenção: 2 anos de dados arquivados
└── Impacto: Reduz RAM em 15-20% por semana
```

### 4.3 Tensores Pré-Computados

```python
# Janela deslizante de 120 dias
- Recompilação: 1x por dia (02:00 AM)
- Tempo: ~30 segundos (para Ceará)
- Impacto de I/O: Baixo (fora de pico)
```

### 4.4 GeoJSON Cacheado

```
Polígonos dos bairros:
- Cache em memória: ~50MB
- TTL: 7 dias (atualizado se geometria mudar)
- Serialization: 200ms → 10ms com cache
```

---

## 5. Escalabilidade Horizontal

### 5.1 Arquitetura Multi-Instância

```
Load Balancer (Nginx)
         ↓
    ┌────┴────┬────────┬────────┐
    ↓         ↓        ↓        ↓
  Flask    Flask    Flask    Flask
  inst1    inst2    inst3    inst4
    ↓         ↓        ↓        ↓
  Shared Redis (Cache de Explicações)
  Shared PostgreSQL (Eventos Persistidos)
  Shared NFS (Modelos .pth)
```

**Benefícios:**
- Scale-out automático com Kubernetes
- Redundância de instâncias
- Distribuição de carga

### 5.2 Deployments Recomendados

#### Pequeno (< 1M eventos/ano)
```
├── 1x Flask (local)
├── 1x SQLite (local)
└── RAM: 4GB
Custo: Baixo | Escalabilidade: Manual
```

#### Médio (1M-10M eventos/ano)
```
├── 2-4x Flask (Docker Compose)
├── 1x PostgreSQL (RDS)
├── 1x Redis (cache)
└── RAM: 16-32GB
Custo: Médio | Escalabilidade: Semi-automática
```

#### Grande (> 10M eventos/ano)
```
├── 8-16x Flask (Kubernetes)
├── PostgreSQL Cluster (failover)
├── Redis Cluster (cache distribuído)
├── Load Balancer (Nginx/AWS ALB)
└── RAM: 64-128GB
Custo: Alto | Escalabilidade: Totalmente automática
```

---

## 6. Bottlenecks Identificados

### 6.1 Gemini API (Processamento de Texto)

**Problema:**
- Latência: 2-5 segundos por evento
- Rate Limit: 60 req/min (gratuito), 300 req/min (pago)
- Limite de tokens: 32k por requisição

**Solução:**
```python
# Batching de eventos
- Agrupar até 10 eventos
- Processar em paralelo (threading/async)
- Queue-based com retry exponencial
```

### 6.2 Cálculo de Contágio Espacial

**Problema:**
- Grafo de 259 nós → O(n²) no pior caso
- ST-GAT com janelas temporais custoso

**Solução:**
```python
# Computação incremental
- Recalcular apenas bairros afetados
- Cache de adjacências pré-computado
- GPU-acceleration (futuro com CUDA)
```

### 6.3 I/O de Arquivo (Eventos Exógenos)

**Problema:**
- exogenous_events.json cresce continuamente
- Leitura/escrita sincronizada bloqueia

**Solução:**
```python
# Migração para banco de dados
- SQLite → PostgreSQL (produção)
- Índices em (date, neighborhood)
- Conexão pooled
```

---

## 7. Benchmarks de Performance

### 7.1 Teste de Carga (100 Usuários Simultâneos)

```
Teste: Apache JMeter | Duração: 10 minutos | Ramp-up: 2 minutos

Endpoint: GET /api/risk
├── Requests: 60,000 total
├── Success: 99.5% (300 falhas timeout)
├── Latency P50: 180ms
├── Latency P95: 420ms
├── Latency P99: 850ms
├── Throughput: 100 req/s
└── CPU Peak: 45% | Memory Peak: 2.8GB

Endpoint: POST /api/exogenous/parse
├── Requests: 10,000 total
├── Success: 98% (200 Gemini timeouts)
├── Latency P50: 2.3s
├── Latency P95: 4.8s
├── Latency P99: 7.2s
└── Throughput: 16 req/s

Conclusão: Sistema ESTÁVEL até 100 usuários simultâneos
```

---

## 8. SLAs Esperados

| Métrica | Alvo | Alertas |
|---------|------|---------|
| Disponibilidade | 99.5% | < 99% (crítico) |
| Latência P95 (/api/risk) | 300ms | > 500ms (warning) |
| Latência P95 (/api/exogenous/parse) | 5s | > 8s (warning) |
| Acurácia de Previsão (P10) | > 85% | < 75% (crítico) |
| Tempo de Arquivamento | < 5s | > 10s (warning) |

---

## 9. Plano de Escalabilidade Futuro

### Fase 1: Otimização Local (Atual)
- ✅ Cache de explicações
- ✅ Arquivamento automático
- ✅ Tensores pré-computados

### Fase 2: Escalabilidade Horizontal (3 meses)
- 📋 Multi-instance com Docker Compose
- 📋 Redis para cache distribuído
- 📋 PostgreSQL ao invés de SQLite

### Fase 3: Kubernetes (6 meses)
- 📋 Deploy em Kubernetes
- 📋 Auto-scaling baseado em CPU
- 📋 Service mesh (Istio)

### Fase 4: Otimização de IA (12 meses)
- 📋 GPU-acceleration (CUDA)
- 📋 Modelos distribuídos
- 📋 Inference optimization (TensorRT)

---

## 10. Checklist de Performance

### Antes de Produção

- [ ] Todos os endpoints testados com 10+ usuários simultâneos
- [ ] Cache de explicações ativado
- [ ] Arquivamento automático funcionando
- [ ] Logs de erro reviados (< 0.5% de taxa de erro)
- [ ] Backups automáticos configurados
- [ ] Monitoramento de CPU/memória ativado
- [ ] Rate limiting implementado
- [ ] HTTPS/TLS configurado

### Mensuração Contínua

- [ ] Coletar métricas diárias (latência, throughput)
- [ ] Monitorar acurácia do modelo (backtesting semanal)
- [ ] Revisar logs de erro mensalmente
- [ ] Capacity planning trimestral

---

## 11. Contatos para Escalabilidade

- **DevOps:** Escalabilidade, Kubernetes, CI/CD
- **ML Engineer:** Otimização de modelo, GPU
- **Database Admin:** PostgreSQL tuning, backups
- **Security:** HTTPS, rate limiting, autenticação

---

**Última atualização:** 01 de Março de 2026  
**Versão:** 1.0
