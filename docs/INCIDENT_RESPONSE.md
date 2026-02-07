# RESPONDEDOR DE INCIDENTES - ST-GCN Crime Prediction System

**Versão**: Week 5  
**Data**: 6 de fevereiro de 2026  
**Propósito**: Guia rápido para responder a incidentes em produção  

---

## Índice de Incidentes

1. [Sistema Completamente Down](#1-sistema-completamente-down)
2. [App Respondendo Lentamente](#2-app-respondendo-lentamente)
3. [Taxa de Erro Elevada](#3-taxa-de-erro-elevada)
4. [Memória ou CPU Crítica](#4-memória-ou-cpu-crítica)
5. [Cache (Redis) Offline](#5-cache-redis-offline)
6. [Explicações Não Geradas](#6-explicações-não-geradas)
7. [Anomaly Detector Falhando](#7-anomaly-detector-falhando)
8. [Disk Space Crítico](#8-disk-space-crítico)
9. [Prometheus/Grafana Down](#9-prometheusgrafana-down)
10. [Rollback Necessário](#10-rollback-necessário)

---

## 1. Sistema Completamente Down

**Indicadores:**
- Website não acessível (timeout)
- `curl http://localhost:5000/` sem resposta
- `docker-compose ps` mostra containers stopped/unhealthy

**Ações Imediatas (< 2 min):**

```bash
# 1. Declarar incidente
Slack: #incidents "INCIDENT: ST-GCN Down - @on-call"

# 2. Verificar status
docker-compose ps

# 3. Ver logs de erro
docker-compose logs --tail=50 app

# 4. Tentar restart rápido
docker-compose restart app
sleep 10
python scripts/health_check.py

# 5. Se ainda down, try clean restart
docker-compose down --timeout 30
docker-compose up -d
sleep 30
python scripts/health_check.py
```

**Se ainda não funciona (2-15 min):**

```bash
# 1. Verificar recursos disponíveis
df -h /var/lib/docker  # Deve ter >1GB
free -h                 # Deve ter >500MB RAM

# 2. Se disco cheio
docker system prune -af --filter "until=24h"
docker-compose up -d

# 3. Se memória crítica
docker system prune -a
killall -9 <process> # Se absolutamente necessário

# 4. Check Docker daemon
docker ps           # Se falhar, Docker daemon está down
sudo systemctl restart docker

# 5. Se nada funciona, rollback
bash scripts/deploy.sh  # Ou deploy-prod-rollback.sh
```

**Root Cause Analysis (após resolução):**

```bash
# 1. Coletar logs
docker-compose logs app > /tmp/incident/logs_app.txt
docker-compose logs > /tmp/incident/logs_all.txt

# 2. Criar issue
# GitHub: https://github.com/repo/issues/new
# Titulo: "[INCIDENT] ST-GCN Down - 2026-02-06 HH:MM"
# Labels: incident, production, p1

# 3. Reunião post-mortem
# Schedule: 24-48 horas após
# Participantes: DevOps, Backend Lead, Product
```

---

## 2. App Respondendo Lentamente

**Indicadores:**
- P95 latência > 500ms
- Requests levando 5-10s
- `curl -v http://localhost:5000/` lento

**Diagnóstico Rápido (< 2 min):**

```bash
# 1. Verificar latência
curl -w "Time: %{time_total}s\n" http://localhost:5000/api/metrics

# 2. Queries Prometheus
# P95 latência: histogram_quantile(0.95, rate(http_requests_duration_seconds_bucket[5m]))
# P99 latência: histogram_quantile(0.99, rate(http_requests_duration_seconds_bucket[5m]))

# 3. CPU/Memory
docker stats st-gcn-app --no-stream

# 4. Redis funcionando?
docker-compose exec cache redis-cli ping
# Deve responder: PONG

# 5. Ver requests lentas em logs
docker-compose logs app | grep -i "slow\|warning\|duration"
```

**Ações (2-10 min):**

```bash
# Opção 1: Reiniciar app
docker-compose restart app

# Opção 2: Limpar cache Redis
docker-compose exec cache redis-cli FLUSHALL
docker-compose logs -f app  # Monitorar recovery

# Opção 3: Aumentar workers (se possível)
# docker-compose.yml: workers: 8 (em vez de 4)
# docker-compose up -d --build app

# Opção 4: Escalar horizontalmente
# Adicionar mais instâncias do app com load balancer
```

**Se não melhorar (10+ min):**

```bash
# 1. Connection pooling issue?
docker-compose exec app python -c \
  "import psutil; print([p.info for p in psutil.virtual_memory()])"

# 2. Modelo muito grande?
ls -lh models/stgcn_model_v2.pth

# 3. Full stack restart
docker-compose down --timeout 30
docker-compose up -d
```

---

## 3. Taxa de Erro Elevada

**Indicadores:**
- >5% de requisições retornando 5xx
- Alertas Prometheus
- Grafana mostra erro rate > limite

**Verificação Rápida (< 2 min):**

```bash
# 1. Qual endpoint está falhando?
docker-compose logs app | grep -i "error\|exception" | tail -20

# 2. Padrão no erro?
docker-compose logs app | grep "ERROR" | awk '{print $NF}' | sort | uniq -c

# 3. Erro específico?
curl -v http://localhost:5000/api/explain/1  # Teste individual

# 4. Health check
python scripts/health_check.py
```

**Causas Comuns:**

| Erro | Causa | Solução |
|------|-------|---------|
| `/api/explain` falha | Modelo não carregou | Restart app |
| `/api/anomaly_status` falha | Redis down | Restart cache |
| Timeout | Recursos escassos | Aumentar memory/CPU |
| 503 Service Unavailable | Momentâneo | Esperar ou restart |

**Resolver (2-10 min):**

```bash
# Opção 1: Erro em endpoint específico
docker-compose logs app | grep -A5 "/api/explain"  # Ver detalhes

# Opção 2: Cache problema
docker-compose restart cache
sleep 10
# Testar: curl http://localhost:5000/api/anomaly_status

# Opção 3: Modelo problema
# Verificar se arquivo existe:
docker exec st-gcn-app ls -lh /app/models/stgcn_model_v2.pth

# Opção 4: Muitos erros, fazer restart
docker-compose restart app
```

---

## 4. Memória ou CPU Crítica

**Indicadores:**
- Docker stats mostra >90% memória
- CPU > 80% sustentado
- OOM killed (out-of-memory)

**Diagnóstico (< 1 min):**

```bash
# 1. Status atual
docker stats st-gcn-app --no-stream

# 2. Detalhes processo
docker top st-gcn-app

# 3. Histórico de memória
# Prometheus: container:memory:percent{id="st-gcn-app"}

# 4. Identificar leak
# Se memória cresce continuamente, pode ser leak
docker-compose logs app | grep -i "memory\|leak"
```

**Ações Imediatas:**

```bash
# Opção 1: Restart (se é spike temporário)
docker-compose restart app

# Opção 2: Aumentar limites
# Editar docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 4G  # Aumentar de 2G
docker-compose up -d --build

# Opção 3: Limpar cache se possível
docker-compose exec cache redis-cli DBSIZE    # Ver tamanho
docker-compose exec cache redis-cli FLUSHALL  # Limpar tudo

# Opção 4: Degradar serviços
# Desabilitar FeatureFlags=não necessários
# - Disable cache
# - Disable explanation
# ENV ENABLE_CACHING=false
```

**Se persiste (Leak Suspeito):**

```bash
# 1. Coletar evidência
docker stats st-gcn-app --no-stream >> /tmp/memory_trend.txt
# Repetir a cada 5 min durante 1 hora

# 2. Analisar trend
head -1 /tmp/memory_trend.txt; tail -1 /tmp/memory_trend.txt

# 3. Se confirmar leak
# Abrir incident - possível memory leak no código
# Considerar rollback para versão anterior
```

---

## 5. Cache (Redis) Offline

**Indicadores:**
- Redis service down
- `/api/anomaly_status` retorna erro
- Logs mostram "redis connection refused"

**Solução Rápida (< 1 min):**

```bash
# 1. Status
docker-compose ps cache

# 2. Logs
docker-compose logs cache | tail -20

# 3. Restart
docker-compose restart cache
sleep 5

# 4. Verificar
docker-compose exec cache redis-cli ping
# Deve retornar: PONG

# 5. Health check
python scripts/health_check.py
```

**Se restart não funciona:**

```bash
# 1. Deletar volume (perderá dados de cache)
docker-compose down
docker volume rm st-gcn-cache_data
docker-compose up -d cache
sleep 10
docker-compose exec cache redis-cli ping

# 2. Full restart
docker-compose down --timeout 30
docker-compose up -d
```

---

## 6. Explicações Não Geradas

**Indicadores:**
- `/api/explain/<id>` retorna erro ou vazio
- Logs mostram ExplanationGenerator failures

**Verificação:**

```bash
# 1. Teste endpoint
curl -v http://localhost:5000/api/explain/1

# 2. Verificar logs
docker-compose logs app | grep -i "explanation\|explain"

# 3. Modelo carregado?
docker-compose logs app | grep -i "model.*loaded"

# 4. Features disponíveis?
docker-compose logs app | head -100 | grep -i "feature"
```

**Solução:**

```bash
# Opção 1: Restart app
docker-compose restart app
sleep 10
curl http://localhost:5000/api/explain/1

# Opção 2: Modelo corrupto?
ls -lh models/stgcn_model_v2.pth
# Se tamanho = 0, arquivo corrompido
# Restaurar de backup

# Opção 3: Feature desabilitada
# Verificar .env: ENABLE_EXPLANATION_API=true
# Se false, mudar e restart
```

---

## 7. Anomaly Detector Falhando

**Indicadores:**
- `/api/anomaly_status` retorna erro
- Predictions sem anomaly info
- Logs: "anomaly detector", "event manager"

**Verificação:**

```bash
# 1. Teste
curl http://localhost:5000/api/anomaly_status

# 2. Logs
docker-compose logs app | grep -i "anomaly"

# 3. Arquivo de eventos existe?
docker exec st-gcn-app ls -lh /app/data/exogenous_events*.json

# 4. JSON válido?
docker exec st-gcn-app python -m json.tool /app/data/exogenous_events_geocoded.json > /dev/null
# Se erro, JSON corrompido
```

**Solução:**

```bash
# Opção 1: Restart app
docker-compose restart app

# Opção 2: Eventos corrompidos
# Restaurar de backup
# cp backups/events_backup.json data/exogenous_events_geocoded.json
docker-compose restart app

# Opção 3: Desabilitar temporariamente
# ENV ENABLE_ANOMALY_DETECTION=false
# Permite app rodar sem detector
```

---

## 8. Disk Space Crítico

**Indicadores:**
- `df -h /var/lib/docker` < 10% livre
- Alertas Disk usado > 90%
- Docker não consegue criar containers

**Ação Imediata (< 2 min):**

```bash
# 1. Ver uso
df -h; du -sh /var/lib/docker/*

# 2. Limpar Docker
docker system prune -a --filter "until=24h"
# Remove imagens não usadas de 24h atrás

# 3. Limpar logs
docker-compose logs --tail=0 > /dev/null  # Limpar logs
docker system prune

# 4. Se ainda crítico, remover backups antigos
ls -lt backups/ | tail -n +5 | awk '{print $NF}' | xargs rm -rf

# 5. Monitorar
df -h
```

**Se não há espaço para cleanup:**

```bash
# 1. Stop app
docker-compose down --timeout 30

# 2. Montar disco externo ou expandir volume

# 3. Restart
docker-compose up -d
```

---

## 9. Prometheus/Grafana Down

**Indicadores:**
- Grafana (3000) não acessível
- Prometheus (9090) não respondendo
- Alertas não funcionam

**Solução Rápida:**

```bash
# 1. Restart
docker-compose restart prometheus grafana
sleep 10

# 2. Verificar
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health

# 3. Check volumes
docker volume ls | grep prometheus
docker volume ls | grep grafana
```

**Se não voltar:**

```bash
# 1. Logs
docker-compose logs prometheus | tail -50
docker-compose logs grafana | tail -50

# 2. Full restart
docker-compose down
docker-compose up -d prometheus grafana
sleep 30

# 3. Monitoramento pode estar DOWN por até 30 min
# app continua funcionando, apenas sem visualização
```

---

## 10. Rollback Necessário

**Quando fazer rollback:**
- Deployment introduziu bug crítico
- Taxa de erro > 10% após deploy
- Serviço não stável por > 5 min

**Ejecutar Rollback (< 10 min):**

```bash
# 1. Declarar rollback
Slack: "#incidents Iniciando rollback - @team"

# 2. Parar sistema atual
docker-compose down --timeout 60

# 3. Restaurar backup
# Se tem deploy_backup:
cp backups/deploy_backup_YYYYMMDD_HHMMSS/docker-compose.backup.yml docker-compose.yml

# 4. Restaurar volumes
for backup in backups/*/;do
  docker run --rm \
    -v st-gcn_cache_data:/data \
    -v $(pwd)/$backup:/backup \
    alpine tar xzf /backup/*.tar.gz -C /data .
done

# 5. Start antiga versão
docker-compose up -d

# 6. Validar
python scripts/health_check.py
```

**Post-Rollback:**

```bash
# 1. Comunicação
Slack: "Rollback concluído - sistema estável"

# 2. Root cause
# Identify o commit problemático
# Revert ou crie hotfix

# 3. Teste em staging
git checkout develop
git revert <bad-commit>
bash scripts/deploy.sh  # Para staging

# 4. Aguarde validação antes de retry em prod
```

---

## Matriz de Decisão Rápida

| Problema | Ação 1 | Ação 2 | Ação 3 |
|----------|--------|--------|--------|
| App down | restart app | full restart | rollback |
| Lento | cache flush | restart app | scale up |
| Erro 5xx | check logs | restart | cache clear |
| Memória | restart | increase limit | cleanup |
| Redis down | restart cache | clear volume | full restart |
| Disk full | prune docker | delete backups | expand FS |
| Monitoring down | restart services | acceptable, app runs | rebuild |

---

## Comunicação de Incidente

### Template Slack

```
🚨 INCIDENT: [Severidade] [Sistema] [HH:MM UTC]

Status: 🔴 DOWN / 🟡 DEGRADED / 🟢 RESOLVED

Impacto:
- [Descrição do que está afetado]
- Usuários afetados: [~quantidade]
- Taxa de erro: [%]

Ação:
- Iniciado: [Ação tomada]
- ETA de resolução: [time]

Lead: @person
Log: [Link para incident issue]
```

### Template Post-Mortem

```
# Post-Mortem: [Título Incidente]

## Timeline
- HH:MM: Alerta disparado
- HH:MM: Lead ativado
- HH:MM: Root cause identificada
- HH:MM: Resolução iniciada
- HH:MM: Sistema restaurado

## Root Cause
[Análise da causa raiz]

## Impacto
- Downtime: X minutos
- Usuários: X
- Receita: $X

## Ações Preventivas
- [ ] Action 1
- [ ] Action 2
- [ ] Action 3

## Lições Aprendidas
1. [Lição 1]
2. [Lição 2]
```

---

## Contatos Emergência

| Papel | Disponível | Telefone | Chat |
|-------|-----------|----------|------|
| On-Call | 24/7 | +55 (XX) | @on-call-person |
| Escalation | Sob demanda | +55 (XX) | @manager |

---

**Última revisão**: 6 de fevereiro de 2026  
**Próxima revisão**: 13 de fevereiro de 2026

