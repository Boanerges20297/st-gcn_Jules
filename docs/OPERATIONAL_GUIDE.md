# GUIA OPERACIONAL - ST-GCN Crime Prediction System

**Versão**: Week 5  
**Data**: 6 de fevereiro de 2026  
**Ambiente**: Production  

---

## Índice

1. [Inicialização & Parada](#inicialização--parada)
2. [Monitoramento](#monitoramento)
3. [Troubleshooting](#troubleshooting)
4. [Backup & Recuperação](#backup--recuperação)
5. [Maintenance Rotineira](#maintenance-rotineira)
6. [Escalabilidade](#escalabilidade)
7. [Segurança](#segurança)
8. [Contatos](#contatos)

---

## Inicialização & Parada

### Iniciar Sistema Completo

```bash
# 1. Verificar arquivo .env
cat .env | grep -E "^[A-Z_]+=" | head -20

# 2. Iniciar containers
docker-compose up -d

# 3. Verificar status
docker-compose ps

# 4. Health check
python scripts/health_check.py

# 5. Verificar logs iniciais
docker-compose logs --tail=100 app
```

**Tempo esperado**: 30-60 segundos até healthy

### Parar Sistema

```bash
# Parada graciosa (com timeout)
docker-compose down --timeout 60

# Parada imediata (apenas se necessário)
docker-compose down -t 0

# Verificar se parou
docker-compose ps
```

### Reiniciar Sistema

```bash
# Opção 1: Reinício completo (recomendado)
docker-compose down --timeout 60
docker-compose up -d
python scripts/health_check.py

# Opção 2: Restart individual
docker-compose restart app

# Opção 3: Rebuild e restart
docker-compose up -d --build
```

---

## Monitoramento

### Janela Operacional de Validação

- A régua vigente para validar melhoria de modelo em produção é de **14 dias corridos**.
- Não promover modelo por pico curto de 1 a 7 dias; exigir sustentação por 14 dias nas métricas operacionais.
- Fortaleza e RMF devem ser acompanhadas principalmente por **P@10**.
- Interior deve ser acompanhado por **P@20 + R@20**, junto com cobertura territorial.

### Dashboards

| Dashboard | URL | Credenciais | Propósito |
|-----------|-----|-------------|----------|
| Grafana | http://localhost:3000 | admin/admin | Visualização de métricas |
| Prometheus | http://localhost:9090 | N/A | Consulta de métricas brutas |

### Métricas Críticas

| Métrica | Alerta | Ação |
|---------|--------|------|
| `up{job="st-gcn-app"}` | 0 | App offline → restart |
| CPU | >80% by 5m | Verificar carga, possível escala horizontal |
| Memória | >90% | Aumentar limite ou otimizar |
| Taxa erro (5xx) | >5% | Verificar logs |
| P95 latência | >500ms | Verificar índices ou otimizar |

### Queries Úteis

```promql
# Taxa de requisição
rate(http_requests_total[5m])

# Taxa de erro
rate(http_requests_total{status=~"5.."}[5m])

# Latência P95
histogram_quantile(0.95, rate(http_requests_duration_seconds_bucket[5m]))

# Uptime em horas
(time() - process_start_time_seconds) / 3600

# Memória usada
container_memory_usage_bytes / 1024 / 1024  # em MB
```

### Logs

```bash
# Logs em real-time
docker-compose logs -f app

# Últimas N linhas
docker-compose logs --tail=100 app

# Logs com timestamp
docker-compose logs --timestamps app

# Buscar por padrão
docker-compose logs app | grep ERROR

# Logs de um serviço específico
docker-compose logs redis
docker-compose logs prometheus
```

---

## Troubleshooting

### Problema: App não inicia

```bash
# 1. Verificar status
docker-compose ps

# 2. Ver logs
docker-compose logs app

# 3. Verificar espaço em disco
df -h /var/lib/docker

# 4. Limpar space
docker system prune -a --volumes

# 5. Retry
docker-compose up -d app
```

### Problema: Memória ou CPU muito alta

```bash
# 1. Identificar recurso
docker stats st-gcn-app

# 2. Processos dentro do container
docker top st-gcn-app

# 3. Aumentar limite (no docker-compose.yml)
# Adicionar:
# deploy:
#   resources:
#     limits:
#       memory: 2G
#       cpus: '2'

# 4. Rebuild e restart
docker-compose up -d --build app
```

### Problema: Endpoints retornando erro

```bash
# 1. Verificar health
python scripts/health_check.py

# 2. Testar endpoint individual
curl -v http://localhost:5000/api/metrics

# 3. Ver logs
docker-compose logs app | grep /api/metrics

# 4. Verificar status do Redis (cache)
docker-compose logs redis

# 5. Restart cache
docker-compose restart cache
```

### Problema: Modelo não carrega

```bash
# 1. Verificar arquivo existe
ls -lh models/stgcn_model_v2.pth

# 2. Verificar permissões
ls -la models/ | grep stgcn

# 3. Ver logs
docker-compose logs app | grep -i "model"

# 4. Verificar espaço em disco
df -h /app/models

# 5. Remount volume
docker-compose down
docker-compose up -d
```

### Problema: Prometheus não coleta dados

```bash
# 1. Verificar Prometheus
docker-compose ps prometheus

# 2. Verificar config
docker exec st-gcn-prometheus cat /etc/prometheus/prometheus.yml

# 3. Ver targets
curl http://localhost:9090/api/v1/targets

# 4. Logs
docker-compose logs prometheus

# 5. Restart
docker-compose restart prometheus
```

---

## Backup & Recuperação

### Backup Manual

```bash
# 1. Backup de volumes
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
docker run --rm \
  -v st-gcn_cache_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/redis_backup.tar.gz -C /data .

# 2. Backup de configuração
docker-compose config > backups/docker-compose_backup.yml

# 3. Backup de logs
docker-compose logs --no-log-prefix app > backups/app_logs_$(date +%s).log

# 4. Listar backups
ls -lh backups/
```

### Recuperação de Backup

```bash
# 1. Stop sistema
docker-compose down --timeout 60

# 2. Restaurar volume
docker run --rm \
  -v st-gcn_cache_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/redis_backup.tar.gz -C /data

# 3. Start sistema
docker-compose up -d

# 4. Validar
python scripts/health_check.py
```

### Scheduled Backup (Cron)

```bash
# Adicionar ao crontab
0 2 * * * /app/scripts/backup.sh

# Ver cron jobs
crontab -l

# Editar cron
crontab -e
```

---

## Maintenance Rotineira

### Diário

- ✅ Verificar alerts no Prometheus/Grafana
- ✅ Revisar logs para warnings/errors
- ✅ Validar health check (idealmente automático)

```bash
# Script diário
python scripts/health_check.py
docker-compose logs --since 1h app | grep -i error
```

### Semanal

- ✅ Revisar uso de recursos (CPU, memória, disco)
- ✅ Verificar taxa de erro e latência
- ✅ Fazer backup manual
- ✅ Testar rollback procedure

```bash
# Verificação semanal
docker stats
df -h
docker-compose logs --since 7d app | grep -i error | wc -l
```

### Mensal

- ✅ Atualizar dependências Python
- ✅ Revisar performance e otimizar
- ✅ Disaster recovery drill
- ✅ Update documentation

```bash
# Atualizar requirements
pip list --outdated
pip install -r requirements.txt --upgrade
docker-compose up -d --build
```

---

## Escalabilidade

### Aumentar Capacidade (Vertical Scaling)

```yaml
# No docker-compose.yml
app:
  deploy:
    resources:
      limits:
        memory: 4G      # Aumentar de 2G
        cpus: '4'       # Aumentar de 2
      reservations:
        memory: 2G
        cpus: '2'
```

Depois:
```bash
docker-compose up -d --build
```

### Múltiplas Instâncias (Horizontal Scaling)

```yaml
# Usar com load balancer
services:
  app1:
    image: st-gcn-app:latest
    ports:
      - "5001:5000"
  
  app2:
    image: st-gcn-app:latest
    ports:
      - "5002:5000"
  
  loadbalancer:
    image: nginx:latest
    ports:
      - "80:80"
```

---

## Segurança

### Secrets & Configuração

```bash
# NUNCA commit .env
echo ".env" >> .gitignore

# Usar secrets seguros em produção
# Opção 1: Docker Secrets
docker secret create db_password -

# Opção 2: Environment variables seguras
export SECRET_KEY=$(openssl rand -hex 32)

# Verificar secrets
docker secret ls
```

### Atualizações de Segurança

```bash
# Verificar vulnerabilidades
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image st-gcn-app:latest

# Atualizar imagem base
docker pull python:3.9-slim
docker-compose up -d --build
```

### Firewall & Network

```bash
# Verificar portas abertas
netstat -tlnp | grep LISTEN

# Restringir acesso a Prometheus (apenas localhost)
# Adicionar no docker-compose.yml:
# ports:
#   - "127.0.0.1:9090:9090"  # Apenas localhost

# HTTPS (com nginx)
# Ver: docs/HTTPS_SETUP.md
```

---

## Contatos & Escalação

### Equipe On-Call

| Papel | Nome | Telefone | Email |
|-------|------|----------|-------|
| Lead | - | +55 (XX) XXXX-XXXX | team@example.com |
| DevOps | - | +55 (XX) XXXX-XXXX | devops@example.com |
| DBA | - | +55 (XX) XXXX-XXXX | dba@example.com |

### Escalação

```
Level 1: Automated alerts + on-call rotation
Level 2: Team lead notificado se >5 min sem resposta
Level 3: Manager notificado se >15 min sem resolução
Level 4: Diretor executivo se sistema down >30 min
```

### SLA Targets

| Métrica | Target | Ação se não atingir |
|---------|--------|-------------------|
| Uptime | 99.9% | Incident review |
| MTTR (Mean Time To Resolve) | <15 min | Root cause analysis |
| MTTD (Mean Time To Detect) | <5 min | Melhorar alertas |
| RTO (Recovery Time Objective) | <1 hora | Teste disaster recovery |
| RPO (Recovery Point Objective) | <1 hora de dados | Backup mais frequente |

---

## Quick Reference

### Comandos Frequentes

```bash
# Status
docker-compose ps
docker-compose logs app -f

# Reiniciar
docker-compose restart app

# Health check
python scripts/health_check.py

# Backup
docker-compose down --timeout 60
# ... backup procedure ...
docker-compose up -d

# View metrics
curl http://localhost:9090/api/v1/query?query=up

# Connect to container
docker exec -it st-gcn-app bash
```

---

**Última revisão**: 6 de fevereiro de 2026  
**Próxima revisão**: 13 de fevereiro de 2026

