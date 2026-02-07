# 🚀 DEPLOY EM DESENVOLVIMENTO - GUIA PRÁTICO

**Data:** 6 de Fevereiro, 2026  
**Sistema:** ST-GCN Crime Prediction  
**Ambiente:** Development → Staging → Production

---

## 📌 TL;DR (Resumo Super Rápido)

```bash
# LOCAL (seu PC) - 2 minutos
git clone <repo>
cd st-gcn_jules
docker-compose up -d
# Pronto! Acesse http://localhost:5050/dashboard

# STAGING (servidor) - 30 minutos
ssh seu-servidor
./scripts/deploy.sh
# Pronto! Acesse http://seu-servidor:5050/dashboard

# PRODUÇÃO - 30 minutos
./scripts/deploy-prod.sh
# Monitorar por 2 horas
```

---

## 🎯 O QUE É DEPLOY?

**Deploy** = Levar código que está funcionando (no seu PC) para um servidor que outras pessoas usam.

### Tipos de Deploy:

| Ambiente | Onde? | Quem Acessa | Risco | Tempo Setup |
|----------|-------|-------------|-------|------------|
| **Development** | Seu PC | Você | Nenhum | 2 min |
| **Staging** | Servidor teste | Time | Baixo | 30 min |
| **Production** | Servidor real | Clientes/Usuários | Alto | 30 min |

---

## 1️⃣ DEPLOY LOCAL (SEU PC/MAC AGORA)

### O Que Você Vai Fazer:
- Subir a aplicação inteira localmente
- Testar tudo funciona
- Ver as métricas em tempo real
- Simular o que estaria em produção

### Passo a Passo:

#### Passo 1: Validar Docker
```bash
docker --version
docker-compose --version

# Se não estiver instalado:
# Windows/Mac: Baixe Docker Desktop
# https://www.docker.com/products/docker-desktop
```

#### Passo 2: Clonar Código
```bash
git clone https://github.com/seu-usuario/st-gcn_jules.git
cd st-gcn_jules
```

#### Passo 3: Subir Tudo com Docker
```bash
docker-compose up -d
```

**O que está acontecendo:**
```
✅ Container 1: app (Flask) na porta 5050
✅ Container 2: redis (Cache) na porta 6379
✅ Container 3: prometheus (Métricas) na porta 9090
✅ Container 4: grafana (Dashboard) na porta 3000
```

#### Passo 4: Verificar Status
```bash
docker-compose ps

# Esperado:
# NAME                    STATUS
# st-gcn_jules-app-1      Up 2 seconds
# st-gcn_jules-redis-1    Up 2 seconds
# st-gcn_jules-prometheus-1 Up 2 seconds
# st-gcn_jules-grafana-1  Up 2 seconds
```

#### Passo 5: Acessar a Aplicação

| URL | O Quê | Login |
|-----|-------|-------|
| **http://localhost:5050/** | App principal | - |
| **http://localhost:5050/dashboard** | 📊 Dashboard cliente (MELHORIAS!) | - |
| **http://localhost:3000** | 📈 Grafana (métricas) | admin/admin |
| **http://localhost:9090** | 🔧 Prometheus (configuração) | - |

#### Passo 6: Ver Logs
```bash
# Logs em tempo real da aplicação
docker-compose logs -f app

# Logs de todos os containers
docker-compose logs -f
```

#### Passo 7: Entrar dentro do Container (se precisar debugar)
```bash
docker-compose exec app bash

# Agora você está dentro do container
python -c "import src.metrics; print('OK')"
exit
```

#### Passo 8: Parar Tudo (quando terminar)
```bash
docker-compose down
```

---

## 2️⃣ DEPLOY EM STAGING (Servidor de Testes)

### O Que É Staging?
Um servidor real (não seu PC) que é uma **cópia exata de produção**, mas apenas para testes. Ninguém acessa, é só pré-validação.

### Por Quê?
- Testar em ambiente "real" (não PC local)
- Validar performance com dados reais
- Descobrir bugs antes de clientes verem
- Ter plano B pronto se der problema

### Arquitetura:

```
SEU PC (Development)
    ↓ git push
GITHUB (Código)
    ↓ git clone
SERVIDOR STAGING
    ↓ docker-compose up
TESTES AUTOMÁTICOS
    ↓ validações...
SERVIDOR PRODUCTION (se tudo OK)
```

### Passo a Passo:

#### Passo 1: Provisionar Servidor Staging

**Opção A: AWS EC2** (Recomendado, $20-50/mês)
```bash
# Criar t3.medium (2vCPU, 4GB RAM)
# S.O.: Ubuntu 20.04 LTS
# Abrir portas: 22 (SSH), 5050, 9090, 3000
```

**Opção B: DigitalOcean** (Barato, $5-10/mês)
```bash
# Criar Droplet (Basic, Ubuntu 20.04)
# Região: São Paulo ou Brasília
# Abrir firewall: SSH, 5050, 9090, 3000
```

**Opção C: VirtualBox Local** (Grátis, para testar)
```bash
# Criar VM Linux (Ubuntu 20.04)
# RAM: 4GB
# Disco: 20GB
```

#### Passo 2: Conectar via SSH
```bash
# AWS/DigitalOcean
ssh ubuntu@seu-ip-do-servidor

# Se usar chave SSH
ssh -i caminho/chave.pem ubuntu@seu-ip-do-servidor
```

#### Passo 3: Instalar Docker no Servidor
```bash
# Download script oficial Docker
curl -fsSL https://get.docker.com -o get-docker.sh

# Executar instalação
sh get-docker.sh

# Verificar
docker --version
docker-compose --version
```

#### Passo 4: Clonar Código
```bash
git clone https://github.com/seu-usuario/st-gcn_jules.git
cd st-gcn_jules
```

#### Passo 5: Copiar Configurações
```bash
# Criar .env baseado no template
cp .env.example .env

# Editar .env com suas variáveis
nano .env

# Variáveis importantes:
# FLASK_ENV=production
# DATABASE_URL=seu-banco-dados
# REDIS_URL=redis://redis:6379/0
# API_KEY=seu-ai-service-key
```

#### Passo 6: Rodar Deploy Automático
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

**O script vai:**
1. ✅ Validar pré-requisitos (Docker, espaço em disco)
2. ✅ Build imagem Docker
3. ✅ Parar containers antigos
4. ✅ Iniciar novos containers
5. ✅ Health check (5 tentativas)
6. ✅ Smoke tests (testes rápidos)
7. ⏮️ **Rollback automático se falhar**

#### Passo 7: Validar Acesso
```bash
# Testa health
curl http://seu-servidor:5050/

# Testa API
curl http://seu-servidor:5050/api/client/dashboard
```

---

## 3️⃣ VALIDAÇÃO PRÉ-PRODUCTION (4-5 HORAS)

### ✅ Testes a Fazer:

#### 1. Testes Funcionais (1 hora)
```bash
python run_simple_tests.py

# Esperado:
# ✅ Imports & Estrutura         PASSOU
# ✅ MetricReporter              PASSOU
# ✅ EventAnomalyDetector        PASSOU
# ✅ ExplanationGenerator        PASSOU
# ✅ Flask App                   PASSOU
# ✅ Data Files                  PASSOU
# TOTAL: 6/6 testes passaram
```

#### 2. Testes de Performance (2 horas)
```bash
pytest tests/test_week5_load.py -v

# Validações:
# □ Tempo resposta < 100ms
# □ CPU < 80%
# □ Memória < 2GB
# □ 100+ requisições simultâneas OK
```

#### 3. Testes de Segurança (1 hora)
```bash
# Manualmente:
# □ Tentar SQL injection - FALHA (esperado)
# □ Tentar XSS - FALHA (esperado)
# □ CORS headers presentes - SIM (esperado)
# □ Rate limiting funciona - SIM (esperado)
```

#### 4. Testes de Monitoring (30 min)
```bash
# Verificar no Grafana (http://seu-servidor:3000)
# □ Dashboard mostrando métricas
# □ CPU/Memory/Disk em verde
# □ Requisições por segundo normais
# □ Sem erros nos logs
```

#### 5. Testes de Database (30 min)
```bash
# □ Conexão funciona
# □ Dados carregam OK
# □ Queries < 100ms
# □ Backup automático funciona
```

## 4️⃣ DEPLOY PARA PRODUÇÃO

### ⚠️ Antes de Fazer:
- [ ] Todos testes passaram em staging
- [ ] Backup recente foi feito
- [ ] Time validou
- [ ] Rollback plan estudado
- [ ] Monitoring configurado

### Passo a Passo:
```bash
# 1. Fazer backup de dados atuais
./scripts/backup.sh

# 2. Deploy com verificações extras
./scripts/deploy-prod.sh

# 3. Script vai pedir confirmação dupla (proteção)
# "Confirmar deploy para PRODUÇÃO? (SIM/NÃO)"

# 4. Se falhar, rollback automático
# Se sucesso, começar monitoramento intensivo

# 5. Nos primeiros 2 dias:
# - Check Grafana a cada 1h
# - Olhar logs para erros
# - Pedir feedback de usuários
```

---

## 🔙 ROLLBACK (Se Der Problema)

Se em produção der erro, é super rápido voltar:

```bash
# Opção 1: Voltar para versão anterior
./scripts/rollback.sh

# Opção 2: Desativar feature problemática (sem redeploy)
# Editar .env: FEATURE_FLAG_X=false
# docker-compose restart app

# Opção 3: Scale down (reduzir carga)
docker-compose up -d --scale app=1
```

---

## 📊 COMANDOS ÚTEIS DIÁRIOS

### Docker Compose
```bash
docker-compose ps                    # Ver status
docker-compose logs -f app           # Ver logs
docker-compose exec app bash         # Entrar
docker-compose down                  # Parar tudo
docker-compose restart app           # Reiniciar um serviço
```

### Monitoring
```bash
curl http://localhost:5050/metrics               # Métricas Prometheus
curl http://localhost:5050/api/client/dashboard # Dashboard JSON
```

### Testes
```bash
python run_simple_tests.py                       # Rápido (1 min)
pytest tests/test_week5_comprehensive.py -v     # Completo (5 min)
```

### Git
```bash
git status                                       # Ver mudanças
git commit -am "Mensagem"                        # Commitar
git push origin main                             # Mandar para repo
```

---

## 🚨 TROUBLESHOOTING

| Erro | Causa | Solução |
|------|-------|---------|
| Port 5050 already in use | Outro app rodando | `docker-compose down && docker-compose up -d` |
| Out of memory | Docker sem RAM | `docker system prune -a` |
| Module not found | Sys.path errado | `cd st-gcn_jules && python run_simple_tests.py` |
| API returns 503 | Health check falhou | `docker-compose logs app` |
| Cannot connect to Redis | Container caído | `docker-compose restart redis` |

---

## 📈 PRÓXIMAS AÇÕES

### Semana 1 (Hoje)
- [ ] Setup local (docker-compose)
- [ ] Rodar testes
- [ ] Ver dashboard funcionar

### Semana 2
- [ ] Criar servidor staging
- [ ] Deploy em staging
- [ ] 5 horas de testes

### Semana 3
- [ ] Deploy em produção
- [ ] Monitoramento 24h nos primeiros 2 dias
- [ ] Feedback dos usuários

### Semana 4+
- [ ] Phase 2C (advanced features)
- [ ] Otimizações baseado em feedback
- [ ] Treinamento do time

---

## 🎯 RESUMO

| Etapa | Tempo | Risco | O Quê |
|-------|-------|-------|-------|
| **Local** | 5 min | ❌ Nenhum | Seu PC funciona |
| **Staging** | 30 min setup + 5h testes | 🟡 Baixo | Servidor teste |
| **Produção** | 30 min | 🔴 Alto | Usuários usam |

**Próximo passo:** Execute `docker-compose up -d` agora! ✨

