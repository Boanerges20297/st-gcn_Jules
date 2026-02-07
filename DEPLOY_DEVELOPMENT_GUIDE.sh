#!/bin/bash
"""
DEPLOY EM DESENVOLVIMENTO - GUIA PRÁTICO
Como subir, testar e validar o sistema antes de ir para produção
"""

# ============================================================================
# PARTE 1: DEPLOY LOCAL (SEU PC/MAC AGORA)
# ============================================================================

echo "
╔════════════════════════════════════════════════════════════════════════════╗
║                  DEPLOY EM DESENVOLVIMENTO - ST-GCN                        ║
║                         6 de Fevereiro, 2026                             ║
╚════════════════════════════════════════════════════════════════════════════╝
"

echo "
[OPÇÃO 1] DEPLOY LOCAL - Rodar tudo aqui no seu PC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"

echo "Passo 1: Validar que Docker está instalado"
if command -v docker &> /dev/null; then
    echo "  ✅ Docker encontrado: $(docker --version)"
else
    echo "  ❌ Docker não instalado!"
    echo "     Baixe em: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo ""
echo "Passo 2: Clonar / Atualizar código"
echo "  $ git clone <repo> st-gcn_jules"
echo "  $ cd st-gcn_jules"

echo ""
echo "Passo 3: Subir stack completa com Docker"
echo "  $ docker-compose up -d"
echo ""
echo "  ✅ Isto vai iniciar 4 containers:"
echo "     • app:5050      (sua API principal)"
echo "     • redis:6379    (cache para performance)"
echo "     • prometheus:9090 (coleta métricas)"
echo "     • grafana:3000  (visualiza métricas)"

echo ""
echo "Passo 4: Validar health"
echo "  $ docker-compose ps"
echo "  $ curl http://localhost:5050/"

echo ""
echo "Passo 5: Acessar aplicação"
echo "  🌐 Dashboard:    http://localhost:5050/dashboard"
echo "  📊 Métricas:     http://localhost:3000 (user: admin, pass: admin)"
echo "  🔧 Prometheus:   http://localhost:9090"

echo ""
echo "Passo 6: Ver logs em tempo real"
echo "  $ docker-compose logs -f app"

echo ""
echo "Passo 7: Parar tudo"
echo "  $ docker-compose down"

# ============================================================================
# PARTE 2: DEPLOY EM STAGING (Servidor pré-produção)
# ============================================================================

echo ""
echo ""
echo "
[OPÇÃO 2] DEPLOY EM STAGING - Servidor isolado para testes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"

echo "📌 O que é Staging?"
echo "   Servidor separado que simula produção, mas é só para testes"
echo "   • Mesma configuração que produção"
echo "   • Dados reais (ou cópia)"
echo "   • Acesso limitado (não publicado)"
echo "   • Pré-vê problemas antes de production"

echo ""
echo "Passo 1: Provisionar servidor staging"
echo "  Opções:"
echo "    A) AWS EC2 (t3.medium, ~\$20/mês)"
echo "    B) DigitalOcean (Droplet, ~\$5/mês)"
echo "    C) VPS (seu datacenter)"
echo "    D) Máquina virtual local (VirtualBox)"

echo ""
echo "Passo 2: SSH para o servidor"
echo "  $ ssh ubuntu@seu-servidor-staging.com"

echo ""
echo "Passo 3: Instalar Docker no servidor"
echo "  $ curl -fsSL https://get.docker.com -o get-docker.sh"
echo "  $ sh get-docker.sh"

echo ""
echo "Passo 4: Clone código"
echo "  $ git clone <seu-repo> st-gcn_jules"
echo "  $ cd st-gcn_jules"

echo ""
echo "Passo 5: Copiar .env"
echo "  $ cp .env.example .env"
echo "  $ nano .env  # editar configurações"

echo ""
echo "Passo 6: Rodar script de deploy automático"
echo "  $ chmod +x scripts/deploy.sh"
echo "  $ ./scripts/deploy.sh"

echo ""
echo "  ✅ Script vai:"
echo "     1. Validar pré-requisitos"
echo "     2. Build Docker image"
echo "     3. Parar containers antigos"
echo "     4. Iniciar novos containers"
echo "     5. Validar saúde (5 tentativas)"
echo "     6. Rodar testes de smoke"
echo "     7. Se falhar, rollback automático"

echo ""
echo "Passo 7: Testar endpoints"
echo "  $ curl http://seu-servidor-staging.com/dashboard"
echo "  $ curl http://seu-servidor-staging.com/api/client/dashboard"

echo ""
echo "Passo 8: Monitorar com Prometheus"
echo "  Abrir no navegador:"
echo "  http://seu-servidor-staging.com:9090"

# ============================================================================
# PARTE 3: VALIDAÇÃO PRÉ-DEPLOY
# ============================================================================

echo ""
echo ""
echo "
[PARTE 3] VALIDAÇÃO - O Que Testar Antes de Ir Para Produção
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"

echo "✅ TESTES A FAZER EM STAGING:"

echo ""
echo "1️⃣  TESTES FUNCIONAIS (1 hora)"
echo "   □ Home page (/) funciona"
echo "   □ Dashboard (/dashboard) carrega"
echo "   □ API metrics (/api/client/dashboard) retorna JSON"
echo "   □ Explicações (/api/explain/1) funcionam"
echo "   □ Anomaly detection funciona"
echo "   $ python run_simple_tests.py"

echo ""
echo "2️⃣  TESTES DE PERFORMANCE (2 horas)"
echo "   □ 100 requisições simultâneas"
echo "   □ Tempo resposta < 100ms"
echo "   □ CPU < 80%"
echo "   □ Memory < 2GB"
echo "   □ Sem memory leaks"
echo "   $ python -m pytest tests/test_week5_load.py -v"

echo ""
echo "3️⃣  TESTES DE SEGURANÇA (1 hora)"
echo "   □ CORS headers presentes"
echo "   □ Sem exposição de dados sensíveis"
echo "   □ Rate limiting funciona"
echo "   □ SQL injection impossível (usando ORM)"
echo "   □ XSS proteção ativa"

echo ""
echo "4️⃣  TESTES DE MONITORING (30 min)"
echo "   □ Prometheus colhendo métricas"
echo "   □ Grafana dashboards atualizando"
echo "   □ Alertas funcionando"
echo "   □ Logs sendo coletados"

echo ""
echo "5️⃣  TESTES DE DATABASE (30 min)"
echo "   □ Conexão com database funciona"
echo "   □ Dados carregam corretamente"
echo "   □ Queries performáticas (< 100ms)"
echo "   □ Backup funcionando"

echo ""
echo "6️⃣  TESTES DE ROLLBACK (15 min)"
echo "   □ Deploy anterior ainda funciona"
echo "   □ Rollback automático se falhar"
echo "   □ Dados não foram perdidos"

echo ""
echo "TEMPO TOTAL: ~5 horas de validação"

# ============================================================================
# PARTE 4: DEPLOY WORKFLOW PRÁTICO
# ============================================================================

echo ""
echo ""
echo "
[PARTE 4] WORKFLOW COMPLETO DE DEPLOY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"

echo "
PASSO A PASSO COMPLETO (Do desenvolvimento para produção):

DIA 1 - DESENVOLVIMENTO LOCAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ 09:00 Clonar código
        \$ git clone <repo>
        
  ✅ 10:00 Rodar local com docker-compose
        \$ docker-compose up -d
        
  ✅ 11:00 Rodar testes locais
        \$ python run_simple_tests.py
        \$ pytest tests/ -v
        
  ✅ 12:00 Validar dashboard funciona
        \$ curl http://localhost:5050/dashboard
        
  ✅ 14:00 Fazer commit
        \$ git add .
        \$ git commit -m 'Phase 2B: Ready for staging'
        \$ git push origin develop

DIA 2-3 - STAGING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ 09:00 Deploy para staging
        \$ ./scripts/deploy.sh
        
  ✅ 10:00 Validações automáticas
        - Health checks ✓
        - Smoke tests ✓
        - Logs check ✓
        
  ✅ 11:00-15:00 Testes manuais (5 horas)
        - Funcional ✓
        - Performance ✓
        - Segurança ✓
        - Monitoring ✓
        - Database ✓
        
  ✅ 16:00 Sign-off técnico
        'Sistema pronto para produção'
        
  ✅ 17:00 Criar release notes
        - Quais mudanças
        - Impacto esperado
        - Rollback plan

DIA 4 - PRODUÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ 09:00 Backup de dados atuais
        \$ ./scripts/backup.sh
        
  ✅ 09:30 Deploy para produção
        \$ ./scripts/deploy-prod.sh
        
  ✅ 10:00 Monitoramento intensivo (primeiras 2 horas)
        - Prometheus/Grafana
        - Logs
        - User reports
        
  ✅ 12:00 Validação final
        - Métricas esperadas ✓
        - Incidentes: 0 ✓
        - Performance: OK ✓
        
  ✅ 13:00 Celebrar! 🎉
"

# ============================================================================
# PARTE 5: COMANDOS ÚTEIS
# ============================================================================

echo ""
echo ""
echo "
[PARTE 5] COMANDOS ÚTEIS PARA DESENVOLVIMENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"

echo "DOCKER COMPOSE (Local)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Subir tudo:                docker-compose up -d"
echo "  Ver status:                docker-compose ps"
echo "  Ver logs:                  docker-compose logs -f app"
echo "  Entrar no container:       docker-compose exec app bash"
echo "  Parar tudo:                docker-compose down"
echo "  Remover volumes:           docker-compose down -v"
echo "  Rebuild imagens:           docker-compose build --no-cache"

echo ""
echo "TESTES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Testes simples:            python run_simple_tests.py"
echo "  Testes unitários:          pytest tests/test_week5_comprehensive.py -v"
echo "  Testes integração:         pytest tests/test_week5_integration.py -v"
echo "  Testes load:               pytest tests/test_week5_load.py -v"
echo "  Cobertura:                 pytest --cov=src tests/"

echo ""
echo "MONITORAMENTO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Prometheus:                http://localhost:9090"
echo "  Grafana:                   http://localhost:3000 (admin/admin)"
echo "  Ver métricas raw:          curl http://localhost:5050/metrics"
echo "  Dashboard cliente:         http://localhost:5050/dashboard"

echo ""
echo "DEBUGGING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ver erros em app:          docker-compose logs app"
echo "  Ver todas requisições:     docker-compose logs -f"
echo "  Health check:              docker-compose exec app python -c \"import health_check; health_check.check()\""
echo "  Entrar em container:       docker-compose exec app bash"

echo ""
echo "DEPLOY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploy staging:            ./scripts/deploy.sh"
echo "  Deploy produção:           ./scripts/deploy-prod.sh"
echo "  Rollback:                  ./scripts/rollback.sh"
echo "  Health check:              ./scripts/health_check.py"

# ============================================================================
# PARTE 6: TROUBLESHOOTING
# ============================================================================

echo ""
echo ""
echo "
[PARTE 6] TROUBLESHOOTING - Erros Comuns e Soluções
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"

echo "❌ Erro: 'docker-compose: command not found'"
echo "   ✅ Solução: Instalar Docker Desktop (inclui docker-compose)"

echo ""
echo "❌ Erro: 'Port 5050 already in use'"
echo "   ✅ Solução: docker-compose down && docker-compose up -d"
echo "   ✅ Ou: Mudar porta em docker-compose.yml (5051:5050)"

echo ""
echo "❌ Erro: 'Out of memory' "
echo "   ✅ Solução: docker system prune -a"
echo "   ✅ Aumentar RAM alocada no Docker Desktop (Settings > Resources)"

echo ""
echo "❌ Erro: 'Permission denied' ao fazer deploy"
echo "   ✅ Solução: chmod +x scripts/*.sh"

echo ""
echo "❌ Error: 'Module not found: src.metrics'"
echo "   ✅ Solução: Estar no diretório correto"
echo "   ✅ Verificar sys.path.insert(0, 'src') no código"

echo ""
echo "❌ Erro: 'Database connection refused'"
echo "   ✅ Solução: docker-compose logs redis"
echo "   ✅ Esperar container redis inicializar (~5s)"

echo ""
echo "❌ Erro: 'API returns 503'"
echo "   ✅ Solução: Health check falhou"
echo "   ✅ Verificar logs: docker-compose logs app"

echo ""
echo ""
echo "
═════════════════════════════════════════════════════════════════════════════
                    FIM DO GUIA DE DEPLOY
═════════════════════════════════════════════════════════════════════════════
"

echo ""
echo "📌 PRÓXIMOS PASSOS:"
echo "   1. Seguir [OPÇÃO 1] para rodar localmente agora"
echo "   2. Testar com: python run_simple_tests.py"
echo "   3. Quando pronto, fazer [OPÇÃO 2] em servidor staging"
echo "   4. Depois de validar, deploy para produção"

echo ""
echo "⏱️  TEMPO ESTIMADO:"
echo "   • Local setup: 5 minutos"
echo "   • Staging setup: 30 minutos"
echo "   • Staging testing: 5 horas"
echo "   • Production deploy: 30 minutos"
echo ""
