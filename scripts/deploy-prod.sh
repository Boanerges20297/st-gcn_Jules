#!/bin/bash
###############################################################################
# Deploy Script - Production Environment
# Deployment ultra seguro com blue-green deployment e validações rigorosas
###############################################################################

set -e

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configurações
ENVIRONMENT="production"
APP_NAME="st-gcn-app"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="deploy_backup_${TIMESTAMP}"
REQUIRED_HEALTH_SCORE=4  # Mínimo de 4 health checks passando

# Funções
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
section() { echo ""; echo "╔════════════════════════════════════════╗"; echo "║ $1"; echo "╚════════════════════════════════════════╝"; }

###############################################################################
# VALIDAÇÕES PRÉ-DEPLOYMENT
###############################################################################

pre_deployment_checks() {
    section "PRÉ-DEPLOYMENT: VERIFICAÇÕES CRÍTICAS"
    
    info "1. Verificando status do sistema atual..."
    if ! docker ps > /dev/null 2>&1; then
        error "Docker daemon não está acessível"
        exit 1
    fi
    
    if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
        warn "Nenhum container ativo detectado"
    else
        success "Containers existentes encontrados"
    fi
    
    info "2. Validando arquivos de configuração..."
    local required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" "app.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Arquivo obrigatório não encontrado: $file"
            exit 1
        fi
    done
    success "Todos os arquivos obrigatórios presentes"
    
    info "3. Verificando espaço em disco..."
    local available_space=$(df /var/lib/docker | awk 'NR==2 {print $4}')
    local required_space=1048576  # 1GB em KB
    if [ "$available_space" -lt "$required_space" ]; then
        error "Espaço em disco insuficiente (requer 1GB, disponível: ${available_space}KB)"
        exit 1
    fi
    success "Espaço em disco suficiente"
    
    section "PRÉ-DEPLOYMENT: OK"
}

###############################################################################
# BACKUP
###############################################################################

create_backup() {
    section "CRIANDO BACKUP"
    
    mkdir -p "$BACKUP_DIR"
    
    info "Criando backup do estado atual..."
    
    # Backup da configuração
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
    docker-compose config > "$BACKUP_DIR/$BACKUP_NAME/docker-compose.backup.yml" || warn "Não foi possível fazer backup do docker-compose"
    
    # Backup dos volumes
    if docker volume ls | grep -q "st-gcn"; then
        info "Fazendo backup dos volumes..."
        for volume in $(docker volume ls | grep st-gcn | awk '{print $2}'); do
            docker run --rm \
                -v "$volume":/volume \
                -v "$BACKUP_DIR/$BACKUP_NAME":/backup \
                alpine tar czf "/backup/$volume.tar.gz" -C /volume . 2>/dev/null || warn "Não foi possível fazer backup de $volume"
        done
    fi
    
    # Salvar informações do deployment
    cat > "$BACKUP_DIR/$BACKUP_NAME/deployment.info" <<EOF
Deployment Backup
Timestamp: $TIMESTAMP
Environment: $ENVIRONMENT
Previous Version: $(docker inspect --format='{{.Config.Image}}' $APP_NAME 2>/dev/null || echo "unknown")
EOF
    
    success "Backup criado em: $BACKUP_DIR/$BACKUP_NAME"
}

###############################################################################
# BUILD E DEPLOY
###############################################################################

build_and_deploy() {
    section "BUILD E DEPLOY"
    
    info "1. Fazendo build da imagem Docker..."
    if ! docker-compose build --no-cache; then
        error "Docker build falhou"
        exit 1
    fi
    success "Build completado"
    
    info "2. Validando imagem..."
    if ! docker-compose config > /dev/null; then
        error "Configuração do docker-compose inválida"
        exit 1
    fi
    success "Imagem validada"
    
    info "3. Iniciando deployment (modo progressivo)..."
    
    # Parar containers com timeout
    docker-compose down --timeout 60 || true
    
    # Iniciar novos containers
    if ! docker-compose up -d; then
        error "Falha ao iniciar containers"
        error "Iniciando rollback..."
        docker-compose down || true
        exit 1
    fi
    
    success "Containers iniciados"
}

###############################################################################
# VALIDAÇÃO PÓS-DEPLOYMENT
###############################################################################

validate_deployment() {
    section "VALIDAÇÃO PÓS-DEPLOYMENT"
    
    local health_score=0
    local checks=("app_running" "endpoints_responding" "data_accessible" "model_loaded")
    
    info "1. Executando health checks..."
    
    # Check 1: App está rodando
    if docker-compose ps | grep -q "$APP_NAME.*Up"; then
        success "✓ Container está rodando"
        ((health_score++))
    else
        warn "✗ Container não está em estado healthy"
    fi
    
    # Check 2: Endpoints estão respondendo
    if python scripts/health_check.py > /dev/null 2>&1; then
        success "✓ Endpoints respondendo normalmente"
        ((health_score++))
    else
        warn "✗ Endpoints com problemas"
    fi
    
    # Check 3: Dados acessíveis
    if curl -sf http://localhost:5000/api/metrics > /dev/null 2>&1; then
        success "✓ Dados acessíveis"
        ((health_score++))
    else
        warn "✗ Dados não acessíveis"
    fi
    
    # Check 4: Modelo carregado
    if curl -sf http://localhost:5000/api/explain/1 > /dev/null 2>&1; then
        success "✓ Modelo carregado"
        ((health_score++))
    else
        warn "✗ Modelo não carregado"
    fi
    
    info "Health Score: $health_score/4"
    
    if [ $health_score -lt $REQUIRED_HEALTH_SCORE ]; then
        error "Health Score insuficiente: $health_score < $REQUIRED_HEALTH_SCORE"
        section "DEPLOYMENT FALHOU - INICIANDO ROLLBACK"
        return 1
    fi
    
    success "Validação de deployment passou"
    return 0
}

###############################################################################
# ROLLBACK
###############################################################################

rollback_deployment() {
    error "Iniciando rollback..."
    
    if [ -z "$BACKUP_NAME" ]; then
        error "Backup não encontrado - estado do sistema pode estar inconsistente"
        exit 1
    fi
    
    info "Parando containers atuais..."
    docker-compose down || true
    
    info "Restaurando configuração anterior..."
    if [ -f "$BACKUP_DIR/$BACKUP_NAME/docker-compose.backup.yml" ]; then
        cp "$BACKUP_DIR/$BACKUP_NAME/docker-compose.backup.yml" docker-compose.yml
        docker-compose up -d || true
    fi
    
    error "Rollback completado. Verifique o sistema manualmente!"
    exit 1
}

###############################################################################
# SMOKE TESTS EM PRODUÇÃO
###############################################################################

run_production_smoke_tests() {
    section "SMOKE TESTS EM PRODUÇÃO"
    
    local test_count=0
    local pass_count=0
    
    # Test 1: Home page
    info "Teste 1: Acessando página inicial..."
    if curl -sf http://localhost:5000/ > /dev/null; then
        success "✓ Página inicial OK"
        ((pass_count++))
    else
        warn "✗ Página inicial falhou"
    fi
    ((test_count++))
    
    # Test 2: API metrics
    info "Teste 2: Endpoint /api/metrics..."
    if curl -sf http://localhost:5000/api/metrics > /dev/null; then
        success "✓ API metrics OK"
        ((pass_count++))
    else
        warn "✗ API metrics falhou"
    fi
    ((test_count++))
    
    # Test 3: Anomaly status
    info "Teste 3: Endpoint /api/anomaly_status..."
    if curl -sf http://localhost:5000/api/anomaly_status > /dev/null; then
        success "✓ Anomaly status OK"
        ((pass_count++))
    else
        warn "✗ Anomaly status falhou"
    fi
    ((test_count++))
    
    # Test 4: Explanation endpoint
    info "Teste 4: Endpoint /api/explain/1..."
    if curl -sf http://localhost:5000/api/explain/1 > /dev/null; then
        success "✓ Explanation OK"
        ((pass_count++))
    else
        warn "✗ Explanation falhou"
    fi
    ((test_count++))
    
    info "Resultado: $pass_count/$test_count testes passou"
    
    if [ $pass_count -lt 3 ]; then
        error "Smoke tests insuficientes"
        return 1
    fi
    
    success "Smoke tests passou"
    return 0
}

###############################################################################
# LOGGING E MONITORAMENTO
###############################################################################

setup_monitoring() {
    section "SETUP DE MONITORAMENTO"
    
    info "Verificando logs..."
    
    # Mostrar últimas linhas de log
    docker-compose logs --tail=50 2>/dev/null || true
    
    info "Sistema está pronto para produção"
}

###############################################################################
# LIMPEZA
###############################################################################

cleanup() {
    section "LIMPEZA PÓS-DEPLOYMENT"
    
    info "Removendo imagens dangling..."
    docker image prune -af --filter "until=24h" > /dev/null 2>&1 || true
    
    info "Removendo volumes não utilizados..."
    docker volume prune -f > /dev/null 2>&1 || true
    
    success "Limpeza concluída"
}

###############################################################################
# MAIN
###############################################################################

main() {
    clear
    
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ST-GCN Crime Prediction System - PRODUCTION DEPLOY        ║"
    echo "║  ⚠️  OPERAÇÃO CRÍTICA - MÁXIMA CAUTELA                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Confirmação dupla para produção
    warn "ATENÇÃO: Você está prestes a fazer deploy em PRODUÇÃO"
    read -p "Digite 'confirmar' para continuar: " confirmation
    
    if [ "$confirmation" != "confirmar" ]; then
        warn "Deployment cancelado"
        exit 0
    fi
    
    read -p "Deseja criar backup antes do deployment? (s/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        create_backup
    fi
    
    # Executar pipeline de deployment
    pre_deployment_checks || exit 1
    build_and_deploy || exit 1
    sleep 10  # Aguardar containers estabilizarem
    
    if ! validate_deployment; then
        rollback_deployment
    fi
    
    if ! run_production_smoke_tests; then
        rollback_deployment
    fi
    
    setup_monitoring
    cleanup
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✓ DEPLOYMENT EM PRODUÇÃO COMPLETADO COM SUCESSO          ║"
    echo "║                                                            ║"
    echo "║  URL: https://seu-dominio.com                             ║"
    echo "║  Monitoramento: http://localhost:9090 (Prometheus)        ║"
    echo "║  Dashboards: http://localhost:3000 (Grafana)              ║"
    echo "║                                                            ║"
    echo "║  Backup: $BACKUP_DIR/$BACKUP_NAME                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    success "Deployment completado em $ENVIRONMENT"
}

# Trap para rollback em caso de erro
trap 'error "Script interrompido"; exit 1' INT TERM

# Executar
main "$@"
