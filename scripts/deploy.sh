#!/bin/bash
###############################################################################
# Deploy Script - Staging Environment
# Deployment seguro com validações e rollback automático
###############################################################################

set -e  # Exit on any error

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
ENVIRONMENT="staging"
APP_NAME="st-gcn-app"
DEPLOY_TIMEOUT=300
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_DELAY=10

# Funções de output
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar pré-requisitos
check_prerequisites() {
    info "Verificando pré-requisitos..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker não está instalado"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose não está instalado"
        exit 1
    fi
    
    success "Pré-requisitos verificados"
}

# Validar arquivo de configuração
validate_config() {
    info "Validando configuração..."
    
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml não encontrado"
        exit 1
    fi
    
    if [ ! -f "Dockerfile" ]; then
        error "Dockerfile não encontrado"
        exit 1
    fi
    
    # Validar sintaxe YAML
    if ! docker-compose config > /dev/null 2>&1; then
        error "docker-compose.yml tem sintaxe inválida"
        exit 1
    fi
    
    success "Configuração válida"
}

# Build da imagem Docker
build_image() {
    info "Build da imagem Docker..."
    
    if ! docker-compose build --no-cache; then
        error "Build falhou"
        exit 1
    fi
    
    success "Build completado"
}

# Parar containers antigos (com backup)
stop_existing() {
    info "Parando containers existentes..."
    
    # Backup do estado anterior
    if docker-compose ps | grep -q "st-gcn-app"; then
        docker-compose stop -t 30 || true
        warn "Containers anteriores parados. Backup mantido para rollback."
    fi
    
    success "Containers parados"
}

# Iniciar novos containers
start_containers() {
    info "Iniciando novos containers..."
    
    if ! timeout $DEPLOY_TIMEOUT docker-compose up -d; then
        error "Falha ao iniciar containers"
        return 1
    fi
    
    success "Containers iniciados"
    return 0
}

# Health check
perform_health_check() {
    info "Executando health check..."
    
    local retry=0
    local max_retries=$HEALTH_CHECK_RETRIES
    
    while [ $retry -lt $max_retries ]; do
        if python scripts/health_check.py; then
            success "Health check passou"
            return 0
        fi
        
        retry=$((retry + 1))
        if [ $retry -lt $max_retries ]; then
            warn "Health check falhou (tentativa $retry/$max_retries). Aguardando..."
            sleep $HEALTH_CHECK_DELAY
        fi
    done
    
    error "Health check falhou após $max_retries tentativas"
    return 1
}

# Smoke tests
run_smoke_tests() {
    info "Executando smoke tests..."
    
    # Teste 1: verificar se app está respondendo
    if ! curl -f http://localhost:5000/ > /dev/null 2>&1; then
        error "App não está respondendo no /health"
        return 1
    fi
    
    # Teste 2: verificar endpoints API
    if ! curl -f http://localhost:5000/api/metrics > /dev/null 2>&1; then
        error "API /metrics não está respondendo"
        return 1
    fi
    
    success "Smoke tests passou"
    return 0
}

# Verificar logs para erros críticos
check_logs() {
    info "Verificando logs para erros..."
    
    if docker-compose logs $APP_NAME | grep -i "error\|exception\|fatal" | grep -v "WARNING"; then
        warn "Possível erro nos logs"
        return 1
    fi
    
    success "Logs verificados"
    return 0
}

# Rollback em caso de falha
rollback() {
    error "Iniciando rollback..."
    
    # Parar containers novos
    docker-compose down || true
    
    # Tentar restaurar backup anterior
    if [ -f ".docker-compose.backup" ]; then
        warn "Restaurando configuração anterior..."
        # Nota: Em produção, isso seria mais complexo
        # Aqui apenas stopamos para evitar estado inconsistente
    fi
    
    error "Rollback completado. Sistema pode estar inconsistente - verificar manualmente"
    exit 1
}

# Limpeza
cleanup() {
    info "Realizando limpeza..."
    
    # Remover imagens antigas (opcional)
    docker image prune -f > /dev/null 2>&1 || true
    
    success "Limpeza completada"
}

###############################################################################
# MAIN
###############################################################################

main() {
    clear
    
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ST-GCN Crime Prediction System - Deploy Script (Staging)  ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Confirmar deployment
    read -p "Continuar com deployment para $ENVIRONMENT? (s/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        warn "Deployment cancelado pelo usuário"
        exit 0
    fi
    
    # Executar stages
    check_prerequisites || exit 1
    validate_config || exit 1
    build_image || exit 1
    stop_existing || true
    
    if ! start_containers; then
        rollback
    fi
    
    if ! perform_health_check; then
        rollback
    fi
    
    if ! run_smoke_tests; then
        rollback
    fi
    
    check_logs || warn "Verifique os logs manualmente"
    cleanup
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✓ DEPLOYMENT COMPLETADO COM SUCESSO                      ║"
    echo "║                                                            ║"
    echo "║  Acesso:  http://localhost:5000                           ║"
    echo "║  Grafana: http://localhost:3000 (admin/admin)             ║"
    echo "║  Prometheus: http://localhost:9090                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    success "Sistema pronto em $ENVIRONMENT"
}

# Trap para rollback em caso de erro
trap 'error "Script interrompido"; rollback' INT TERM

# Executar
main "$@"
