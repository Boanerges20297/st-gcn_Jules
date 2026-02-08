#!/usr/bin/env python3
"""
DIAGNÓSTICO RÁPIDO - ST-GCN Docker Deploy
Identifica por que localhost não está respondendo
"""

import subprocess
import json
import time
import sys
from datetime import datetime

def run_command(cmd, silent=False):
    """Executar comando e retornar output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def check_docker():
    """Verificar se Docker está instalado e rodando"""
    print_header("1️⃣  VERIFICANDO DOCKER")
    
    print("Verificando Docker...")
    code, out, err = run_command("docker --version")
    if code != 0:
        print("  ❌ Docker não está instalado")
        print("     Baixe em: https://www.docker.com/products/docker-desktop")
        return False
    print(f"  ✅ {out.strip()}")
    
    print("\nVerificando se Docker daemon está rodando...")
    code, out, err = run_command("docker ps")
    if code != 0:
        print("  ❌ Docker daemon não está rodando")
        print("     Solução: Abra Docker Desktop")
        return False
    print("  ✅ Docker daemon está rodando")
    
    return True

def check_docker_compose():
    """Verificar docker-compose"""
    print_header("2️⃣  VERIFICANDO DOCKER COMPOSE")
    
    code, out, err = run_command("docker-compose --version")
    if code != 0:
        print("  ❌ docker-compose não encontrado")
        print("     Solução: Docker Desktop já inclui ele")
        return False
    print(f"  ✅ {out.strip()}")
    
    return True

def check_containers():
    """Verificar status dos containers"""
    print_header("3️⃣  VERIFICANDO CONTAINERS")
    
    print("Listando containers em execução...")
    code, out, err = run_command("docker-compose ps")
    
    if code != 0:
        print("  ❌ Erro ao rodar docker-compose ps")
        print(f"     Erro: {err}")
        print("\n  Possível problema:")
        print("    • Você está no diretório st-gcn_jules?")
        print("    • docker-compose.yml existe aqui?")
        return False
    
    print(out)
    
    # Analisar output
    print("\nAnalisando containers...")
    
    if "st-gcn_jules-app-1" not in out or "Up" not in out:
        print("  ⚠️  Container app não está rodando!")
        print("\n  Solução 1: Subir docker-compose")
        print("    $ docker-compose up -d")
        
        print("\n  Solução 2: Se erro, ver logs")
        print("    $ docker-compose logs app")
        return False
    
    print("  ✅ Container app está rodando")
    
    if "st-gcn_jules-redis-1" not in out:
        print("  ⚠️  Redis não está rodando (cache)")
    else:
        print("  ✅ Redis está rodando")
    
    if "st-gcn_jules-prometheus-1" not in out:
        print("  ⚠️  Prometheus não está rodando (métricas)")
    else:
        print("  ✅ Prometheus está rodando")
    
    if "st-gcn_jules-grafana-1" not in out:
        print("  ⚠️  Grafana não está rodando")
    else:
        print("  ✅ Grafana está rodando")
    
    return "app" in out and "Up" in out

def check_ports():
    """Verificar se portas estão abertas"""
    print_header("4️⃣  VERIFICANDO PORTAS")
    
    ports = [
        (5050, "App Principal"),
        (3000, "Grafana"),
        (9090, "Prometheus"),
        (6379, "Redis")
    ]
    
    for port, name in ports:
        # Windows
        if sys.platform == "win32":
            code, out, err = run_command(f'netstat -ano | findstr :{port}')
            if code == 0 and out.strip():
                print(f"  ✅ Porta {port} ({name}) está aberta")
            else:
                print(f"  ❌ Porta {port} ({name}) não está aberta")
        
        # Linux/Mac
        else:
            code, out, err = run_command(f'lsof -i :{port}')
            if code == 0 and out.strip():
                print(f"  ✅ Porta {port} ({name}) está aberta")
            else:
                print(f"  ❌ Porta {port} ({name}) não está aberta")

def check_app_health():
    """Verificar se app está respondendo"""
    print_header("5️⃣  TESTANDO ENDPOINTS")
    
    endpoints = [
        ("http://localhost:5050/", "Home"),
        ("http://localhost:5050/dashboard", "Dashboard"),
        ("http://localhost:5050/api/metrics", "Métricas"),
        ("http://localhost:3000", "Grafana"),
    ]
    
    print("Testando endpoints...")
    
    for url, name in endpoints:
        print(f"\n  Testando {name} ({url})...")
        
        # Tentar com curl
        code, out, err = run_command(f'curl -s -w "Status: %%{{http_code}}" -o /dev/null {url}')
        
        if "200" in code or "200" in out:
            print(f"    ✅ Respondendo (200 OK)")
        elif "404" in code or "404" in out:
            print(f"    ⚠️  Encontrado mas erro 404 (conteúdo não existe)")
        elif "Connection refused" in err or "Failed to connect" in err:
            print(f"    ❌ Conexão recusada")
            print(f"       └─ Porta pode não estar aberta ou app caiu")
        elif "timeout" in err.lower():
            print(f"    ⏱️  Timeout (servidor muito lento)")
        else:
            print(f"    ⚠️  Erro desconhecido: {err[:60]}")

def check_logs():
    """Verificar logs para erros"""
    print_header("6️⃣  ANALISANDO LOGS")
    
    print("Últimas 20 linhas de logs do app...")
    code, out, err = run_command("docker-compose logs app | tail -20")
    
    if code == 0:
        print(out)
        
        # Procurar por erros comuns
        if "ModuleNotFoundError" in out:
            print("\n  ❌ ERRO: Módulo não encontrado")
            print("     Solução: Verificar imports em app.py")
        
        if "Address already in use" in out:
            print("\n  ❌ ERRO: Porta já está em uso")
            print("     Solução: docker-compose down && docker-compose up -d")
        
        if "Segmentation fault" in out:
            print("\n  ❌ ERRO: Crash na aplicação")
            print("     Solução: Verificar recursos disponíveis")
    else:
        print("  Erro ao ler logs:", err)

def check_docker_resources():
    """Verificar recursos do Docker"""
    print_header("7️⃣  VERIFICANDO RECURSOS")
    
    print("Uso de memória e CPU dos containers...")
    code, out, err = run_command("docker stats --no-stream")
    
    if code == 0:
        print(out)
        print("\n  ✅ Se % valores estão normais, recursos OK")
        print("  ⚠️  Se algum container usa > 80% CPU/MEM, pode ser problema")
    else:
        print("  Não foi possível coletar stats")

def main():
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  DIAGNÓSTICO DOCKER - ST-GCN".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    print(f"\nHora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Diagnóstico completo do setup Docker...")
    
    # Rodando todas as verificações
    docker_ok = check_docker()
    if not docker_ok:
        print("\n" + "!"*80)
        print("PARAR AQUI: Docker não está funcionando")
        print("!"*80)
        return
    
    compose_ok = check_docker_compose()
    containers_ok = check_containers()
    check_ports()
    
    # Esperar um pouco se containers foram iniciados agora
    if containers_ok:
        print("\n  Aguardando app inicializar (5 segundos)...")
        time.sleep(5)
    
    check_app_health()
    check_logs()
    check_docker_resources()
    
    # RESUMO FINAL
    print_header("📋 RESUMO DO DIAGNÓSTICO")
    
    print("Verificações realizadas:")
    print(f"  • Docker instalado e rodando: {'✅' if docker_ok else '❌'}")
    print(f"  • Docker Compose disponível: {'✅' if compose_ok else '❌'}")
    print(f"  • Containers em execução: {'✅' if containers_ok else '⚠️'}")
    
    # SOLUÇÕES POSSÍVEIS
    print("\n" + "-"*80)
    print("\n🔧 SOLUÇÕES COMUNS:\n")
    
    print("PROBLEMA: 'Connection refused' no localhost:5050")
    print("  SOLUÇÃO 1: Containers não estão rodando")
    print("    $ docker-compose down")
    print("    $ docker-compose up -d")
    print()
    
    print("  SOLUÇÃO 2: App crashed (erro nos logs)")
    print("    $ docker-compose logs app")
    print("    └─ Ver mensagem de erro e corrigir em app.py")
    print()
    
    print("PROBLEMA: 'Port 5050 already in use'")
    print("  SOLUÇÃO: Docker ainda usando porta de deploy anterior")
    print("    $ docker-compose down -v")
    print("    $ docker-compose up -d")
    print()
    
    print("PROBLEMA: Very slow / timeout")
    print("  SOLUÇÃO: Docker Desktop sem RAM suficiente")
    print("    1. Abra Docker Desktop")
    print("    2. Settings > Resources")
    print("    3. Aumentar CPU: 4, RAM: 4-6GB")
    print("    4. Reiniciar Docker")
    print()
    
    print("PROBLEMA: Módulo não encontrado")
    print("  SOLUÇÃO: Verificar sys.path no código")
    print("    $ docker-compose exec app python -c \"import src.metrics\"")
    print()
    
    print("-"*80)
    
    # TESTE FINAL
    print("\n\n✨ TESTAR AGORA:\n")
    print("1. Abra navegador: http://localhost:5050/dashboard")
    print("2. Deve mostrar gráficos e métricas")
    print()
    print("3. Se ainda não funcionar, execute:")
    print("   $ docker-compose down -v")
    print("   $ docker-compose up -d")
    print("   $ python diagnostic.py  # rodar este script novamente")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnóstico cancelado.")
    except Exception as e:
        print(f"\n\n❌ Erro durante diagnóstico: {e}")
        import traceback
        traceback.print_exc()
