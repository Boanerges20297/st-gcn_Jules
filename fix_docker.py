#!/usr/bin/env python3
"""
SOLUÇÃO RÁPIDA - Rodar app.py direto sem Docker para testar
"""

print("""
════════════════════════════════════════════════════════════════════════════════
                    SOLUÇÃO RÁPIDA - SEM DOCKER
════════════════════════════════════════════════════════════════════════════════

Se Docker não está funcionando, você pode rodar direto do Python:

OPÇÃO 1: RODAR APP.PY DIRETO (mais rápido para testar)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ python app.py
  
  app rodará em: http://localhost:5050

OPÇÃO 2: PARAR DOCKER-COMPOSE E LIMPAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ docker-compose down -v
  $ docker system prune -a
  $ docker-compose up -d
  
  (isto vai redownload imagens e reiniciar tudo)

OPÇÃO 3: VERIFICAR SE APARECE ERRO ESPECÍFICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ docker logs st-gcn-app
  
  (mostra erros dentro do container)

OPÇÃO 4: DEBUGAR MANUALMENTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ docker ps -a
  
  (ver TODOS containers, incluindo os parados)
  
  Se houver container parado com erro:
  
  $ docker logs <CONTAINER_ID>
  
  (ver logs do container parado)

RECOMENDADO: Fazer isto AGORA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Passo 1: Para primeiro
  $ docker-compose down

Passo 2: Limpar tudo
  $ docker system prune -a --volumes
  
  (Responda 'y' para confirmar)

Passo 3: Reiniciar
  $ docker-compose up -d
  
  (vai refazer tudo do zero, diferente desta vez)

Passo 4: Depois de subir, espera 30 segundos e acessa:
  http://localhost:5000/dashboard
  
  (NOTA: Porta é 5000, não 5050 neste docker-compose.yml)

AINDA NÃO FUNCIONA? Vai pro Plano B:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ docker-compose down
  $ python app.py
  
  App vai rodar em: http://localhost:5050
  
  (sim, porta diferente porque .env default é 5050, docker-compose.yml usa 5000)

════════════════════════════════════════════════════════════════════════════════
""")

import subprocess
import time

print("\n[EXECUTANDO] Limpeza e reinício automático...\n")

# Passo 1: Parar
print("[1/4] Parando docker-compose...")
subprocess.run("docker-compose down", shell=True, capture_output=True)
time.sleep(2)

# Passo 2: Limpar (com confirmação automática)
print("[2/4] Limpando sistema Docker...")
subprocess.run("docker system prune -a --volumes -f", shell=True, capture_output=True)
time.sleep(2)

# Passo 3: Subir
print("[3/4] Subindo docker-compose novamente...")
result = subprocess.run("docker-compose up -d", shell=True, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"❌ Erro: {result.stderr}")
time.sleep(10)

# Passo 4: Verificar
print("[4/4] Verificando containers...")
result = subprocess.run("docker-compose ps", shell=True, capture_output=True, text=True)
print(result.stdout)

# Teste final
print("\n" + "="*80)
print("TESTE FINAL")
print("="*80)

import requests
import time

endpoints = [
    ("http://localhost:5000/", "HOME"),
    ("http://localhost:5000/dashboard", "DASHBOARD"),
]

print("\nAguardando app inicializar (15 segundos)...")
time.sleep(15)

for url, name in endpoints:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: {url} (200 OK)")
        else:
            print(f"⚠️  {name}: {url} ({response.status_code})")
    except Exception as e:
        print(f"❌ {name}: {url} - {str(e)[:50]}")

print("\n" + "="*80)
print("FIM DO SCRIPT DE RECUPERAÇÃO")
print("="*80)
print("\nSe ainda não funcionar:")
print("  1. Abra Docker Desktop (Application > Docker)")
print("  2. Aguarde até ver 'Docker is running'")
print("  3. Tente novamente: docker-compose ps")
