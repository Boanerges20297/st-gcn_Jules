#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Paths definition
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_ENRIQUECIDO = BASE_DIR / 'data' / 'raw' / 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    SIBLING_DIR = Path(r"C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico")
    DST_ENRIQUECIDO = SIBLING_DIR / 'data' / 'processed' / 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    SIBLING_PREPARE_SCRIPT = SIBLING_DIR / 'src' / 'prepare_data.py'
    SIBLING_PYTHON = SIBLING_DIR / '.venv' / 'Scripts' / 'python.exe'
else:
    # Linux VPS paths (Hostinger)
    SIBLING_DIR = Path("/root/crime-predict_mosaico")
    DST_ENRIQUECIDO = SIBLING_DIR / 'data' / 'processed' / 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    SIBLING_PREPARE_SCRIPT = SIBLING_DIR / 'src' / 'prepare_data.py'
    # Remote runs through Docker

def main():
    print("=== Sincronizacao de Artefatos para o Crime-Predict ===")
    
    if not SRC_ENRIQUECIDO.exists():
        print(f"Erro: Base de origem nao encontrada em {SRC_ENRIQUECIDO}")
        sys.exit(1)
        
    if not SIBLING_DIR.exists():
        print(f"Erro: Pasta do projeto destino nao encontrada em {SIBLING_DIR}")
        sys.exit(1)
        
    print(f"Copiando base enriquecida atualizada...")
    print(f"Origem: {SRC_ENRIQUECIDO}")
    print(f"Destino: {DST_ENRIQUECIDO}")
    
    os.makedirs(DST_ENRIQUECIDO.parent, exist_ok=True)
    shutil.copy2(SRC_ENRIQUECIDO, DST_ENRIQUECIDO)
    print("Copia da base enriquecida concluida com sucesso.")
    
    if IS_WINDOWS:
        if not SIBLING_PREPARE_SCRIPT.exists():
            print(f"Aviso: Script de preparacao nao encontrado em {SIBLING_PREPARE_SCRIPT}")
            return
            
        print("\nExecutando script de preparacao de dados no Crime-Predict (Windows Local)...")
        python_exec = str(SIBLING_PYTHON) if SIBLING_PYTHON.exists() else "python"
        
        try:
            result = subprocess.run(
                [python_exec, str(SIBLING_PREPARE_SCRIPT)],
                cwd=str(SIBLING_DIR),
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            print("Script prepare_data.py executado com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar o prepare_data.py:")
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)
            
        # Copiar fortaleza_crimes.csv para fortaleza_crimes_normalizado.csv
        fortaleza_crimes = SIBLING_DIR / 'data' / 'processed' / 'fortaleza_crimes.csv'
        fortaleza_crimes_normalizado = SIBLING_DIR / 'data' / 'processed' / 'fortaleza_crimes_normalizado.csv'
        
        if fortaleza_crimes.exists():
            print(f"\nSincronizando {fortaleza_crimes.name} para {fortaleza_crimes_normalizado.name}...")
            shutil.copy2(fortaleza_crimes, fortaleza_crimes_normalizado)
            print("Sincronizacao de normalizacao concluida.")
            
    else:
        # Linux VPS via Docker
        print("\nSincronizando base e executando preparacao via Docker no Crime-Predict (Linux VPS)...")
        try:
            # 1. Copiar o arquivo atualizado para dentro do container
            print("Copiando arquivo de dados para o container crime-predict...")
            subprocess.run(
                ["docker", "cp", str(DST_ENRIQUECIDO), "crime-predict:/app/data/processed/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"],
                check=True
            )
            
            # 2. Executar prepare_data.py no container
            print("Executando prepare_data.py no container...")
            result = subprocess.run(
                ["docker", "exec", "crime-predict", "python", "src/prepare_data.py"],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            
            # 3. Copiar fortaleza_crimes.csv para fortaleza_crimes_normalizado.csv no container
            print("Normalizando arquivos no container...")
            subprocess.run(
                ["docker", "exec", "crime-predict", "cp", "data/processed/fortaleza_crimes.csv", "data/processed/fortaleza_crimes_normalizado.csv"],
                check=True
            )
            
            # 4. Reiniciar os containers
            print("Reiniciando containers para aplicar alteracoes...")
            subprocess.run(
                ["docker", "restart", "crime-predict", "report-preview-telegram-gateway"],
                check=True
            )
            print("Containers reiniciados e atualizados.")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar comandos Docker na VPS:")
            print(e.stderr if hasattr(e, 'stderr') else str(e), file=sys.stderr)
            sys.exit(1)
            
    print("\nSincronizacao e preparacao concluidas com sucesso!")

if __name__ == '__main__':
    main()
