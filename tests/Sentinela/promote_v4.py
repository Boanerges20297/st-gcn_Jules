"""
====================================================================
SENTINELA V4 — SCRIPT DE PROMOÇÃO (Challenger)
====================================================================
Promove o modelo candidato V4 de tests/Sentinela/ para models/active/
e atualiza o sistema Champion/Challenger.
====================================================================
"""

import os
import shutil
import pickle
from datetime import datetime

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CANDIDATE = os.path.join(BASE_PATH, "tests", "Sentinela", "sentinela_v4_model.pkl")
ACTIVE_DIR = os.path.join(BASE_PATH, "models", "active")
TARGET = os.path.join(ACTIVE_DIR, "sentinela_v4_model.pkl")
BACKUP_DIR = os.path.join(BASE_PATH, "models", "archive")

def promote():
    print(f"--- Promoção Sentinela V4 ---")
    
    if not os.path.exists(CANDIDATE):
        print(f"❌ Erro: Modelo candidato não encontrado em {CANDIDATE}")
        return

    # Backup do anterior se existir
    if os.path.exists(TARGET):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        shutil.copy2(TARGET, os.path.join(BACKUP_DIR, f"{ts}_sentinela_v4_model.pkl"))
        print(f"[OK] Backup do V4 anterior realizado.")

    # Promoção física
    shutil.copy2(CANDIDATE, TARGET)
    print(f"[OK] Modelo V4 copiado para {TARGET}")

    # Atualizar o ChampionChallenger.py via substituição de string (ou informar o usuário)
    # Aqui vamos apenas promover o arquivo. A lógica de troca de path no código 
    # será feita separadamente para garantir integridade.
    
    print(f"--- Promoção Concluída com Sucesso ---")

if __name__ == "__main__":
    promote()
