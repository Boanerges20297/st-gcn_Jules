"""
Monitor de progresso do treinamento
"""

import os
import time
from pathlib import Path

BASE_DIR = Path('.')
MODEL_PATH = BASE_DIR / 'models' / 'stgcn_model_v2.pth'

print("\n" + "="*80)
print("📊 MONITOR DE TREINAMENTO")
print("="*80)
print("\nAguardando conclusão do treinamento...")
print("(Isto pode levar 20-30 minutos com 60 épocas)")

last_size = 0
check_interval = 10  # verificar a cada 10 segundos

while True:
    time.sleep(check_interval)
    
    if MODEL_PATH.exists():
        current_size = MODEL_PATH.stat().st_size
        
        if current_size > last_size:
            size_mb = current_size / (1024 * 1024)
            print(f"✅ Modelo sendo atualizado... ({size_mb:.2f} MB)")
            last_size = current_size
        
        # Se o tamanho não mudou por 60 segundos, provavelmente acabou
        if last_size > 0:
            time.sleep(60)
            if MODEL_PATH.stat().st_size == last_size:
                print("\n✅ TREINAMENTO CONCLUÍDO!")
                final_size = MODEL_PATH.stat().st_size / (1024 * 1024)
                print(f"   Modelo salvo: {final_size:.2f} MB")
                print(f"   Arquivo: models/stgcn_model_v2.pth")
                break
    else:
        print("⏳ Aguardando início do treinamento...")
