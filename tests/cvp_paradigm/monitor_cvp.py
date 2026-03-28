import os
import time

LOG_PATH = 'logs/training_CVP_PARADIGM.log'

def monitor():
    if not os.path.exists(LOG_PATH):
        print("Aguardando inicio do log...")
        return

    print("--- MONITORAMENTO PARADIGMA CVP ---")
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            print("Log vazio.")
            return
        
        # Mostrar as ultimas 5 linhas
        for line in lines[-5:]:
            print(line.strip())
        
        # Buscar o melhor recorde no arquivo
        records = [line for line in lines if "RECORD:" in line]
        if records:
            print(f"\nMelhor Performance ate agora: {records[-1].strip()}")

if __name__ == "__main__":
    monitor()
