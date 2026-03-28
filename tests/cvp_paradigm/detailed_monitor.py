import os
import time
import re

LOG_PATH = 'logs/training_CVP_PARADIGM.log'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def parse_log():
    if not os.path.exists(LOG_PATH):
        return [], None

    epochs = []
    best_record = None
    
    # Regex para capturar EP, LOSS e P@10
    pattern = re.compile(r"EP (\d+) \| LOSS: ([\d.]+) \| P@10: ([\d.]+)%")
    
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append({
                    'ep': int(match.group(1)),
                    'loss': float(match.group(2)),
                    'p10': float(match.group(3))
                })
            if "RECORD:" in line:
                best_record = line.strip()
    
    return epochs, best_record

def monitor():
    last_count = 0
    while True:
        epochs, best = parse_log()
        
        if not epochs:
            clear_screen()
            print("--- AGUARDANDO DADOS DE TREINO (CVP) ---")
            time.sleep(2)
            continue

        clear_screen()
        print("="*60)
        print(f"📊 MONITOR DE TREINAMENTO - PARADIGMA CVP")
        print("="*60)
        
        if best:
            print(f"🏆 {best}")
        print("-"*60)
        print(f"{'EP':<5} | {'LOSS':<10} | {'P@10':<10} | {'TENDENCIA P@10'}")
        print("-"*60)

        # Mostrar as últimas 15 épocas
        display_epochs = epochs[-15:]
        for i, ep in enumerate(display_epochs):
            trend = ""
            if i > 0:
                diff = ep['p10'] - display_epochs[i-1]['p10']
                if diff > 0:
                    trend = f"▲ +{diff:.2f}%"
                elif diff < 0:
                    trend = f"▼ {diff:.2f}%"
                else:
                    trend = "• estabilizado"
            
            print(f"{ep['ep']:02d}    | {ep['loss']:.4f}    | {ep['p10']:>6.2f}%   | {trend}")

        print("-"*60)
        print(f"Atualizado em: {time.strftime('%H:%M:%S')}")
        print("Pressione Ctrl+C para sair do monitor.")
        
        time.sleep(5) # Atualiza a cada 5 segundos

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoramento encerrado.")
