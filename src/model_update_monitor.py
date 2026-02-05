"""
Monitor de atualizações de dados e retreinamento de modelos.
Detecta mudanças em data/raw/ e dispara reprocessamento + retreinamento.
"""

import os
import json
import hashlib
import threading
import time
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# Estados globais
UPDATE_STATE = {
    'status': 'idle',  # idle, processing, training, updating_models, error
    'progress': 0,     # 0-100
    'message': '',
    'last_check': None,
    'last_update': None,
    'error_message': None
}

STATE_LOCK = threading.Lock()
MONITOR_ACTIVE = False

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHECKSUM_FILE = os.path.join(BASE_DIR, '.data_checksum')

def get_directory_hash(directory):
    """Calcula hash MD5 de todos os arquivos em um diretório."""
    if not os.path.exists(directory):
        return None
    
    hasher = hashlib.md5()
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if file.startswith('.'):
                continue
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
            except:
                pass
    
    return hasher.hexdigest()

def save_checksum(checksum):
    """Salva o checksum atual."""
    with open(CHECKSUM_FILE, 'w') as f:
        json.dump({'hash': checksum, 'timestamp': datetime.now().isoformat()}, f)

def load_checksum():
    """Carrega o checksum anterior."""
    if os.path.exists(CHECKSUM_FILE):
        try:
            with open(CHECKSUM_FILE, 'r') as f:
                return json.load(f).get('hash')
        except:
            return None
    return None

def update_state(status=None, progress=None, message=None, error=None):
    """Atualiza o estado global thread-safe."""
    with STATE_LOCK:
        if status:
            UPDATE_STATE['status'] = status
        if progress is not None:
            UPDATE_STATE['progress'] = progress
        if message:
            UPDATE_STATE['message'] = message
        if error:
            UPDATE_STATE['error_message'] = error
        if not error:
            UPDATE_STATE['error_message'] = None

def get_state():
    """Retorna cópia do estado atual."""
    with STATE_LOCK:
        return dict(UPDATE_STATE)

def run_data_processing():
    """Executa data_processing.py."""
    try:
        update_state(status='processing', progress=10, message='Reprocessando dados brutos...')
        
        script_path = os.path.join(BASE_DIR, 'src', 'data_processing.py')
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"data_processing.py failed: {result.stderr}")
        
        update_state(progress=30, message='Dados reprocessados com sucesso')
        return True
    except Exception as e:
        update_state(
            status='error',
            message='Erro ao reprocessar dados',
            error=str(e)
        )
        return False

def run_model_training():
    """Executa scripts de treinamento."""
    try:
        update_state(status='training', progress=40, message='Retreinando ST-GCN...')
        
        # 1. Treinar ST-GCN
        script_path = os.path.join(BASE_DIR, 'src', 'train.py')
        result = subprocess.run(
            ['python', script_path, '--epochs', '50', '--batch_size', '8'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hora timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"ST-GCN training failed: {result.stderr}")
        
        update_state(progress=70, message='Retreinando Ranking Model...')
        
        # 2. Treinar Ranking Model
        ranking_script = os.path.join(BASE_DIR, 'scripts', 'training', 'train_ranking_window30_final.py')
        if os.path.exists(ranking_script):
            result = subprocess.run(
                ['python', ranking_script],
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"Ranking training failed: {result.stderr}")
        
        update_state(progress=90, message='Modelos atualizados com sucesso')
        return True
    except Exception as e:
        update_state(
            status='error',
            message='Erro ao treinar modelos',
            error=str(e)
        )
        return False

def check_and_update():
    """Verifica mudanças e dispara atualização se necessário."""
    try:
        current_hash = get_directory_hash(DATA_RAW_DIR)
        previous_hash = load_checksum()
        
        with STATE_LOCK:
            UPDATE_STATE['last_check'] = datetime.now().isoformat()
        
        if current_hash != previous_hash:
            print(f"[MONITOR] Mudanças detectadas em {DATA_RAW_DIR}")
            
            # Executa pipeline de atualização
            if run_data_processing() and run_model_training():
                save_checksum(current_hash)
                
                with STATE_LOCK:
                    UPDATE_STATE['status'] = 'updating_models'
                    UPDATE_STATE['progress'] = 95
                    UPDATE_STATE['message'] = 'Sincronizando modelos...'
                    UPDATE_STATE['last_update'] = datetime.now().isoformat()
                
                # Aguarda modelo ser recarregado (app.py detecta automaticamente)
                time.sleep(2)
                
                update_state(status='idle', progress=100, message='Atualização concluída!')
                print("[MONITOR] Atualização concluída com sucesso!")
            else:
                update_state(status='error', progress=0)
        else:
            update_state(message='Sem mudanças detectadas')
    except Exception as e:
        print(f"[MONITOR ERROR] {e}")
        traceback.print_exc()
        update_state(
            status='error',
            message='Erro no monitor de atualização',
            error=str(e)
        )

def start_monitor(check_interval=300):
    """
    Inicia o monitor em thread separada.
    check_interval: segundos entre verificações (padrão: 5 min)
    """
    global MONITOR_ACTIVE
    
    if MONITOR_ACTIVE:
        print("[MONITOR] Monitor já está ativo")
        return
    
    MONITOR_ACTIVE = True
    
    def monitor_loop():
        print(f"[MONITOR] Iniciado - verificando a cada {check_interval}s")
        while MONITOR_ACTIVE:
            try:
                check_and_update()
            except Exception as e:
                print(f"[MONITOR LOOP ERROR] {e}")
                traceback.print_exc()
            
            time.sleep(check_interval)
    
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    print("[MONITOR] Thread iniciada em background")

def stop_monitor():
    """Para o monitor."""
    global MONITOR_ACTIVE
    MONITOR_ACTIVE = False
    print("[MONITOR] Parado")

if __name__ == '__main__':
    # Teste manual
    print("Verificando mudanças...")
    check_and_update()
    print(f"Estado: {get_state()}")
