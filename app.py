from flask import Flask, jsonify, render_template, request
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import pickle
import json
import warnings
import logging
import unicodedata
from datetime import datetime, timedelta
import re
import sys
import threading
import time

# --- Orquestrador Regional ST-GAT ---
try:
    from src.core.orchestrator import StateOrchestrator, normalize_name
    from src.core.efficiency_monitor import EfficiencyMonitor
    orchestrator = None 
except ImportError:
    # Fallback se o PYTHONPATH não incluir a raiz corretamente
    import sys
    sys.path.append(os.getcwd())
    from src.core.orchestrator import StateOrchestrator
    from src.core.efficiency_monitor import EfficiencyMonitor
    def normalize_name(text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
        import re
        return re.sub(r'\s*-\s*AIS.*$', '', text).strip()

warnings.filterwarnings('ignore')

# Configurações globais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)

# Cache de nomes para busca rápida
nodes_gdf = None
nodes_map = {}
last_update = None
efficiency_metrics = None
efficiency_monitor = None

def load_data_and_models():
    """Carrega os dados geoespaciais e inicializa o orquestrador neural."""
    global nodes_gdf, nodes_map, orchestrator, efficiency_monitor
    
    try:
        # 1. Carregar GeoDataFrame de Fortaleza para nomes e coordenadas
        pkl_path = os.path.join(DATA_DIR, 'processed_fortaleza.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data_load = pickle.load(f)
                nodes_gdf = data_load.get("nodes_gdf")
                if nodes_gdf is not None:
                    # Normalizar nomes para o mapa
                    for i, row in nodes_gdf.iterrows():
                        name_norm = normalize_name(row['name'])
                        nodes_map[name_norm] = {
                            'id': i,
                            'lat': row['latitude'],
                            'lng': row['longitude'],
                            'original_name': row['name']
                        }
            print(f"✅ Dados geoespaciais carregados: {len(nodes_map)} localidades.")
        
        # 2. Inicializar Cérebro Neural
        orchestrator = StateOrchestrator(BASE_DIR)
        
        # 3. Iniciar Monitor de Eficiência e Relatórios
        efficiency_monitor = EfficiencyMonitor(BASE_DIR, orchestrator, nodes_gdf)
        
        # Disparar Monitor em Segundo Plano (Thread Paralela)
        threading.Thread(target=run_background_efficiency_monitor, daemon=True).start()
        
    except Exception as e:
        print(f"❌ Erro crítico ao carregar sistema: {e}")
        import traceback
        traceback.print_exc()

def run_background_efficiency_monitor():
    """Tarefa em background que executa a cada 7 dias ou no start."""
    global efficiency_monitor
    # Aguarda o sistema inicializar completamente
    time.sleep(10)
    while True:
        if efficiency_monitor is not None:
            try:
                num_loc = len(nodes_gdf) if nodes_gdf is not None else 0
                print(f"\n" + "="*60)
                print(f"🛡️  [MONITOR DE EFICIÊNCIA] Iniciando Avaliação ({num_loc} localidades)")
                print("="*60)
                
                metrics = efficiency_monitor.run_evaluation()
                
                if metrics:
                    print(f"📅 Data da Avaliação: {metrics.get('date')}")
                    print(f"📊 Eventos Detectados: {metrics.get('total_events', 0)} ({metrics.get('brute_cvli', 0)} Brutos + {metrics.get('exogenous', 0)} Exógenos)")
                    
                    # Exibir Global
                    if 'global' in metrics:
                        m = metrics['global']
                        print(f"\n🌍 REGIONALIZAÇÃO: GLOBAL")
                        print(f"   P5:  {m.get('p5', 0)*100:.1f}% | Hits: {', '.join(m.get('hits5', []))}")
                        print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                        print(f"   P20: {m.get('p20', 0)*100:.1f}% | Hits: {', '.join(m.get('hits20', []))}")
                    
                    # Exibir Fortaleza
                    if 'fortaleza' in metrics:
                        m = metrics['fortaleza']
                        print(f"\n🏙️  REGIONALIZAÇÃO: FORTALEZA")
                        print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                    
                    # Exibir RMF e Interior se houver acertos
                    for reg in ['rmf', 'interior']:
                        if reg in metrics and metrics[reg].get('p10', 0) > 0:
                            m = metrics[reg]
                            reg_name = "REGIÃO METROPOLITANA" if reg == 'rmf' else "INTERIOR"
                            print(f"\n📍 REGIONALIZAÇÃO: {reg_name}")
                            print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                    
                    print("\n" + "="*60 + "\n")
                else:
                    print("📊 [Monitor] Sem eventos suficientes para avaliação hoje.")
            except Exception as e:
                print(f"⚠️ [Monitor] Erro na thread de eficiência: {e}")
        
        # Dorme por 7 dias antes da próxima rodada (604800 segundos)
        time.sleep(604800)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    if orchestrator is None:
        return jsonify({'error': 'Sistema em inicialização...'}), 503
    
    # Obter região (opcional)
    region = request.args.get('region', None)
    
    # Obter scores combinados
    scores = orchestrator.get_combined_risk(None)
    
    # Formatar resposta para o Leaflet
    results = []
    for name, score in scores.items():
        if name in nodes_map:
            node_info = nodes_map[name]
            results.append({
                'name': node_info['original_name'],
                'lat': node_info['lat'],
                'lng': node_info['lng'],
                'risk': float(score)
            })
    
    # Sort por risco
    results = sorted(results, key=lambda x: x['risk'], reverse=True)
    
    # Obter métricas do monitor
    metrics = efficiency_monitor.get_latest_metrics() if efficiency_monitor else None
    
    return jsonify({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'predictions': results,
        'metrics': metrics
    })

if __name__ == '__main__':
    load_data_and_models()
    app.run(host='0.0.0.0', port=5050, debug=True)
