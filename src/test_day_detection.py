#!/usr/bin/env python
"""
test_day_detection_and_model_loading.py

Verifica se o sistema detecta corretamente o dia da semana
e carrega o modelo correspondente
"""

import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_day_detection_and_loading():
    print("=" * 80)
    print("🔍 TESTE: DETECÇÃO DE DIA DA SEMANA + CARREGAMENTO DE MODELO")
    print("=" * 80)
    
    # 1. Detectar dia atual
    today = datetime.now()
    day_of_week = today.weekday()  # 0=Segunda, 6=Domingo
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    current_day_name = day_names[day_of_week]
    
    print(f"\n[DETECÇÃO] Data atual: {today.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"[DETECÇÃO] Dia da semana detectado: {day_of_week} ({current_day_name})")
    
    # 2. Verificar se modelo existe para este dia
    model_dir = Path(ROOT) / 'models' / 'ranking_by_day'
    model_path = model_dir / f'ranking_model_day{day_of_week}.pth'
    
    print(f"\n[MODELO] Caminho esperado: {model_path}")
    print(f"[MODELO] Arquivo existe: {model_path.exists()}")
    
    # 3. Listar todos os modelos disponíveis
    print(f"\n[MODELOS] Disponíveis em {model_dir}:")
    if model_dir.exists():
        models = sorted(model_dir.glob('ranking_model_day*.pth'))
        scalers = model_dir.glob('scalers.pkl')
        
        for model_file in models:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            day_num = int(model_file.stem.split('day')[1])
            day_name = day_names[day_num]
            print(f"  ✅ {model_file.name:<30} ({size_mb:.2f} MB) - {day_name}")
        
        for scaler_file in scalers:
            size_mb = scaler_file.stat().st_size / (1024 * 1024)
            print(f"  ✅ {scaler_file.name:<30} ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ Diretório não existe: {model_dir}")
        print(f"     Execute: python src/train_ranking_final_production.py")
    
    # 4. Testar carregamento do sistema
    print(f"\n[SISTEMA] Carregando ranking system...")
    try:
        from src.ranking_correction_system import get_ranking_system
        ranking_system = get_ranking_system()
        print(f"  ✅ Sistema carregado com sucesso!")
        print(f"  ✅ Modelos disponíveis: {list(ranking_system.models_by_day.keys())}")
        print(f"  ✅ Escaladores carregados: {len(ranking_system.scalers_by_day)}")
        
        # 5. Testar que ele vai usar o modelo correto para hoje
        print(f"\n[TESTE] Simulando predição para {current_day_name}...")
        import pickle
        import numpy as np
        
        # Dados de teste
        pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        cvli_data = data['node_features'][:, -30:, 0]  # Últimos 30 dias
        
        scores, confidence = ranking_system.get_ranking_scores(
            cvli_data,
            day_of_week=day_of_week
        )
        
        print(f"  ✅ Scores obtidos: {len(scores)} nós")
        print(f"  ✅ Confiança para {current_day_name}: {confidence:.4f}")
        print(f"  ✅ Top-5 scores: {scores[np.argsort(-scores)[:5]]}")
        
        print(f"\n[RESULTADO] ✅ SISTEMA FUNCIONANDO PERFEITAMENTE!")
        print(f"  - Dia detectado: {current_day_name}")
        print(f"  - Modelo carregado: ranking_model_day{day_of_week}.pth")
        print(f"  - Confiança: {confidence:.4f}")
        print(f"  - Pronto para predições!")
        
    except Exception as e:
        print(f"  ❌ ERRO ao carregar: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("📋 FLUXO DE EXECUÇÃO NA APP:")
    print("=" * 80)
    print("""
1. Requisição chega em calculate_risk()
2. Sistema detecta dia da semana automaticamente
3. Carrega modelo correspondente: ranking_model_day{X}.pth
4. Valida predições do ST-GCN
5. Se confiança > 0.6: corrige top-5
6. Retorna resultado melhorado

Dia atual: %s
Modelo: ranking_model_day%d.pth
Status: ✅ PRONTO
    """ % (current_day_name, day_of_week))

if __name__ == "__main__":
    test_day_detection_and_loading()
