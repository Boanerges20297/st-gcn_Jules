#!/usr/bin/env python3
"""
TEST: Viabilidade de integrar dados de prisões policiais como feature exógena
SEM interferir no treinamento em andamento
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def load_prison_data():
    """Carrega dados de prisões do JS convertido para JSON"""
    js_file = os.path.join(BASE_DIR, 'data', 'raw', 'data_with_coordinates.js')
    
    if not os.path.exists(js_file):
        print(f"❌ Arquivo não encontrado: {js_file}")
        return None
    
    print(f"[1/5] Carregando prisões de {js_file}...")
    
    # Ler arquivo JS
    with open(js_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extrair JSON (remove "module.exports = ")
    json_str = content.replace('module.exports = ', '').strip()
    if json_str.endswith(';'):
        json_str = json_str[:-1]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSON: {e}")
        return None
    
    print(f"✅ Carregados {len(data)} registros de prisão")
    return data

def aggregate_prison_data(prison_data):
    """Agrega prisões por data e bairro"""
    print(f"\n[2/5] Agregando prisões por data/bairro...")
    
    prison_by_date_bairro = {}
    
    for record in prison_data:
        try:
            date_str = record.get('Data', '')
            bairro = record.get('BairroOcor', '').strip().upper()
            
            if not date_str or not bairro:
                continue
            
            # Parse data
            date_obj = pd.to_datetime(date_str)
            date_key = date_obj.strftime('%Y-%m-%d')
            
            key = (date_key, bairro)
            prison_by_date_bairro[key] = prison_by_date_bairro.get(key, 0) + 1
        except Exception as e:
            continue
    
    print(f"✅ Agregadas {len(prison_by_date_bairro)} combinações únicas data+bairro")
    print(f"   Período: {min(k[0] for k in prison_by_date_bairro)} até {max(k[0] for k in prison_by_date_bairro)}")
    
    # Estatísticas
    dates_unique = len(set(k[0] for k in prison_by_date_bairro))
    bairros_unique = len(set(k[1] for k in prison_by_date_bairro))
    print(f"   {dates_unique} dias únicos, {bairros_unique} bairros únicos")
    print(f"   Média prisões/dia: {len(prison_by_date_bairro) / dates_unique:.1f}")
    
    return prison_by_date_bairro

def load_crime_nodes():
    """Carrega nós de crime do grafo"""
    pkl_file = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
    
    if not os.path.exists(pkl_file):
        print(f"❌ Arquivo não encontrado: {pkl_file}")
        return None, None
    
    print(f"\n[3/5] Carregando grafo de crime...")
    with open(pkl_file, 'rb') as f:
        data_pack = pickle.load(f)
    
    node_features = data_pack['node_features']
    dates = data_pack['dates']
    
    print(f"✅ Grafo carregado:")
    print(f"   Shape: {node_features.shape} (nós, dias, features)")
    print(f"   Período: {dates[0].date()} até {dates[-1].date()}")
    
    return node_features, dates

def test_mapping(prison_data):
    """Testa mapeamento de prisões para bairros do grafo"""
    print(f"\n[4/5] Testando mapeamento prisões → bairros...")
    
    # Carregar nós para ter nomes dos bairros
    bairros_file = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
    if not os.path.exists(bairros_file):
        print(f"⚠️  Arquivo de bairros não encontrado")
        return
    
    with open(bairros_file, 'r', encoding='utf-8') as f:
        bairros_data = json.load(f)
    
    node_names = set()
    for feature in bairros_data.get('features', []):
        props = feature.get('properties', {})
        name = props.get('name', '').strip().upper()
        if name:
            node_names.add(name)
    
    print(f"   Grafo tem {len(node_names)} bairros únicos")
    
    # Verificar intersecção
    prison_bairros = set()
    for record in prison_data:
        bairro = record.get('BairroOcor')
        if bairro and isinstance(bairro, str):
            bairro = bairro.strip().upper()
            if bairro:
                prison_bairros.add(bairro)
    
    print(f"   Prisões têm {len(prison_bairros)} bairros únicos")
    
    intersection = prison_bairros & node_names
    print(f"   ✅ Intersecção: {len(intersection)} bairros mapeados ({len(intersection)/max(1, len(node_names))*100:.1f}%)")
    
    # Top 10 bairros com mais prisões
    print(f"\n   Top 10 bairros com mais prisões:")
    prison_counts = {}
    for record in prison_data:
        bairro = record.get('BairroOcor')
        if bairro and isinstance(bairro, str):
            bairro = bairro.strip().upper()
            if bairro:
                prison_counts[bairro] = prison_counts.get(bairro, 0) + 1
    
    for bairro, count in sorted(prison_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        mapped = "✅" if bairro in intersection else "❌"
        print(f"      {mapped} {bairro}: {count} prisões")

def correlation_analysis(prison_data, node_features, dates):
    """Analisa correlação entre prisões e crimes"""
    print(f"\n[5/5] Analisando correlação prisões ↔ crimes...")
    
    # Agregação
    prison_by_date_bairro = {}
    for record in prison_data:
        try:
            date_str = record.get('Data', '')
            bairro = record.get('BairroOcor', '').strip().upper()
            
            if not date_str or not bairro:
                continue
            
            date_obj = pd.to_datetime(date_str)
            date_key = date_obj.strftime('%Y-%m-%d')
            
            key = (date_key, bairro)
            prison_by_date_bairro[key] = prison_by_date_bairro.get(key, 0) + 1
        except:
            continue
    
    # Por dia (agregado em Fortaleza)
    prison_by_date = {}
    for (date_key, bairro), count in prison_by_date_bairro.items():
        prison_by_date[date_key] = prison_by_date.get(date_key, 0) + count
    
    # Crimes por dia (agregado)
    crimes_by_date = {}
    for date_idx, date_obj in enumerate(dates):
        date_key = date_obj.strftime('%Y-%m-%d')
        crimes_by_date[date_key] = node_features[:, date_idx, 0].sum()  # CVLI
    
    # Intersecção de datas
    common_dates = sorted(set(prison_by_date.keys()) & set(crimes_by_date.keys()))
    print(f"   Datas com ambos dados: {len(common_dates)}")
    
    if len(common_dates) < 30:
        print(f"   ⚠️  Poucas datas com ambos dados, análise limitada")
        return
    
    prison_vals = np.array([prison_by_date[d] for d in common_dates])
    crime_vals = np.array([crimes_by_date[d] for d in common_dates])
    
    # Correlação sem lag
    corr_0 = pearsonr(prison_vals, crime_vals)[0]
    print(f"   Correlação (lag=0 dias): {corr_0:.4f}")
    
    # Correlação com lag (prisões preveem crimes 3 dias depois)
    if len(common_dates) > 3:
        corr_3 = pearsonr(prison_vals[:-3], crime_vals[3:])[0]
        print(f"   Correlação (lag=3 dias): {corr_3:.4f}")
    
    # Conclusão
    print(f"\n💡 CONCLUSÃO:")
    if abs(corr_0) > 0.3 or (len(common_dates) > 3 and abs(corr_3) > 0.3):
        print(f"   ✅ VIÁVEL! Prisões mostram correlação com crimes")
        print(f"      → Pode ser integrada como canal 8 exógeno")
    else:
        print(f"   ⚠️  Correlação fraca - talvez melhor como lag-feature")

def main():
    print("=" * 80)
    print("TEST: Viabilidade de Feature Exógena - Prisões Policiais")
    print("=" * 80)
    
    # Carregar
    prison_data = load_prison_data()
    if not prison_data:
        return
    
    aggregate_prison_data(prison_data)
    test_mapping(prison_data)
    
    node_features, dates = load_crime_nodes()
    if node_features is not None:
        correlation_analysis(prison_data, node_features, dates)
    
    print("\n" + "=" * 80)
    print("✅ TEST CONCLUÍDO - Sem modificações nos arquivos de treinamento")
    print("=" * 80)

if __name__ == "__main__":
    main()
