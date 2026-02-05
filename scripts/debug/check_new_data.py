"""
Verifica dados novos de 2026 e prepara para merge
"""
import json
import pandas as pd
from datetime import datetime

print("="*60)
print("ANÁLISE DE DADOS NOVOS - 2026")
print("="*60)

# Dados atuais processados
import pickle
with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)
    dates = data.get('dates')
    print(f"\n📊 DADOS ATUAIS NO MODELO:")
    print(f"  - Período: {dates[0]} até {dates[-1]}")
    print(f"  - Total: {len(dates)} dias")

# Arquivo de dados novos
print(f"\n📁 ARQUIVO NOVO: data/raw/dados_status_020226.json")
try:
    with open('data/raw/dados_status_020226.json', 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    # Pular headers do PHPMyAdmin
    records = [r for r in content if isinstance(r, dict) and 'data_fato' in r]
    
    if records:
        df = pd.DataFrame(records)
        df['data_fato'] = pd.to_datetime(df['data_fato'])
        
        print(f"  ✓ {len(records)} registros encontrados")
        print(f"  - Período: {df['data_fato'].min()} até {df['data_fato'].max()}")
        
        # Quantos em 2026?
        df_2026 = df[df['data_fato'] >= '2026-01-01']
        print(f"\n📅 DADOS DE 2026:")
        print(f"  - Total: {len(df_2026)} ocorrências")
        
        if len(df_2026) > 0:
            print(f"\n  Distribuição por mês:")
            monthly = df_2026.groupby(df_2026['data_fato'].dt.to_period('M')).size()
            for month, count in monthly.items():
                print(f"    {month}: {count} ocorrências")
            
            # Últimas ocorrências
            print(f"\n  Últimas 5 ocorrências:")
            recent = df_2026.nlargest(5, 'data_fato')[['data_fato', 'natureza_principal', 'bairro']].values
            for r in recent:
                print(f"    {r[0]} - {r[1]} - {r[2]}")
    else:
        print("  ⚠️ Nenhum registro com data_fato encontrado")
        print(f"\n  Estrutura do arquivo:")
        print(f"  {content[:5]}")
        
except Exception as e:
    print(f"  ❌ Erro ao ler arquivo: {e}")
    import traceback
    traceback.print_exc()

# Verificar dados CSV existentes
print(f"\n📋 DADOS CSV EXISTENTES:")
try:
    df_csv = pd.read_csv('outputs/occurrences_with_bairro_geo.csv')
    df_csv['data_fato'] = pd.to_datetime(df_csv['data_fato'])
    print(f"  - Total: {len(df_csv)} registros")
    print(f"  - Período: {df_csv['data_fato'].min()} até {df_csv['data_fato'].max()}")
    
    df_csv_2026 = df_csv[df_csv['data_fato'] >= '2026-01-01']
    print(f"  - Dados de 2026: {len(df_csv_2026)} registros")
except Exception as e:
    print(f"  ❌ Erro: {e}")

print(f"\n{'='*60}")
print(f"🔍 DIAGNÓSTICO:")
print(f"{'='*60}")
print(f"\n⚠️ PROBLEMA IDENTIFICADO:")
print(f"  Hoje é {datetime.now().strftime('%Y-%m-%d')}")
print(f"  Modelo usando dados até {dates[-1]}")
print(f"  Gap de {(datetime.now() - datetime.strptime(str(dates[-1]), '%Y-%m-%d %H:%M:%S')).days} dias!")

print(f"\n💡 RECOMENDAÇÃO:")
print(f"  ✓ SIM, deve mergear dados de 2026")
print(f"  ✓ SIM, deve retreinar o modelo")
print(f"  ✓ Caso contrário, predições continuarão defasadas")

print(f"\n🔧 PRÓXIMOS PASSOS:")
print(f"  1. Verificar se dados novos estão formatados corretamente")
print(f"  2. Executar: python scripts/merge_and_retrain.py")
print(f"  3. Aguardar retreinamento (~3-5 minutos)")
print(f"  4. Reiniciar aplicação")
