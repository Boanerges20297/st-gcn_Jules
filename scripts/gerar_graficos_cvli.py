import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Configurações de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

path_csv = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\analise_cvli_fortaleza_completa.csv'
output_dir = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\outputs\analises\cvli'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import re

# Função para encontrar a linha do cabeçalho
def find_header_row(path):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 'Bairro' in line:
                return i
    return 5 # Fallback

try:
    header_row = find_header_row(path_csv)
    print(f"Cabeçalho encontrado na linha: {header_row}")
    
    # 1. Carregar dados
    df = pd.read_csv(path_csv, skiprows=header_row, encoding='utf-8')
    
    # Limpar nomes de colunas
    df.columns = [c.strip() for c in df.columns]
    print(f"Colunas detectadas: {list(df.columns)}")
    
    # Identificar colunas corretas dinamicamente
    col_total = [c for c in df.columns if 'Total Geral' in c][0]
    col_faccao = [c for c in df.columns if 'Facção' in c][0]
    year_cols = [c for c in df.columns if 'Total 20' in c and ('Bruto' in c or 'Total 20' in c) and 'Parcial' not in c and 'Projeção' not in c]
    years = [re.search(r'20\d{2}', c).group() for c in year_cols]
    
    # --- GRÁFICO 1: Evolução Temporal dos Top 10 Bairros ---
    top_10 = df.nlargest(10, col_total)
    df_melted = top_10.melt(id_vars=['Bairro'], value_vars=year_cols, var_name='Ano_Col', value_name='CVLI')
    df_melted['Ano'] = df_melted['Ano_Col'].apply(lambda x: re.search(r'20\d{2}', x).group())
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_melted, x='Ano', y='CVLI', hue='Bairro', marker='o', linewidth=2.5)
    plt.title('Evolução Temporal de CVLI - Top 10 Bairros Críticos', fontsize=15)
    plt.ylabel('Contagem de Mortes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_evolucao_temporal_top10.png'), dpi=300)
    plt.close()

    # --- GRÁFICO 2: Distribuição de CVLI por Facção (Outliers e Variabilidade) ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=col_faccao, y=col_total, palette='viridis')
    sns.stripplot(data=df, x=col_faccao, y=col_total, color=".3", alpha=0.5)
    plt.title('Distribuição de CVLI por Facção - Identificação de Outliers', fontsize=15)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, '02_distribuicao_faccao_outliers.png'), dpi=300)
    plt.close()

    # --- GRÁFICO 3: Heatmap de Intensidade Criminal (Top 30 Bairros x Anos) ---
    top_30 = df.nlargest(30, col_total)
    heatmap_data = top_30.set_index('Bairro')[year_cols]
    heatmap_data.columns = years
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='g', cbar_kws={'label': 'Nº de Ocorrências'})
    plt.title('Heatmap de Intensidade Criminal (Anual)', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_heatmap_intensidade.png'), dpi=300)
    plt.close()

    # --- GRÁFICO 4: Análise de Anomalias (Crescimento 2024 vs 2025) ---
    try:
        col_2024 = [c for c in year_cols if '2024' in c][0]
        col_2025 = [c for c in year_cols if '2025' in c][0]
        
        df['Crescimento_24_25'] = df[col_2025] - df[col_2024]
        anomalias = df.sort_values(by='Crescimento_24_25', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=anomalias, x='Crescimento_24_25', y='Bairro', palette='Reds_r')
        plt.title('Maiores Saltos de Violência (Aumento Bruto 2024 -> 2025)', fontsize=15)
        plt.xlabel('Diferença de Mortes (2025 - 2024)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_anomalias_crescimento.png'), dpi=300)
        plt.close()
    except:
        print("Aviso: Falha ao gerar gráfico de anomalias (colunas 2024/2025 não identificadas).")

    # --- GRÁFICO 5: Correlação Periodicidade x Total Geral ---
    try:
        col_per = [c for c in df.columns if 'Periodicidade' in c][0]
        df['Periodicidade_Num'] = df[col_per].str.extract('(\d+\.?\d*)').astype(float)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=col_total, y='Periodicidade_Num', hue=col_faccao, size=col_total, sizes=(20, 200), alpha=0.7)
        plt.title('Correlação: Volume Total vs Periodicidade', fontsize=15)
        plt.ylabel('1 Crime a cada X dias (Menor é mais frequente)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_correlacao_periodicidade.png'), dpi=300)
        plt.close()
    except:
        print("Aviso: Falha ao gerar gráfico de correlação periodicidade.")

    print(f"Análise gráfica concluída. Gráficos salvos em: {output_dir}")

except Exception as e:
    print(f"Erro ao gerar gráficos: {e}")
    import traceback
    traceback.print_exc()
