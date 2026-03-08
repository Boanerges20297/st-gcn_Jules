import pandas as pd
import os

path_csv = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\analise_cvli_fortaleza_completa.csv'
path_xlsx = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\TABELA_ANALISE_CVLI_FORTALEZA.xlsx'

try:
    # A linha que contém "Bairro" é a linha 5 (índice 4)
    # Ao pular 4 linhas, a linha 5 torna-se o cabeçalho
    df = pd.read_csv(path_csv, skiprows=4, encoding='utf-8')
    
    # Limpar nomes de colunas que podem vir com erros de encoding do CSV original
    # (Tratando caracteres especiais como ç e ã)
    df.columns = [
        'Bairro', 'Facção Predominante', 'Total Geral CVLI', 'Dias com Ocorrência',
        '% do Período Total', 'Periodicidade (1 a cada X dias)', 
        'Total 2022', 'Total 2023', 'Total 2024', 'Total 2025', 'Total 2026', 'Projeção 2026'
    ]
    
    # Criar Excel formatado
    with pd.ExcelWriter(path_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Análise CVLI Fortaleza')
        
        # Ajustar largura das colunas
        worksheet = writer.sheets['Análise CVLI Fortaleza']
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 4
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)

    print(f"Excel atualizado com sucesso com TODAS as colunas: {path_xlsx}")
except Exception as e:
    print(f"Erro ao exportar Excel: {e}")
