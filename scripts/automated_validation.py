import os
import sys
import pandas as pd
import numpy as np
import json
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime

# Garantir que a raiz do projeto esteja no path
sys.path.append(os.getcwd())

from src.core.orchestrator import StateOrchestrator, normalize_name

def run_validation(new_records_df):
    """
    Valida o desempenho dos modelos atuais contra os novos registros (Gabarito).
    Calcula P@10 e P@20 regionalizado.
    """
    print(f"--- Iniciando Validação de Desempenho (Shadow Validation) ---")
    
    if new_records_df.empty:
        print("⚠ Sem registros novos para validar.")
        return

    # 1. Inicializar Orquestrador
    try:
        orch = StateOrchestrator('.')
    except Exception as e:
        print(f"❌ Erro ao inicializar Orquestrador para validação: {e}")
        return

    # 2. Obter Predições Atuais (Snapshot de Risco)
    risk_results = orch.get_combined_risk()

    # 3. Filtrar apenas CVLIs dos dados novos (Canal Principal de Validação)
    # Tenta várias colunas possíveis (tipo, nature, tipo_evento)
    new_cvlis = pd.DataFrame()
    for col in ['tipo', 'nature', 'tipo_evento']:
        if col in new_records_df.columns:
            mask = new_records_df[col].astype(str).str.contains('CVLI|HOMICIDIO', na=False, case=False)
            new_cvlis = new_records_df[mask].copy()
            if not new_cvlis.empty: break
    
    if new_cvlis.empty:
        print("ℹ Nenhum CVLI encontrado nos novos dados para cálculo de P@K.")
        return

    # 4. Cálculo de Precisão Regionalizada
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_new_cvlis': len(new_cvlis),
        'regions': {}
    }

    # Separar resultados por região
    region_results = {}
    for node_name, score in risk_results.items():
        # Descobrir a região do nó
        owner_region = orch._node_owners.get(normalize_name(node_name), 'interior')
        if owner_region not in region_results:
            region_results[owner_region] = []
        region_results[owner_region].append((node_name, score))

    for region in ['fortaleza', 'rmf', 'interior']:
        if region not in region_results: continue
        
        # Ordenar Top nós da região
        sorted_nodes = sorted(region_results[region], key=lambda x: x[1], reverse=True)
        top10 = [normalize_name(n[0]) for n in sorted_nodes[:10]]
        top20 = [normalize_name(n[0]) for n in sorted_nodes[:20]]
        
        # Filtrar CVLIs desta região (assumindo que o DF novo tem coluna bairro/municipio)
        # Para simplificar, vamos ver se o bairro do crime está no top 10/20
        reg_cvlis = []
        if 'bairro' in new_cvlis.columns:
             # Nota: Isso assume que o bairro no CVLI bate com o nome do nó no modelo
             # O merge_new_data já faz o find_closest_bairro, então deve bater.
             for _, row in new_cvlis.iterrows():
                 b_norm = normalize_name(str(row['bairro']))
                 # Verificação grosseira de região (precisaria de PIP real para ser perfeito)
                 # Mas se o bairro pertence a essa região no orquestrador, validamos aqui
                 if orch._node_owners.get(b_norm) == region:
                     reg_cvlis.append(b_norm)
        
        if not reg_cvlis:
            continue
            
        hits10 = sum(1 for b in reg_cvlis if b in top10)
        hits20 = sum(1 for b in reg_cvlis if b in top20)
        
        p10 = (hits10 / len(reg_cvlis)) * 100
        p20 = (hits20 / len(reg_cvlis)) * 100
        
        report['regions'][region] = {
            'cvlis_count': len(reg_cvlis),
            'hits_p10': hits10,
            'hits_p20': hits20,
            'p10': round(p10, 2),
            'p20': round(p20, 2)
        }
        print(f"📊 [{region.upper()}] P@10: {p10:.1f}% | P@20: {p20:.1f}% ({hits10}/{len(reg_cvlis)} acertos)")

    # 5. Salvar Log no VALIDATION_LOG.md
    save_to_markdown(report)
    return report

def save_to_markdown(report):
    log_path = 'VALIDATION_LOG.md'
    exists = os.path.exists(log_path)
    
    with open(log_path, 'a', encoding='utf-8') as f:
        if not exists:
            f.write("# 🛡️ Log de Validação Automatizada - Sentinela Tactical\n\n")
            f.write("Este arquivo registra o desempenho shadow dos modelos contra novos dados inseridos.\n\n")
        
        f.write(f"## 🕒 Validação: {report['timestamp']}\n\n")
        f.write(f"**Total de Novos CVLIs Analisados:** {report['total_new_cvlis']}\n\n")
        
        f.write("| Região | Ocorrências | Acertos P@10 | Acertos P@20 | P@10 (%) | P@20 (%) |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for reg, stats in report['regions'].items():
            f.write(f"| **{reg.upper()}** | {stats['cvlis_count']} | {stats['hits_p10']} | {stats['hits_p20']} | **{stats['p10']}%** | **{stats['p20']}%** |\n")
        
        f.write("\n---\n\n")

if __name__ == "__main__":
    # Teste rápido se rodado diretamente (usando o CSV oficial como exemplo de dados "novos")
    OFFICIAL_CSV = os.path.join('data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
    if os.path.exists(OFFICIAL_CSV):
        df = pd.read_csv(OFFICIAL_CSV).tail(50) # Simula as últimas 50
        run_validation(df)
