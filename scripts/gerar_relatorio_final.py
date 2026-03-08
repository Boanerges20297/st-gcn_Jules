import pandas as pd
import os

path_csv = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\analise_cvli_fortaleza_completa.csv'
report_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\RELATORIO_CVLI_FORTALEZA_FINAL.md'

try:
    # Carregar dados (pular as linhas de metadados iniciais)
    df = pd.read_csv(path_csv, skiprows=3, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    
    # Criar Cabeçalho do Relatório
    header = """# Relatório de Análise CVLI - Fortaleza (Revisão Final 2026)
**Período:** 01/01/2022 a 27/02/2026
**Mapeamento Territorial:** KML ORCRIMS 2026 (Estado do Ceará)

---

## 1. Top 40 Bairros Críticos e Domínio Territorial

| Bairro | Facção Predominante | Total CVLI | Periodicidade | Projeção 2026 |
| :--- | :---: | :---: | :---: | :---: |
"
    
    table_rows = ""
    for i, row in df.head(40).iterrows():
        table_rows += f"| {row['Bairro']} | **{row['Facção Predominante']}** | {row['Total Geral CVLI']} | {row['Periodicidade (1 a cada X dias)']} | {row['Projeção 2026']} |\n"

    footer = """
---

## 2. Análise de Destaques e Tendências

### 2.1. Hegemonia Territorial
O **Comando Vermelho (CV)** consolidou seu mapeamento na maioria absoluta dos bairros de alta incidência. Bairros como **Barra do Ceará**, **Barroso** e **Granja Lisboa** apresentam domínio estável conforme os registros territoriais de 2026.

### 2.2. Áreas em Alerta Máximo (2026)
- **Lagoa Redonda:** Apresenta a maior projeção de crescimento do ano (53.1), indicando um cenário de guerra ativa ou instabilidade aguda.
- **Prefeito José Walter:** Mantém uma frequência alarmante, com projeção de aproximadamente 20 mortes para o período parcial atual (19.9 ajustado).
- **Barroso:** Segue com alta periculosidade, registrando projeção de 26.5 para o fechamento deste ano.

### 2.3. Reduções Notáveis
Bairros como **Cristo Redentor** e **Bom Jardim** mostram uma tendência de queda em 2026 no recorte parcial, indicando mudanças na dinâmica local.

---
## 3. Arquivos de Saída
- **Excel Detalhado:** `data/raw/TABELA_ANALISE_CVLI_FORTALEZA.xlsx`
- **Gráficos Analíticos:** `outputs/analises/cvli/`
- **Dataset de Apoio:** `data/raw/analise_cvli_fortaleza_completa.csv`

*Relatório consolidado para o sistema de monitoramento Report Preview.*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(header + table_rows + footer)

    print(f"Relatório Markdown final gerado em: {report_path}")

except Exception as e:
    print(f"Erro ao gerar relatório MD: {e}")
