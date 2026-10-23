# CVLI: auditoria SSPDS e integridade de fontes (outubro/2026)
Este diretorio fica no **repositorio unico Report Preview / main**.
## Proveniencia
- Planilha publica SSPDS `CVLI_2009_a_2025.xlsx`, municipio Fortaleza, contagem diaria de linhas de vitimas. Os dados foram lidos no STI de um arquivo historico da SSPDS, inicialmente preservado no projeto experimental CRTF.
- Arquivos `*_pre_fix_*.csv`: comparativos **antes** de corrigir copias de vitimas; sua contagem bruta nao serve para treinar ou avaliar modelos.
- Arquivos `*_post_fix_*.csv`: comparativo da planilha oficial com o **CSV corrigido**.
- Fonte de janeiro a setembro de 2026: extratos mensais SSPDS Status CVLI, com recorte parcial. O mes outubro nao possui comparacao correspondente nessa fonte.
- `mysql_repair_applied_report.json`: trilha do reparo do CSV, sem nomes de pessoas.
- `mysql_repair_idempotence_report.json`: prova de que a segunda execucao nao mudou os registros.
Para atualizar as tabelas de comparacao apos uma nova atualizacao do CSV, a partir da raiz do Report Preview:
```powershell
python research/benchmarks/cvli_sspds_audit_20261022/compare_enriched_to_official_aggregates.py
```
Relatorio geral: `docs/AUDITORIA_CVLI_MYSQL_SSPDS_20261023.md`.
**Nao eliminar os 26 grupos ambiguos por heuristica nem forcar coincidencia estatistica com a SSPDS**. Os benchmarks anteriores a esta auditoria estao invalidados para alegacoes de acuracia; devem ser repetidos com a base corrigida.
