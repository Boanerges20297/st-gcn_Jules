# Auditoria CVLI: MySQL, SSPDS e idempotencia (23/10/2026)
**Repositorio unico:** C:\\Users\\STI01\\Desktop\\Projetos\\Report Preview, branch main. A base can├┤nica para modelagem e o codigo residem AQUI, nao no fork "Report Preview CRTF".
## Causa identificada
A tabela MySQL dados_status tinha 132.217 IDs unicos; o CSV enriquecido tambem tinha os mesmos IDs. Porem diversas linhas MySQL representavam **a mesma vitima CVLI em agencias/AIS distintas**. O merge por ID nao identificava espelhos; o agrupamento historico por apenas geografia/hora poderia apagar vitimas diferentes. A nova rotina usa identidade da vitima *apenas na memoria do STI*, data/hora, municipio, coordenadas e campos compativeis para reunir espelhos comprovados. A coluna id preserva todos os IDs relacionados ("1 | 2"); duas vitimas distintas permanecem em linhas separadas. Casos ambiguos nao sao descartados.
## Reparacao aplicada, sem mudar o MySQL
- Antes: 132.217 linhas; depois: **129.765**; representacoes redundantes de CVLI: **2.452**, formando 2.446 grupos.
- Grupos ambiguos mantidos: **26**. Quantidades CVLI ausentes/zero corrigidas: **199**.
- TODOS os 132.217 IDs originais preservados/rastreaveis. **114.820 registros de outros crimes identicos** antes e depois.
- CSV can├┤nico: data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv.
- SHA256 anterior: 144d3cf0f74809028ec4f8e156620f850705919e27f1f29989a0f1c372e972b2.
- SHA256 apos reparo: 9416f1b709e27739f700daa613cc2219f2ed90362c8b947059b8b683bbc07df1.
- Backup local: data/runtime/research/cvli_mysql_integrity_20261022/backups/; MySQL nunca recebeu UPDATE/DELETE.
- Segunda execucao do reparo: **zero mudancas**. Merge com ID conhecido: **zero mudancas**.
## Conferencia SSPDS independente em Fortaleza
Fonte: arquivo publico SSPDS CVLI 2009-2025, extraido em forma agregada. Comparativos antes/depois e script local de verificacao estao todos em research/benchmarks/cvli_sspds_audit_20261022/.
| Ano | SSPDS | CSV antigo | CSV corrigido | Dias com contagem identica |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 850 | 850 | 850 | 321/321 |
| 2023 | 738 | 1476 | 739 | 313/314 |
| 2024 | 834 | 1668 | 848 | 305/317 |
| 2025 | 743 | 1491 | 745 | 308/310 |
Diferencas restantes (+1, +14, +2 de 2023 a 2025) exigem auditoria de fonte e classificacao, nao exclusao automatica. 2026 contem apenas dados parciais, com distintas coberturas entre SSPDS Status e CSV.
## Arquivos e controle de regressao
- src/core/cvli_source_canonical.py: identificar aliases de uma vitima; derivar indicador de multiplas vitimas por evento sem duplicar mortes.
- scripts/repair_enriched_cvli_from_mysql.py: conciliacao fonte MySQL, backup, troca atomica, verificacao SHA256; dry-run como padrao.
- scripts/merge_new_data.py: remove append direto, filtra TODOS os IDs (inclusive aliases), concilia antes de gravar, sem regravacao quando lote vazio.
- scripts/calculate_multi_deaths_historical.py: antigo agrupamento geometrico perigoso substituido por reconciliacao verificada.
- src/core/data_processing.py: contabiliza vitimas e identifica multiplas vitimas antes do filtro tatico.
- tests/test_cvli_mysql_canonical.py: **11 testes de cenarios** aprovados em execucao direta; sintaxe e imports Python aprovados.
- **Sem alteracao** dos arquivos de pesos de rede neural, deploy e MySQL; sem tocar VPN, caches do projeto CRTF ou producao.
## Tensores isolados (na mesma pasta de trabalho)
data/runtime/research/cvli_rebuilt_20261022/processed/ e .../all_official/. Rebuild top40 e 121 bairros OK. Malha top40 mudou 1 bairro e a ordem dos nos. Tensor all_official de 121 bairros, canal CVLI: 2022=840; 2023=733; 2024=840; 2025=743; 2026 ate 21/10=230. A diferenca para CSV por ano inclui filtros e mapeamento territorial. **Nao reutilizar pesos antigos sobre tensor novo**.
## Importante para pesquisa
Os benchmarks de ST-GAT/Poisson calculados antes de corrigir as duplicacoes sao **historicos e nao resultados validados na base correta**. Manter resultados por rastreabilidade, mas retreinar/reevaluar modelos e baselines de 30/60/90/120 dias para conclusoes da disserta├º├úo. Nenhum resultado antigo constitui validacao dessa correcao.
Os CSV agregados da SSPDS foram trazidos da pasta CRTF para ESTE repositorio. Os dados de runtime e backups ficam apenas locais, na mesma raiz de trabalho, e nao devem ser publicados junto com codigo ou benchmarks.
