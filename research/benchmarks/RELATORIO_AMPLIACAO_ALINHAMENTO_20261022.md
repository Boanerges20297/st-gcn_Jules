# Experimentos de amplia├º├úo territorial e alinhamento de horizontes ÔÇö Report Preview original
**Data:** 22/10/2026. **Ambiente:** STI01, malha Fortaleza, apenas CVLI. **Status:** experimento isolado, sem altera├º├úo de produ├º├úo, deploy ou merge. N├úo interpretar estes resultados como valida├º├úo para promo├º├úo autom├ítica.
## Sum├írio executivo
1. **A integridade do modelo atual foi preservada.** Foi reconstru├¡do o tensor oficial do Top-40 em `research/benchmarks/exp_data/top40_rebuild/`. Nomes e ordem dos 40 bairros, `node_features`, `adj_geo` e `adj_conflict` s├úo id├¬nticos aos artefatos ativos (m├íxima diferen├ºa 0).
2. **Top-60 fixo** (40 atuais + 20 seguintes no ranking hist├│rico at├® 2025): m├®dia de **97,1%** do CVLI futuro observado nos 121 bairros oficiais, ante 70,9% no Top-40 fixo; em 12 janelas de 30 dias, 300/311 contagens acumuladas em janelas sobrepostas versus 223/311.
3. **Top-40 m├│vel** (├║ltimos 730 dias anteriores a cada corte): m├®dia de **79,7%** de cobertura; alcan├ºou **55,0% de P@10** com ST-GAT 30% / Poisson 70%, superior ao Top-60 fixo em P@10 (53,3% com pesos 20/80), mas com cobertura total inferior.
4. Com horizontes de **30 dias alinhados**, **Top-60 fixo 20/80** obteve P@10 de 53,3% e Captura@20 de 64,7% (ambas com denominador de eventos dos 121 bairros). **Top-60 fixo 40/60** obteve P@10 de 52,5% e Captura@20 de 65,7%.
5. ST-GAT puro n├úo venceu o h├¡brido. **Mais bairros n├úo s├úo automaticamente melhor ranking:** 121 bairros a 40/60 tiveram P@10 de 50,8% e Captura@20 de 62,0%.
## Detalhes da execu├º├úo
- **Original:** Poisson treinado at├® 31/12/2025 para horizonte de **14 dias**; ST-GAT v5 (DeepSTGAT_v5) treinado at├® 30/06/2026 para horizonte de **30 dias**, com valida├º├úo do checkpoint abrangendo 01/07ÔÇô29/08/2026.
- **Experimental:** novo Poisson 30 dias treinado sem alterar o oficial, com dados de 2022ÔÇô2025 dos 121 bairros, amostragem temporal a cada 14 dias e 11.616 amostras; 162.382 registros candidatos de treinamento antes da subamostragem. Mesmo Poisson experimental usado nas quatro malhas para compara├º├úo.
- ST-GAT v5 reaplicado **zero-shot** ├ás malhas 40, 60 e 121. Os pesos e a arquitetura do checkpoint original **n├úo foram retreinados**, apesar da mudan├ºa de topologia do grafo; consequentemente, os resultados de expans├úo s├úo apenas provas explorat├│rias de transfer├¬ncia, n├úo checkpoints especializados validados.
- O blend usa percentis, nos pesos ST-GAT `0%,20%,30%,40%,50%,70%,100%`, e reatribui valores da distribui├º├úo Poisson ap├│s reordenar, como em `src/core/orchestrator.py`. **Scores n├úo s├úo probabilidades calibradas.**
- Foram usadas **12 datas de corte semanais**, de 01/07 a 16/09/2026, cada uma prevendo os pr├│ximos 30 dias. As janelas se sobrep├Áem. A janela de setembro (3 cortes) n├úo se sobrep├Áe ao per├¡odo de valida├º├úo at├® agosto informado pelo checkpoint, mas ainda ├® amostra reduzida.
- Todas as m├®tricas de **Captura@10 e Captura@20** t├¬m denominador comum: CVLIs futuros nos **121 bairros**, n├úo somente dentro de cada malha. `P@10` ├® a propor├º├úo dos 10 priorizados com pelo menos uma ocorr├¬ncia futura. **Cobertura do pool** ├® a fra├º├úo do CVLI futuro que ocorreu em bairros inclu├¡dos entre os candidatos, independentemente da classifica├º├úo dada pelo modelo.
- Sele├º├úo Top-40 ativo usa 2024ÔÇô2025, com cutoff fixo em 31/12/2025. Rolling usa s├│ os 730 dias anteriores ao corte (sem sele├º├úo com dados posteriores). Aumentar a malha conservou os 40 nomes originais e acrescentou 20 de maior CVLI hist├│rico segundo a sele├º├úo oficial.
## Cobertura territorial ÔÇö janelas de 30 dias
| Sele├º├úo de bairros | Cobertura m├®dia | Eventos dentro / eventos totais agregados por janela |
|---|---:|---:|
| Top-40 fixo | 70,9% | 223/311 |
| Top-40 m├│vel (730d) | 79,7% | 253/311 |
| Top-60 m├│vel (730d) | 88,5% | 278/311 |
| Top-60 fixo (40 + 20 hist├│ricos) | **97,1%** | **300/311** |
| 121 oficiais | 100% | 311/311 |
As somas de 311 incluem duplica├º├úo temporal porque 12 janelas de 30 dias se sobrep├Áem.
### Meses independentes (sem sobreposi├º├úo temporal)
| M├¬s de 2026 | 121 bairros | Top-40 fixo | Top-60 fixo |
|---|---:|---:|---:|
| Julho | 37 | 25 | 33 |
| Agosto | 25 | 19 | 25 |
| Setembro | 20 | 14 | 20 |
| Outubro (at├® 21/10) | 11 | 4 | 10 |
A diferen├ºa n├úo representa necessariamente falha de ingest├úo: ├® consequ├¬ncia sobretudo da decis├úo deliberada de manter uma malha seletiva. Dados brutos e tensores podem diferir em outros meses devido a filtros t├íticos e reconcilia├º├úo.
## Compara├º├úo de previs├úo ÔÇö horizontes de 30 dias alinhados, 12 cortes
| Malha | ST-GAT / Poisson | P@10 | Captura@20 | Captura@10 |
|---|---|---:|---:|---:|
| 40 fixo | 20/80 | 49,2% | 59,3% | 42,0% |
| 40 fixo | 40/60 | 50,8% | 58,9% | 43,0% |
| 40 m├│vel | 30/70 | **55,0%** | 66,3% | **48,2%** |
| 60 fixo | 20/80 | 53,3% | 64,7% | 47,1% |
| 60 fixo | 40/60 | 52,5% | 65,7% | 45,8% |
| 60 m├│vel | 30/70 | 53,3% | 62,5% | 46,7% |
| 121 | 40/60 | 50,8% | 62,0% | 45,0% |
| 60 fixo | 100/0 | 45,8% | 60,6% | 42,9% |
**Observa├º├úo:** em 40 m├│vel, a escolha dos candidatos varia por corte e o grafo da rede tamb├®m muda, mas os par├ómetros ST-GAT foram transferidos sem retreino.
## Controles de 30 dias sem sobreposi├º├úo ÔÇö dois pain├®is, tr├¬s cortes por painel
Painel A = 01/07, 05/08, 09/09; painel B = 08/07, 12/08, 16/09.
| Modelo | P@10 A | P@10 B | Captura@20 A | Captura@20 B |
|---|---:|---:|---:|---:|
| Top-40 fixo 20/80 | 50,0% | 50,0% | 58,6% | 61,3% |
| Top-40 m├│vel 30/70 | **56,7%** | **53,3%** | **67,6%** | **67,3%** |
| Top-60 fixo 20/80 | 53,3% | 50,0% | 64,6% | 64,0% |
| Top-60 fixo 40/60 | 53,3% | 50,0% | 67,2% | 66,4% |
| 121 40/60 | 53,3% | 53,3% | 62,4% | 60,6% |
Esses pain├®is fornecem uma verifica├º├úo de consist├¬ncia, **n├úo** signific├óncia estat├¡stica: cada um cont├®m apenas tr├¬s per├¡odos.
## Ressalvas e decis├úo segura
1. O ST-GAT foi **transferido** para novas quantidades e configura├º├Áes de n├│s, sem retreino espec├¡fico. Para uma candidatura real, retreinar o ST-GAT v5 para Top-60 e 30 dias, com vers├Áes de data congeladas e valida├º├úo independente.
2. Mesmo com o Poisson experimental treinado em 30 dias, os modelos t├¬m **janelas de treinamento diferentes**: Poisson at├® dezembro/2025; ST-GAT at├® junho/2026. N├úo h├í paridade de informa├º├úo hist├│rica entre treinos.
3. A geometria e `tension_index` s├úo do snapshot mais recente; pode haver vazamento de vari├íveis est├íticas no backtest. Refazer backtests com snapshots verdadeiramente hist├│ricos antes de qualquer promo├º├úo.
4. A cobertura mensal confirma que 60 bairros ret├¬m mais eventos do que 40, mas n├úo prova que todos os 20 adicionais devam receber patrulhamento proporcional.
5. **Proposta:** manter produ├º├úo intacta em 40/20-80. Pr├│xima fase em shadow: (a) Top-40 m├│vel 30/70 para prioridade operacional de alta concentra├º├úo; (b) Top-60 fixo 20/80 ou 40/60 para cobertura territorial ampliada, com m├®tricas de custo e quantidade de alarmes. Somente reconsiderar produ├º├úo ap├│s backtest e valida├º├úo prospectiva.
## Artefatos
Na pasta `research/benchmarks/`:
- `expand_align_30d_20261022.py`: treino experimental Poisson 30d, infer├¬ncia ST-GAT v5 transferida e benchmark de malhas est├íticas; gera CSVs e JSON com limita├º├Áes.
- `rolling_alignment_30d_20261022.py`: benchmark de sele├º├úo Top-40 e Top-60 m├│veis.
- `nonoverlap_monthly_20261022.py`: pain├®is sem sobreposi├º├úo e cobertura mensal por malha.
- `exp_models/poisson30_all121_stride14.pkl`: modelo Poisson experimental isolado (n├úo promovido).
- `exp_data/top40_rebuild/` e `exp_data/all_official/`: tensores reconstru├¡dos exclusivamente para os testes.
- `expand_align_30d_*_20261022.csv`, `rolling_risk_*_20261022.csv`, `pool_coverage_dynamic_20261022.csv`, `nonoverlap_30d_20261022.csv`, `monthly_coverage_20261022.csv`: resultados reprodut├¡veis.
**Prote├º├úo de produ├º├úo:** scripts criados somente em `research/benchmarks/` e tensores somente na subpasta `exp_data/`. `src/`, `data/processed/` original e `models/active/` n├úo foram gravados pelos experimentos. Sem commit/push/PR/deploy. Persist├¬ncia can├┤nica remota pendente.
