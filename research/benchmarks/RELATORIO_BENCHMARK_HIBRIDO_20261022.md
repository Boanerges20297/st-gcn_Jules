# Benchmark de pesos Poisson x ST-GAT v5 ÔÇö Report Preview ORIGINAL
**Data da auditoria:** 22/10/2026. **Escopo:** somente CVLI de Fortaleza, nos 40 n├│s de `data/processed/processed_fortaleza.pkl`. **Execu├º├úo:** STI01, c├│pia de trabalho local, modelos/checkpoints ativos, sem retreinar e sem alterar produ├º├úo.
## Implementa├º├úo verificada
- `src/core/orchestrator.py:1245-1275`: `HYBRID_STGAT_WEIGHT=0.20`. O peso combina **percentis de ranking**, n├úo probabilidades; depois, a distribui├º├úo do Poisson ├® reatribu├¡da ├ás novas posi├º├Áes. Logo os n├║meros de score permanecem Poisson mesmo com ST-GAT majorit├írio.
- Poisson ativo: `models/active/production/poisson/fortaleza_poisson_regressor.pkl`, treinado 2022ÔÇô2025 e `horizon_days=14`.
- ST-GAT `DeepSTGAT_v5`: checkpoint `models/active/legacy_torch/fortaleza_model_active.pth`, treino at├® 30/06/2026, `predict_horizon=30`; janela de valida├º├úo informada 01/07ÔÇô29/08/2026.
- Processado em 21/10/2026: 40 bairros e 1.755 dias. Outros bairros de Fortaleza **n├úo** est├úo no conjunto de candidatos.
## Avalia├º├Áes executadas
**Crit├®rio:** scores calculados somente com os dias anteriores ao corte; CVLI observado nos 14 ou 30 dias seguintes (canal 0 do processado). Poisson chamado pelo runtime ativo; ST-GAT pelo mesmo c├│digo de prepara├º├úo, checkpoint e p├│s-processamento usados na valida├º├úo oficial; blend id├¬ntico ao `orchestrator.py`. Recalcularam-se os atributos est├íticos `total_cvli` e `recent_cvli` de cada corte, mas outros atributos do snapshot ainda podem carregar informa├º├úo posterior.
`P@10`: bairros com CVLI entre os dez priorizados / 10. `Captura@20`: CVLIs ocorridos nos vinte priorizados / CVLIs dos 40 bairros. N├úo s├úo probabilidades calibradas.
### Horizonte real 14 dias ÔÇö 15 cortes semanais de 01/07 a 07/10/2026
| ST-GAT / Poisson | P@10 | Captura@20 |
|---|---:|---:|
| 0 / 100 | 25,3% | 81,7% |
| 20 / 80 ÔÇö atual | **28,0%** | 81,7% |
| 30 / 70 | **28,0%** | 85,1% |
| 40 / 60 | **28,0%** | 85,7% |
| 50 / 50 | 27,3% | **86,9%** |
| 60 / 40 | 27,3% | 85,3% |
| 70 / 30 | 26,7% | 81,2% |
| 80 / 20 | 27,3% | 79,9% |
| 100 / 0 | 24,7% | 80,5% |
### Controle sem sobreposi├º├úo ÔÇö 8 cortes de 14 em 14 dias
| ST-GAT / Poisson | P@10 | Captura@20 |
|---|---:|---:|
| 20 / 80 | **25,0%** | 79,1% |
| 30 / 70 | **25,0%** | 83,3% |
| 40 / 60 | **25,0%** | 84,5% |
| 50 / 50 | 23,8% | 85,1% |
| 60 / 40 | 23,8% | 85,1% |
| 70 / 30 | 23,8% | 80,0% |
| 100 / 0 | 21,3% | 78,6% |
**Interpreta├º├úo:** o ganho de 30ÔÇô40% ├® sobretudo na cobertura do Top-20; a precis├úo Top-10 permaneceu igual ao peso atual neste controle. S├úo apenas oito cortes independentes; n├úo h├í evid├¬ncia estat├¡stica suficiente para mudan├ºa autom├ítica.
### Horizonte de refer├¬ncia do ST-GAT, 30 dias ÔÇö 12 cortes semanais
| ST-GAT / Poisson | P@10 | Captura@20 |
|---|---:|---:|
| 20 / 80 | 49,2% | 84,1% |
| 30 / 70 | **50,8%** | 84,5% |
| 40 / 60 | **50,8%** | **84,7%** |
| 50 / 50 | 50,0% | 83,9% |
| 60 / 40 | 49,2% | 82,4% |
| 70 / 30 | 48,3% | 80,2% |
| 100 / 0 | 44,2% | 79,7% |
Neste experimento, 30ÔÇô40% tamb├®m superam marginalmente 20% em P@10. Por├®m o Poisson foi treinado para 14 dias; a compara├º├úo de 30 dias ├® an├ílise de sensibilidade, n├úo condi├º├úo homog├¬nea de treinamento.
## Cobertura operacional e consist├¬ncia
- Contagem de CVLI bruto Fortaleza: **julho 37**, **agosto 25**, **setembro 20**, **outubro 11** (at├® 21/10). Compara├º├úo apenas descritiva: bruto e processado podem diferir por geocodifica├º├úo, agrega├º├úo e regras de contagem.
- Por igualdade de nome de bairro, dos **11** registros brutos de outubro, somente **2** pertencem aos 40 nomes escolhidos. H├í alias/grafia divergente, p.ex. **JANGURURSSU** versus **JANGURUSSU**, portanto o n├║mero n├úo comprova falha no pipeline; demonstra que o universo fixo de 40 bairros e a normaliza├º├úo devem ser auditados.
- A s├®rie processada dos 40 bairros registra 25 CVLI em julho, 19 em agosto, 14 em setembro e 4 em outubro, sem prova de que represente toda Fortaleza.
- O Poisson foi serializado com **scikit-learn 1.8.0**, mas o ambiente usado para reproduzir a infer├¬ncia tem **1.5.2**; houve aviso de incompatibilidade. Os resultados devem ser repetidos em ambiente equivalente ao da produ├º├úo.
- O grafo e `tension_index` s├úo do snapshot atual e podem introduzir vazamento temporal em cortes anteriores; s├│ os CVLIs usados na janela foram truncados. JulhoÔÇôagosto tamb├®m integrou a janela de valida├º├úo do ST-GAT. SetembroÔÇôoutubro tem poucas ocorr├¬ncias; conclus├Áes estat├¡sticas limitadas.
## Decis├úo t├®cnica
**N├úo inverter a propor├º├úo em produ├º├úo agora.** Promover **30% ST-GAT / 70% Poisson** a primeiro *challenger em shadow*, **40/60** como segundo e **50/50** para teste de sensibilidade. Testes de maioria ST-GAT (60ÔÇô100%) n├úo demonstraram superioridade robusta, incluindo o ST-GAT puro. Antes da promo├º├úo, alinhar horizontes 14 vs 30 dias, construir snapshots temporais sem vazamento, controlar cobertura territorial ampla e validar estabilidade por janelas independentes em per├¡odos adicionais. Reconsiderar tamb├®m o modelo de exibi├º├úo de score: mesmo num h├¡brido com ST-GAT majorit├írio, o score exibido continua herdado do Poisson.
## Artefatos reproduz├¡veis
- `hybrid_weight_sweep_20261022.py` ÔÇö 14 dias, stride 7
- `hybrid_weight_sweep_stride14.py` ÔÇö 14 dias, stride 14
- `hybrid_weight_sweep_h30.py` ÔÇö 30 dias, stride 7
- `window_metrics_20261022*.csv` ÔÇö m├®tricas por corte
- `summary_20261022*.csv` ÔÇö agregados
- `top10_20261022*.csv` ÔÇö ranking por corte
- `metadata_20261022*.json` ÔÇö fonte, limita├º├Áes e condi├º├Áes de execu├º├úo
**Estado de integra├º├úo:** apenas arquivos novos em `research/benchmarks/`; nenhum arquivo de modelos, data, c├│digo da produ├º├úo ou branch Git foi alterado por este benchmark. N├úo houve commit, push nem deploy.
