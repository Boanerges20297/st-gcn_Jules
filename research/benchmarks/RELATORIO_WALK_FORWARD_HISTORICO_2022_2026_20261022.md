> **ATENCAO (23/10/2026):** Este benchmark foi executado antes da consolidacao de CVLI por vitima. A auditoria MySQL/SSPDS identificou 2.452 representacoes redundantes na base enriquecida. Numeros e conclusoes deste relatorio sao **historicos, nao validacao na base corrigida**. Ver `docs/AUDITORIA_CVLI_MYSQL_SSPDS_20261023.md`.

﻿# Report Preview ÔÇö Walk-forward hist├│rico CVLI 2022ÔÇô2026 (controle temporal)
**Data da execu├º├úo:** 22/10/2026 ┬À **STI01/NEXO** ┬À **Status:** experimentos conclu├¡dos; N├âO promovido ├á produ├º├úo.
## Por que este experimento?
Aguardar 30 dias para a previs├úo selada de outubro/2026 ├® ├║til como prova prospectiva adicional, mas n├úo substitui a avalia├º├úo poss├¡vel **agora** com CVLI desde 2022. Retreinamos ST-GAT e Poisson conforme o momento hist├│rico para n├úo comparar resultados de 2024 com checkpoints treinados at├® junho de 2026.
## Base e consist├¬ncia
- Fonte analisada: `research/benchmarks/exp_data/all_official/processed_fortaleza.pkl`: **121 bairros**, datas di├írias de 01/01/2022 a 21/10/2026, **5.820 unidades de CVLI agregadas no canal-alvo**.
- Contagens anuais nesse tensor: **2022=840, 2023=1.462, 2024=1.653, 2025=1.487, 2026=378 (at├® 21/10)**.
- O CSV bruto `data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`, filtrando cidade Fortaleza e tipo `cvli`, cont├®m **381 linhas em 2026**. A diferen├ºa para o tensor ├® pequena; a queda de 2026 **j├í est├í na origem**, n├úo ├® perda maci├ºa da convers├úo para bairros.
- **Alerta de comparabilidade:** janeiroÔÇôsetembro de 2025 somam cerca de 1.176 CVLIs no tensor; janeiroÔÇôsetembro de 2026, **367**. N├úo demonstramos se a origem de 2026 est├í incompleta, alterou o escopo ou se houve redu├º├úo efetiva ÔÇö at├® esclarecer, 2026 ├® somente sensibilidade.
## Protocolo temporal executado
- Origem 2024: treinamento com alvos terminando at├® **31/12/2023**; escolha da melhor ├®poca em janeiroÔÇômar├ºo/2024; testes **abrilÔÇôdezembro/2024**, nove per├¡odos independentes de 30 dias.
- Origem 2025: treinamento com alvos terminando at├® **31/12/2024**; valida├º├úo em janeiroÔÇômar├ºo/2025; testes **abrilÔÇôdezembro/2025**, nove per├¡odos de 30 dias.
- Origem 2026: treinamento com alvos terminando at├® **31/12/2025**; valida├º├úo em janeiroÔÇômar├ºo/2026; testes **abrilÔÇôsetembro/2026**, seis per├¡odos de 30 dias.
- Cada corte de teste inicia no dia 1┬║ do m├¬s e inclui os **30 dias subsequentes**. Eventuais dias 31 n├úo entram nos per├¡odos avaliados. Janelas de teste **n├úo se sobrep├Áem**; h├í depend├¬ncia temporal residual entre meses.
- ST-GAT v5 **reiniciado com pesos aleat├│rios em cada origem** e treinado na malha m├│vel de 60 bairros calculada pela soma de CVLI dos **730 dias anteriores** ao in├¡cio do fold. A melhor ├®poca foi escolhida **somente pelos tr├¬s cortes de valida├º├úo anteriores aos testes**. Melhores ├®pocas: **2024=1, 2025=2, 2026=6**.
- Na predi├º├úo de cada m├¬s, as malhas m├│veis Top-40 e Top-60 s├úo **recalculadas usando apenas os 730 dias anteriores ├á data da previs├úo**. O mesmo ST-GAT treinado com 60 n├│s calcula scores para os 60 candidatos atuais; o Top-40 ├® uma **restri├º├úo da sa├¡da dessa mesma rede de 60 n├│s**, n├úo um ST-GAT treinado exclusivamente em 40 n├│s.
- Grafo neural usa coordenadas de bairros e dist├óncias geogr├íficas est├íveis; grafo de conflito ├® identidade. Foram mantidos **CVLI (canal 0), calend├írio (3ÔÇô22) e momentum causal (33ÔÇô36)**; demais canais s├úo zerados. Nenhum dado recente de fac├º├Áes, clima, CVP, intelig├¬ncia ou tens├úo t├ítica ├® reutilizado retrospectivamente.
- Poisson original da fam├¡lia do projeto ├® **refeito desde zero para cada origem**, usando os **121 bairros**, `build_current_frame` oficial, treino de janelas hist├│ricas de 30 dias e rec├ílculo de atributos est├íticos de CVLI no exato instante da amostra. Tens├úo t├ítica ├® zero. As previs├Áes s├úo scores de ranking, n├úo probabilidades calibradas.
- M├®tricas: **P@10** (propor├º├úo de dez bairros antecipados com pelo menos um CVLI nos trinta dias), captura@20 (fra├º├úo de todos os CVLIs da cidade ocorridos nos vinte bairros mais bem classificados), cobertura do pool 40/60 e falsos positivos Top-10. Denominador CVLI: **todos os 121 bairros**, sempre.
## Principal: Top-60 m├│vel, 18 janelas de 2024ÔÇô2025
| Peso ST-GAT / Poisson | P@10 | Captura@20 | Cobertura potencial do Top-60 |
|---|---:|---:|---:|
| **0/100 (Poisson puro)** | 73,9% | 46,5% | 86,4% |
| **20/80 (h├¡brido refer├¬ncia)** | 74,4% | 48,2% | 86,4% |
| **40/60** | 78,9% | 49,1% | 86,4% |
| **70/30** | **81,1%** | **49,6%** | **86,4%** |
| **100/0 (score operacional ST-GAT)** | 79,4% | 48,9% | 86,4% |
H├í **2.260 CVLIs somados** nos 18 horizontes de 30 dias (n├úo ├® total anual; dias 31 s├úo exclu├¡dos). No Top-20, 70/30 concentrou 1.113 eventos somados contra 1.083 no 20/80, mas os percentuais acima s├úo **m├®dias por janela**, n├úo divis├úo desses totais.
### Diferen├ºas pareadas 70/30
- Frente ao **Poisson puro**: P@10 **+7,22 pp**, intervalo bootstrap descritivo de 95% **[+2,22; +12,78] pp**; captura@20 **+3,03 pp**, **[+0,25; +5,88] pp**.
- Frente ao h├¡brido **20/80**: P@10 **+6,67 pp**, intervalo descritivo **[+2,22; +11,67] pp**; captura@20 **+1,42 pp**, **[-0,35; +3,26] pp**.
- Foram usados 10 mil reamostragens dos **18 meses pareados**. Esses intervalos N├âO corrigem multiplicidade de pesos tentados, autocorrela├º├úo residual, sele├º├úo anterior de hip├│teses nem diferen├ºas estruturais de 2026; n├úo sustentam alega├º├Áes conclusivas de superioridade.
### Controle com baselines simples
| Ranking sobre 60 candidatos | P@10 | Captura@20 |
|---|---:|---:|
| Poisson puro | 73,9% | 46,5% |
| EWMA de CVLI dos ├║ltimos 7/14/30/60 dias | 73,9% | 47,6% |
| **Somente contagem de CVLI nos ├║ltimos 30 dias** | **78,9%** | **48,5%** |
| H├¡brido operacional 70/30 | **81,1%** | **49,6%** |
Contra o simples CVLI-30d, **70/30 aumentou apenas +2,22 pp de P@10**, bootstrap **[-1,11; +6,11] pp**, e **+1,04 pp de captura@20**, **[-0,53; +2,71] pp**. Ambos incluem diferen├ºa zero. Portanto **o valor incremental da rede sobre a persist├¬ncia recente ainda N├âO foi demonstrado com confian├ºa**.
O Poisson aplicado livremente aos **121 bairros** obteve P@10 **73,9%** e captura@20 **44,1%**; o CVLI-30d aplicado aos **121** obteve **78,3%** e **48,9%**, respectivamente. Comparar com 121 oferece outra refer├¬ncia sem limitar previamente os bairros.
### Top-40 x Top-60
No teste de 2024ÔÇô2025, **Top-40 m├│vel (calculado a partir da rede de 60)** teve cobertura de **74,4%**, contra **86,4%** do Top-60. Com peso 70/30, ambos produziram **P@10=81,1%**; captura@20 foi **50,3% no Top-40** e **49,6% no Top-60**. **Mais cobertura n├úo implica melhor ordena├º├úo no Top-20**. Isto n├úo compara uma rede exclusiva de 40 n├│s com outra exclusiva de 60 ÔÇö ├® o mesmo ST-GAT60 restringido.
### Abla├º├úo: neural bruto
Com 70% de **sa├¡da neural bruta** (sem a regra de suporte de CVLI que comp├Áe o score operacional), o Top-60 alcan├ºou P@10 de **80,6%** e captura@20 de **51,3%**, frente a **81,1% e 49,6%** com o score operacional. Isso sugere que o componente auxiliar pode reduzir captura em algumas configura├º├Áes; requer testes pr├®-registrados independentes.
## Sensibilidade 2026 (seis janelas, abrilÔÇôsetembro)
| Peso ST-GAT | P@10 | Captura@20 |
|---|---:|---:|
| 0% | 45,0% | 54,2% |
| **20%** | 50,0% | **57,2%** |
| 40% | **51,7%** | 54,0% |
| 70% | 50,0% | 54,2% |
| 100% | 50,0% | 52,2% |
**N├úo houve dom├¡nio consistente do ST-GAT majorit├írio em 2026**, e o tamanho da s├®rie caiu drasticamente na origem.
## Interpreta├º├úo e decis├úo
O backtest hist├│rico de 2024ÔÇô2025 ├® **promissor** para ST-GAT majorit├írio, sobretudo no P@10, mas ganhos de captura s├úo modestos, e um baseline de **CVLI dos ├║ltimos 30 dias** ainda compete muito bem. A amplia├º├úo a 60 bairros elevou cobertura potencial mas **n├úo** garantiu melhor captura@20 diante da op├º├úo Top-40.
**N├âO alterar produ├º├úo.** Mant├¬-la na arquitetura aprovada enquanto avan├ºamos em:
1. Auditoria da origem dos CVLIs de 2026 (fonte, classifica├º├úo, completude e compara├º├úo mensal consistente).
2. Repeti├º├úo de treinamento com **sementes diferentes** e compara├º├úo com 40 n├│s treinados nativamente, n├úo somente restri├º├úo de 60.
3. Avalia├º├úo prospectiva selada de 22/10ÔÇô20/11/2026 apenas como valida├º├úo externa adicional. Os experimentos hist├│ricos est├úo conclu├¡dos **sem necessidade de aguardar um m├¬s**.
4. Prefer├¬ncia por indicadores de decis├úo operacional: P@10, captura@20, falsos positivos, estabilidade, custo de cobrir 10 ou 20 bairros. Scores continuam **n├úo calibrados como probabilidades de CVLI**.
## Reprodutibilidade e isolamento
Scripts:
- `research/benchmarks/rolling_origin_cvli_retrain_20261022.py`: execu├º├úo de cada origem (`--fold 2024`, `--fold 2025`, `--fold 2026`).
- `research/benchmarks/aggregate_rolling_origin_cvli_only_20261022.py`: m├®tricas globais e intervalos bootstrap.
- `research/benchmarks/historical_city_baselines_20261022.py`: controles Poisson total 121, EWMA e CVLI-30d.
- `research/benchmarks/historical_rolling_cvli_only_20261022/`: CSVs por fold com m├®tricas agregadas, ├®pocas e datas de treinamento/valida├º├úo, sem ranking nominal de bairros.
- `research/benchmarks/rolling_origin_cvli_only_pooled_summary_20261022.csv`, `rolling_origin_cvli_only_paired_20261022.csv`, `rolling_origin_cvli_only_baselines_20261022.csv` e metadados.
Nenhuma produ├º├úo, modelos ativos, pesos operacionais ou pipeline de ingest├úo foi alterado. N├úo houve promo├º├Áes ou deploy. O snapshot prospectivo selado permanece separado e privado.
