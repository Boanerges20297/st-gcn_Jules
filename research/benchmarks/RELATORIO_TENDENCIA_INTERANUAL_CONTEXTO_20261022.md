> **ATENCAO (23/10/2026):** Este benchmark foi executado antes da consolidacao de CVLI por vitima. A auditoria MySQL/SSPDS identificou 2.452 representacoes redundantes na base enriquecida. Numeros e conclusoes deste relatorio sao **historicos, nao validacao na base corrigida**. Ver `docs/AUDITORIA_CVLI_MYSQL_SSPDS_20261023.md`.

﻿# Report Preview ÔÇö Tend├¬ncia interanual como sinal contextual do h├¡brido ST-GAT/Poisson
**Executado em 22/10/2026 ÔÇó STI01/NEXO ÔÇó pesquisa isolada ÔÇó sem altera├º├Áes em produ├º├úo**
## Hip├│tese proposta
Substituir "um peso do ST-GAT para cada ano" por **uma regra reutiliz├ível entre anos**: extrair a mudan├ºa na participa├º├úo relativa de CVLI dos bairros entre os dois anos civis completos anteriores e verificar, com os ├║ltimos **60, 90 ou 120 dias j├í observados**, se essa trajet├│ria est├í ganhando ou perdendo apoio. Aplicar esse fator na escolha de malha, horizonte de tend├¬ncia e/ou peso do ST-GAT. Horizonte de ALVO segue **30 dias** (n├úo confundir com o per├¡odo do sinal).
Exemplo verific├ível: **2022ÔåÆ2023 para prever 2024**, **2023ÔåÆ2024 para prever 2025**, **2024ÔåÆ2025 para prever 2026**. Uma contagem de CVLI de julho de 2025 jamais pode influenciar a escolha aplicada a julho de 2025.
## Base e grupos
- Fonte: tensor CVLI de **121 bairros**, 01/01/2022ÔÇô21/10/2026. Cada previs├úo ├® avaliada com denominador de **toda Fortaleza**, nunca somente os 60 bairros selecionados.
- Malha prim├íria de **60 bairros m├│veis**, escolhidos em cada data pelo CVLI dos **730 dias anteriores**. Controles para 40 bairros e todos os 121.
- Avalia├º├úo: nove janelas de 30 dias sem sobreposi├º├úo em 2024 (abrilÔÇôdezembro), nove em 2025, seis em 2026 (abrilÔÇôsetembro).
- 2026 ├® **teste de sensibilidade** por anomalia de volume de CVLI j├í presente no CSV bruto.
- Nos resultados de ST-GAT/Poisson, usam-se previs├Áes hist├│ricas previamente calculadas em treino por ano, com checkpoint do ST-GAT **iniciado do zero** para cada origem e Poisson retreinado por corte de ano. O experimento atual analisa contexto e combina├º├Áes; **ainda N├âO treinou uma nova rede com a tend├¬ncia interanual inclu├¡da nos canais da arquitetura**.
## C├ílculo causal do sinal
Se `sA` representa a **participa├º├úo relativa de cada bairro** nos CVLIs do pen├║ltimo ano e `sB` sua participa├º├úo no ├║ltimo ano completo, a previs├úo estrutural testada foi:
`estrutura = max(epsilon, sB + 0.5 * (sB - sA))`
A participa├º├úo no total da cidade evita interpretar o salto de intensidade de toda Fortaleza como se fosse crescimento espec├¡fico de cada bairro. As participa├º├Áes foram suavizadas com pseudocontagem de 0,5 por bairro. Testou-se ainda tend├¬ncia interanual integral (multiplicador 1) e extrapola├º├úo logar├¡tmica com amortecimento.
O contexto recente foi calculado somente de `[data_corte - L, data_corte)` para L = 60, 90, 120 dias; a combina├º├úo de escores foi por percentis/ranking com **25%, 50% ou 75% de tend├¬ncia anual**. Todos s├úo *scores para prioriza├º├úo*, n├úo probabilidades CVLI.
A vers├úo adaptativa de janela escolheu L pela **maior concord├óncia ordinal entre a distribui├º├úo estrutural interanual e a distribui├º├úo CVLI recente**. Isso foi calculado antes do per├¡odo alvo, sem olhar resultados futuros.
## Resultado A ÔÇö for├ºa dos sinais, 2024ÔÇô2025 (18 janelas)
| Sinal de ranking, Top-60 | P@10 | Captura@20 |
|---|---:|---:|
| Tend├¬ncia interanual de 50% (sem rec├¬ncia) | 72,2% | 47,3% |
| ├Ültimo ano completo (sem extrapolar) | 73,3% | 48,8% |
| CVLI ├║ltimos 30 dias | 78,9% | 48,5% |
| Bimestre 60d | 77,8% | 51,7% |
| Trimestre 90d | 82,8% | 49,3% |
| **Quadrimestre 120d** | **83,9%** | 50,6% |
| Tend├¬ncia interanual + 120d (50% para cada sinal) | 80,6% | **51,9%** |
| Janela adaptada por concord├óncia interanual | 78,3% | 51,9% |
**Compara├º├úo contextual**: ST-GAT/Poisson **70/30 fixo** anterior, mesmos cortes 2024ÔÇô2025 e mesma malha m├│vel 60, registrou **P@10=81,1%; captura@20=49,6%**. Portanto, *quadrimestre puro* teve P@10 superior, e *tend├¬ncia interanual + quadrimestre* apresentou captura@20 superior neste backtest. N├úo ├® prova de ganho independente da rede: esta parte usa rankings estat├¡sticos e n├úo reestrutura a rede neural.
Diferen├ºas pareadas para tend├¬ncia interanual + 120d (50/50) **vs CVLI-30d puro**:
- P@10 **+1,67 pp**, bootstrap de 18 janelas (intervalo descritivo 95% **[-3,33; +7,22] pp**) ÔÇö n├úo conclusivo.
- Captura@20 **+3,39 pp**, intervalo **[+0,65; +6,11] pp** ÔÇö sinal explorat├│rio favor├ível; n├úo ajustado por buscas m├║ltiplas nem depend├¬ncias.
**Janela preferida pela concord├óncia de n├¡veis com tend├¬ncia anual (sem consultar resultados):** 120d em **8/9 meses de 2024** e **8/9 de 2025**. Em 2026, 120d em **4/6 meses**.
## Resultado B ÔÇö generaliza├º├úo entre anos (sem escolher estrat├®gia com dados do ano testado)
Ao selecionar a melhor combina├º├úo **pelas nove janelas de 2024** e s├│ ent├úo aplic├í-la a 2025, a regra escolhida foi **tend├¬ncia interanual + quadrimestre 50/50**. Na avalia├º├úo independente de 2025 atingiu **P@10=76,7%; captura@20=51,8%**, enquanto o **quadrimestre puro** em 2025 obteve **84,4%; 52,2%**. A regra mais sofisticada **n├úo generalizou como melhor escolha**.
## Resultado C ÔÇö pol├¡tica contextual de pesos ST-GAT
A regra testada mede **concord├óncia direcional**: se os bairros que ganharam participa├º├úo do ano A para B s├úo tamb├®m os que ganharam participa├º├úo entre os dois blocos recentes de L dias. Seleciona entre **20% ou 70% de ST-GAT** de acordo com um limiar. Foram avaliados 60/90/120d, cinco limiares e dois sentidos do gatilho, **30 pol├¡ticas sobre apenas 9 janelas de desenvolvimento de 2024** ÔÇö risco elevado de sobreajuste.
A pol├¡tica com maior utilidade em 2024 foi 120d, limiar -0,15 de correla├º├úo ordinal das *mudan├ºas de participa├º├úo*, com 70% ST-GAT quando a correla├º├úo >= -0,15, sen├úo 20%. N├úo ├® crit├®rio a promover; ├® somente melhor candidato retrospectivo.
| Ano | Pol├¡tica | P@10 | Captura@20 |
|---|---|---:|---:|
| **2025 (teste fora do ajuste)** | **Adapta├º├úo contextual** | **78,9%** | **51,1%** |
| 2025 | Fixo ST-GAT 20% | 74,4% | 50,4% |
| 2025 | **Fixo ST-GAT 70%** | **81,1%** | **51,4%** |
| 2026 (sensibilidade) | Adapta├º├úo contextual | 51,7% | 56,5% |
| 2026 | Fixo ST-GAT 20% | 50,0% | 57,2% |
| 2026 | Fixo ST-GAT 70% | 50,0% | 54,2% |
O gatilho selecionou 70% ST-GAT em **6 de 9** meses de 2025. A pol├¡tica adaptativa **n├úo superou a configura├º├úo fixa 70/30** no teste de 2025. Os intervalos pareados em nove meses s├úo largos; n├úo h├í demonstra├º├úo robusta de benef├¡cio. N├úo se deve otimizar peso por ano com base no mesmo ano avaliado.
## Leitura e pr├│xima hip├│tese test├ível
1. **A no├º├úo de dois ritmos temporais ├® v├ílida como hip├│tese operacional**: ├║ltimo par de anos representa dire├º├úo estrutural; ├║ltimos 60/90/120 dias indicam contexto do momento.
2. O ├║ltimo quadrimestre tem sinal preditivo de P@10 forte **mesmo sem rede neural**; isso exige uma compara├º├úo justa com ST-GAT, n├úo apenas com Poisson.
3. Extrapolar tend├¬ncia anual sozinha mostrou desempenho pior que rec├¬ncia pura. Incorpor├í-la com **peso moderado** pode ajudar captura, mas seu papel ainda ├® de regularizador/contexto, n├úo substituto de sinais recentes.
4. **A pol├¡tica adaptativa de pesos ainda falhou contra ST-GAT 70/30 fixo em 2025.** N├úo confundir plausibilidade metodol├│gica com ganho comprovado.
5. **Pr├│ximo experimento recomendado:** usar a tend├¬ncia interanual **como atributo causal de n├│ dentro do ST-GAT** (e opcionalmente do Poisson), treinando com snapshots recriados por ├®poca e revisando 60/90/120d; comparar modelo id├¬ntico com e sem esse canal, com m├║ltiplas sementes, sele├º├úo somente em passado e anos reservados. N├úo inserir essa feature diretamente no checkpoint ativo.
6. Fonte de 2026 tem forte descontinuidade de CVLI j├í no CSV bruto (381 linhas CVLI, 378 registros no tensor at├® 21/10). Manter ressalva de completude.
## Scripts/resultados reprodut├¡veis
- `interannual_signal_context_20261022.py`: rankings por tend├¬ncias anuais + contexto de 60/90/120d, 1.368 linhas de m├®tricas, sem exporta├º├úo de ranking nominal de bairros.
- `contextual_hybrid_year_to_year_20261022.py`: regra de pesos ajustada em 2024, teste independente em 2025 e sensibilidade em 2026.
- `interannual_context_cvli_20261022/interannual_window_rows.csv`, `interannual_window_summary.csv`, `interannual_yearly_comparison.csv`, `selected_contexts.csv`.
- `interannual_context_cvli_20261022/context_policy_yearly_test_summary.csv`, `context_weight_policy_candidates_2024.csv`, `yearly_context_slope_alignment.csv`, `interannual_metadata.json`, `context_policy_metadata.json`.
Todos os artefatos foram produzidos na pasta experimental `research/benchmarks/`. Nenhuma produ├º├úo, modelo ativo, pipeline ou regra h├¡brida operacional foi modificada.
