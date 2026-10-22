# Report Preview ORIGINAL ÔÇö Experimento A/B/C ST-GAT v5 em 60 bairros
**Data:** 22/10/2026. **Execu├º├úo:** STI01 via NEXO. **Status:** conclu├¡do e auditado; pesquisa isolada, **nenhuma promo├º├úo ├á produ├º├úo**.
## Pergunta
A expans├úo de 40 para 60 bairros exige reaprender os pesos do ST-GAT v5 do zero ou a transfer├¬ncia do modelo ativo conserva sinais ├║teis? O experimento preserva os 37 canais dispon├¡veis do processamento original (+ padding para 41 canais, conforme infer├¬ncia can├┤nica), os grafos territorial e de conflito originais, e utiliza horizonte de 30 dias.
## Malhas e s├®ries
- Regi├úo-alvo: Fortaleza, 121 bairros oficiais; **todos os eventos futuros da cidade** s├úo o denominador de captura@10 e captura@20.
- Top-40: nomes do checkpoint/dados ativos, na ordem original.
- Top-60: os 40 anteriores + 20 bairros restantes ordenados por CVLI hist├│rico de 2024ÔÇô2025. **N├úo** foi feita sele├º├úo de bairros pelo alvo de setembro.
- Base experimental congelada em `research/benchmarks/exp_data/all_official/processed_fortaleza.pkl`, datas 01/01/2022ÔÇô21/10/2026.
- Houve diverg├¬ncia entre o conte├║do do tensor Top-40 ativo, que evolui com o pipeline, e o subconjunto correspondente dessa c├│pia experimental. Por isso **todas as infer├¬ncias A/B/C foram feitas sobre a mesma c├│pia experimental**, preservando checkpoint original e IDs/ordem dos 40 bairros. N├úo confundir com execu├º├úo paralela do sistema em produ├º├úo.
## M├®todo
- Arquitetura: `DeepSTGAT_v5`, 120 dias de contexto, 41 canais, horizonte **30 dias**, grafos geogr├ífico e de conflito.
- Entrada: transforma├º├úo `build_input` da **infer├¬ncia can├┤nica**, com momentum causal recalculado na data do corte, padding de canais e normaliza├º├úo por janela. Os treinos transferido e aleat├│rio receberam as **mesmas entradas**.
- Perda: `NegBinomRankingLoss` do treinador oficial, com par├ómetros de ranking e honestidade, m├íscara bimestral de CVLI, dispers├úo aprend├¡vel e acumula├º├úo de gradiente de 4 amostras.
- Treinamento B/C: **98 janelas** de alvos de 30 dias (subamostragem a cada 14 dias), alvos de 2022ÔÇôjunho/2026, todos terminando **at├® 30/06/2026**.
- Valida├º├úo para escolha de checkpoint: **01/07, 15/07, 22/07 e 01/08**, com horizonte finalizando at├® agosto. Crit├®rio: 0,6├ùP@10 + 0,4├ùP@20 (n├úo ├® probabilidade).
- Holdout: **01/09, 08/09 e 15/09** com 30 dias futuros e refer├¬ncia dos **121 bairros**. Nunca usado para escolha de ├®poca.
- A: checkpoint ativo sobre Top-40 e sua transfer├¬ncia *zero-shot* ao Top-60.
- B: warm start desse checkpoint, 9 ├®pocas, learning rate experimental 6e-5; **melhor ├®poca 7**.
- C: inicializa├º├úo aleat├│ria comprovada diferente, learning rate experimental 2,5e-4; treinou 5 ├®pocas, interrompido por paci├¬ncia; **melhor ├®poca 1**. Isso ├® treino real, por├®m com curta converg├¬ncia ├║til.
- Poisson usado para o h├¡brido: modelo experimental **30 dias point-in-time**, salvo no controle anterior, id├¬ntico entre A/B/C. Os scores s├úo combinados por ranking percentil; nenhum score ├® probabilidade CVLI calibrada.
## Valida├º├úo antes de setembro
O checkpoint zero-shot Top-60 iniciou com P@10=45,0%, P@20=36,25%, score de sele├º├úo 0,415.
O treino B atingiu **score 0,440** no melhor checkpoint (P@10=47,5%). C tamb├®m obteve **score 0,440** no checkpoint selecionado (P@10=47,5%). O treinamento do zero, portanto, n├úo dominou a transfer├¬ncia na m├®trica de valida├º├úo.
## Setembro ÔÇö tr├¬s horizontes semanais sobrepostos
**H├¡brido 70% ST-GAT / 30% Poisson:**
| Variante | P@10 | Captura@10 dos CVLIs da cidade | Captura@20 dos CVLIs da cidade |
|---|---:|---:|---:|
| A: original Top-40 | 43,3% | 45,1% | 57,6% |
| A: original zero-shot Top-60 | 53,3% | 50,9% | 70,8% |
| **B: transfer learning Top-60** | **60,0%** | **61,0%** | **79,4%** |
| C: do zero Top-60 | **60,0%** | **61,0%** | **81,0%** |
Outros pesos Top-60:
- A zero-shot, 20% ST-GAT: P@10 **56,7%**, captura@20 **66,3%**.
- B ajustado, 20% ST-GAT: P@10 **53,3%**, captura@20 **68,5%**.
- B ajustado, 40% ST-GAT: P@10 **56,7%**, captura@20 **72,7%**.
- C do zero, 40% ST-GAT: P@10 **56,7%**, captura@20 **72,7%**.
- B ajustado, 100% score operacional ST-GAT: P@10 **60,0%**, captura@20 **75,2%**.
- C do zero, 100% score operacional ST-GAT: P@10 **60,0%**, captura@20 **79,4%**.
**Por que o Top-60 alcan├ºou 100% de cobertura no holdout?** Os CVLIs em cada janela foram 20 (01/09), 17 (08/09) e 15 (15/09), todos em bairros dos 60 candidatos selecionados. Isso vale **somente para essas tr├¬s janelas de setembro/outubro**; n├úo ├® cobertura integral anual nem previs├úo perfeita. As tr├¬s janelas se sobrep├Áem, logo **n├úo s├úo 52 eventos independentes**.
### Abla├º├úo: sinal neural puro vs score operacional
A pontua├º├úo operacional ST-GAT incorpora a rede neural e componentes de suporte hist├│rico/tens├úo. Para verificar se realmente houve aprendizado neural, medimos rankings por `softplus(sa├¡da_bruta_da_rede)`, sem esses componentes:
| Variante | P@10 neural puro | Captura@20 neural puro | Captura@20 score ST-GAT operacional |
|---|---:|---:|---:|
| Original Top-40 | 33,3% | 53,2% | 57,6% |
| Original zero-shot Top-60 | 33,3% | 56,8% | 70,8% |
| **B transferido** | **60,0%** | **82,2%** | 75,2% |
| C do zero | **60,0%** | 71,3% | 79,4% |
**Leitura:** B apresenta o maior sinal neural isolado no teste; C alcan├ºa captura@20 um pouco maior no **h├¡brido 70/30**, mas depende proporcionalmente mais da regra operacional do score. N├úo se pode dizer que treinamento do zero ├® superior.
## Interpreta├º├úo operativa
O sinal mais forte ├® que **a geometria ampliada, o fine-tuning e um ST-GAT majorit├írio merecem um novo teste real**, em vez de fixar Poisson 80% por tradi├º├úo. Por├®m o benchmark estrito anterior, que excluiu poss├¡veis vari├íveis retrospectivas, mostrou queda de captura quando a parcela ST-GAT cresceu. Os dois experimentos n├úo s├úo equivalentes: as vari├íveis e o grafo tamb├®m mudaram. Logo, melhoria aparente **n├úo implica causalidade nem valida├º├úo operacional externa**.
## Limita├º├Áes cr├¡ticas
1. **Vazamento temporal potencial:** grafo de fac├º├Áes/conflitos e `tension_index` t├¬m origem em snapshot contempor├óneo, n├úo em snapshots fi├®is a 2022ÔÇô2026 por data. A entrada CVLI e momentum ├® constru├¡da por corte, mas **n├úo h├í garantia de que todos os sinais hist├│ricos realmente existiam naquele instante**.
2. Top-60 fixo selecionado por hist├│rico at├® fim de 2025; v├ílido para teste de 2026, mas introduz sele├º├úo retrospectiva na parte inicial do treino (2022ÔÇô2025).
3. O checkpoint original j├í foi selecionado com valida├º├úo de julhoÔÇôagosto/2026, portanto esse intervalo n├úo ├® holdout independente para ele. Setembro permanece reservado nesta rodada.
4. Apenas **tr├¬s janelas de setembro, todas sobrepostas**; tamanho pequeno e pouco poder estat├¡stico.
5. C converge por escolha de checkpoint com apenas uma ├®poca ├║til; random init poderia melhorar com outro desenho de treino. LRs B/C diferentes foram definidos antes de observar setembro, para acomodar inicializa├º├úo diversa.
6. O procedimento de treino deste controle usa a transforma├º├úo da infer├¬ncia can├┤nica nos dois bra├ºos treinados. O treino hist├│rico original inclu├¡a outros elementos, como mem├│ria operacional; logo, isto **n├úo reproduz exatamente o pipeline anterior de treinamento**.
7. Captura ├® fra├º├úo de CVLIs futuros, n├úo taxa de preven├º├úo nem probabilidade calibrada; Top-20 representa 20 bairros da cidade.
## Pr├│ximo gate
**N├úo promover ├á produ├º├úo ainda.** Preparar uma avalia├º├úo em shadow dos Top-60 com snapshots territoriais congelados por data e teste prospectivo mais longo. Comparar 20%, 40%, 70% de ST-GAT **pr├®-definidos**. Medir P@10, captura@20, falsos alertas, calibra├º├úo e estabilidade por m├¬s.
## Artefatos reproduz├¡veis no STI
- `research/benchmarks/threeway_stgat60_30d_20261022.py`: treinos A/B/C, sele├º├úo e avalia├º├úo
- `research/benchmarks/threeway_stgat60_neural_audit_20261022.py`: auditoria independente do componente neural
- `research/benchmarks/threeway_stgat60_30d_rows_20261022.csv` e `...summary...`: h├¡brido por janela e por peso
- `research/benchmarks/threeway_stgat60_neural_audit_rows_20261022.csv` e `...summary...`: neural puro versus score operacional
- `research/benchmarks/train60_threeway_{transfer,scratch}_learning_20261022.csv`: evolu├º├úo por ├®poca
- `research/benchmarks/exp_models/stgat_v5_60_30d_{transfer,scratch}_original_features_20261022.pth`: checkpoints **locais, ignorados pelo Git**, marcados experimental_only e do_not_promote
- `research/benchmarks/exp_data/`: base experimental local, tamb├®m fora do Git
**Seguran├ºa:** nenhum checkpoint ativo, orquestrador, rotina de dados ou implanta├º├úo foi alterado; commit, se feito, ser├í limitado aos scripts e relat├│rios de pesquisa.
