# Relatório de Validação de Modelos — 2026-06-10

## Escopo
Este relatório consolida a análise da arquitetura atual, o histórico em `TRAINING_LOG.md` e uma validação exaustiva em holdout longo, sem promover nenhum modelo em produção.

## Arquitetura confirmada em produção
- **Champion regional Fortaleza:** checkpoint `models/active/fortaleza_model_active.pth`
- **Classe real carregada:** `PureSTGCN_64`
- **Meta do checkpoint:** `in_channels=41`, `window=90`
- **Challenger carregado:** `models/active/lgbm_lean_v3_freeze.pkl`
- **Estado do árbitro Champion/Challenger:** `data/cc_state.json` com `cc_weight=0.0`
- **Conclusão operacional:** o challenger está presente, mas hoje **não influencia** a previsão final da API.

## Leitura do histórico (`TRAINING_LOG.md`)

### Marcos relevantes
- **T57**: melhor validação histórica fora da amostra do paradigma Sentinela V3.
- **T57b**: freeze e promoção do `lgbm_lean_v3_freeze.pkl`.
- **T58**: integração Champion/Challenger na arquitetura de produção.
- **T112**: novo freeze com 15 features e contexto adicional.
- **T120–T123**: rodada recente de Fortaleza com degradação material de performance no pipeline ST.

### Conclusão histórica
O log mostra que o paradigma Sentinela foi promissor em abril, mas as iterações mais recentes não sustentam, por si só, evidência suficiente para promoção automática de um novo challenger sem backtest temporal mais duro.

## Metodologia da validação exaustiva

### Objetivo
Evitar esperar novos 14 dias e validar imediatamente com holdout temporal realista.

### Corte usado
- **Treino:** até `2026-02-28`
- **Holdout:** de `2026-03-01` até `2026-05-25`
- **Horizonte por previsão:** 14 dias
- **Total de janelas avaliadas:** 86 janelas diárias com ground truth disponível

### Modelos comparados
- `EWMA`
- `Active_LGBM`
- `Active_Ensemble`
- `FrozenFeb_LGBM`
- `FrozenFeb_Ensemble`

### Métricas
- `P@10`: precisão nos 10 bairros mais altos
- `P@20`: precisão nos 20 bairros mais altos
- `R@10`: recall sobre bairros com CVLI no top-10
- `R@20`: recall sobre bairros com CVLI no top-20

## Resultado consolidado do holdout longo

| Modelo | P@10 | P@20 | R@10 | R@20 | Janelas |
|---|---:|---:|---:|---:|---:|
| Active_Ensemble | 21.98% | 13.84% | 48.24% | 62.18% | 86 |
| FrozenFeb_Ensemble | 21.28% | 14.53% | 50.40% | 66.27% | 86 |
| EWMA | 20.23% | 15.93% | 47.40% | 69.66% | 86 |
| Active_LGBM | 19.07% | 12.85% | 46.57% | 59.84% | 86 |
| FrozenFeb_LGBM | 16.63% | 11.69% | 41.59% | 55.31% | 86 |

## Leitura técnica dos resultados
- **Melhor P@10 médio:** `Active_Ensemble` com `21.98%`
- **Melhor P@20 médio:** `EWMA` com `15.93%`
- **Melhor recall amplo:** `EWMA` também lidera `R@20` com `69.66%`
- **Melhor equilíbrio entre candidato congelado e cobertura:** `FrozenFeb_Ensemble` melhora recall vs ativo, mas perde em P@10
- **LGBM puro** é o pior bloco entre os comparados no holdout longo

## Resultado mensal

### Março/2026
- `Active_Ensemble`: `P@10=34.19%`, `P@20=21.13%`
- `FrozenFeb_Ensemble`: `P@10=29.68%`, `P@20=20.65%`
- `EWMA`: `P@10=27.42%`, `P@20=20.81%`

### Abril/2026
- `FrozenFeb_Ensemble`: `P@10=15.00%`, `P@20=9.50%`
- `EWMA`: `P@10=15.00%`, `P@20=9.50%`
- `Active_Ensemble`: `P@10=12.67%`, `P@20=7.83%`

### Maio/2026
- `FrozenFeb_Ensemble`: `P@10=18.40%`, `P@20=13.00%`
- `Active_Ensemble`: `P@10=18.00%`, `P@20=12.00%`
- `EWMA`: `P@10=17.60%`, `P@20=17.60%`

## Comparação direta: challenger ativo vs novo candidato
- O candidato novo retreinado em `tests/Sentinela/lgbm_lean_v3_freeze.pkl` gera top-10 diferente do ativo em `3/10` posições.
- Overlap do top-10 entre ativo e candidato: `7/10`
- O candidato novo é **mais recente**, mas não mostra ganho de P@10 no holdout longo suficiente para justificar promoção imediata.

## Ranking de alto nível do candidato novo
Fonte: `tests/Sentinela/ranking_atual_v3_freeze.csv`

1. `MESSEJANA`
2. `BARRA DO CEARA`
3. `JANGURURSSU`
4. `JOSE DE ALENCAR`
5. `MONDUBIM`
6. `PARQUE DOIS IRMAOS`
7. `CAJAZEIRAS`
8. `PRAIA DO FURUTO II`
9. `BARROSO`
10. `SIQUEIRA`

## Recomendação

### Decisão
**Não promover agora.**

### Justificativa
- O novo challenger não supera o ativo de forma clara em `P@10`, que é a métrica mais sensível para decisão tática.
- O `EWMA` segue extremamente competitivo em cobertura e recall amplo.
- O holdout longo de março até maio mostra que os ganhos do candidato são marginais e mais concentrados em recall do que em precisão de topo.
- Como `cc_weight=0.0`, qualquer promoção do `.pkl` sem revisão da regra de arbitragem ainda teria impacto operacional limitado.

## Próximos passos recomendados
1. Validar se o blend `EWMA + regras de guardrail` deve ganhar prioridade sobre o ensemble atual.
2. Rodar uma otimização explícita de pesos do ensemble com alvo em `P@10` e restrição mínima de `R@20`.
3. Criar um critério formal de promoção com thresholds simultâneos para `P@10`, `P@20` e estabilidade mensal.
4. Só promover challenger quando houver vantagem consistente em holdout longo, não apenas em uma sombra curta.

## Artefatos gerados
- `outputs/validation_holdout_mar_to_jun_2026.csv`
- `outputs/validation_holdout_mar_to_jun_2026_summary.csv`
- `outputs/validation_holdout_mar_to_jun_2026_monthly.csv`
- `outputs/validation_holdout_mar_to_jun_2026.json`
## Addendum — Hipótese dos bairros recorrentes e otimização de blend

### Validação da hipótese
A hipótese de que “90% dos CVLI ocorrem nos mesmos bairros” foi apenas **parcialmente confirmada**.

#### No histórico acumulado dos 40 bairros monitorados
- 50% dos CVLI históricos estão em **12 bairros**
- 70% dos CVLI históricos estão em **21 bairros**
- 80% dos CVLI históricos estão em **26 bairros**
- 90% dos CVLI históricos estão em **32 bairros**

Isso significa que 90% do volume histórico não está concentrado em um núcleo pequeno: ele ocupa **80% dos 40 bairros monitorados**.

#### No holdout de `2026-03-01` a `2026-05-25`
- Em média, **78.65%** dos bairros positivos de cada janela de 14 dias caem dentro do **top-32 histórico**
- Mediana: **80%**
- Apenas **14 de 86** janelas tiveram **100%** dos bairros positivos dentro do top-32 histórico

### Teste direto: ranking histórico fixo
Se usarmos sempre o mesmo top-10 histórico, sem deixar o modelo reordenar dinamicamente:
- **P@10 médio = 17.79%**
- **R@10 médio = 44.41%**

Conclusão: **não basta só trocar posições**. A composição do conjunto de risco também muda.

### Otimização de blend para P@10
Foi feita uma busca em grade com pesos para:
- `EWMA`
- `LGBM`
- `hist_pct`
- `target_enc`

Melhor blend encontrado no holdout longo:
- `60% EWMA`
- `20% LGBM`
- `20% histórico (`hist_pct`)`
- `0% target_enc`

#### Resultado desse blend
- **P@10 = 23.26%**
- **P@20 = 14.53%**
- **R@10 = 54.51%**
- **R@20 = 65.59%**

Esse foi o melhor blend testado sob a restrição de manter `R@20 >= 65%`.

### Decisão prática
Mesmo após otimização, o melhor blend continua **muito distante da meta de 45%+ em P@10**.
A meta desejada **não é atingível** com o conjunto atual de sinais, pelo menos nesse protocolo temporal de março até maio.

### Artefatos adicionais
- `outputs/blend_gridsearch_p10.csv`
- `outputs/blend_gridsearch_p10_summary.json`
- `outputs/historical_fixed_set_validation.csv`
