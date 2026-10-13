# Experimento corrigido: decomposicao de sinais do ST-GAT v5

Data: 2026-08-15

## Objetivo

Separar quatro fontes de previsibilidade:

1. persistencia estrutural de longo prazo;
2. estado recente de curto prazo;
3. memoria recente ponderada;
4. ganho adicional do ST-GAT v5.

Esta nota substitui a leitura anterior que chamava um baseline movel de `Structural_mu_baseline`.

## Script e comando

Script alterado:

- `scripts/validate_stgat_v5_walk_forward.py`

Comando principal:

```powershell
.\.venv\Scripts\python.exe scripts\validate_stgat_v5_walk_forward.py --start 2026-03-01 --end 2026-07-03 --horizon 30 --stride 1
```

## Definicoes

`StructuralLongTerm_mu`: media diaria historica de cada territorio entre 2022-01-01 e 2024-12-31, multiplicada pelo horizonte de 30 dias. Fica fixa durante toda a avaliacao de 2026.

`RecentMean30`: media diaria dos 30 dias imediatamente anteriores a `t`, multiplicada pelo horizonte de 30 dias.

`EWMA`: media ponderada de taxas diarias recentes, com janelas 7, 14, 30 e 60 dias e pesos 0.40, 0.30, 0.20 e 0.10, multiplicada pelo horizonte de 30 dias.

`ST-GAT v5`: inferencia do checkpoint ativo, sem retreino, tuning, nova feature ou alteracao arquitetural.

## Resultado principal

| Modelo | Escopo | P@5 | P@10 | P@20 | R@20 |
|---|---|---:|---:|---:|---:|
| StructuralLongTerm_mu | Fortaleza | 0.4624 | 0.3600 | 0.2308 | 0.5073 |
| RecentMean30 | Fortaleza | 0.4256 | 0.2744 | 0.2640 | 0.5868 |
| EWMA | Fortaleza | 0.4064 | 0.3480 | 0.3096 | 0.6822 |
| ST-GAT v5 | Fortaleza | 0.3872 | 0.2936 | 0.2660 | 0.5817 |
| StructuralLongTerm_mu | Global | 0.7600 | 0.6672 | 0.6104 | 0.3974 |
| RecentMean30 | Global | 0.8896 | 0.8384 | 0.6396 | 0.4180 |
| EWMA | Global | 0.8304 | 0.7496 | 0.6444 | 0.4217 |
| ST-GAT v5 | Global | 0.8368 | 0.7736 | 0.6464 | 0.4214 |

## Leitura provisoria

No global, o ST-GAT v5 supera o `StructuralLongTerm_mu`, mas nao supera o `RecentMean30` em P@5/P@10. O ganho global do ST-GAT parece menor do que a memoria recente simples.

Em Fortaleza isoladamente, `StructuralLongTerm_mu` supera o ST-GAT em P@5 e P@10; EWMA supera o ST-GAT em P@20/R@20. Assim, para bairros de Fortaleza, a persistencia historica e a memoria recente ainda explicam muito do desempenho.

No Interior, ST-GAT e `RecentMean30` ficam praticamente empatados em P@10/R@10; `StructuralLongTerm_mu` e melhor em P@5.

Na RMF, ST-GAT empata com `RecentMean30`/EWMA em P@5 e fica proximo do `StructuralLongTerm_mu` em P@10.

## Efeito entre escopos

Composicao media do Top-K global:

- ST-GAT Top-10: Fortaleza 2.040, RMF 1.256, Interior 6.704.
- RecentMean30 Top-10: Fortaleza 1.088, RMF 2.248, Interior 6.664.
- EWMA Top-10: Fortaleza 1.520, RMF 2.168, Interior 6.312.
- StructuralLongTerm Top-10: Fortaleza 0.000, RMF 6.000, Interior 4.000.

O ST-GAT redistribui mais posicoes globais para Fortaleza do que os baselines recentes, mas isso nao implica ganho dentro de Fortaleza, onde o desempenho isolado ficou abaixo dos baselines principais.

## Estabilidade e ganho

Correlacoes com `RelativeGain_vs_StructuralLongTerm_mu`:

- historical_cv Pearson: -0.525315
- historical_cv Spearman: -0.633482
- historical_mean Pearson: 0.338463
- historical_mean Spearman: 0.618727

Leitura: nesta rodada, o ganho relativo do ST-GAT contra o baseline estrutural tende a ser maior em territorios historicamente mais frequentes e menor em territorios com maior coeficiente de variacao. Isto e associacao observacional, nao causalidade.

## Leakage e ressalvas

Nao foi identificado uso de contagem futura nos baselines:

- `StructuralLongTerm_mu` usa somente 2022-2024;
- `RecentMean30` usa somente os 30 dias anteriores a `t`;
- `EWMA` usa somente janelas passadas;
- o alvo futuro entra apenas na avaliacao.

Ressalva: o checkpoint ST-GAT ativo continua sendo inferido sem retreino por corte temporal. Portanto, a avaliacao nao deve ser apresentada como prova prospectiva estrita se o checkpoint ativo tiver usado informacao posterior ao corte durante o treino.

## Proxima decisao

Nao iniciar ST-GAT residual ainda. Primeiro interpretar esta decomposicao: o sinal recente simples aparece forte, e a pergunta agora e se o ST-GAT residual deve ser desenhado para superar `RecentMean30` e EWMA, nao apenas `StructuralLongTerm_mu`.
