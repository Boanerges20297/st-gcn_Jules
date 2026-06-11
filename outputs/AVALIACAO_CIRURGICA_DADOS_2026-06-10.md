# Avaliação Cirúrgica dos Dados — Estratégia de Modelagem

## Diagnóstico central
A principal limitação atual não parece ser falta de modelos sofisticados. O problema está na **estrutura do sinal** e na **forma como o alvo é formulado**.

## 1. Estrutura dos eventos
- Série com `40` bairros monitorados e `1620` dias.
- Média diária de CVLI total: `1.40`
- Mediana diária de CVLI total: `1.0`
- Média diária de bairros positivos: `1.36`
- Mediana diária de bairros positivos: `1.0`

### Janela operacional de 14 dias
- Média de bairros positivos por janela: `13.97`
- Mediana de bairros positivos por janela: `15`
- Média de CVLI total por janela: `19.46`
- Mediana de CVLI total por janela: `20`

### Leitura
Para previsão diária, o problema é esparso demais. Para janela de 14 dias, ele deixa de ser “evento raro por bairro” e vira um problema de **ordenação densa**, com muitos bairros positivos por janela.

## 2. Força real dos sinais
Associação monotônica com alvo binário no treino até `2026-02-28`:

| Feature | Spearman alvo binário |
|---|---:|
| `hist_pct` | `0.264` |
| `target_enc` | `0.242` |
| `cvp_ewma_30d` | `0.114` |
| `cvp_ewma_14d` | `0.110` |
| `cvp_ewma_7d` | `0.103` |
| `cvp_cvli_ratio` | `0.088` |
| `inter_chuva_hist` | `0.064` |
| `inter_intel_cvli` | `0.028` |
| `intel_ewma_14d` | `0.013` |
| `nbr_cvli_30d` | `0.011` |

### Leitura
- O que mais explica o alvo é **histórico estrutural** (`hist_pct`, `target_enc`).
- Os sinais dinâmicos mais úteis são os derivados de **CVP recente**.
- `intel` e `vizinhança` quase não aparecem como drivers gerais nesse protocolo.
- Isso explica por que vários modelos diferentes acabam convergindo para resultados parecidos: o limite informacional está concentrado em poucos sinais medianos.

## 3. Persistência histórica dos hotspots
### No histórico acumulado
- `50%` dos CVLI estão em `12` bairros
- `70%` em `21` bairros
- `80%` em `26` bairros
- `90%` em `32` bairros

### No futuro recente (holdout)
- Parcela média dos bairros positivos futuros dentro do top-10 histórico: `44.33%`
- Parcela média dentro do top-20 histórico: `49.19%`

### Leitura
A ideia de que “os mesmos bairros explicam quase tudo” é **forte demais** para o dado real. O histórico ajuda bastante, mas sozinho cobre só metade do fenômeno futuro recente.

## 4. Drift temporal
A taxa de positivos mensal (`pos_rate`) varia muito ao longo do tempo.
Exemplos:
- `2024-03`: `0.436`
- `2025-10`: `0.313`
- `2026-01`: `0.278`
- `2026-02`: `0.150`

### Leitura
Há **mudança de regime temporal** relevante. Um modelo que aprende um ranking estrutural estável sofre quando a densidade e a composição dos bairros positivos mudam de forma abrupta.

## 5. Vizinhança e contágio espacial
- Correlação média entre `nbr_cvli_30d` e alvo futuro: `-0.0168`

### Leitura
No agregado, o sinal de retaliação/vizinhança está praticamente nulo. Isso não significa que ele nunca ajude, mas sim que **não sustenta sozinho** uma estratégia principal global.

## Conclusão estratégica
O problema atual não é “escolher entre modelo simples e deep learning”.
O problema é que, com o alvo atual, o dado mistura dois fenômenos:
1. **risco estrutural persistente**
2. **deslocamento tático recente**

Quando os dois são jogados num único rank de 40 bairros para horizonte de 14 dias, os modelos capturam bem o estrutural, mas mal o deslocamento fino. Por isso o teto de `P@10` fica travado.

## Estratégia recomendada
### Melhor formulação
Trocar o problema de “um ranking único puro” para uma arquitetura em camadas:
- **Camada 1 — Base estrutural:** histórico (`hist_pct`, `target_enc`, CVP acumulado)
- **Camada 2 — Ajuste tático:** EWMA recente, CVP curto prazo, gatilhos contextuais
- **Camada 3 — Regime:** detectar mudança de densidade / mês / fase operacional

### Em termos práticos
A estratégia mais promissora não é “mais deep” por si só. É:
- **modelo híbrido com separação explícita entre estrutural e tático**
- ou **dois estágios**:
  - estágio A: detectar shortlist provável
  - estágio B: reranquear só dentro dela

## O que evitar
- Não insistir em um ranking global simples como se todos os 40 bairros competissem no mesmo regime.
- Não assumir que o histórico top-10 ou top-20 resolve sozinho.
- Não esperar que um modelo profundo ganhe muito se os sinais dinâmicos continuarem fracos e pouco estáveis.

## Recomendação final
A melhor estratégia agora é desenhar uma formulação que respeite a natureza do dado:
- **estrutural + tático + regime**
- e validar isso com holdout temporal longo

## Artefatos gerados
- `outputs/data_surgical_summary.json`
- `outputs/data_surgical_feature_associations.csv`
- `outputs/data_surgical_monthly_target.csv`
- `outputs/data_surgical_monthly_features.csv`
- `outputs/data_surgical_hotspot_persistence.csv`
- `outputs/data_surgical_window_stats.csv`
