# Tabela de Estratégias e Modelos — CVLI

## Objetivo
Documentar, de forma direta, quais estratégias e modelos foram testados para previsão de `CVLI`, o que cada um usa e quando faz sentido utilizá-los.

| Estratégia / Modelo | Tipo | Sinal principal | Usa `CVP`? | Vantagem | Limitação | Quando usar |
|---|---|---|---|---|---|---|
| `Naive_EWMA` | Simples | `CVLI` recente | Não | Muito robusto, rápido e explicável | Não captura nuances estruturais | Baseline mínimo e comparação diária |
| `LightGBM_Rank` | Árvore / ranking | Mix de features tabulares | Sim, mas não deve liderar | Boa cobertura ampla | Não foi melhor no topo (`P@10`) | Cobertura operacional e apoio |
| `ST-GAT_Active` | Deep learning | Dinâmica espaço-temporal | Indireto | Arquitetura mais rica do projeto | Sensível à formulação e à disponibilidade de canais | Referência de arquitetura avançada |
| `LSTM` | Deep learning | Sequência temporal | Não diretamente | Capta memória temporal | Não superou os baselines simples | Pesquisa secundária |
| `TCN` | Deep learning | Sequência temporal convolucional | Não diretamente | Bom candidato temporal simples | Ainda abaixo do desejado no topo | Pesquisa secundária |
| `Histórico puro` | Heurística | `hist_pct` / `target_enc` | Não | Muito explicável | Congela demais o passado | Nunca usar sozinho |
| `CVLI_STRUCTURAL_ONLY` | Híbrido simples | Estrutural de `CVLI` | Não | Boa leitura territorial de base | Fraco no curto prazo | Componente estrutural de blend |
| `CVLI_TACTICAL_ONLY` | Híbrido simples | `CVLI` recente (`EWMA 7/14/30d`) | Não | Melhor `P@10` recente entre os candidatos testados | Pode oscilar e perder cobertura | Melhor candidato para foco em topo |
| `CVLI_STRUCT_TACTICAL` | Híbrido | Estrutural + tático de `CVLI` | Não | Equilibra memória e curto prazo | Não bateu o tático puro | Quando quiser suavizar volatilidade |
| `CVLI_FIRST_WITH_WEAK_CVP_CONTEXT` | Híbrido | `CVLI` estrutural+tático | Sim, fraco | Respeita `CVLI` como motor | `CVP` não trouxe ganho claro | Só se precisar de contexto auxiliar |
| `SHORT20_MIX` | 2 estágios | Shortlist tática + rerank | Não | Melhor equilíbrio entre cobertura e controle | `P@10` abaixo do tático puro | Operação com foco em top-20 / recall |
| `SHORT20_RERANK` | 2 estágios | Shortlist por `CVLI`, rerank estrutural | Não | Fácil de explicar operacionalmente | Perde topo quando rerank pesa demais | Cenários de triagem |

## Leitura objetiva
- Se a meta principal é **acertar o top-10 (`P@10`)**, o melhor candidato atual é `CVLI_TACTICAL_ONLY`.
- Se a meta principal é **equilíbrio entre topo e cobertura**, o melhor candidato atual é `SHORT20_MIX`.
- `CVP` deve permanecer **apenas contextual**, nunca como motor principal do score.

## Recomendação implementável agora
### Modelo principal de acompanhamento
- **`CVLI_TACTICAL_ONLY`**

### Modelo secundário de acompanhamento
- **`SHORT20_MIX`**

### Estratégia de monitoramento
- Gerar os dois rankings diariamente
- Validar ambos contra os próximos dados reais
- Acompanhar:
  - `P@10`
  - `P@20`
  - `R@10`
  - `R@20`
  - estabilidade do top-10

## Decisão atual
- **Não promover automaticamente para produção ainda**
- **Implementar e acompanhar em paralelo** como candidatos operacionais

## Operação no frontend
- `ST-GAT` permanece como **modelo padrão** do sistema.
- `CVLI_TACTICAL_ONLY` e `SHORT20_MIX` ficam como **modos opcionais** no frontend para comparação intuitiva.
- Nos modos opcionais, **Fortaleza** usa o ranking experimental de `CVLI`; **RMF** e **Interior** continuam no `ST-GAT`.
