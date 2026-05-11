# 📝 Plano de Fase — Fase 2: Retreino com Inteligência Profunda e Memória de Longo Prazo (4 Anos)

## 🎯 Objetivo
Retreinar o modelo Challenger (Sentinela V3) utilizando todo o histórico disponível (2022-2026), implementando um sistema de **Pesos Dinâmicos por Recência** para garantir que o modelo capture padrões históricos profundos, mas priorize a volatilidade e as mudanças de contexto do cenário atual de 2026.

## 🛠️ Tarefas
1. **[T-1] Expansão do Horizonte de Treino (2022-2026):**
   - Alterar `start_d` em `freeze_total_v3.py` para `2022-01-01`.
   - Garantir que o pipeline de features lide corretamente com 4 anos de dados (memória e tempo de processamento).

2. **[T-2] Implementação de Context Sensing (Sample Weighting):**
   - Adicionar lógica de `sample_weight` ao `LGBMRanker`.
   - Utilizar decaimento exponencial: $w = e^{-(T_{atual} - T_{amostra}) / \tau}$, onde amostras recentes têm peso máximo e amostras antigas (2022) servem como "ruído de fundo estrutural". Isso permite ao modelo "sentir" a transição de contexto dinamicamente.

3. **[T-3] Refinamento de Hiperparâmetros (Incorporando REVIEWS.md):**
   - Manter rigorosamente as **10 features originais** (Requisito #20).
   - Aumentar `num_leaves` para **127** e `n_estimators` para **1500**.
   - Ajustar `min_data_in_leaf=3` para permitir que o modelo capture microssinais em áreas de baixa densidade (inteligência cirúrgica).

4. **[T-4] Execução e Auditoria de Estocasticidade:**
   - Rodar o treino completo.
   - Analisar o `freeze_report.txt` verificando se o ganho de memória longa (4 anos) melhorou o P@10 em relação ao modelo de 2 anos.
   - Validar se a "Aceleração Criminal" está sendo capturada mesmo com o ruído dos anos anteriores.

## ✅ Critérios de Aceitação (UAT)
- [ ] O dataset de treino inicia em Jan/2022.
- [ ] O script utiliza `sample_weight` no método `.fit()` do ranker.
- [ ] P@10 ≥ 52% na validação sombra (Abr/Maio 2026).
- [ ] O modelo gerado preserva compatibilidade com 10 features (Requisito #20).
- [ ] Registro da tentativa 58 no `TRAINING_LOG.md`.

## 🚨 Riscos e Mitigações
- **Risco:** Dados de 2022 muito diferentes de 2026 (Drift de conceito).
- **Mitigação:** O peso exponencial prioriza o contexto atual, usando o passado apenas para estabilizar a "geografia do crime".
- **Risco:** Estouro de memória com 4 anos de features.
- **Mitigação:** Uso de `int8` e `float32` agressivo no processamento das matrizes.
