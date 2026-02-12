# Histórico de Experimentos do Ranking e Híbrido

## Objetivo
Melhorar o desempenho do ranking puro e da arquitetura híbrida (ST-GCN + ranking) para atingir P@5 ≥ 0.6 de forma consistente em todos os dias da semana.

---

## 1. Ajustes de Loss e Regularização
- **Loss:**
  - MSELoss (original)
  - BCEWithLogitsLoss com pos_weight ajustável (valores testados: 2, 5, 10)
- **Dropout:**
  - Testados valores 0.1, 0.2, 0.3
- **Épocas:**
  - Testados 100 e 200 epochs
- **Resultados:**
  - Ranking puro melhorou, atingindo P@5 até 0.8 em raros cenários (ex: dia 0), mas na maioria dos dias ficou entre 0.2 e 0.4.
  - Híbrido (0.6 ST-GCN + 0.4 ranking) não superou o ranking puro.

---

## 2. Grid Search de Hiperparâmetros
- **Parâmetros:**
  - pos_weight: 2, 5, 10
  - dropout: 0.1, 0.2, 0.3
  - epochs: 100, 200
- **Critério de parada:**
  - Só considerar ranking útil se P@5 ≥ 0.6
- **Resultados:**
  - Apenas dia 0 atingiu P@5 ≥ 0.6 de forma consistente.
  - Demais dias ficaram abaixo do limiar.

---

## 3. Avaliação do Híbrido
- **Pesos testados:**
  - 0.6 ST-GCN + 0.4 ranking ajustado
- **Resultados:**
  - P@5 híbrido variou entre 0.15 e 0.25, sem ganho relevante sobre o ranking puro.

---

## 4. Diagnóstico e Próximos Passos
- **Diagnóstico:**
  - Ranking ainda não é competitivo na maioria dos dias.
  - Híbrido só será útil quando ranking puro superar 0.6.
- **Próximos passos:**
  1. Expandir features do ranking (estatísticas temporais, tendências, autocorrelação, etc.)
  2. Testar arquiteturas mais profundas e ensemble
  3. Experimentar loss customizadas e validação cruzada
  4. Automatizar ciclo de busca e só parar quando todos os dias atingirem P@5 ≥ 0.6

---

## Observações
- ST-GCN puro segue como baseline até o ranking ser competitivo.
- Todos os scripts, logs e resultados intermediários estão salvos em `outputs/` e `models/ranking_by_day/`.
- Documentação e scripts de avaliação estão em `src/` e `scripts/`.

---

*Atualizado em 12/02/2026*

Recomendações de experimentos avançados:
Features temporais e estatísticas avançadas:

Adicione médias, desvios, tendências, autocorrelação, rolling windows, outliers, etc.
Inclua features de contexto: feriados, eventos, sazonalidade, clima.
Arquitetura do modelo:

Teste redes mais profundas, residual connections, batch/layer normalization extra.
Experimente modelos não-lineares: RandomForest, XGBoost, LightGBM (como baseline rápido).
Loss customizada:

Implemente loss baseada em ranking (ex: pairwise ranking loss, focal loss, hinge loss).
Ajuste o pos_weight dinamicamente conforme a proporção de positivos.
Ensemble/Stacking:

Combine diferentes modelos (MLP, árvore, regressão) via stacking/blending.
Use ensemble de modelos treinados com seeds diferentes.
Data augmentation:

Gere janelas sintéticas, adicione ruído, shuffle temporal, bootstrapping.
Validação cruzada:

Use cross-validation estratificada por dia da semana para evitar overfitting.