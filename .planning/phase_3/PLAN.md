# Plano de Fase 3: Upgrade Neural do Champion (ST-GAT)

## 🎯 Objetivo
Elevar a performance do modelo principal (**Champion**) para o patamar do Challenger, corrigindo a inércia temporal e expandindo a base de conhecimento estrutural para 4 anos.

## 🛠️ Intervenções Técnicas

### 1. Context Sensing (Recency Bias) no PyTorch
Diferente do LightGBM, o PyTorch não possui `sample_weight` nativo no `.fit()`. Vamos implementar manualmente no loop de treino de `train_all_specialists.py`:
- Calcular um multiplicador exponencial de gradiente baseado na recência da janela.
- `weight = exp(idx / tau)`, onde janelas de 2026 terão peso máximo.

### 2. Expansão de Memória (4 Anos)
- Alterar o carregamento de dados para iniciar em **Janeiro de 2022**.
- Permitir que o GAT aprenda padrões de longo prazo (AIS/Bairros) enquanto foca no "calor" recente via pesos.

### 3. Ajuste de Reatividade (LR & Scheduler)
- Aumentar o Learning Rate base para `0.001` (atualmente `0.0003`).
- Utilizar `CosineAnnealingWarmRestarts` para forçar a saída de mínimos locais de "estagnação histórica".

### 4. Expansão de Capacidade (IQ do Modelo)
- Aumentar o número de **Cabeças de Atenção para 16** (atualmente 4-8).
- Foco em extrair sinais mais complexos dos 4 anos de dados.

## 📋 Critérios de Aceite (UAT)
- [ ] Treino concluído com 4 anos de dados.
- [ ] P@20 (Validação Sombra) >= 50% (Atualmente 38%).
- [ ] Gradientes estáveis (sem explosão por causa dos pesos).

## 🚀 Execução
1. Modificar `scripts/training/Active/train_all_specialists.py`.
2. Iniciar treino do especialista de Fortaleza.
3. Validar métricas contra o Sentinela V3.
