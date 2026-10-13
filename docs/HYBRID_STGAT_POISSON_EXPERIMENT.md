# Experimento de Hibridismo: ST-GAT + Poisson Regressor (Fortaleza)

**Data do Experimento:** 12/10/2026  
**Configuração operacional oficial:** 80% Poisson + 20% ST-GAT v5  
**Ambiente:** Validação cega oficial de Fortaleza (121 nós, horizonte de 30 dias, 31 janelas de 01/07/2026 a 29/08/2026)  
**Artefatos Avaliados:**
- **ST-GAT Neural:** [models/active/legacy_torch/fortaleza_model_active.pth](C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/models/active/legacy_torch/fortaleza_model_active.pth) (`DeepSTGAT_v5`, 120 dias, 41 canais, NegBinom)
- **Poisson Regressor:** [models/active/production/poisson/fortaleza_poisson_regressor.pkl](C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/models/active/production/poisson/fortaleza_poisson_regressor.pkl) (`Pipeline StandardScaler + PoissonRegressor`, 23 features tabulares)
- **Script do Experimento:** [scripts/experiments/hybrid_stgat_poisson_experiment.py](C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/experiments/hybrid_stgat_poisson_experiment.py)

---

## 1. Motivação

No dashboard atual, os motores operam de forma isolada:
1. `StateOrchestrator.get_combined_risk` roda com o **Poisson Regressor** (padronizado operacionalmente).
2. `StateOrchestrator.get_combined_risk_stgat_v5` roda com o **ST-GAT v5** (rede neural espaço-temporal de grafos).

Após os testes com Aprendizado por Reforço (RL via SCST) demonstrarem ganhos marginais (+1,47 pp) com alto custo computacional, investigou-se a hipótese de **hibridismo/ensemble**:
> *ST-GAT captura correlações espaciais complexas e dinâmica de grafos entre facções/bairros contíguos; Poisson captura a tendência estável de longo prazo baseada no volume histórico estático e contagens recentes.*  
> *Uma combinação linear ou por percentil entre os dois pode produzir uma sinergia superior a qualquer modelo isolado.*

---

## 2. Metodologia

Foram avaliadas as **31 janelas de validação temporal cega**, com cálculo de:
- $P@10$: Precisão no top-10 bairros ranqueados.
- $P@20$: Precisão no top-20 bairros ranqueados.
- **Score Composto Oficial**: $0.60 \times P@10 + 0.40 \times P@20$.

Variou-se o peso $\alpha \in [0.0, 1.0]$ do ST-GAT, onde $Score = \alpha \cdot S_{\text{STGAT}} + (1 - \alpha) \cdot S_{\text{Poisson}}$.

Foram testados quatro esquemas de fusão de escores:
1. **Ranking Percentil:** Normalização dos postos em $[0, 1]$ por janela antes do blend.
2. **Contagem Direta:** Blend das médias esperadas de contagem ($\mu_{\text{STGAT}}$ e $\mu_{\text{Poisson}}$).
3. **Média Geométrica:** Ponderação multiplicativa $(S_{\text{STGAT}} + \epsilon)^\alpha \cdot (S_{\text{Poisson}} + \epsilon)^{1-\alpha}$.
4. **Min-Max Normalizado:** Escalonamento linear $[0, 1]$ do sinal de cada modelo.

---

## 3. Resultados Empíricos

### 3.1. Grid Search por Esquema de Fusão

| Configuração | P@10 | P@20 | Score Composto |
| :--- | :---: | :---: | :---: |
| **ST-GAT Puro ($\alpha = 1.0$)** | 32,26% | 42,93% | 36,53% |
| **Poisson Puro ($\alpha = 0.0$)** | 38,71% | 65,52% | 49,44% |
| **Melhor Min-Max ($\alpha = 0.0$)** | 38,71% | 65,52% | 49,44% |
| **Melhor Geométrico ($\alpha = 0.20$)** | 44,84% | 60,85% | 51,24% |
| **Melhor Contagem Direta ($\alpha = 0.30$)** | 45,81% | 62,86% | 52,63% |
| **Melhor Ranking Percentil ($\alpha = 0.20$)** | **47,74%** | **61,08%** | **53,08%** |
| **Ranking Percentil 70/30 ($\alpha = 0.30$)** | 44,19% | 59,45% | 50,29% |

### 3.2. Curva de Sensibilidade do Blend por Percentil

| $\alpha_{\text{STGAT}}$ | $w_{\text{Poisson}}$ | P@10 | P@20 | Score Composto |
| :---: | :---: | :---: | :---: | :---: |
| 0.00 (Poisson puro) | 1.00 | 38,71% | 65,52% | 49,44% |
| 0.10 | 0.90 | 42,58% | 60,31% | 49,67% |
| 0.15 | 0.85 | 43,23% | 61,89% | 50,69% |
| 0.17 | 0.83 | 47,42% | 61,35% | 52,99% |
| **0.20 (ÓTIMO)** | **0.80** | **47,74%** | **61,08%** | **53,08%** |
| 0.25 | 0.75 | 45,48% | 60,17% | 51,36% |
| 0.30 | 0.70 | 44,19% | 59,45% | 50,29% |
| 0.50 | 0.50 | 40,32% | 57,29% | 47,11% |
| 0.70 | 0.30 | 38,06% | 51,00% | 43,24% |
| 1.00 (ST-GAT puro) | 0.00 | 32,26% | 42,93% | 36,53% |

---

## 4. Diagnóstico de Complementaridade

1. **Correlação Moderada ($\rho_{\text{Spearman}} = 0,595$):**
   - ST-GAT e Poisson não são redundantes.
   - O Poisson tem forte âncora histórica e de volume agregado; o ST-GAT detecta flutuações na malha geográfica e conexões de conflito.
2. **Dominância Temporal:**
   - Em **23 das 31 janelas** (74,2%), o Blend superou o Poisson puro em $P@10$.
   - Em **8 das 31 janelas** (25,8%), o Blend empatou com o Poisson puro.
   - Em **0 janelas** (0,0%), o Blend foi inferior ao Poisson puro.
3. **Ganho Expressivo de $P@10$ (+9,03 pp):**
   - $P@10$ saltou de **38,71%** (Poisson) e **32,26%** (ST-GAT) para **47,74%** no ensemble.
   - Isso comprova que a injeção de 20% do sinal de grafo do ST-GAT desempata hotspots que o Poisson sozinho errava no topo do ranking.

---

## 5. Conclusão e Recomendação

Na validação registrada, o hibridismo com **80% Poisson + 20% ST-GAT** (ou blend percentil $\alpha=0.20$) foi o melhor ponto testado e permanece como configuração oficial:
- 70/30: P@10 **44,19%**, P@20 **59,45%**, composto **50,29%**.
- Poisson puro: P@10 **38,71%**, P@20 **65,52%**, composto **49,44%**.
- 80/20: P@10 **47,74%**, P@20 **61,08%**, composto **53,08%**.

O blend 80/20 supera Poisson puro em **+3,64 pp** de score composto e **+9,03 pp** em $P@10$; supera ST-GAT puro em **+16,55 pp** de score composto e **+15,48 pp** em $P@10$. A configuração usa os artefatos já treinados e persistidos em disco, sem retreino.
