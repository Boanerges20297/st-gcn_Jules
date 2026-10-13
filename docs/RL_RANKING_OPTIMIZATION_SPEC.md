# Especificação Técnica: Otimização de Ranking via Aprendizado por Reforço (RL)

**Data de Elaboração / Checkpoint de Referência:** 12/10/2026 19:30:00 (UTC-03:00)  
**Status do Treinamento Base no Momento:** Época 1, Batch ~1145/1493 (`fortaleza_model_active.pth`)  
**Modelo Espaço-Temporal Base:** DeepSTGAT_v5 (121 nós de Fortaleza, janela histórica 120 dias, horizonte 30 dias)  
**Métrica Incumbente Alvo:** Score Composto $S = 0{,}60 \cdot P@10 + 0{,}40 \cdot P@20$ (Piso de referência incumbente: $42{,}73\%$)

---

## 1. Contexto e Motivação Operacional

O pipeline de especialistas neurais utiliza atualmente a loss `NegBinomRankingLoss`, combinando:
1. NLL de Binomial Negativa com dispersão aprendível $r$ (captura overdispersão e inflação de zeros de CVLI).
2. Entropia cruzada listwise (aproximação contínua da distribuição espacial de eventos).
3. Penalidade de indecisão / gap (afasta o top-K da média de predições).
4. Penalidade de *cold streak* / honestidade (evita falsos alertas em nós com longo período inativo).

### O Gargalo
Métricas de corte operacional como $P@10$ e $P@20$ dependem de operadores de ordenação discretos ($\text{topk}$, $\text{argsort}$), que possuem derivada zero quase em toda parte. Enquanto a perda contínua aproxima o ranqueamento geral, pequenos ruídos nas pontuações de bairros intermediários alteram a fronteira do Top-10/Top-20 sem que a função de perda tradicional receba sinal de gradiente específico para o corte.

O objetivo do Aprendizado por Reforço (RL) é permitir a **otimização direta de objetivos discretos com restrição orçamentária de patrulhamento**.

---

## 2. Abordagem Selecionada: Self-Critical Policy Gradient (SCST)

Optou-se pelo **Self-Critical Policy Gradient com amostragem Gumbel-Top-$K$** como estratégia mais eficiente (Filosofia Ponytail / YAGNI):
- **Sem rede de valor separada (sem Critic):** Utiliza a própria predição determinística (gulosa) do modelo como linha de base (*baseline*).
- **Zero overhead em produção:** Na inferência, o ST-GAT continua sendo executado exatamente da mesma forma ($\mu = \text{softplus}(\hat{y}) \rightarrow \text{topk}(10)$).
- **Baixa variância:** Se a amostragem exploratória obtiver a mesma recompensa que a predição gulosa, a vantagem é zero e nenhum gradiente ruidoso é aplicado.

---

## 3. Formulação Matemática

### 3.1 Espaço de Ação e Política
Dado o tensor de entrada $X_t$ e as matrizes de adjacência (geográfica e de conflito), o ST-GAT produz scores não-normalizados $s_i = \text{softplus}(f_\theta(X_t)_i)$ para cada nó $i \in \{1, \dots, N\}$.

A probabilidade de um nó ser escolhido em um ranking de Plackett-Luce é parametrizada via softmax:
$$P(i \mid s) = \frac{\exp(s_i)}{\sum_{j} \exp(s_j)}$$

### 3.2 Amostragem Exploratória (Gumbel-Top-$K$)
Para explorar permutações sem viés:
$$\tilde{s}_i = \log(s_i + \epsilon) + G_i, \quad G_i \sim \text{Gumbel}(0, 1)$$
- **Ação Amostrada ($A_{\text{sample}}$):** Índices dos $K$ maiores valores de $\tilde{s}$.
- **Ação Gulosa / Baseline ($A_{\text{greedy}}$):** Índices dos $K$ maiores valores de $s$ determinísticos.

### 3.3 Função de Recompensa Operacional ($R$)
A recompensa reflete a eficácia da alocação policial no horizonte de 30 dias:
$$R(A) = w_{10} \cdot P@10(A) + w_{20} \cdot P@20(A) + \alpha \cdot \frac{\sum_{i \in A_{10}} y_i}{\sum_j y_j} - \lambda \cdot \text{FP}@10(A)$$

Onde:
- $P@10(A), P@20(A)$: Precisão no Top-10 e Top-20 avaliada sobre os nós ativos com $y > 0$.
- $\frac{\sum_{i \in A_{10}} y_i}{\sum_j y_j}$: Cobertura ponderada de vítimas (prioriza capturar chacinas/eventos múltiplos).
- $\text{FP}@10(A)$: Proporção de bairros no Top-10 que registraram 0 homicídios (penalidade por viatura ociosa).
- Valores de calibração padrão: $w_{10} = 0{,}6$, $w_{20} = 0{,}4$, $\alpha = 0{,}2$, $\lambda = 0{,}1$.

### 3.4 Gradiente de Política com Vantagem Self-Critical
$$\Delta R = R(A_{\text{sample}}) - R(A_{\text{greedy}})$$
$$\mathcal{L}_{\text{RL}}(\theta) = - \Delta R \cdot \sum_{i \in A_{\text{sample}}} \log P(i \mid s)$$

- Se $\Delta R > 0$: A exploração encontrou um conjunto de bairros melhor que o ranking atual $\rightarrow$ aumenta a probabilidade desses nós.
- Se $\Delta R < 0$: A exploração foi pior $\rightarrow$ desencoraja a escolha.
- Se $\Delta R = 0$: Nenhum ajuste desnecessário é realizado.

---

## 4. Estratégia de Treinamento em Duas Fases (Two-Stage Transfer)

```
[Fase 1: Treino Base Supervisionado] (EM ANDAMENTO)
  ├── Entrada: Janelas históricas de 120 dias
  ├── Perda: NegBinomRankingLoss (NLL + Listwise CE + Gap + Honesty)
  ├── Meta: Aprender representação espaço-temporal profunda + dispersão r ≈ 2.0
  └── Saída: Checkpoint com pesos consolidados (fortaleza_model_active.pth)
             │
             ▼
[Fase 2: Fine-Tuning por Reforço - SCST]
  ├── Carga: Carrega os pesos de fortaleza_model_active.pth da Fase 1
  ├── Taxa de Aprendizado: Reduzida (ex: lr = 5e-6)
  ├── Perda Híbrida: L = L_NegBinom + beta * L_RL (com beta = 0.15)
  ├── Épocas: 3 a 5 épocas apenas para ajuste fino na fronteira de corte
  └── Critério de Parada: Early stopping no Score Composto P@10/P@20 da validação
```

---

## 5. Arquitetura do Módulo de Código (`RLRankingModule`)

A implementação planejada pode ser plugada como subclasse ou função utilitária dentro do ecossistema de treinamento:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfCriticalRankingLoss(nn.Module):
    def __init__(self, k1=10, k2=20, w10=0.6, w20=0.4, fp_penalty=0.10, rl_weight=0.15):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.w10 = w10
        self.w20 = w20
        self.fp_penalty = fp_penalty
        self.rl_weight = rl_weight

    def _compute_reward(self, selected_indices, target):
        positive_mask = (target > 0)
        pos_count = int(positive_mask.sum().item())
        if pos_count == 0:
            return 0.0

        target_top10 = set(torch.topk(target, min(self.k1, pos_count)).indices.tolist())
        target_top20 = set(torch.topk(target, min(self.k2, pos_count)).indices.tolist())

        sel10 = set(selected_indices[:self.k1].tolist())
        sel20 = set(selected_indices[:self.k2].tolist())

        p10 = len(sel10 & target_top10) / min(self.k1, pos_count)
        p20 = len(sel20 & target_top20) / min(self.k2, pos_count)

        # Penalidade por falso positivo no Top-10
        fp10 = sum(1 for idx in sel10 if target[idx] == 0) / self.k1

        return (self.w10 * p10) + (self.w20 * p20) - (self.fp_penalty * fp10)

    def forward(self, pred, target, active_mask=None):
        if active_mask is not None and active_mask.sum() > 0:
            pred = pred[active_mask]
            target = target[active_mask]

        scores = F.softplus(pred.squeeze())
        n_nodes = scores.numel()
        if n_nodes < self.k1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        log_probs = F.log_softmax(scores, dim=0)

        # 1. Trajetória Gulosa (Baseline)
        greedy_indices = torch.topk(scores, min(self.k2, n_nodes)).indices
        with torch.no_grad():
            r_greedy = self._compute_reward(greedy_indices, target)

        # 2. Trajetória Amostrada (Exploração Gumbel)
        gumbel = -torch.empty_like(scores).exponential_().log()
        perturbed_scores = (scores + 1e-6).log() + gumbel
        sampled_indices = torch.topk(perturbed_scores, min(self.k2, n_nodes)).indices

        with torch.no_grad():
            r_sample = self._compute_reward(sampled_indices, target)
            advantage = r_sample - r_greedy

        # 3. Policy Gradient Loss
        sampled_log_probs = log_probs[sampled_indices]
        policy_loss = -advantage * sampled_log_probs.sum()

        return self.rl_weight * policy_loss
```

---

## 6. Critérios de Validação e Sucesso

Na etapa pós-treino supervisionado, a validação do fine-tuning via RL deve atender:
1. **Piso de Não-Degradação:** $P@10 \ge 40\%$ e $P@20 \ge 45\%$ no conjunto de validação de 31 janelas.
2. **Estabilidade de Calibração:** Preservação do erro de contagem ($MAE \le 1{,}50$).
3. **Redução de Falsos Positivos Críticos:** Bairros com probabilidade zero na janela não podem figurar no Top-10 em favor de bairros moderadamente ativos.

---

## 7. Como Executar o Fine-Tuning RL

A implementação já foi integrada ao script oficial `scripts/training/Active/train_all_specialists.py`.

Para executar o fine-tuning RL sobre o checkpoint de Fortaleza:

```powershell
.\.venv\Scripts\python.exe scripts/training/Active/train_all_specialists.py --region fortaleza --fine-tune-rl
```

Esse comando:
1. Carrega os pesos de `models/active/legacy_torch/fortaleza_model_active.pth` como warm-start.
2. Ativa o `SelfCriticalRankingLoss` com taxa de aprendizado suave (`lr = 5e-6`) e escalonador estático.
3. Avalia cada época contra o piso incumbente de validação ($42{,}73\%$).
4. Se houver novo recorde composto, salva o modelo atualizado preservando compatibilidade total com o pipeline de produção.

---

## 8. Resultados da Primeira Rodada Experimental de RL (12/10/2026)

A execução do comando acima completou 4 épocas completas (`rl_patience=4`), encerrando às 22:47:58 via early stopping com os seguintes resultados na validação (31 janelas de 60 dias):

| Época | Val P@10 | Val P@20 | Score Composto | Loss Treino | Dispersão $r$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **E001** | 33,23% | 44,65% | 37,80% | 7,0130 | 1,967 |
| **E002** | 34,19% | 44,42% | 38,28% | 6,9699 | 1,933 |
| **E003** | 34,84% | 44,36% | 38,65% | 6,8999 | 1,901 |
| **E004** | **36,13%** | 43,99% | **39,27%** | 6,8934 | 1,870 |

### Conclusões da Rodada e Veredito Operacional:
1. **Ganho Marginal e Insuficiente:** Embora o $P@10$ tenha tido ligeira subida (de $33{,}23\%$ para $36{,}13\%$), o score composto final estagnou em **$39{,}27\%$**, ficando quase 3,5 pontos percentuais abaixo do baseline supervisionado incumbente ($42{,}73\%$).
2. **Custo Computacional Desproporcional:** Foram necessárias mais de 1h30 de processamento para um ganho prático mínimo, sem superação do modelo em produção.
3. **Decisão Operacional (Princípio Ponytail / YAGNI):** O Aprendizado por Reforço (RL) adiciona complexidade e pontos de falha que não compensam o retorno marginal obtido. O RL é formalmente **descartado como caminho principal de otimização de ranking** do projeto, mantendo-se a arquitetura estável oficial baseada no ST-GAT com Negative Binomial supervisionada padrão + Regressores Poisson calibrados (`scripts/promote_statewide_poisson_regressors.py`). O código e esta especificação permanecem arquivados apenas como registro de benchmarking e rastreabilidade técnica.
