# 🛡️ Log de Validação de Dados (Gabarito) - Report Preview

Este arquivo registra a performance das predições do sistema (Champion + Challenger + Orquestrador Regional) confrontadas com os novos dados reais (gabarito) assim que são mesclados na base oficial.

---

## Estrutura de Métricas
- **N_CVLI Bruto:** Total de ocorrências de CVLI confirmadas na nova entrada de dados para a região.
- **Hits Bruto:** Quantidade de bairros/localidades com CVLI real que estavam no Top 10 de risco predito.
- **P@10:** Precisão nos Top 10 (Hits / 10). Responde: "dos 10 preditos, quantos acertei?"
- **P@20:** Precisão nos Top 20 (Hits / 20). Responde: "dos 20 preditos, quantos acertei?"
- **R@10:** Recall nos Top 10 (Hits / N_CVLI Bruto). Responde: "dos crimes reais, quantos cobri no Top 10?"
- **R@20:** Recall nos Top 20 (Hits / N_CVLI Bruto). Responde: "dos crimes reais, quantos cobri no Top 20?"

---

## Histórico de Validação Regional Detalhada

### 🔄 Sessão de Validação: 2026-05-13 07:33
**Período Gabarito:** 2026-05-11 a 2026-05-11

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      1       |     1      | 10.0% |  5.0% | 100.0% | 100.0% |   🚨   |
| RMF       |      1       |     1      | 10.0% |  5.0% | 100.0% | 100.0% |   🚨   |
| INTERIOR  |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |

---

### 🔄 Sessão de Validação: 2026-05-14 21:02
**Período Gabarito:** 2026-05-12 a 2026-05-12

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |
| RMF       |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

### 🔄 Sessão de Validação: 2026-05-15 15:26
**Período Gabarito:** 2026-05-13 a 2026-05-14

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |
| RMF       |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

### 🔄 Sessão de Validação: 2026-05-15 15:48
**Período Gabarito:** 2026-05-01 a 2026-05-14

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      6       |     5      | 50.0% | 25.0% | 83.3%  | 83.3%  |   ✅    |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      5       |     4      | 40.0% | 20.0% | 80.0%  | 80.0%  |   ✅    |

---

### 🔄 Sessão de Validação: 2026-05-22 14:56
**Período Gabarito:** 2026-05-06 a 2026-05-19

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      5       |     3      | 30.0% | 15.0% | 60.0%  | 60.0%  |   ⚠️   |
| RMF       |      1       |     0      | 0.0%  | 5.0%  |  0.0%  | 100.0% |   🚨    |
| INTERIOR  |      5       |     4      | 40.0% | 20.0% | 80.0%  | 80.0%  |   ✅    |

---

### 🔄 Sessão de Validação: 2026-05-22 15:23
**Período Gabarito:** 2026-05-06 a 2026-05-19

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      5       |     3      | 30.0% | 15.0% | 60.0%  | 60.0%  |   ⚠️   |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      5       |     4      | 40.0% | 20.0% | 80.0%  | 80.0%  |   ✅    |

---

### 🔄 Sessão de Validação: 2026-05-25 13:55
**Período Gabarito:** 2026-05-11 a 2026-05-24

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      6       |     2      | 20.0% | 15.0% | 33.3%  | 50.0%  |   ⚠️   |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      6       |     3      | 30.0% | 20.0% | 50.0%  | 66.7%  |   ⚠️   |

---

### ðŸ”„ SessÃ£o de ValidaÃ§Ã£o: 2026-05-25 15:17
**PerÃ­odo Gabarito:** 2026-05-11 a 2026-05-24

| RegiÃ£o    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      6       |     5      | 50.0% | 30.0% | 83.3%  | 100.0% |  âœ…   |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% | âš ï¸ |
| INTERIOR  |      7       |     5      | 50.0% | 25.0% | 71.4%  | 71.4%  |  âœ…   |

---
