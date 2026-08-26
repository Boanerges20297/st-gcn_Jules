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

### 🔄 Sessão de Validação: 2026-05-27 15:26
**Período Gabarito:** 2026-05-12 a 2026-05-25

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      5       |     5      | 50.0% | 25.0% | 100.0% | 100.0% |   ✅    |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      5       |     4      | 40.0% | 20.0% | 80.0%  | 80.0%  |   ✅    |

---

### 🔄 Sessão de Validação: 2026-06-04 17:58
**Período Gabarito:** 2026-05-19 a 2026-05-30

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      7       |     4      | 40.0% | 25.0% | 57.1%  | 71.4%  |   ✅    |
| RMF       |      2       |     1      | 10.0% | 10.0% | 50.0%  | 100.0% |   🚨    |
| INTERIOR  |      5       |     3      | 30.0% | 20.0% | 60.0%  | 80.0%  |   ⚠️   |

---

### 🔄 Sessão de Validação: 2026-06-09 13:49
**Período Gabarito:** 2026-05-25 a 2026-06-07

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      7       |     3      | 30.0% | 25.0% | 42.9%  | 71.4%  |   ⚠️   |
| RMF       |      1       |     0      | 0.0%  | 5.0%  |  0.0%  | 100.0% |   🚨    |
| INTERIOR  |      4       |     3      | 30.0% | 15.0% | 75.0%  | 75.0%  |   ⚠️   |

---

### 🔄 Sessão de Validação: 2026-06-12 15:32
**Período Gabarito:** 2026-05-29 a 2026-06-11

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      8       |     2      | 20.0% | 20.0% | 25.0%  | 50.0%  |   ⚠️   |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      6       |     4      | 40.0% | 25.0% | 66.7%  | 83.3%  |   ✅    |

---

<!-- validation-session: startup|2026-05-29|2026-06-11|Fortaleza Poisson Ranker + RMF/Interior ST-GAT -->

### 🔄 Sessão de Validação: 2026-06-13 08:42
**Período Gabarito:** 2026-05-29 a 2026-06-11

**Origem:** startup

**Arquitetura:** Fortaleza Poisson Ranker + RMF/Interior ST-GAT

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      8       |     4      | 40.0% | 20.0% | 50.0%  | 50.0%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      5       |     1      | 10.0% | 20.0% | 20.0%  | 80.0%  |   🚨    |

---

<!-- validation-session: startup|2026-05-29|2026-06-11|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-13 08:51
**Período Gabarito:** 2026-05-29 a 2026-06-11

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      8       |     4      | 40.0% | 20.0% | 50.0%  | 50.0%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      5       |     2      | 20.0% | 20.0% | 40.0%  | 80.0%  |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-06-01|2026-06-15|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-16 16:05
**Período Gabarito:** 2026-06-01 a 2026-06-15

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      11      |     7      | 70.0% | 35.0% | 63.6%  | 63.6%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 20.0% | 50.0%  | 100.0% |   ⚠️   |

---

<!-- validation-session: startup|2026-06-01|2026-06-15|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-16 16:06
**Período Gabarito:** 2026-06-01 a 2026-06-15

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      12      |     7      | 70.0% | 35.0% | 58.3%  | 58.3%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 20.0% | 50.0%  | 100.0% |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-06-04|2026-06-15|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-22 15:53
**Período Gabarito:** 2026-06-04 a 2026-06-15

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      9       |     6      | 60.0% | 30.0% | 66.7%  | 66.7%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 20.0% | 50.0%  | 100.0% |   ⚠️   |

---

<!-- validation-session: startup|2026-06-04|2026-06-15|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-22 15:54
**Período Gabarito:** 2026-06-04 a 2026-06-15

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      10      |     6      | 60.0% | 30.0% | 60.0%  | 60.0%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 20.0% | 50.0%  | 100.0% |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-06-09|2026-06-23|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-26 14:21
**Período Gabarito:** 2026-06-09 a 2026-06-23

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      10      |     7      | 70.0% | 40.0% | 70.0%  | 80.0%  |   ✅    |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      5       |     3      | 30.0% | 20.0% | 60.0%  | 80.0%  |   ⚠️   |

---

<!-- validation-session: startup|2026-06-09|2026-06-23|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-06-26 14:26
**Período Gabarito:** 2026-06-09 a 2026-06-23

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      11      |     7      | 70.0% | 40.0% | 63.6%  | 72.7%  |   ✅    |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      5       |     3      | 30.0% | 20.0% | 60.0%  | 80.0%  |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-06-19|2026-06-28|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-07 10:03
**Período Gabarito:** 2026-06-19 a 2026-06-28

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      7       |     6      | 60.0% | 30.0% | 85.7%  | 85.7%  |   ✅    |
| RMF       |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |
| INTERIOR  |      1       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |

---

<!-- validation-session: startup|2026-06-19|2026-06-28|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-07 16:26
**Período Gabarito:** 2026-06-19 a 2026-06-28

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      7       |     6      | 60.0% | 30.0% | 85.7%  | 85.7%  |   ✅    |
| RMF       |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |
| INTERIOR  |      0       |     0      | 0.0%  | 0.0%  |  0.0%  |  0.0%  |   🚨    |

---

<!-- validation-session: merge_new_data|2026-06-26|2026-07-09|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-11 17:02
**Período Gabarito:** 2026-06-26 a 2026-07-09

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      8       |     5      | 50.0% | 30.0% | 62.5%  | 75.0%  |   ✅    |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

<!-- validation-session: startup|2026-06-26|2026-07-09|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-11 17:02
**Período Gabarito:** 2026-06-26 a 2026-07-09

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      8       |     5      | 50.0% | 30.0% | 62.5%  | 75.0%  |   ✅    |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

<!-- validation-session: merge_new_data|2026-07-03|2026-07-16|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-19 09:45
**Período Gabarito:** 2026-07-03 a 2026-07-16

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      9       |     5      | 50.0% | 25.0% | 55.6%  | 55.6%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

<!-- validation-session: startup|2026-07-03|2026-07-16|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-19 09:47
**Período Gabarito:** 2026-07-03 a 2026-07-16

**Origem:** startup

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      9       |     5      | 50.0% | 25.0% | 55.6%  | 55.6%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

<!-- validation-session: startup|2026-07-03|2026-07-16|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-07-20 09:31
**Período Gabarito:** 2026-07-03 a 2026-07-16

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      9       |     5      | 50.0% | 30.0% | 55.6%  | 66.7%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |

---

<!-- validation-session: startup|2026-06-17|2026-07-16|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-07-24 11:06
**Período Gabarito:** 2026-06-17 a 2026-07-16

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      18      |     9      | 90.0% | 55.0% | 50.0%  | 61.1%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      3       |     1      | 10.0% | 10.0% | 33.3%  | 66.7%  |   🚨    |

---

<!-- validation-session: merge_new_data|2026-07-10|2026-07-23|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-24 11:18
**Período Gabarito:** 2026-07-10 a 2026-07-23

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |

---

<!-- validation-session: startup|2026-06-24|2026-07-23|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-07-24 11:18
**Período Gabarito:** 2026-06-24 a 2026-07-23

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      13      |     8      | 80.0% | 40.0% | 61.5%  | 61.5%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 10.0% | 50.0%  | 50.0%  |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-07-13|2026-07-26|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-27 20:37
**Período Gabarito:** 2026-07-13 a 2026-07-26

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      4       |     3      | 30.0% | 15.0% | 75.0%  | 75.0%  |   ⚠️   |
| RMF       |      1       |     1      | 10.0% | 5.0%  | 100.0% | 100.0% |   🚨    |
| INTERIOR  |      3       |     2      | 20.0% | 10.0% | 66.7%  | 66.7%  |   ⚠️   |

---

<!-- validation-session: startup|2026-06-27|2026-07-26|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-07-27 20:38
**Período Gabarito:** 2026-06-27 a 2026-07-26

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      12      |     6      | 60.0% | 35.0% | 50.0%  | 58.3%  |   ✅    |
| RMF       |      3       |     2      | 20.0% | 10.0% | 66.7%  | 66.7%  |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 10.0% | 50.0%  | 50.0%  |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-07-17|2026-07-30|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-07-31 16:29
**Período Gabarito:** 2026-07-17 a 2026-07-30

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      7       |     4      | 40.0% | 20.0% | 57.1%  | 57.1%  |   ✅    |
| RMF       |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |
| INTERIOR  |      3       |     1      | 10.0% | 10.0% | 33.3%  | 66.7%  |   🚨    |

---

<!-- validation-session: startup|2026-07-01|2026-07-30|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-07-31 16:30
**Período Gabarito:** 2026-07-01 a 2026-07-30

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      16      |     7      | 70.0% | 35.0% | 43.8%  | 43.8%  |   ✅    |
| RMF       |      4       |     3      | 30.0% | 15.0% | 75.0%  | 75.0%  |   ⚠️   |
| INTERIOR  |      4       |     2      | 20.0% | 10.0% | 50.0%  | 50.0%  |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-07-22|2026-08-04|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-08-08 11:32
**Período Gabarito:** 2026-07-22 a 2026-08-04

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      8       |     5      | 50.0% | 25.0% | 62.5%  | 62.5%  |   ✅    |
| RMF       |      4       |     3      | 30.0% | 15.0% | 75.0%  | 75.0%  |   ⚠️   |
| INTERIOR  |      2       |     2      | 20.0% | 10.0% | 100.0% | 100.0% |   ⚠️   |

---

<!-- validation-session: startup|2026-07-06|2026-08-04|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-08-08 12:19
**Período Gabarito:** 2026-07-06 a 2026-08-04

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      15      |     8      | 80.0% | 40.0% | 53.3%  | 53.3%  |   ✅    |
| RMF       |      5       |     3      | 30.0% | 15.0% | 60.0%  | 60.0%  |   ⚠️   |
| INTERIOR  |      5       |     3      | 30.0% | 15.0% | 60.0%  | 60.0%  |   ⚠️   |

---

<!-- validation-session: merge_new_data|2026-08-10|2026-08-23|Poisson Ranker Estadual -->

### 🔄 Sessão de Validação: 2026-08-26 13:44
**Período Gabarito:** 2026-08-10 a 2026-08-23

**Origem:** merge_new_data

**Arquitetura:** Poisson Ranker Estadual

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      6       |     4      | 40.0% | 20.0% | 66.7%  | 66.7%  |   ✅    |
| RMF       |      1       |     0      | 0.0%  | 5.0%  |  0.0%  | 100.0% |   🚨    |
| INTERIOR  |      4       |     3      | 30.0% | 20.0% | 75.0%  | 100.0% |   ⚠️   |

---

<!-- validation-session: startup|2026-07-25|2026-08-23|ST-GAT v5 (DeepSTGAT_v5 Ativo) -->

### 🔄 Sessão de Validação: 2026-08-26 13:59
**Período Gabarito:** 2026-07-25 a 2026-08-23

**Origem:** startup

**Arquitetura:** ST-GAT v5 (DeepSTGAT_v5 Ativo)

| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |
|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|
| FORTALEZA |      18      |     8      | 80.0% | 45.0% | 44.4%  | 50.0%  |   ✅    |
| RMF       |      4       |     4      | 40.0% | 20.0% | 100.0% | 100.0% |   ✅    |
| INTERIOR  |      6       |     3      | 30.0% | 20.0% | 50.0%  | 66.7%  |   ⚠️   |

---
