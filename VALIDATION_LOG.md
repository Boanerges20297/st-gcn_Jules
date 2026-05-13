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
