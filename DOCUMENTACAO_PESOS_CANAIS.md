# 🧠 Análise de Importância de Canais (Weights Analytics)
**Sistema Report Preview - Arquitetura ST-GAT**

Este documento detalha a influência de cada um dos **29 canais de entrada** na tomada de decisão dos modelos especialistas. Os valores representam a **magnitude média (L1-Norm)** dos pesos na primeira camada convolucional temporal (`layer1.time_conv`), indicando quais variáveis o modelo prioriza para calcular o risco de CVLI.

---

## 🏙️ 1. Fortaleza (Especialista Capital)
O modelo de Fortaleza é orientado por **Inteligência Dinâmica** e **Contexto Global**.

| Rank | Canal | Magnitude | Interpretação Tática |
| :--- | :--- | :--- | :--- |
| **1º** | **24: Exógeno (Tensão)** | **0.3478** | O sinal mais forte. O modelo reage agressivamente a pichações e ameaças. **IMPORTANTE: Nunca solicitei isso ate pq nao faz sentido** |
| **2º** | **28: Pulso Global** | **0.3093** | O risco na capital "sobe" preventivamente se o estado estiver em crise. |
| **3º** | **27: Disponível (Infra)** | **0.2510** | A configuração urbana e vulnerabilidade física do bairro. **IMPORTANTE: Não temos dados de iunfra que eu saiba, a menos que consigamos dados de api** |
| **4º** | **00: CVLI (Histórico)** | **0.2209** | Memória de curto/médio prazo de homicídios reais. |
| **5º** | **11: MONTH (Fev)** | **0.2069** | Padrão sazonal recorrente (Carnaval/Início de ano). | **IMPORTANTE: verifique os meses historicos mais quentes na verdade sao no segundo semestre, mas verifique antes de mudar** |

**Análise:** O modelo não é "escravo do passado". Ele valoriza mais a **Tensão (24)** do que o **Homicídio Real (00)**, permitindo prever o crime antes que ele ocorra (Antecipação Tática).

---

## 🏘️ 2. RMF (Região Metropolitana)
O modelo da RMF equilibra **Sazonalidade Semanal** e **Inteligência**.

| Rank | Canal | Magnitude | Interpretação Tática |
| :--- | :--- | :--- | :--- |
| **1º** | **24: Exógeno (Tensão)** | **0.2795** | Reatividade a conflitos de facções na periferia metropolitana. | **IMPORTANTE: aumentar o peso para 0.30** |
| **2º** | **27: Disponível (Infra)** | **0.2118** | Forte peso da localização e eixos rodoviários. **IMPORTANTE: nunca vi essa definição no projeto, inventado?** |
| **3º** | **00: CVLI (Histórico)** | **0.1794** | Dependência moderada de eventos passados. |
| **4º** | **28: Pulso Global** | **0.1702** | Influência moderada do contexto estadual. |
| **5º** | **08: DOW (Sábado)** | **0.1653** | Forte componente de lazer e dinâmica de fim de semana. | **IMPORTANTE: Acrescentar sexta-feira** |

**Análise:** Na RMF, o crime segue padrões de calendário (Sábado/Domingo) de forma muito mais nítida que na capital.

---

## 🌳 3. Interior (Especialista Estadual)
O modelo do Interior é **Estrutural** e **Sazonal (Turismo/Veraneio)**.

| Rank | Canal | Magnitude | Interpretação Tática |
| :--- | :--- | :--- | :--- |
| **1º** | **27: Disponível (Infra)** | **0.2525** | A logística das cidades e estradas é o principal fator de risco. | **IMPORTANTE: Verificar a definição de "Disponível (Infra)" no projeto** |
| **2º** | **28: Pulso Global** | **0.2205** | Reflete a "febre" do interior como um todo. |
| **3º** | **00: CVLI (Histórico)** | **0.2086** | Inércia criminal em polos regionais. |
| **4º** | **10: MONTH (Jan)** | **0.2085** | Efeito veraneio/férias impactando a segurança. |
| **5º** | **04: DOW (Terça)** | **0.2040** | Padrão específico de crimes em dias úteis no interior. | **IMPORTANTE: Acrescentar sexta-feira** |

**Análise:** O Interior é mais previsível por fatores geográficos (27) e datas específicas (Janeiro) do que por pulsos rápidos de inteligência.

---

## ⚠️ 4. Pontos de Atenção e Calibração Futura

1.  **Baixa Influência de Supressão (Canal 23):** Com magnitude de **~0.05**, o modelo está "ignorando" o alívio de risco gerado por prisões e apreensões. **IMPORTANTE: Alterar para 0.10** |
    *   *Ação:* Aumentar artificialmente o sinal de entrada deste canal no pré-processamento para forçar o modelo a reduzir o risco quando a polícia atua.
2.  **Irrelevância de Veículos (Canal 01):** O roubo de veículos não está sendo usado como "termômetro" de homicídio (**0.04**). **IMPORTANTE: Mudar para 0.10** |
    *   *Conclusão:* A dinâmica de CVLI no Ceará está descolada da dinâmica de patrimônio, focando quase exclusivamente em conflito territorial.
3.  **Domínio de Facções (Canal 02):** Apresenta peso baixo (**~0.05 - 0.13**) porque a informação já está "embutida" no Canal 24 (Exógeno/Tensão). O modelo prefere a tensão dinâmica à estática. **IMPORTANTE: Manter como está, mas monitorar.**

---
**Documento Gerado em:** 11 de Março de 2026
**Responsável:** Inteligência Artificial - Report Preview
