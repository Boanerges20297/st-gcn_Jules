# 🧠 Guia de Canais e Inteligência: Sistema Report Preview (ST-GAT)

O Sistema Report Preview opera utilizando um tensor de entrada de **26 canais**, onde cada canal representa uma "camada de realidade" diferente capturada pelo modelo de Redes Neurais em Grafo (ST-GAT).

---

## 📊 1. Mapa Geral de Canais (Tensores)

O cérebro do Report Preview processa 26 informações simultâneas para cada bairro, todos os dias:

| Canal | Nome | Tipo | Descrição |
| :--- | :--- | :--- | :--- |
| **0** | **CVLI** | Histórico | Ocorrências reais de crimes letais (Homicídios). |
| **1** | **CVP** | Histórico | Crimes contra o patrimônio (Roubos/Furtos). |
| **2** | **Tensão** | Estrutural | Índice estático de vulnerabilidade e presença de facções. |
| **3-9** | **DOW** | Sazonal | Dia da semana (Segunda a Domingo). |
| **10-21** | **MONTH** | Sazonal | Mês do ano (Janeiro a Dezembro). |
| **22** | **Weekend** | Sazonal | Flag de final de semana (Sexta noite a Domingo). |
| **23** | **Supressão** | **Dinâmico** | **Ação Policial Positiva (Alívio de Risco).** |
| **24** | **Exógeno** | **Dinâmico** | **Eventos de Tensão Padrão (Alertas comuns).** |
| **25** | **Crítico** | **Dinâmico** | **Alertas de Alta Periculosidade (Rupturas).** |

---

## 🛡️ 2. Canais de Inteligência em Tempo Real (23, 24, 25)

Estes são os canais mais importantes para a decisão tática, pois são alimentados pelo rádio (CIOPS) e Inteligência.

### 🟢 Canal 23: Supressão (O "Alívio")
Este canal sinaliza ao modelo que a polícia está atuando fortemente na área. Ele gera um contrapeso ao risco.
*   **Apreensão de Fuzil/Metralhadora:** Intensidade **1.0** (Máximo alívio).
*   **Prisão de Liderança (Torre/Frente):** Intensidade **0.9**.
*   **Recuperação de Veículos/Drogas:** Intensidade **0.4 - 0.6**.

### 🟡 Canal 24: Exógeno Padrão (A "Tensão")
Eventos que aumentam a vigilância, mas não indicam guerra iminente.
*   Pichações de novas facções, ameaças em redes sociais, movimentações suspeitas de rotina.

### 🔴 Canal 25: Exógeno Crítico (O "Choque")
Eventos que possuem alto poder de gerar retaliação ou morte imediata.
*   **Keywords Gatilho:** *"Execução"*, *"Facção"*, *"Chacina"*, *"Tortura"*.
*   **Transferência de Líderes:** Sempre gera um pulso de risco máximo no modelo.

---

## ⚙️ 3. Funcionamento Matemático (O fluxo do Risco)

O Report Preview não apenas "soma" os crimes. Ele processa a informação em 3 etapas:

### Passo A: Injeção Espacial
Quando um evento entra no **Canal 25** no bairro **BOM JARDIM**, o modelo ativa a **Atenção em Grafo (GAT)**.
*   **Efeito Dominó:** O risco sobe no Bom Jardim e "vaza" para **Granja Lisboa** e **Granja Portugal** automaticamente, pois o modelo sabe que são vizinhos e compartilham dinâmicas criminais.

### Passo B: Normalização Robusta (Z-Score + Sigmoide)
Para evitar que um único bairro "esmage" o ranking, usamos uma curva não-linear:
1.  **Z-Score:** Mede o quão fora do comum está aquele bairro em relação à média do estado.
2.  **Sigmoide:** Comprime o valor para uma curva suave, garantindo que o Top 10 seja bem distribuído.
3.  **Dampening (Amortecimento):** Aplicamos a fórmula `50 + (raw - 50) * 0.85` para evitar que o dashboard fique saturado em 100% sem necessidade real.

### Passo C: O "Arquivo Morto" (Cleanup)
Para manter o modelo focado no **Horizonte de 7 dias**, o sistema executa uma limpeza automática:
*   Eventos com mais de 7 dias são movidos para o arquivo `exogenous_events_(data).json`.
*   Isso garante que o Canal 25 "esvazie", permitindo que o risco de um bairro caia se a poeira baixar.

---

## 🚦 4. Réguas de Decisão (Status)

O Dashboard classifica o risco final conforme as faixas definidas para operação tática:

*   💀 **CRÍTICO (>= 90%):** Área em ruptura. Necessidade de intervenção imediata e saturação de área.
*   ⚠️ **ALTO (80% - 89%):** Área em pré-conflito ou com histórico extremamente instável.
*   🔍 **MODERADO (50% - 79%):** Área sob monitoramento preventivo (Atenção).
*   ✅ **BAIXO (< 50%):** Área dentro da normalidade estatística.

---

## 📋 5. Estatísticas de Performance
No sidebar, o gestor acompanha a saúde do modelo:
*   **Pressão nos Hotspots:** Média de perigo no Top 5.
*   **Calor do Estado:** Nível de febre criminal de todo o Ceará.
*   **Precisão de Captura:** Quão nítido está o sinal de risco hoje (Confiança).

---
**Autor:** Inteligência Report Preview / Gemini CLI
**Versão:** 2.0 (Arquitetura Estadual Unificada)
**Última Atualização:** 16 de Fevereiro de 2026
