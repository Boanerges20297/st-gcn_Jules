# 🔄 Documentação do Fluxo Completo: Sistema Report Preview

Esta documentação detalha o fluxo de dados "end-to-end" do sistema Report Preview, desde a ingestão de ocorrências brutas e inteligência em tempo real até a geração de previsões de risco e visualização no dashboard.

---

## 📐 Visão Geral da Arquitetura

O sistema opera sob uma arquitetura híbrida que combina **Deep Learning em Grafos (ST-GAT)** para análise de tendências espaço-temporais e **Large Language Models (LLM - Google Gemini)** para processamento de linguagem natural e geração de relatórios táticos.

### O Fluxo em 5 Etapas:
1.  **Ingestão de Dados:** Coleta de dados históricos e eventos em tempo real (CIOPS).
2.  **Engenharia de Features:** Transformação de dados em tensores espaço-temporais (29 canais).
3.  **Orquestração Regional:** Processamento segregado por dinâmicas regionais (Fortaleza, RMF, Interior).
4.  **Inferência & Refinamento:** Execução do modelo neural e aplicação de viés tático (TAG-Bias).
5.  **Apresentação & Explicação:** Visualização de risco e geração de justificativas em linguagem natural.

---

## 1. Ingestão de Dados (Data Ingestion)

O sistema consome três tipos primários de dados:

### A. Dados Históricos e Estruturais (Base Fria)
*   **Crime Data:** `data/processed/dados_status_ocorrencias_gerais_ENRIQUECIDO.json`
    *   Contém o histórico de CVLIs (Crimes Violentos Letais Intencionais), roubos, etc.
*   **Dados Geoespaciais:** Arquivos GeoJSON (`data/processed/*.geojson`)
    *   Definem as fronteiras dos bairros e as zonas de influência de facções (CV, GDE, etc.).
*   **Metadados:** Informações estáticas sobre vulnerabilidade social e infraestrutura.

### B. Dados de Inteligência em Tempo Real (Base Quente)
*   **Fonte:** Mensagens de rádio da polícia (CIOPS), denúncias ou relatórios de campo.
*   **Entrada:** Endpoint API `/api/exogenous/parse`.
*   **Processamento LLM:** O texto bruto é enviado ao Google Gemini (`src/llm_service.py`) que extrai:
    *   **Localização:** Bairro/Município.
    *   **Tipo de Evento:** Tiroteio, Homicídio, Pichação, Invasão.
    *   **Criticidade:** Classificação automática para os canais de "Tensão" ou "Choque".
*   **Armazenamento:** Salvo em `data/exogenous_events.json`.

---

## 2. Processamento e Engenharia de Features

Antes de alimentar a rede neural, os dados são transformados em tensores multidimensionais.

*   **Responsável:** `src/core/data_processing.py`
*   **Estrutura do Tensor:** `(Batch, Time_Steps, Nodes, Channels)`
    *   **Nodes:** Bairros ou localidades (Nós do grafo).
    *   **Time_Steps:** Janela de observação histórica (ex: últimos 30 dias).
    *   **Channels (29 canais):**
        *   `0-2`: Crimes históricos (CVLI, Veículos) e Tensão Estática.
        *   `3-22`: Features Sazonais (Dia da semana, Mês, Fim de semana).
        *   `23-25`: **Canais Dinâmicos** (Supressão Policial, Tensão Exógena, Eventos Críticos).
        *   `26-28`: Canais Livres/Globais para calibração futura.

### Construção do Grafo (Adjacency Matrix)
O sistema calcula uma matriz de adjacência `A` que define como o risco flui entre os bairros.
*   **Conexão Física:** Bairros vizinhos geograficamente têm peso 1.
*   **Conexão Semântica:** Bairros dominados pela mesma facção ou facções rivais podem ter conexões reforçadas (configurável).

---

## 3. Orquestração Regional (State Orchestrator)

Para aumentar a precisão, o Ceará não é tratado como um monólito. O `StateOrchestrator` (`src/core/orchestrator.py`) divide o problema:

1.  **Especialista Fortaleza:** Modelo treinado na dinâmica urbana densa e conflito de facções intenso.
2.  **Especialista RMF:** Focado na Região Metropolitana, onde a dinâmica é híbrida.
3.  **Especialista Interior:** Focado em grandes distâncias e manchas criminais dispersas.

Cada região possui seu próprio modelo `.pth` carregado em `models/active/`.

---

## 4. Inferência e TAG-Bias (O Coração do Modelo)

Quando o usuário solicita uma previsão (`/api/risk`), o seguinte processo ocorre:

### A. Carregamento e Injeção
O Orchestrator carrega os últimos dados conhecidos e "injeta" os eventos recentes do arquivo `exogenous_events.json` diretamente nos canais 23, 24 e 25 dos tensores de entrada.

### B. Execução da Rede Neural (Deep ST-GAT)
A arquitetura `DeepSTGAT` (`src/core/architectures.py`) processa o tensor.
*   **Camadas Temporais:** Analisam a sequência de dias para detectar tendências de alta/baixa.
*   **Camadas Espaciais (GAT):** Analisam como o risco de um bairro influencia seus vizinhos (mecanismo de atenção).

### C. Pós-Processamento e TAG-Bias
A saída bruta do modelo (raw logits) passa por refinamentos:
1.  **TAG-Bias (Trigger Alert Graph Bias):** Se houve um evento de "Choque" (Canal 25) nas últimas 24h, o sistema força um aumento no score de risco, independentemente da previsão histórica do modelo. Isso garante reatividade imediata a crises.
2.  **Normalização (Z-Score + Sigmoid):** Os valores são normalizados estatisticamente para garantir uma distribuição legível (0% a 100%).
3.  **Amortecimento (Dampening):** Evita alarmismo excessivo, suavizando picos extremos que não sejam sustentados por múltiplos fatores.

---

## 5. Geração de Explicações (Explainability)

O sistema não apenas diz "Onde", mas "Por que".

*   **Endpoint:** `/api/explain`
*   **Análise Técnica:** O `ExplanationGenerator` (`src/explanation_generator.py`) analisa os gradientes e valores de entrada para identificar o que mais contribuiu para o risco:
    *   Foi a tendência histórica? (Série temporal subindo).
    *   Foi o contágio vizinho? (Bairros adjacentes com alto risco).
    *   Foi um evento exógeno? (Ataque recente registrado).
*   **Tradução Gerencial (LLM):** Os dados técnicos são enviados ao Gemini com um prompt especializado para gerar um texto como:
    > *"Alta probabilidade de conflito no bairro Bom Jardim devido a ruptura recente no pacto local e atividade intensa em bairros vizinhos (Granja Lisboa)."*

---

## 6. Apresentação (Dashboard)

A interface final (`templates/index.html`) consome os JSONs gerados:

*   **Mapa Interativo:** Renderiza os bairros coloridos pela escala de risco (Verde -> Vermelho).
*   **Sidebar de Métricas:** Exibe a "Temperatura do Estado" e alertas imediatos.
*   **Feedback Loop:** O sistema permite que operadores validem as previsões, gerando dados para re-treino futuro.

---

**Arquivos Chave no Fluxo:**
- `app.py`: API Gateway.
- `src/core/orchestrator.py`: Lógica de controle e especialistas.
- `src/core/data_processing.py`: ETL e tensores.
- `src/core/architectures.py`: Definição da Rede Neural.
- `src/llm_service.py`: Interface com IA Generativa.
