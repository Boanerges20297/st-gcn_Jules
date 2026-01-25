# Documentação Técnica e Operacional - Sistema de Inteligência Preditiva

## 1. Visão Geral e Objetivos

Este documento descreve a arquitetura, fluxo operacional e capacidades do Sistema de Inteligência Preditiva desenvolvido para apoio à decisão tática em segurança pública.

O objetivo primário do sistema é antecipar manchas criminais e identificar áreas de alto risco para Crimes Violentos Letais Intencionais (CVLI) e Crimes Violentos Contra o Patrimônio (CVP) em um horizonte de curto prazo (24h a 72h).

O sistema integra análise histórica de longo prazo, padrões espaço-temporais aprendidos por Inteligência Artificial (ST-GCN) e dados exógenos em tempo real para fornecer uma consciência situacional dinâmica.

## 2. Arquitetura do Modelo (IA)

O núcleo preditivo utiliza uma **Rede Neural Convolucional em Grafos Espaço-Temporais (ST-GCN)**.

*   **Grafos:** A cidade é modelada como um grafo onde cada nó representa uma área monitorada (bairro/comunidade) e as arestas representam a proximidade geográfica.
*   **Convolução Espacial:** Permite que o modelo entenda como o risco em uma área influencia seus vizinhos (efeito de contágio/difusão).
*   **Convolução Temporal:** Analisa a série histórica de cada nó para capturar tendências, sazonalidade e padrões recorrentes.

### Configuração dos Modelos
Existem dois modelos distintos operando em paralelo:

1.  **Modelo CVLI (Crimes Violentos):**
    *   **Input:** Janela histórica de 180 dias.
    *   **Output:** Previsão de risco para 3 dias (média móvel).
    *   **Justificativa:** Padrões de violência letal tendem a ter ciclos mais longos e dependências históricas complexas (ex: retaliações).

2.  **Modelo CVP (Patrimônio/Drogas):**
    *   **Input:** Janela histórica de 30 dias.
    *   **Output:** Previsão de risco para 1 dia.
    *   **Justificativa:** Crimes de oportunidade e tráfico respondem a dinâmicas mais imediatas e deslocamentos rápidos.

## 3. Fluxo de Funcionamento e Dados

### 3.1. Ingestão de Dados
O sistema processa dados brutos provenientes de:
1.  **Polígonos (GeoJSON):** Definição geográfica das áreas de interesse (Territórios, AIS, Bairros).
2.  **Ocorrências (CSV/JSON):** Histórico de eventos criminais normalizados.

### 3.2. Normalização de Risco
O sistema aplica uma normalização híbrida sobre as predições brutas da IA:
*   **Score Base:** Normalização linear (0-100) das saídas do modelo.
*   **Sensibilidade (Boost):** Aplicação de multiplicadores (1.5x) para garantir que sinais fracos em áreas críticas não sejam ignorados.
*   **Histórico Mínimo:** Áreas com atividade criminal recente recebem um "piso" de risco (mínimo 25-30%) para evitar falsos negativos (risco zero) em zonas ativas.

### 3.3. Gatilhos de Alerta (Triggers)
O sistema categoriza o risco em quatro níveis visuais:
*   🔴 **CRÍTICO (> 80%):** Ação imediata recomendada. Saturação tática.
*   🟠 **ALTO (> 60%):** Alerta elevado. Patrulhamento direcionado.
*   🔵 **MÉDIO (> 20%):** Atenção. Monitoramento padrão.
*   🟢 **BAIXO (<= 20%):** Estabilidade.

## 4. Funcionalidades Dinâmicas

### 4.1. Inserção de Dados Exógenos (Novo)
O sistema permite a ingestão direta de "blocos" de texto da CIOPS (Coordenadoria Integrada de Operações de Segurança) para reavaliação instantânea do risco.

**Fluxo:**
1.  O operador cola o texto bruto das ocorrências (ex: despachos de rádio, relatórios de campo).
2.  **Processamento NLP:** O sistema utiliza Regex e heurísticas para extrair:
    *   Natureza do evento.
    *   Localização (Bairro, Rua, AIS).
3.  **Geolocalização:**
    *   Tenta associar a localização a um nó conhecido do grafo (Área monitorada/Bairro).
    *   *Fallback 1 (Bairros de Fortaleza):* Se não encontrar no grafo, busca em uma base estática de bairros oficiais de Fortaleza (IBGE).
    *   *Fallback 2 (Municípios do Ceará):* Se não encontrar o bairro, busca na lista de 184 municípios do Ceará (IBGE) e utiliza as coordenadas da sede municipal.
    *   *Fallback 3 (Geométrico):* Último recurso, calcula o centróide geométrico dos nós pertencentes à cidade detectada (se houver correspondência parcial de nome).
4.  **Reavaliação (Simulação):**
    *   Os eventos são tratados como "Conflitos Ativos".
    *   O sistema simula uma "explosão" de risco nos nós afetados, aumentando artificialmente a conectividade e o score de risco para refletir a instabilidade em tempo real.

### 4.2. Simulação de Cenários
O painel permite ao gestor simular intervenções:
*   **Supressão (Equipe Tática):** Simula a presença de policiamento. Reduz drasticamente o risco na área e bloqueia a difusão para vizinhos (isolamento do nó no grafo).
*   **Conflito (Exógeno):** Simula um ataque ou disputa. Amplifica o risco e a difusão para áreas vizinhas.

## 5. Requisitos Técnicos
*   **Backend:** Python (Flask), PyTorch (Incrência), GeoPandas (Geoprocessamento).
*   **Frontend:** HTML5, Bootstrap 5, Leaflet JS (Mapas Interativos).
*   **Infraestrutura:** Requer suporte a operações vetoriais (NumPy/Torch) e memória suficiente para manter o grafo da cidade carregado (~2000 nós).

---
*Documento gerado para fins de apresentação técnica e validação de requisitos.*
