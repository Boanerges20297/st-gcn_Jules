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

### 3.3. Gatilhos de Alerta e Comunicação
O sistema categoriza o risco em quatro níveis visuais, utilizando linguagem gerencial para facilitar a compreensão:

*   🔴 **CRÍTICO (> 80%):** Ação imediata recomendada.
    *   *Descrição:* "Tendência de agravamento recente".
*   🟠 **ALTO (> 60%):** Alerta elevado.
    *   *Descrição:* "Valor histórico alto para o período" (quando estável) ou alta probabilidade preditiva.
*   🔵 **MÉDIO (> 20%):** Atenção.
    *   *Descrição:* "Manutenção do padrão de risco médio".
*   🟢 **BAIXO (<= 20%):** Estabilidade.
    *   *Descrição:* "Estabilidade (Baixo Risco)".

> **Nota:** O sistema não exibe mais percentuais estatísticos complexos nos motivos, focando em descrições qualitativas diretas.

## 4. Funcionalidades Dinâmicas

### 4.1. Inserção de Dados Exógenos (IA Generativa)
O sistema permite a ingestão direta de relatórios não estruturados da CIOPS (Coordenadoria Integrada de Operações de Segurança) e utiliza **Inteligência Artificial Generativa (Google Gemini)** para estruturar e localizar os eventos.

**Fluxo:**
1.  O operador cola o texto bruto das ocorrências (ex: despachos de rádio, relatórios de campo).
2.  **Processamento LLM (Gemini 1.5):**
    *   O sistema envia o texto para a nuvem do Google Gemini (requer chave de API configurada).
    *   A IA extrai e estrutura: **Natureza**, **Localização Completa**, **Bairro** e **Município**.
3.  **Geolocalização Inteligente (Hierárquica):**
    *   *Nível 1 (Endereço Completo):* Busca correspondência do endereço específico na malha viária/nós.
    *   *Nível 2 (Bairro):* Se falhar, utiliza o bairro extraído pela IA para centralizar no nó correspondente.
    *   *Nível 3 (Município):* Em último caso, centraliza na sede do município identificado.
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
