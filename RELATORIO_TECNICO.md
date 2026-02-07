# Relatório Técnico: Eficiência dos Modelos e Arquitetura

**Data:** 25 de Fevereiro de 2026
**Autor:** Jules (Assistente de Engenharia de Software)
**Assunto:** Análise de Eficiência, Performance e Aderência à Realidade

---

## 1. Sumário Executivo

A arquitetura atual, baseada em **ST-GCN (Spatio-Temporal Graph Convolutional Networks)**, demonstra **alta eficiência computacional** (baixa latência), permitindo inferência em tempo real mesmo em CPUs. No entanto, a **aderência à realidade** (precisão na identificação de hotspots) está limitada (P@5 ~31%), prejudicada principalmente pela qualidade do grafo espacial (geometria de pontos vs polígonos) e pela falta de variáveis dinâmicas exógenas.

O sistema opera com um "gap de realidade" significativo: ele "vê" a cidade como uma nuvem de pontos estáticos conectados por linhas retas (distância euclidiana), ignorando a malha viária, barreiras físicas e a dinâmica fluida das facções criminosas.

---

## 2. Diagnóstico Técnico

### 2.1. Métricas de Performance (Modelo `stgcn_model_v2.pth`)

A avaliação realizada no conjunto de validação (últimos 20% da série temporal) apresentou os seguintes resultados:

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **MSE (Erro Quadrático Médio)** | **0.5673** | Erro de magnitude moderado, considerando a esparcidade dos dados. |
| **RMSE (Raiz do EQM)** | **0.7523** | Em média, o modelo erra por ~0.75 crimes/dia por área. |
| **MAE (Erro Absoluto Médio)** | **0.6800** | Erro absoluto robusto a outliers. |
| **Precision@5 (P@5)** | **31.30%** | **Crítico.** De cada 5 áreas apontadas como "Top Risco", apenas ~1.5 realmente confirmam o risco alto. |
| **Latência de Inferência** | **12.99 ms** | **Excelente.** O sistema é extremamente rápido e leve. |

### 2.2. Eficiência Computacional

*   **Arquitetura:** Leve. O uso de camadas convolucionais gráficas (GCN) é muito mais eficiente que abordagens puramente visuais (CNNs em grids) ou recorrentes profundas (LSTMs densas).
*   **Escalabilidade:** O modelo escala linearmente com o número de nós (atualmente 319). Suporta expansão para ~1000 nós sem degradação perceptível de latência.

---

## 3. Análise de Arquitetura e Dados

### 3.1. O Modelo (ST-GCN)
*   **Pontos Fortes:** Captura bem a dependência temporal (histórico de 30 dias) e espacial (vizinhos próximos). A camada de *Temporal Attention* ajuda a focar nos dias mais recentes.
*   **Pontos Fracos:** A matriz de adjacência é **estática**. Se uma facção toma um território, o modelo não "sabe" que a conexão mudou até que um novo treino ocorra ou a matriz seja recarregada.

### 3.2. Os Dados (Feature Engineering)
*   **Features (26 Canais):** CVLI, CVP, Tensão, Calendário (Dia da semana, Mês).
    *   *Crítica:* O canal de "Tensão" é estático (baseado em arquivos GeoJSON fixos). A realidade das ruas muda semanalmente.
*   **Geometria:** O sistema falhou em carregar polígonos (`0/319` polígonos atribuídos), operando puramente com centróides (Pontos).
*   **Atribuição de Ocorrências:**
    *   **99.6%** das ocorrências foram atribuídas via **KDTree (Proximidade Espacial)**.
    *   Apenas **0.4%** foram atribuídas por match exato de nome de Bairro/Cidade.
    *   *Impacto:* Isso gera "ruído espacial". Uma ocorrência na borda de um bairro pode ser atribuída ao bairro vizinho errado apenas por proximidade geométrica do centróide.

---

## 4. Análise de "Realidade" (O Gap)

O modelo atual é uma **abstração matemática eficiente**, mas distante da **realidade física e social**.

1.  **A "Cidade Abstrata" vs "Cidade Real":**
    *   O modelo usa distância euclidiana (raio de 2km). Na realidade, dois bairros podem ser vizinhos visuais mas separados por uma rodovia ou muro, impedindo o fluxo de crime.
    *   A falta de polígonos impede o cálculo preciso de densidade e fronteiras.

2.  **Cegueira Contextual:**
    *   O modelo não sabe se está chovendo (reduz crime de rua), se há um jogo de futebol (concentra multidões) ou se houve uma operação policial ontem. Ele olha apenas para o passado estatístico.

3.  **Dinâmica de Facções:**
    *   A "Guerra" é modelada como uma matriz estática (`adj_conflict`). Na rua, alianças e disputas mudam rapidamente. O modelo reage *lento demais* a essas mudanças.

---

## 5. Plano de Ação: Aproximando-se da Realidade

Para elevar a precisão (P@5) de 30% para patamares operacionais (>60%) sem sacrificar a eficiência:

### Curto Prazo (Imediato)
1.  **Correção Geométrica:** Garantir a ingestão dos arquivos `.geojson` de Bairros e Municípios para que os nós tenham área e fronteiras reais, não apenas pontos.
2.  **Refinamento de Atribuição (NLP):** Melhorar o algoritmo de *matching* de nomes para reduzir a dependência do KDTree espacial. Usar bibliotecas de similaridade de texto (Fuzzy Matching) para ligar "Bairro X" da ocorrência ao "Bairro X" do grafo.
3.  **Peso na Recência:** Aumentar o peso da *Loss Function* para erros em dias recentes (D-1, D-2), forçando o modelo a priorizar o "agora" sobre o "mês passado".

### Médio Prazo (Estratégico)
1.  **Graph Attention Networks (GATv2):** Substituir a GCN estática por GAT. Isso permite que o modelo aprenda *quais* vizinhos são importantes dinamicamente a cada dia, simulando a fluidez das fronteiras do crime.
2.  **Variáveis Exógenas Dinâmicas:** Injetar dados de **Clima (Chuva/Temp)** e **Calendário de Eventos (Feriados, Jogos)** como features globais ou nodais.
3.  **Grafo Viário:** Substituir a distância euclidiana por distância de rede (tempo de deslocamento via ruas) para definir a matriz de adjacência `adj_geo`.

### Longo Prazo (Visão de Futuro)
1.  **Feedback Loop em Tempo Real:** Permitir que operadores corrijam predições na interface, re-treinando o modelo incrementalmente ("Human-in-the-loop").
2.  **Simulação Multi-Agente:** Criar um "Digital Twin" simplificado onde agentes (polícia, facções) interagem sobre o grafo para simular cenários "E se?".

---

**Conclusão:** O sistema é robusto e rápido, mas precisa ser "conectado" à geografia real e ao contexto dinâmico da cidade para se tornar uma ferramenta tática de precisão.
