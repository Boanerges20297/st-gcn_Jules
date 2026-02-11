# Documentação Detalhada da Aplicação ST-GCN Jules

---

## 1. Arquitetura Geral

- **Núcleo**: `app.py` (Flask), módulos auxiliares em `src/` e `analysis/`.
- **Modelos**: ST-GCN v2 (Graph Convolutional Network temporal) e RankingModel (MLP para ranking).
- **Combinação de scores**: 70% ST-GCN + 30% RankingModel (ajustável).
- **Pipeline**: Dados brutos → pré-processamento → extração de features → matriz de adjacência → inferência → scores → classificação de risco.

---

## 2. Dados e Tipos de Dados

- **Entradas**:
  - Dados de crimes (CVLI, CVP, etc.), eventos exógenos, datas, localização (nós/microrregiões).
  - Features: 26 canais (ex: dia da semana, mês, tensões, derivados).
  - Matrizes de adjacência (grafo de vizinhança).
  - Dados de facção (GeoJSON, KML, CSV).
- **Tipos**:
  - Numéricos (contagens, scores, ranks)
  - Categóricos (tipo de crime, facção)
  - Temporais (timestamp, janela de 30 dias)
  - Geoespaciais (coordenadas, polígonos)

---

## 3. Inferência e Scores

- **Inferência**:
  - ST-GCN processa séries temporais e relações espaciais para prever risco futuro por nó.
  - RankingModel refina a ordenação dos nós, focando em precisão no top-5 e top-20.
- **Scores**:
  - Cada nó recebe um score de risco (float, 0-1).
  - Classificação de criticidade: CRÍTICO (≥0.9), ALTO (≥0.7), MÉDIO (≥0.4), BAIXO (<0.4).
  - Métricas de avaliação: P@5, P@10, P@20, NDCG@5, NDCG@20, Recall@K, MRR, erro absoluto de ranking.
  - Exemplo de baseline: P@5=0.80, P@20=0.50 (alvo ≥0.55).

---

## 4. Influência e Relações

- **Quem influencia quem**:
  - A matriz de adjacência define a influência espacial: um nó é influenciado pelos vizinhos (grafo).
  - Eventos exógenos (ex: operações policiais, eventos climáticos) amplificam scores de risco em regiões afetadas.
  - Facções: cada nó pode ser associado a uma facção dominante, influenciando análises de domínio territorial.
- **Explicabilidade**:
  - O módulo `explanation_generator.py` decompõe fatores de risco, mostrando quais features mais contribuíram para o score.
  - Classificação de risco pode ser acompanhada de "caveats" (avisos) e explicações automáticas.

---

## 5. Métricas e Análise de Erros

- **Análise de erros**:
  - Scripts como `ranking_error_analysis.py` e `long_tail_analysis.py` identificam padrões de erro, nós problemáticos e sugerem ajustes (ex: regularização, feature engineering).
  - Relatórios detalham taxas de undershooting/overshooting, análise por tiers (top-5, long-tail, etc).
- **Critérios de seleção de modelo**:
  - Modelos só são aprovados se atingirem thresholds mínimos em P@5, P@20, NDCG, etc.
  - Framework de comparação automática entre modelos.

---

## 6. Dados de Facção e Territorialidade

- **Mapeamento**:
  - Cada nó pode ter facção atribuída via integração de dados KML/GeoJSON.
  - Estatísticas de domínio: % de território por facção, fragmentação, espalhamento geográfico.
  - Visualizações: radar charts, mapas coloridos, rankings de facção.
- **Exemplo**:
  - COMANDO VERMELHO controla ~80% dos placemarks mapeados, espalhado por 5 cidades.

---

## 7. API e Endpoints

- **Principais endpoints**:
  - `/api/risk`: retorna scores de risco por nó.
  - `/api/top20_micro_nodes`: retorna top-20 nós mais críticos.
  - `/api/polygons`: retorna polígonos e metadados.
  - `/api/risk-forecast`: previsão de risco futura.
- **Fluxo**:
  - Requisição → pré-processamento → inferência → pós-processamento (ex: amplificação exógena) → resposta JSON.

---

## 8. Referências e Documentação

- **Leitura recomendada**:
  - `README.md`: visão geral, scripts, features, exemplos.
  - `TECHNICAL_SUMMARY.md`: arquitetura, rationale, hyperparameters.
  - `QUICK_START.md`: execução, endpoints, troubleshooting.
  - `DOCUMENTATION_INDEX.md`: índice detalhado, matriz de cobertura.
  - `STRUCTURE.md`: estrutura de pastas e arquivos.

---

*Gerado automaticamente em 11/02/2026.*
