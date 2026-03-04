# 📚 Glossário Técnico - REPORT PREVIEW

## Introdução

Este glossário define os termos técnicos e conceitos usados no sistema REPORT PREVIEW. Útil tanto para técnicos quanto para gestores que desejam entender a linguagem do sistema.

---

## A

### API (Application Programming Interface)
Interface de programação que permite comunicação entre sistemas.

**REPORT PREVIEW:** Endpoints RESTful em Flask que expõem funcionalidades (ex: `/api/risk`, `/api/exogenous/parse`).

### Anomalia
Desvio do padrão esperado detectado nos dados.

**REPORT PREVIEW:** O sistema detecta anomalias em risco (picos inesperados) e dados (valores faltantes, duplicatas).

---

## B

### Backtesting
Processo de validação de modelo comparando previsões passadas com resultados reais.

**REPORT PREVIEW:** O Monitor de Eficiência realiza backtesting semanal, calculando P10, P20 para validar precisão.

**Fórmula:** Taxa Acerto = (Crimes Previstos no Top 10 ÷ Total de Crimes Reais) × 100%

### Bairro
Unidade geográfica mínima do modelo (também chamada "nó" ou "localidade").

**REPORT PREVIEW:** O Ceará tem 259 bairros definidos (Fortaleza: 127, RMF: 43, Interior: 89).

---

## C

### Canal
Tipo de evento exógeno que afeta o risco.

**REPORT PREVIEW:**
- **Canal 25 (Crítico/Conflito):** Aumenta risco (homicídios, conflitos)
- **Canal 23 (Supressão/Ação Policial):** Reduz risco (prisões, apreensões)

### CIOPS (Centro de Informações Operacionais da Polícia)
Sistema de registros policiais do Ceará.

**REPORT PREVIEW:** Fonte de eventos exógenos que alimentam o modelo em tempo real.

### Confiança do Modelo
Métrica agregada (0-100%) que indica clareza e robustez das previsões.

**REPORT PREVIEW:** 
- Scores acima de 80% = Alta previsibilidade
- Baseada em convergência, precisão e recall
- Calculada por região e globalmente

### Contágio Espacial / Sombra de Risco
Propagação do risco de um bairro para seus vizinhos através da conectividade urbana.

**REPORT PREVIEW:** Se o risco sobe em Bom Jardim, vizinhos como Pirambu têm aumento secundário (ex: +3%).

**Mecanismo:** Grafo de adjacência + pesos de proximidade.

### CVLI (Crimes Violentos Letais Intencionais)
Classificação estatística de crimes graves (homicídios, latrocínios).

**REPORT PREVIEW:** Evento base para treinamento do modelo.

---

## D

### Dashboard
Interface visual para visualização de dados em tempo real.

**REPORT PREVIEW:**
- **Dashboard Principal** (`/`): Mapa de risco, metricas, top 10
- **Admin Dashboard** (`/admin/health`): Monitoramento de saúde, alertas, métricas

### Degradação de Modelo
Queda em performance / precisão do modelo.

**REPORT PREVIEW:** Detectada quando P10 < 70% ou P20 < 75%. Gera alerta automático.

---

## E

### Evento Exógeno
Informação externa (em tempo real) que ajusta as previsões.

**REPORT PREVIEW:** Eventos policiais (prisões, homicídios) que são processados via Gemini LLM.

### Explicabilidade
Capacidade de justificar decisões do modelo em linguagem natural.

**REPORT PREVIEW:** `/api/explain/<id>` retorna:
- Fatores primários (histórico, contágio, eventos recentes)
- Contribuição de cada fator em %
- Eventos que afetaram o risco

---

## F

### F1-Score
Métrica balanceada de performance (média harmônica de Precision e Recall).

**REPORT PREVIEW:** Fórmula: `2 × (Precision × Recall) / (Precision + Recall)`
- Ideal para datasets desbalanceados
- Varia de 0 a 1 (ou 0% a 100%)

### Fortaleza
Capital do Ceará e maior região do modelo (127 bairros).

**REPORT PREVIEW:** Região com maior granularidade de previsão.

---

## G

### GeoJSON
Formato de representação de dados geoespaciais (geometrias + propriedades).

**REPORT PREVIEW:** Polígonos dos bairros em GeoJSON servidos em `/api/polygons`.

### Gemini (Google Gemini 2.0 Flash)
Modelo de linguagem grande (LLM) usado para processar texto de eventos.

**REPORT PREVIEW:** Extrai estrutura (data, hora, bairro, natureza) de textos policiais brutos.

### Grafo
Estrutura de dados com nós (bairros) e arestas (conexões).

**REPORT PREVIEW:** Grafo espacial define vizinhança dos bairros para cálculo de contágio.

---

## H

### Health Check
Verificação de disponibilidade e saúde do sistema.

**REPORT PREVIEW:** `/api/model-update-status` retorna status do servidor, modelos e confiança.

---

## I

### Interior
Região que compreende municípios do interior do Ceará (89 bairros/cidades).

**REPORT PREVIEW:** Terceira região com menor granularidade de previsão.

---

## J

### JSON (JavaScript Object Notation)
Formato de texto estruturado para dados.

**REPORT PREVIEW:** Formato padrão de requisições e respostas da API.

---

## K

### KPI (Key Performance Indicator)
Métrica chave de desempenho.

**REPORT PREVIEW:**
- Temperatura do Estado
- Confiança do Ranking
- Taxa de Erro de API
- P10/P20 (precisão)

---

## L

### Latência P95
Percentil 95% de latência (95% das requisições respondem em menos que este tempo).

**REPORT PREVIEW:** Métrica de performance. Alvo: < 500ms para `/api/risk`.

### LLM (Large Language Model)
Modelo de inteligência artificial treinado em grandes corpus de texto.

**REPORT PREVIEW:** Usado para processar eventos policiais (Gemini).

---

## M

### Mapa de Risco
Visualização geográfica de scores de risco (0-100%).

**REPORT PREVIEW:**
- **Vinho (≥90%):** Crítico
- **Vermelho (80-89%):** Alto
- **Laranja (50-79%):** Moderado
- **Azul (<50%):** Baixo

### Monitor de Eficiência
Sistema que avalia performance do modelo automaticamente (a cada 7 dias).

**REPORT PREVIEW:** Calcula P10/P20, precision, recall, f1-score.

---

## N

### Nó
Sinônimo de bairro. Unidade base da rede de predição.

**REPORT PREVIEW:** 259 nós no Ceará.

---

## O

### Orquestrador
Gerenciador que coordena múltiplos modelos regionais.

**REPORT PREVIEW:** `StateOrchestrator` em `src/core/orchestrator.py` gerencia Fortaleza, RMF e Interior.

### Outlier
Valor anômalo significativamente diferente do esperado.

**REPORT PREVIEW:** Detectados e logados em "Qualidade de Dados".

---

## P

### P10 (Precision top 10)
Percentual de crimes reais que caem nos 10 bairros com maior risco previsto.

**REPORT PREVIEW:**
- Métrica principal de precisão
- Alvo: > 85%
- Calculada semanalmente via backtesting

### P20
Análogo ao P10, mas para top 20 bairros.

**REPORT PREVIEW:**
- Alvo: > 88%
- Menos rigoroso que P10

### Pipeline
Sequência de etapas de processamento de dados.

**REPORT PREVIEW:**
1. Ingestão → 2. Processamento → 3. Validação → 4. Persistência → 5. Cálculo de Risco → 6. Explicação

### Precision
Percentual de previsões de alto risco que realmente tiveram crimes.

**REPORT PREVIEW:** Fórmula: (Verdadeiros Positivos) / (Verdadeiros + Falsos Positivos)

### Previsão
Output do modelo: score de risco para cada bairro em data/hora específica.

**REPORT PREVIEW:** Retorna score 0-100% e nível (BAIXO/MODERADO/ALTO/CRÍTICO).

---

## Q

### Qualidade de Dados
Métrica agregada de completude, consistência e acurácia dos dados.

**REPORT PREVIEW:** Dashboard exibe taxa completude (alvo: > 95%).

---

## R

### Recall
Percentual de crimes reais capturados pelas previsões de alto risco.

**REPORT PREVIEW:** Fórmula: (Verdadeiros Positivos) / (Verdadeiros Positivos + Falsos Negativos)

### RMF (Região Metropolitana de Fortaleza)
Região ao redor da capital (43 bairros/cidades como Maracanaú, Caucaia, etc).

**REPORT PREVIEW:** Segunda região com média granularidade.

### RNN (Recurrent Neural Network)
Tipo de rede neural com memória para dados sequenciais.

**REPORT PREVIEW:** ST-GAT é baseado em mecanismos de RNN para contexto temporal.

---

## S

### ST-GAT (Spatial-Temporal Graph Attention Network)
Arquitetura de rede neural que combina atenção espacial (entre bairros) e temporal (ao longo do tempo).

**REPORT PREVIEW:** Motor principal do modelo. Localizado em `src/core/architectures.py`.

### Sombra de Risco
Sinônimo de "Contágio Espacial" (ver C).

### Supressão
Ação policial que reduz risco (prisão, apreensão de armas).

**REPORT PREVIEW:** Canal 23. Comparado a "apagar fogo" no mapa.

---

## T

### TAG-Bias (Tactical Attention Bias)
Mecanismo que injeta "choque" de eventos recentes no modelo para reatividade imediata.

**REPORT PREVIEW:** Permite resposta rápida a eventos críticos sem esperar retraining.

### Temperatura do Estado
Métrica ponderada que indica nível geral de tensão de segurança.

**REPORT PREVIEW:**
- 0-30: Calmo
- 30-60: Normal
- 60-80: Tenso
- 80-100: Crítico

**Cálculo:** Média ponderada de riscos por população/importância de bairros.

### Tensor
Estrutura matemática multi-dimensional que alimenta o modelo.

**REPORT PREVIEW:** 
- 29 canais (tipos de variáveis)
- Dimensões: [Tempo, Espaço, Canais]
- Exemplo: [120 dias, 259 bairros, 29 features]

### Threshold
Limite de corte para decisão.

**REPORT PREVIEW:**
- Risco ≥ 90% = Crítico
- Taxa erro > 2% = Alerta
- Latência > 500ms = Warning

### Top 10 / Top 20
Ranking dos 10 ou 20 bairros com maior risco.

**REPORT PREVIEW:** Atualizado em tempo real. Foco operacional para gestores.

### Treino (Training)
Processo de ajustar pesos do modelo com dados históricos.

**REPORT PREVIEW:**
- Frequência: Semanal ou sob demanda
- Duração: ~30 minutos para Ceará
- Dados: Últimos 120 dias

---

## V

### Validação
Processo de conferência de dados antes de persistência.

**REPORT PREVIEW:**
- Checa integridade (datas, bairros válidos)
- Detecta duplicatas
- Normaliza valores

### Vizinhança
Conjunto de bairros adjacentes a um nó.

**REPORT PREVIEW:** Definida pelo grafo de adjacência geográfica.

---

## W

### What-If Analysis
Simulador de cenários (ver "Simulador de Cenários").

**REPORT PREVIEW:** `/api/simulate` endpoint.

---

## X

*(Sem termos com X)*

---

## Y

*(Sem termos com Y)*

---

## Z

### Zona
Sinônimo de bairro ou região.

**REPORT PREVIEW:** "Zona crítica", "zona de contágio".

---

## Siglas Comuns

| Sigla | Significado | Contexto |
|-------|-------------|---------|
| CVLI | Crimes Violentos Letais Intencionais | Classificação estatística |
| CIOPS | Centro Informações Operacionais Polícia | Sistema de registros |
| ST-GAT | Spatial-Temporal Graph Attention Network | Arquitetura IA |
| LLM | Large Language Model | Gemini, processamento texto |
| RNN | Recurrent Neural Network | Tipo de rede neural |
| API | Application Programming Interface | Endpoints REST |
| JSON | JavaScript Object Notation | Formato dados |
| GeoJSON | Geography JSON | Formato geoespacial |
| RMF | Região Metropolitana Fortaleza | Região |
| P10 | Precision top 10 | Métrica performance |
| KPI | Key Performance Indicator | Métrica chave |
| ETL | Extract, Transform, Load | Pipeline dados |
| JWT | JSON Web Token | Autenticação |
| TLS | Transport Layer Security | Criptografia |
| CSV | Comma-Separated Values | Formato arquivo |

---

## Fórmulas Matemáticas

### Score de Risco
```
Risk_Score = α × Historical_Trend + β × Spatial_Contagion + γ × Recent_Events

Onde:
- α ≈ 0.5 (peso histórico)
- β ≈ 0.3 (peso contágio)
- γ ≈ 0.2 (peso eventos)
```

### P10 Score
```
P10 = (Crimes_Reais_em_Top_10 ÷ Total_Crimes_Reais) × 100%

Exemplo:
- Se 8 de 10 crimes aconteceram nos top 10 bairros previstos
- P10 = 80%
```

### Temperatura do Estado
```
Temp = Σ(Risk_Score_i × Population_Weight_i) ÷ Σ(Population_Weight_i)

Escala: 0-100 (linear)
```

### F1-Score
```
F1 = 2 × (Precision × Recall) ÷ (Precision + Recall)

Intervalo: [0, 1] ou [0%, 100%]
Máximo: 1 (perfeito)
```

---

## Referências Externas

- [Graph Neural Networks](https://arxiv.org/abs/1901.00596)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
- [Time Series Forecasting](https://en.wikipedia.org/wiki/Time_series)
- [Spatial Statistics](https://en.wikipedia.org/wiki/Spatial_statistics)

---

**Última atualização:** 01 de Março de 2026  
**Versão:** 1.0
