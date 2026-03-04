# 🛡️ REPORT PREVIEW: Sistema de Inteligência Geospacial e Preditiva

## 1. Visão Geral
**REPORT PREVIEW** é uma plataforma avançada de suporte à decisão estratégica para segurança pública, especificamente calibrada para a realidade do Estado do Ceará (CPRAIO/CIOPS). O sistema utiliza redes neurais de grafos (**ST-GAT - Spatial-Temporal Graph Attention Networks**) para prever manchas de criminalidade (CVLI) e integrar inteligência em tempo real via eventos exógenos, fornecendo análises preditivas com base em padrões históricos e eventos dinâmicos.

## 2. Arquitetura de Inteligência (O Motor)
A aplicação opera em três camadas de dados sincronizadas:

### A. Camada Estatística (Base Histórica)
- **Algoritmo:** ST-GAT (Deep Learning em Grafos) - Redes Neurais de Atenção Espacial-Temporal.
- **Janela de Inteligência:** Analisa os últimos **120 dias** de dados históricos brutos para capturar padrões sazonais e tendências.
- **Propagação Espacial:** O modelo não olha apenas para um bairro isolado; ele entende a conectividade urbana através de grafos geoespaciais. Se o risco sobe em um bairro, o sistema calcula a "sombra de risco" nas localidades vizinhas e rotas de fuga prováveis, proporcionando previsões contextualizadas.

### B. Camada de Inteligência Viva (Choques Exógenos)
Processa informações em tempo real de logs policiais (CIOPS) para ajustar dinamicamente as previsões estatísticas:
- **Canal 25 (Crítico/Conflito):** Aumenta o risco local com ponderação contextual. Ativado por homicídios, expulsões, conflitos de facções, execuções e violência interpessoal severa.
- **Canal 23 (Supressão/Ação Policial):** Reduz ou estabiliza o risco através de intervenção. Ativado por prisões qualificadas, apreensões de armas (com pesos maiores para fuzis e armamento pesado), recuperação de veículos roubados, e patrulhamento intensivo em zonas críticas.

### C. Processamento de Linguagem Natural (LLM)
- Utiliza **Gemini 2.0 Flash** para extrair dados estruturados e semânticos de textos brutos com alta precisão.
- **Detecção Inteligente de Cabeçalho:** Identifica automaticamente blocos de "Ações Policiais", "Ocorrências" ou "Eventos", propagando contexto temporal e natureza para todos os eventos associados, garantindo precisão cronológica e consistência no mapa de risco.
- **Normalização Contextual:** Padroniza nomenclaturas de bairros, naturezas de crimes e categorias, reduzindo ambiguidades nos dados.

---

## 3. Funcionalidades Principais (Features)

### 🗺️ Mapa de Risco Dinâmico
Representação visual geográfica do score de risco (0-100%) com atualizações em tempo real:
- **Vinho (>= 90%):** CRÍTICO. Zona de máxima sensibilidade. Necessidade de intervenção/saturação imediata com recursos táticos.
- **Vermelho (80-89%):** ALTO. Prioridade de patrulhamento preventivo. Monitoramento contínuo e prontidão operacional.
- **Laranja (50-79%):** MODERADO. Monitoramento ativo com barreiras inteligentes e reforço situacional.
- **Azul (< 50%):** BAIXO. Patrulhas preventivas normais. Atividades de rotina e presença comunitária.

### 🧪 Simulador de Cenários (What-If Analysis)
Ferramenta de análise prospectiva que permite ao gestor projetar o impacto de ações estratégicas antes da execução:
- **Simulação de Supressão:** Modela o efeito de ações policiais (envio de equipes, reforço tático) no padrão de risco. Responde: "O que acontece com a mancha de risco se eu enviar 5 equipes para este ponto?"
- **Simulação de Conflito:** Projeta o impacto regional de eventos críticos (confrontos, atentados). Responde: "Qual o impacto cascata em zonas vizinhas se houver um ataque em determinada área?"
- **Validação Preditiva:** Permite testes de hipóteses sem exposição operacional.

### 📈 Painel do Gestor (Métricas de Confiança e Situação)
Dashboard executivo com KPIs estratégicos:
- **Confiança do Ranking:** Indica a clareza e robustez estatística do modelo. Scores acima de 80% indicam previsibilidade alta com baixa incerteza. Baseado em índices de convergência do modelo.
- **Temperatura do Estado:** Métrica agregada ponderada que indica se o estado está em dia de calmaria (baixa volatilidade de risco) ou tensão generalizada (padrões elevados de risco distribuído). Útil para alocação regional de recursos.
- **Top 10 Regional:** Foco operacional imediato nos 10 pontos de maior sensibilidade por macrorregião (Fortaleza, RMF, Interior). Identificação de focos críticos com priorização automática.
- **Série Temporal de Confiança:** Histórico de confiança do sistema ao longo do tempo, permitindo análise de degradação ou melhoria do modelo.

### 📊 Módulo de Análise Histórica
- **Backtesting Contínuo:** Avaliação comparativa entre previsões do modelo e eventos reais.
- **Análise de Padrões:** Identificação de ciclos sazonais, tendências de longo prazo e eventos anomálicos.
- **Relatórios Exportáveis:** Geração de relatórios em múltiplos formatos para documentação e auditoria.

---

## 4. Confiança e Precisão do Sistema

A confiança do sistema é validada através de múltiplos mecanismos:

### Monitor de Eficiência (Backtesting Contínuo)
- O sistema realiza rodadas de autoavaliação a cada 7 dias.
- Compara previsões históricas com eventos reais ocorridos (CVLIs confirmadas).
- **Métrica P10/P20:** Se os crimes reais ocorrerem dentro dos bairros listados no Top 10/Top 20 do sistema em determinado período, a precisão é confirmada e a confiança aumenta.

### Indicadores de Qualidade
- **Taxa de Acerto (Precision):** Percentual de previsões confirmadas vs. total de previsões.
- **Cobertura (Recall):** Percentual de eventos reais capturados pela previsão.
- **F1-Score:** Métrica balanceada de desempenho geral do modelo.

---

## 5. Ciclo de Vida e Arquitetura de Dados

### Fluxo de Processamento
1. **Ingestão:** Via API REST, colagem de texto no dashboard ou integração direta com sistemas policiais (CIOPS).
2. **Processamento:** Extração estruturada de data, hora, bairro e natureza do crime usando IA (Gemini) + regras determinísticas.
3. **Validação:** Conferência de integridade, detecção de anomalias e normalização de dados.
4. **Persistência:** Armazenamento em `exogenous_events.json` com `date` como âncora cronológica primária. Sincronização com base de dados histórica.
5. **Arquivamento:** Eventos com mais de 7 dias são movidos para `data/archives/` automaticamente, otimizando performance do motor de busca e análise.
6. **Retroalimentação:** Dados processados alimentam o modelo ST-GAT para refinamento contínuo de pesos e previsões.

### Estrutura de Armazenamento
- **Banco de Dados Principal:** `data/` (histórico completo, índices otimizados)
- **Cache Temporal:** Últimos 120 dias em memória para cálculos de previsão
- **Arquivo Morto:** `data/archives/` (consulta sob demanda, auditoria)
- **Logs de Treino:** `logs/training_*.log` (rastreamento de experimentos e ajustes)

---

## 6. Configuração, Manutenção e Operação

### Infraestrutura Técnica
- **Servidor Principal:** Flask rodando em `localhost:5050` (produção: configurável)
- **Base de Dados:** Localizada em `data/` com estrutura organizada por períodos e regiões
- **Logs de Treino:** `logs/training_...log` para rastreamento de experimentos ML
- **Ambiente:** Python 3.8+, dependências em `requirements.txt`

### Monitoramento e Saúde do Sistema
- **Health Check:** Endpoint dedicado para verificação de disponibilidade
- **Logs de Erro:** Rastreamento centralizado de falhas e exceções
- **Métricas de Performance:** Tempo de resposta de API, latência de previsão, utilização de recursos

### Integração com Sistemas Externos
- **CIOPS:** Sincronização de eventos policiais via API
- **Gemini 2.0 Flash:** Processamento de linguagem natural para extração de dados
- **Geolocalização:** Integração de mapas e bases cartográficas

---

## 7. Casos de Uso e Aplicações

### 🚔 Para Gestores Operacionais
- Priorização dinâmica de zonas de patrulhamento
- Alocação otimizada de recursos táticos
- Resposta rápida a mudanças de padrão de risco

### 📋 Para Planejadores Estratégicos
- Análise de tendências de criminalidade de longo prazo
- Identificação de padrões sazonais e ciclos
- Modelagem de cenários para planejamento de operações

### 🔍 Para Analistas de Inteligência
- Rastreamento de "sombras de risco" e propagação de ameaças
- Detecção de anomalias e padrões anormais
- Correlação entre eventos críticos e mudanças de risco

---

## 8. Limitações Conhecidas e Considerações

- **Dependência de Qualidade de Dados:** A precisão do modelo depende da completude e precisão dos dados de entrada (CIOPS).
- **Viés Histórico:** Padrões históricos podem refletir decisões operacionais anteriores, não apenas atividade real.
- **Latência de Eventos:** Há defasagem natural entre ocorrência de crime e entrada de dados no sistema.
- **Granularidade Geográfica:** Precisão limitada pela granularidade de bairros definidos no modelo.

---
*Documentação gerada em 01 de Março de 2026 - Versão 2.0 (REPORT PREVIEW - Referência Completa de Sistema e Funcionalidades).*
