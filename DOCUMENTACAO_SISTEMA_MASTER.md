# 🛡️ REPORT PREVIEW: Sistema de Inteligência Geospacial e Preditiva

## 1. Visão Geral
**REPORT PREVIEW** é uma plataforma avançada de suporte à decisão estratégica para segurança pública, especificamente calibrada para a realidade do Estado do Ceará (CPRAIO/CIOPS). O sistema utiliza redes neurais de grafos (**ST-GAT - Spatial-Temporal Graph Attention Networks**) para prever manchas de criminalidade (CVLI) e integrar inteligência em tempo real via eventos exógenos, fornecendo análises preditivas com base em padrões históricos, aceleração criminal dinâmica e auto-ajuste temporal autônomo.

## 2. Arquitetura de Inteligência (O Motor)
A aplicação opera através de um **Orquestrador Dinâmico** que gerencia especialistas regionalizados (Fortaleza, RMF, Interior) operando com a arquitetura de alta eficiência **DeepSTGAT_64**. A inteligência é dividida nas seguintes camadas:

### A. Camada Estatística e Vetorial (DeepSTGAT_64)
- **Algoritmo:** Redes Neurais de Atenção Espacial-Temporal baseadas em Grafos (64 Neurônios).
- **Janela de Inteligência Base:** Analisa os últimos **120 dias** de dados históricos brutos em matrizes de adjacência (Geografia + Conflito de Facções).
- **Multi-Scale Temporal Momentum (32 Canais):** A rede não olha apenas para o volume de crimes, mas para a **velocidade e aceleração da violência**. O Orquestrador calcula em tempo real (runtime) 3 escalas temporais de aceleração (7 dias, 14 dias e 30 dias) e as injeta no tensor preditivo, permitindo que a rede diferencie um "revide de fim de semana" de uma "guerra de expansão de longo prazo".

### B. Camada de Inteligência Viva (Choques Exógenos)
Processa informações em tempo real de logs policiais para ajustar dinamicamente as previsões estatísticas:
- **Canal 25 (Crítico/Conflito):** Aumenta o risco local com ponderação contextual. Ativado por homicídios, expulsões, conflitos de facções e execuções.
- **Canal 23 (Supressão/Ação Policial):** Reduz ou estabiliza o risco através de intervenções qualificadas (prisões de líderes, apreensão de fuzis, saturação).

### C. Auto-Curriculum Temporal (Temporal Shrinkage)
O sistema possui um termostato de inteligência artificial autônomo que reage a quedas de performance:
- Se o **Monitor de Eficiência** detectar que a precisão do modelo (P@10) caiu abaixo de **50%** em uma região, o Orquestrador aplica uma **Máscara de Atenção Dinâmica**.
- Essa máscara encolhe a janela temporal de visão do modelo (ex: de 120 dias para 90, 60 ou 30 dias), "apagando" o passado distante para forçar a rede a focar exclusivamente no "calor" do conflito recente, eliminando a inércia histórica que possa estar gerando ruído.
- Quando a performance se restabelece, a janela volta a expandir gradativamente.

---

## 3. Funcionalidades Principais (Features)

### 🗺️ Mapa de Risco Dinâmico e Capilaridade de Ruas
Representação visual geográfica do score de risco (0-100%) com drill-down para logradouros:
- **Micro-Inteligência de Ruas:** O sistema cruza os pontos exatos de CVLI dos últimos 30 dias com os micronodos de inteligência de facções. Ao clicar em um bairro (ex: Messejana), o gestor visualiza exatamente quais logradouros concentram a violência real.
- **Vinho (>= 90%):** CRÍTICO. Necessidade de intervenção tática imediata.
- **Vermelho (80-89%):** ALTO. Prioridade de patrulhamento preventivo.
- **Laranja (50-79%):** MODERADO. Monitoramento ativo com barreiras inteligentes.
- **Azul (< 50%):** BAIXO. Patrulhas preventivas normais.

### 🧪 Simulador de Cenários (What-If Analysis)
Ferramenta de análise prospectiva que permite ao gestor projetar o impacto de ações estratégicas antes da execução:
- **Simulação de Supressão:** Modela o efeito de ações policiais no padrão de risco e no efeito "contágio" espacial.
- **Simulação de Conflito:** Projeta o impacto regional de eventos críticos (confrontos, atentados) em áreas de domínio rival.

### 📈 Painel Executivo (Métricas e Explicabilidade)
Dashboard executivo com KPIs estratégicos e inteligência interpretável:
- **Explicabilidade LLM (Gemini):** Cada bairro do Top 10 possui uma análise gerada por IA que explica o porquê do bairro estar em risco (baseado em momentum de 14 dias, eventos exógenos e vizinhança perigosa).
- **Temperatura do Estado:** Métrica agregada ponderada que indica a volatilidade do risco estadual para calibração de tropas de recobrimento.
- **Top 10 Regional:** Foco operacional subdividido entre Fortaleza, RMF e Interior.

---

## 4. Ciclo de Vida e Validação

### Monitor de Eficiência (Backtesting Autônomo)
O sistema não é uma "caixa preta" estática; ele se audita em background:
- Roda avaliações comparando previsões anteriores com os eventos reais do presente.
- Gera métricas de precisão (P@5, P@10, P@20).
- Em caso de degradação da cobertura territorial (ex: se uma facção em guerra sumir do radar), dispara alertas críticos de "Subestimação Territorial" para os administradores no Painel de Saúde (Health Dashboard).

### Processamento de LLM para Eventos (Gemini 2.0 Flash)
- A ingestão de boletins e resumos do CIOPS passa por um parsing avançado usando LLM.
- O modelo identifica datas, municípios, bairros, extrai o nível de intensidade (Low/Medium/High) e determina se a ação foi de "Conflito" ou "Supressão", alimentando a rede ST-GAT em segundos de forma estruturada.

---

## 5. Estrutura e Operação Técnica

### Infraestrutura
- **Servidor Principal:** Flask (`app.py`), operando na porta 5050.
- **Base de Dados/Tensores:** Pickles limpos e padronizados em `data/processed/`, lidos diretamente no runtime (`pd.read_pickle`) de forma agnóstica a versões do pandas para evitar falhas de `StringDtype`.
- **Inteligência Desacoplada:** Treinamentos ocorrem paralelamente em scripts da pasta `scripts/training/`. Modelos aprovados (Recordistas) são promovidos para a pasta `models/active/`, e o Orquestrador os assume automaticamente sem necessidade de restart forçado ou quebras de arquitetura.

### Configuração de Treinamento Atual (Março/2026)
- **Status:** DeepSTGAT_64 (Multi-Scale Momentum).
- **Meta:** Superar consistentemente a barreira de 50% P@10 em Fortaleza.
- **Prevenção de Colapso:** Logs extremos com captura de norma de gradiente (GradNorm) e Gradient Accumulation dinâmico.

---

## 6. Limitações Conhecidas e Considerações

- **Qualidade de Dados Reais:** A ferramenta de "Inteligência de Ruas" reflete as coordenadas de ocorrência exatas; locais subnotificados ou registrados com endereço da delegacia e não do fato afetam a capilaridade micro.
- **Latência de Eventos LLM:** Há defasagem natural entre ocorrência de crime e entrada de dados no sistema via colagem manual de relatórios.
- **Granularidade do Polígono:** A rede neural enxerga o bairro como um nó único e contínuo, distribuindo o risco uniformemente em sua área, mesmo que o crime se concentre apenas em uma favela específica dentro dele (a não ser que os eventos exógenos apontem a rua via LLM).

---
*Documentação atualizada em 13 de Março de 2026 - Versão 3.0 (Integração de Multi-Scale Momentum e Auto-Curriculum Temporal).*