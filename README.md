# 🛡️ Report Preview — Sistema de Predição de Risco Criminal

> **Inteligência Criminal Preditiva para o Estado do Ceará**
> Paradigma Híbrido: ST-GAT (champion) + Sentinela V3 (challenger controlado) + guardrails operacionais para evitar falsos positivos territoriais

---

## 📌 Visão Geral

O **Report Preview** é uma plataforma operacional de previsão de crimes violentos letais intencionais (CVLI) desenvolvida para apoiar o planejamento tático de segurança pública no estado do Ceará. O sistema prediz os bairros com maior probabilidade de registrar homicídios nos próximos **14 dias**, com foco na capital **Fortaleza** (40 bairros monitorados).

### Métricas de Performance (Validação Sombra — Abr/2026)

| Modelo | P@10 | P@20 | Janela | Status |
|--------|------|------|--------|--------|
| ST-GAT (fortaleza_model_active) | ~42% | — | 120d | ✅ Oficial (champion) |
| **Sentinela V3 — EWMA-Multi** | **50%** | 65% | 14d | Shadow / laboratório |
| **Sentinela V3 — LGBM Lean** | 30% | **70%** | 14d | Shadow / laboratório |
| **Sentinela V3 — Ensemble** | 30% | **70%** | 14d | Challenger carregado; blend pode permanecer em `0%` |
| Baseline aleatório | 25% | 50% | — | Referência |

> **P@K** = Precisão no Top-K: proporção dos K bairros previstos que de fato registraram CVLI no horizonte de 14 dias.
>
> **Guardrail operacional atual:** o score final prioriza `CVLI` recente/histórico e vizinhança. `CVP` permanece apenas como contexto interno do modelo; ele não deve promover diretamente um bairro para risco moderado/alto. Pressão territorial por facção também só entra quando existe suporte `CVLI` recente real ou lastro histórico relevante.

### Documentacao Operacional

- Deploy completo em VPS/Hostinger: [IMPLEMENTACAO_NUVEM_HOSTINGER.md](IMPLEMENTACAO_NUVEM_HOSTINGER.md)
- Checklist rapido de implantacao: [CHECKLIST_DEPLOY_HOSTINGER.md](CHECKLIST_DEPLOY_HOSTINGER.md)

---

## 🏗️ Arquitetura do Sistema

O sistema opera em duas camadas paralelas que convergem num blend dinâmico:

```
┌─────────────────────────────────────────────────────────────┐
│                     CAMADA 1: ST-GAT                        │
│                                                             │
│  Dados Históricos (37 canais, 120 dias)                     │
│      ↓                                                      │
│  DeepSTGAT_64 (3 camadas GAT + Atenção Temporal)            │
│      ↓                                                      │
│  scores_stgat = {bairro: score}  ← orchestrator.py         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               CHAMPION/CHALLENGER (src/core/                │
│                champion_challenger.py)                      │
│                                                             │
│  Avalia P@10 de ambos contra CVLI real (últimos 14 dias)    │
│  Ajusta blend via EMA:                                      │
│    score_final = (1-w) × ST-GAT + w × LGBM Lean            │
│    w começa em 0%, sobe até 50% se LGBM provar vantagem     │
│    Decisão persistida em data/cc_state.json                 │
│    Auditoria em logs/cc_decisions.jsonl                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 CAMADA 2: SENTINELA V3 / CC                 │
│                                                             │
│  Ranking challenger usado apenas sob arbitragem segura      │
│  Features contextuais: target_enc, intel_ewma, nbr_cvli,    │
│  hist_pct, sinais de contexto e validação por P@10          │
│      ↓                                                      │
│  LightGBM LambdaRank (300 árvores, reg forte)               │
│      +                                                      │
│  EWMA-Multi (7d×0.4 + 14d×0.35 + 30d×0.15 + 90d×0.1)      │
│      ↓                                                      │
│  Blend dinâmico → somente se o challenger provar vantagem   │
└─────────────────────────────────────────────────────────────┘
```

### Guardrails do Score Final

1. O alvo primário é `CVLI`; `CVP` é canal auxiliar de contexto e não driver direto do risco final.
2. Facção/tensão territorial não sobe bairro frio sozinha: o reforço territorial exige `CVLI` recente no tensor do modelo ou lastro histórico relevante.
3. O challenger pode ser carregado sem influenciar a API: o peso efetivo vem de `data/cc_state.json` e pode ficar em `0%`.

### Componentes Principais

| Componente | Arquivo | Responsabilidade |
|-----------|---------|-----------------|
| **Orquestrador ST-GAT** | `src/core/orchestrator.py` | Modelo oficial — 3 regiões (Fortaleza, RMF, Interior) |
| **Champion/Challenger** | `src/core/champion_challenger.py` | Blend dinâmico ST-GAT ↔ LGBM, arbitra com dados reais |
| **Inferência Sentinela** | `tests/Sentinela/sentinela_inference.py` | API limpa do LGBM Lean + alertas de Intel |
| **Fine-tuner Tempo Real** | `tests/Sentinela/finetune_realtime_v1.py` | LGBM ajustado nos últimos 30 dias (janela deslizante) |
| **Freeze / Re-treino** | `tests/Sentinela/freeze_total_v3.py` | Re-treino completo ao receber novos dados |
| **Validação Sombra** | `tests/Sentinela/train_validate_v3.py` | Auditoria out-of-sample periódica |
| **Promoção de Modelo** | `tests/Sentinela/promote_model.py` | Promoção segura com backup e log |
| **Monitor de Eficiência** | `src/core/efficiency_monitor.py` | P@10/P@20/Recall avaliados automaticamente |

---

## 🧠 Sentinela V3 — O Novo Paradigma (Tentativas 55–57b)

### Por que migramos do ST-GAT puro para o paradigma híbrido?

Após 54 tentativas de otimização do ST-GAT (T1→T54), o modelo atingiu um teto de ~42% P@10 com alta variância cross-fold. A principal limitação era a **instabilidade com dados esparsos de CVLI** — homicídios são eventos raros, e redes neurais profundas sofrem com esse regime.

O **Sentinela V3** adota uma filosofia diferente:

1. **Menos é mais**: 10 features de alta importância > 42 features ruidosas
2. **Ranking explícito**: LightGBM LambdaRank otimiza diretamente para NDCG@10, não MSE
3. **Momentum EWMA**: captura tendências de curto prazo que o GAT (janela 120d) perde
4. **Guardrails de produção**: sinais contextuais só entram no score externo quando encontram suporte em `CVLI` recente/histórico relevante

### Features históricas do Challenger (treino original)

As importâncias abaixo descrevem o treinamento do challenger em laboratório. No caminho externo atual servido pela API, features derivadas de `CVP` são neutralizadas para que não empurrem diretamente o score final.

| # | Feature | Peso% | O que captura |
|---|---------|-------|---------------|
| 1 | `cvp_cvli_ratio` (calibrado) | 18.8% | Feature histórica do challenger; neutralizada no score externo |
| 2 | `target_enc` | 15.7% | Média histórica expanding de CVLI por bairro |
| 3 | `intel_ewma_14d` | 10.8% | Pressão de operações policiais (armas/drogas/veículos) |
| 4 | `cvp_ewma_30d` | 10.5% | Feature histórica de contexto; neutralizada no score externo |
| 5 | `intel_ewma_7d` | 8.8% | Intel de tropa de curto prazo |
| 6 | `inter_intel_cvli` | 8.5% | Interação simultânea: Intel alta + CVLI recente |
| 7 | `nbr_cvli_30d` | 8.3% | CVLI em bairros geograficamente vizinhos (retaliação) |
| 8 | `hist_pct` | 6.9% | Percentil histórico do bairro no ranking de CVLI |
| 9 | `cvp_ewma_14d` | 6.5% | Feature histórica de contexto; neutralizada no score externo |
| 10 | `cvp_ewma_7d` | 5.2% | Feature histórica de contexto; neutralizada no score externo |

### Motor de Inteligência de Tropa (score_intel)

O score de intel por operação policial é calculado assim:

```python
score_intel = (
    qtd_armas               × 15.0   # maior preditor de violência
  + log1p(qtd_drogas)       ×  4.0   # escalonado para evitar outliers
  + qtd_drogas_itens        ×  2.0
  + qtd_veiculos_apreendidos×  3.0
  + peso_natureza                    # APREENSÃO ARMA=15, TRÁFICO=8, etc.
)
```

---

## 📁 Estrutura do Projeto

```
Report Preview/
│
├── app.py                          # API Gateway Flask — ponto de entrada
│
├── src/core/
│   ├── orchestrator.py             # ST-GAT: 3 modelos regionais (champion)
│   ├── architectures.py            # DeepSTGAT_64 — definição da rede neural
│   ├── champion_challenger.py      # 🆕 Blend dinâmico ST-GAT + LGBM Lean
│   ├── efficiency_monitor.py       # Avaliação automática P@10/P@20
│   └── model_calibrator.py         # Auto-ajuste de janela temporal
│
├── models/
│   ├── active/
│   │   ├── fortaleza_model_active.pth   # ST-GAT oficial (champion)
│   │   ├── lgbm_lean_v3_freeze.pkl      # 🆕 LGBM V3 promovido (challenger)
│   │   ├── ranking_atual.csv            # Ranking atual dos 40 bairros
│   │   └── ranking_atual.json           # Idem, formato JSON
│   └── archive/                         # Modelos anteriores (backup automático)
│
├── tests/Sentinela/                # 🆕 Laboratório Sentinela V3
│   ├── lgbm_lean_v3_freeze.pkl     # Modelo candidato (pré-promoção)
│   ├── freeze_total_v3.py          # Re-treino com dados completos
│   ├── train_validate_v3.py        # Validação sombra out-of-sample
│   ├── sentinela_inference.py      # Interface de inferência + alertas
│   ├── finetune_realtime_v1.py     # Fine-tuner janela deslizante 30d
│   ├── promote_model.py            # Promoção segura para models/active/
│   └── ROADMAP.md                  # Roadmap detalhado das próximas fases
│
├── data/
│   ├── raw/
│   │   ├── dados_status_ocorrencias_gerais_ENRIQUECIDO.csv  # CVLI + CVP
│   │   ├── ocorrencias_tropa_limpo_fortaleza.csv            # Intel de tropa
│   │   └── bairros_centros_latlong.json                     # Centroides
│   ├── processed/                  # Pkls pré-processados para ST-GAT
│   ├── cc_state.json               # 🆕 Estado Champion/Challenger (pesos)
│   └── exogenous_events.json       # Eventos exógenos de tempo real
│
├── logs/
│   ├── cc_decisions.jsonl          # 🆕 Auditoria de cada decisão CC
│   ├── predict_p10.jsonl           # Top-10 predito por região (histórico)
│   ├── rankings/                   # Relatórios diários por região
│   └── manual/                     # Logs operacionais manuais e capturas avulsas
│
└── docs/
  └── ai/GEMINI.md               # Contexto do projeto para IA-assistente

├── TRAINING_LOG.md              # Histórico completo de experimentos (T1→T57b)
└── VALIDATION_LOG.md            # Histórico de validação e checkpoints
```

---

## 🔄 Fluxo de Produção

### Ciclo diário (já automatizado)

```
[Startup do app.py]
    ↓
1. StateOrchestrator carrega ST-GAT (fortaleza, rmf, interior)
2. ChampionChallenger carrega LGBM Lean e estado CC persistido
3. EfficiencyMonitor avalia P@10/P@20 das últimas duas semanas
4. Relatório diário Markdown gerado em logs/rankings/

[A cada requisição /api/risk]
    ↓
1. ST-GAT gera scores_map {bairro: score} para todas as regiões
2. CC.apply(scores_map) só blenda Fortaleza quando `w_cc > 0`
3. Pesos CC ajustados automaticamente 1x/hora via EMA
4. JSON retornado ao frontend (mesmo formato de sempre)
```

### Ciclo semanal (manual)

```bash
# 1. Fine-tuner: captura padrões emergentes dos últimos 30 dias
.\.venv\Scripts\python.exe tests/Sentinela/finetune_realtime_v1.py --janela 30

# 2. Consultar ranking atual com explicações por bairro
.\.venv\Scripts\python.exe tests/Sentinela/sentinela_inference.py
```

### Ciclo mensal (ao receber novos dados)

```bash
# 1. Re-treinar com histórico completo atualizado
.\.venv\Scripts\python.exe tests/Sentinela/freeze_total_v3.py

# 2. Validar fora da amostra (shadow validation)
.\.venv\Scripts\python.exe tests/Sentinela/train_validate_v3.py

# 3. Revisar ranking_atual_v3_freeze.csv operacionalmente

# 4. Promover se aprovado (interativo, com backup automático)
.\.venv\Scripts\python.exe tests/Sentinela/promote_model.py
```

---

## 🏆 Champion/Challenger — Como Funciona

O CC arbitra qual modelo pode influenciar o score final sem depender de uma decisão humana a cada ciclo:

```
Novo período (1x/hora):
  1. Coleta CVLI real dos últimos 14 dias
  2. Calcula P@10 do ST-GAT  → p10_champ
  3. Calcula P@10 do LGBM    → p10_chal
  4. Se p10_chal > p10_champ + 3pp → aumenta w_cc gradualmente (EMA 30%)
  5. Se p10_champ > p10_chal + 3pp → diminui w_cc gradualmente
  6. Blend: score = (1-w) × ST-GAT + w × LGBM     [w ∈ 0%, 50%]
  7. Persiste decisão em logs/cc_decisions.jsonl
```

**Garantias de segurança:**
- Começa em `w=0%` (100% ST-GAT) — transição gradual e pode permanecer em `0%`
- Máximo de `w=50%` — ST-GAT nunca é eliminado
- Se LGBM falhar: fallback imediato para 100% ST-GAT
- Features derivadas de `CVP` não entram diretamente no score externo do challenger
- Sem alteração na interface da API — transparente para o frontend

---

## 🔍 Explicabilidade por Bairro

O `sentinela_inference.py` expõe uma razão principal por bairro, mas a leitura operacional deve respeitar o guardrail do score final:

```
ANCURI       → CVLI recente + pressão territorial válida
BARROSO      → Pressão Intel + CVLI simultâneos
JANGURUSSU   → Histórico de CVLI elevado
CURIO        → Intel de tropa recente com suporte espacial
CENTRO       → Mantido em baixo quando não há CVLI recente relevante
```

---

## 🛠️ Instalação e Configuração

### Pré-requisitos

- Python 3.9+
- Chave de API do Google Gemini (funcionalidades LLM)
- PyTorch (CPU ou CUDA) compatível com o ambiente

### Setup

```bash
# 1. Clonar
git clone <url-do-repositorio>
cd "Report Preview"

# 2. Criar ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env e adicionar GOOGLE_API_KEY

# 5. Iniciar
python app.py
```

O sistema estará acessível em `http://localhost:5000`.

---

## 📡 Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Dashboard — mapa de risco interativo |
| `GET` | `/api/risk` | Scores de risco por bairro (blendados CC) |
| `POST` | `/api/exogenous/parse` | Ingerir evento em linguagem natural (LLM) |
| `GET` | `/api/explain?node_id=X` | Explicação tática do risco de um bairro |
| `GET` | `/api/admin/health` | Dashboard de saúde do sistema |

---

## 📊 Histórico de Evolução do Modelo

| Fase | Tentativas | Paradigma | Melhor P@10 |
|------|-----------|-----------|-------------|
| ST-GAT Básico | T1–T20 | GAT + RankLoss | ~28% |
| ST-GAT Avançado | T21–T45 | DeepSTGAT_64 + Intel | ~38% |
| ST-GAT Elite | T46–T54 | Momentum + Z-Score Local | ~42.9% |
| **Sentinela V3** | **T55–T57b** | **LGBM Lean + EWMA-Multi** | **50% P@10** |

> Histórico completo com análise de cada tentativa em [`TRAINING_LOG.md`](TRAINING_LOG.md).

---

## 🗺️ Roadmap

| Fase | Status | Descrição |
|------|--------|-----------|
| Promoção manual | 🔵 Disponível | `promote_model.py` com checklist e backup |
| Re-treino periódico | ✅ Implementado | `freeze_total_v3.py` |
| Fine-tuning tempo real | ✅ Implementado | `finetune_realtime_v1.py` |
| Champion/Challenger | ✅ Integrado | `champion_challenger.py` no app |
| Hibridismo ST-GAT+LGBM | 🔴 Exploratório | Fase 7 — aguarda ST-GAT estabilizar ≥40% consistente |

> Roadmap detalhado em [`tests/Sentinela/ROADMAP.md`](tests/Sentinela/ROADMAP.md).

---

## 🛡️ Segurança e Licença

Este software é de **uso restrito** para fins de análise de segurança pública e inteligência criminal. Todos os dados de ocorrências, localidades e inteligência de facções devem ser tratados com o nível de confidencialidade adequado à sua classificação.

**Desenvolvido por:** Equipe de Inteligência Artificial & Tática — Ceará, 2026.
