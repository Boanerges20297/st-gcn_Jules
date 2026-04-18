# 🗺️ SENTINELA — ROADMAP DE FASES

> **Estado atual:** Modelo `lgbm_lean_v3_freeze.pkl` em `tests/Sentinela/`  
> **Validação sombra (Abr/2026):** P@10=50% (EWMA-Multi) · P@20=70% (Ensemble)  
> **Pendente:** Revisão manual → Promoção para `models/active/`

---

## ✅ Sprint Concluída (Fases 1–3)

| Tentativa | Script | Resultado |
|-----------|--------|-----------|
| T55 | `benchmark_v2.py` | 42 features, 6 folds → P@20=65.6% ✅ |
| T56 | `benchmark_v3.py` | LGBM Lean + EWMA-Multi → P@20=69.1% ✅ |
| T57 | `train_validate_v3.py` | Sombra real → P@10=50% ✅ · P@20=70% ✅ |
| T57b | `freeze_total_v3.py` | Treino total + correção false positive |

## ✅ Fase 4.2 — Inferência (Implementada)
> Script: `sentinela_inference.py` · Status: **FUNCIONAL**  
> Saída validada: ranking top-40 + alertas intel + JSON para integração

## ✅ Fase 6 — Fine-Tuning em Tempo Real (Implementada)
> Script: `finetune_realtime_v1.py` · Status: **FUNCIONAL**  
> Fine-tuner ativado (+10pp vs base no período recente) · Saídas: JSON + CSV + pkl

---

## 🔵 Fase 4 — Promoção e Integração (Próxima)

### 4.1 Promoção Manual

**Critério:** Revisar `ranking_atual_v3_freeze.csv` operacionalmente  
**Comando de promoção:**

```powershell
# Executar após aprovação
copy "tests\Sentinela\lgbm_lean_v3_freeze.pkl" "models\active\lgbm_lean_v3_freeze.pkl"
```

**O que verificar antes de promover:**
- [ ] Top-10 bairros fazem sentido operacionalmente (sem bairros comerciais sem CVLI)
- [ ] Alertas de Intel (MONDUBIM, JANGURUSSU, CURIO) são conhecidos do campo
- [ ] Bairros ausentes do top-10 que deveriam estar (validar com analistas)

### 4.2 Integração com o Sistema

O modelo expõe uma interface simples de inferência:

```python
import pickle, pandas as pd, numpy as np

# Carregar
with open("models/active/lgbm_lean_v3_freeze.pkl", "rb") as f:
    payload = pickle.load(f)

ranker       = payload["ranker"]
feat_names   = payload["feat_names_lgbm"]
top_bairros  = payload["top_bairros"]
ewma_weights = payload["ewma_weights"]

# Inferência: dado df_features (40 linhas × 10 cols), obter ranking
scores_lgbm = ranker.predict(df_features[feat_names])  # → array(40,)
# Ensemble com EWMA-Multi já calculado separadamente
```

**Script de integração a criar:** `sentinela_inference.py`  
Função principal: `get_ranking(data_ref: date) → List[{bairro, score, rank}]`

---

## 🟡 Fase 5 — Re-treino Periódico (Manutenção)

### Quando re-treinar

| Gatilho | Ação |
|---------|------|
| Novos dados chegam (semanal/mensal) | Rodar `freeze_total_v3.py` → novo `.pkl` |
| P@10 monitorado cai abaixo de 35% | Executar `train_validate_v3.py` para diagnóstico |
| Mudança de padrão territorial detectada | Re-treino + revisão dos top-40 bairros |

### Protocolo de re-treino

```
1. Atualizar CSVs em data/raw/
2. .\.venv\Scripts\python.exe tests/Sentinela/freeze_total_v3.py
3. Revisar ranking_atual_v3_freeze.csv
4. Se aprovado → copiar para models/active/
5. Arquivar modelo anterior com timestamp
```

### Monitoramento de drift

Adicionar ao `freeze_total_v3.py` (futuramente):
- Comparar ranking atual vs ranking semana anterior (overlap @ top-10)
- Alertar se mudança > 4 posições nos top-5 sem justificativa de intel

---

## 🟠 Fase 6 — Fine-Tuning em Tempo Real (Segunda Fase)

> **Contexto:** Mencionado como segunda fase pelo usuário. Usa o LGBM Lean como modelo base  
> para ajuste incremental com dados dos últimos 30 dias (janela deslizante curta).

### Objetivo

Capturar **quebras de padrão imediatas** — ex: disputa territorial emergindo em bairro historicamente calmo — sem esperar o próximo ciclo de re-treino mensal.

### Estratégia

```
Modelo base:  lgbm_lean_v3_freeze.pkl  (treinado no histórico completo)
              ↓ Carregado com pesos fixos
Fine-tuner:   LGBM Lean com janela deslizante de 30 dias
              ↓ Treina toda semana em dados recentes
Ensemble:     70% base + 30% fine-tuner
              ↓ Produz ranking final
```

### Por que o LGBM Lean é ideal para fine-tuning

- **10 features apenas** → treina em < 5 segundos com 30 dias de dados
- **Regularização forte** (reg_lambda=2.0) → não colapsa em overfitting com poucos dados
- **Sem estado** → pode ser substituído a qualquer momento sem afetar o modelo base
- **Explicável** → feature importance legível para operadores de campo

### Script a criar: `finetune_realtime_v1.py`

**Inputs:**
- `lgbm_lean_v3_freeze.pkl` (modelo base congelado)
- Dados dos últimos 30 dias (janela deslizante)
- Peso do fine-tuner no ensemble (padrão: 30%)

**Outputs:**
- `lgbm_finetune_current.pkl` (sobrescrito a cada rodada)
- `ranking_realtime.csv` com scores combinados
- Delta de posições vs ranking base (alertas de mudança)

**Trigger:** Automático ao receber novos dados, ou manual via comando

```powershell
.\.venv\Scripts\python.exe tests/Sentinela/finetune_realtime_v1.py --dias 30 --peso-ft 0.30
```

### Critério de ativação do fine-tuner

O fine-tuner só contribui para o ensemble se:
- Tem ≥ 3 eventos CVLI nos últimos 30 dias nos top-40 bairros (sinal suficiente)
- Sua P@10 nos últimos 14 dias > P@10 do modelo base puro (validação interna)
- Caso contrário: ensemble retorna 100% modelo base (fallback seguro)

---

## 🔴 Fase 7 — Hibridismo com ST-GAT (Exploratório)

> **Contexto:** O ST-GAT (Tentativas 1–54) atingia 42.9% P@10 internamente mas  
> sofria com overfitting e instabilidade. O LGBM Lean oferece base estável para hibridismo.

### Hipótese

```
Score_final = α × ST_GAT_score + β × LGBM_score + γ × EWMA_score
```

Com calibração dinâmica de α, β, γ baseada na performance recente de cada componente.

### Pré-requisitos

- [ ] ST-GAT estabilizado (P@10 > 40% consistente por 5 folds)
- [ ] Infraestrutura de inferência GPU disponível em produção
- [ ] Comparação justa: mesmo protocolo de folds que o LGBM

### Scripts existentes relevantes

| Script | Localização | Status |
|--------|-------------|--------|
| `train_t33_elite.py` | `logs/training_ELITE_P10.log` | Treinado até Mar/2026 |
| `st_gat_inference.py` | A criar | Pendente |
| `hybrid_ensemble.py` | A criar | Fase 7 |

---

## 📁 Estado Final da Pasta `tests/Sentinela/`

| Arquivo | Papel | Modificar? |
|---------|-------|------------|
| `lgbm_lean_v3_freeze.pkl` | 🟢 Modelo candidato produção | Nunca editar |
| `freeze_total_v3.py` | Re-treino periódico | Manutenção |
| `train_validate_v3.py` | Auditoria/validação sombra | Manutenção |
| `benchmark_correto.py` | Baseline V1 (referência) | Arquivar futuramente |
| `ranking_atual_v3_freeze.csv` | Ranking atual com explicações | Regenerado a cada treino |
| `freeze_report.txt` | Relatório do treino final | Regenerado a cada treino |
| `BENCHMARK_EXPLICACAO.txt` | Protocolo de benchmark | Referência permanente |
| `ROADMAP.md` | **Este arquivo** | Atualizar a cada fase |

---

## 📊 Métricas de Referência Consolidadas

| Modelo | P@10 | P@20 | Tipo |
|--------|------|------|------|
| Chance aleatória | 25.0% | 50.0% | Baseline |
| EWMA simples (hl=14) | 36.8% | 58.0% | V1 |
| LightGBM V1 | 30.8% | 61.3% | V1 |
| ST-GAT completo | 42.9% | — | Interno |
| **EWMA-Multi (V3)** | **50.0%** | 65.0% | **Sombra real** |
| **LGBM Lean (V3)** | 30.0% | **70.0%** | **Sombra real** |
| **Ensemble V3** | 30.0% | **70.0%** | **Sombra real** |

> **Nota:** Métricas de sombra são 1 único período (Abr/2026, 2 eventos CVLI).  
> Benchmarks de 6 folds: EWMA-Multi P@10=41.5% médio · Ensemble P@20=69.1% médio.

---

## 🔑 Features do Modelo (por importância)

| # | Feature | Peso% | Descrição |
|---|---------|-------|-----------|
| 1 | `cvp_cvli_ratio` (calibrado) | 18.8% | Razão CVP/CVLI × sqrt(hist_pct) — escalada de crime patrimonial para homicídio |
| 2 | `target_enc` | 15.7% | Média histórica de CVLI por bairro (expanding) |
| 3 | `cvp_ewma_30d` | 10.5% | Tendência de longo prazo de CVP |
| 4 | `intel_ewma_14d` | 10.8% | Score de intel de tropa (armas+drogas+veículos) com pesos por natureza |
| 5 | `inter_intel_cvli` | 8.5% | Interação intel × CVLI (pressão simultânea) |
| 6 | `nbr_cvli_30d` | 8.3% | CVLI vizinhos nos últimos 30 dias (retaliação espacial) |
| 7 | `intel_ewma_7d` | 8.8% | Intel de tropa de curto prazo |
| 8 | `hist_pct` | 6.9% | Percentil histórico do bairro no ranking de CVLI |
| 9 | `cvp_ewma_14d` | 6.5% | Tendência CVP médio prazo |
| 10 | `cvp_ewma_7d` | 5.2% | Tendência CVP curto prazo |

---

*Última atualização: 14/04/2026 — Sprint T55→T57b concluída*  
*Próxima revisão: ao promover modelo ou iniciar Fase 5/6*
