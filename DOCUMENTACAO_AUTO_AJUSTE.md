# Auto-Ajuste de Modelos — REPORT PREVIEW

Sistema de calibração automática dos parâmetros de inferência do motor ST-GAT, ativado quando o modelo deixa de cobrir adequadamente os territórios de tensão conhecida.

---

## Filosofia: Termômetro Territorial

O REPORT PREVIEW **não prevê onde haverá crime**. Prevê **tensão territorial** — zonas sob pressão de facções, rotas de disputa, territórios de domínio ativo.

Um território controlado por facção pode ter 0 crimes recentes justamente por ter controle total (sem resistência). O score alto nesse caso é **informação correta**, não outlier.

Por isso:
- Outliers de score alto são **preservados** (dampening removido)
- Territórios com domínio de facção têm piso mínimo de tensão garantido
- O ground truth de avaliação combina CVLI + eventos exógenos + presença de facção

---

## Visão Geral

O auto-ajuste **não retreina** o modelo neural. Ele ajusta os parâmetros pós-inferência do `StateOrchestrator` em tempo real para garantir que zonas de tensão conhecida apareçam no topo do ranking.

```
Avaliação → Cobertura territorial < 80% → Calibração automática → Nova avaliação
                                                ↓ (se recuperou)
                                           Rollback completo
                                                ↓ (se não recuperou após 3 tentativas)
                                           Alerta CRITICAL → Intervenção manual
```

---

## Trigger de Calibração

**Métrica:** Cobertura de Territórios de Facção no Top-20%

> Percentual dos territórios com domínio de facção confirmado que aparecem no top-20% do ranking de tensão territorial.

| Limiar | Ação |
|--------|------|
| ≥ 80%  | Normal — nenhuma ação |
| < 80%  | Calibração automática (1 passo) |
| < 80% após 3 passos | Alerta CRITICAL — intervenção manual necessária |

Configuração em `app.py`:
```python
_FACTION_COVERAGE_MIN = 0.80
```

---

## Parâmetros Ajustados

`dampening` foi removido — comprimir outliers de tensão é irracional para o propósito da aplicação.

| Parâmetro           | Padrão | Mín   | Máx  | Efeito                                              |
|---------------------|--------|-------|------|-----------------------------------------------------|
| `tension_factor`    | 0.50   | 0.50  | 2.00 | Peso do `tension_index` (facções) nos logits       |
| `tag_bias_direct`   | 1.50   | 1.50  | 3.50 | Boost no nó com gatilho INTEL_TRIGGER              |
| `tag_bias_neighbor` | 0.50   | 0.50  | 1.20 | Vazamento do TAG-Bias para nós vizinhos            |
| `min_risk`          | 30.0   | 15.0  | 30.0 | Piso de tensão (%) para territórios com CVLI recente **ou** domínio de facção |

---

## Passos de Ajuste

Cada degradação detectada aplica **1 passo**. Máximo de **3 passos** por região.

| Parâmetro           | Delta P20 (normal) | Delta P10 (agressivo) |
|---------------------|--------------------|-----------------------|
| `tension_factor`    | +0.20              | +0.30                 |
| `tag_bias_direct`   | +0.30              | +0.45                 |
| `tag_bias_neighbor` | +0.10              | +0.15                 |

**Exemplo — 3 passos (P20):**
```
Passo 0: tension=0.50, tag_bias=1.50
Passo 1: tension=0.70, tag_bias=1.80
Passo 2: tension=0.90, tag_bias=2.10
Passo 3: tension=1.10, tag_bias=2.40
```

---

## Ciclo de Vida de uma Degradação

```
1. efficiency_monitor.run_evaluation() constrói ground truth composto
   (CVLI + exógenos + territórios de facção)
2. _check_faction_coverage_alerts() calcula: facções no top-20% / total facções
3. Se cobertura < 80%:
   - Alerta HIGH: "Tensão territorial subestimada — FORTALEZA: 65%"
   - ModelCalibrator.on_degradation(): tension_factor +0.20, tag_bias +0.30
   - Salvo em data/calibration_state.json
4. Próxima inferência usa parâmetros ajustados

Cenário A — Recuperação:
   Cobertura volta a ≥ 80%
   → Rollback COMPLETO para valores originais (todos os parâmetros)
   → steps = 0

Cenário B — Sem melhora após 3 passos:
   Cobertura ainda < 80%
   → Alerta CRITICAL: "INTERVENÇÃO MANUAL NECESSÁRIA"
   → Suppression 12h
```

---

## Rollback

O rollback é **sempre completo** — todos os parâmetros voltam ao valor original simultaneamente, independente de quantos passos foram aplicados.

```python
# Valores restaurados no rollback
{
    'tension_factor':     0.50,
    'min_risk':          30.0,
    'tag_bias_direct':    1.50,
    'tag_bias_neighbor':  0.50,
}
```

O rollback só ocorre quando a cobertura territorial volta a ≥ 80% na próxima avaliação.

---

## Persistência

Estado de calibração salvo em `data/calibration_state.json`. Os parâmetros são **reaplicados automaticamente** ao reiniciar o servidor via `model_calibrator.reapply_on_startup(orchestrator)`.

---

## API — Endpoint de Status

```
GET /api/admin/health/calibration-status
```

**Resposta:**
```json
{
  "available": true,
  "regions": {
    "fortaleza": {
      "steps": 2,
      "max_steps": 3,
      "is_degraded": true,
      "is_critical": false,
      "last_event": {
        "timestamp": "2026-03-01T20:30:00",
        "trigger": "fortaleza.p20=43.0% < 50%",
        "step": 2
      }
    },
    "rmf": {
      "steps": 0,
      "is_degraded": false,
      "is_critical": false,
      "last_event": null
    }
  }
}
```

---

## Alertas Gerados

| Tipo de Alerta                    | Severidade | Condição                                      |
|-----------------------------------|------------|-----------------------------------------------|
| `model_degraded_{region}_{metric}`| HIGH       | Métrica abaixo do limite (supressão 24h)      |
| `auto_calibration_{region}`       | MEDIUM     | Passo de ajuste aplicado                      |
| `calibration_rollback_{region}`   | LOW        | Rollback completo após recuperação            |
| `calibration_maxed_{region}`      | CRITICAL   | 3 passos sem recuperação (supressão 12h)      |

---

## Arquivos Relacionados

| Arquivo                            | Função                                          |
|------------------------------------|-------------------------------------------------|
| `src/core/model_calibrator.py`     | Lógica de calibração, rollback e persistência  |
| `src/core/orchestrator.py`         | Motor de inferência com `calib_params` por região |
| `src/core/admin_health_routes.py`  | Endpoint `/calibration-status`                 |
| `app.py`                           | `_CONFIDENCE_THRESHOLDS`, `_check_model_confidence_alerts()` |
| `data/calibration_state.json`      | Estado persistido entre reinícios              |

---

## Limitações

- O auto-ajuste melhora o **ranking relativo** dos nós, não retreina os pesos neurais.
- Se a degradação for causada por **deriva de dados** (ex.: novo padrão de criminalidade não visto no treino), o retreinamento é necessário.
- Após 3 passos sem melhora, o sistema sinaliza mas **não reverte automaticamente** — aguarda confirmação via alerta CRITICAL.
- Os parâmetros não são reaplicados automaticamente após reinício do servidor (ver nota acima).
