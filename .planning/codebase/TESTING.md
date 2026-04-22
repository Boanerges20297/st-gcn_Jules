# Testing Patterns

**Analysis Date:** 2026-04-19

## Test Framework

**Runner:**
- Scripts customizados de treinamento e validação em `tests/Sentinela/`.
- pytest (detectado via `.pytest_cache`).

**Assertion Library:**
- Assertions manuais e métricas de ML (`P@k`, `NDCG`, `Recall`).

**Run Commands:**
```bash
# Treinar e validar modelo Challenger (Sentinela V3)
python tests/Sentinela/freeze_total_v3.py

# Validar modelo candidato em cenário sombra
python tests/Sentinela/train_validate_v3.py

# Promover modelo para produção (com checks de segurança)
python tests/Sentinela/promote_model.py
```

## Test File Organization

**Location:**
- Testes de ML e Lab: `tests/Sentinela/`
- Testes unitários/funcionais: `tests/` (não detalhado mas presente na estrutura).

**Naming:**
- `freeze_total_v3.py`
- `train_validate_v3.py`
- `promote_model.py`

## Test Structure

**ML Validation Pattern:**
```python
# Em tests/Sentinela/train_validate_v3.py
def evaluate_model(model, features, targets):
    # Split temporal ou holdout
    # Predição
    # Cálculo de P@10 e P@20
    # Comparação com baseline (Champion)
    pass
```

**Promotion Pattern:**
```python
# Em tests/Sentinela/promote_model.py
CHECKLIST = [
    ("Modelo candidato existe",           lambda: os.path.exists(CANDIDATE)),
    ("Relatório de treino existe",        lambda: os.path.exists(REPORT)),
    ("Modelo >= 500 KB (não corrompido)", lambda: os.path.getsize(CANDIDATE) >= 500_000),
]
```

## Mocking

**Framework:** `unittest.mock` (implícito) ou substituição manual de arquivos.

**Patterns:**
- Não detectado uso extensivo de Mocks; o sistema prefere rodar contra snapshots de dados reais em `data/raw/`.

## Fixtures and Factories

**Test Data:**
- CSVs de amostra em `tests/Sentinela/` (ex: `ranking_atual_v3_freeze.csv`).

**Location:**
- `data/raw/` serve como fonte de "fixtures" reais para validação.

## Coverage

**Requirements:** Não há metas formais de cobertura de código, o foco é na performance preditiva (P@10 e P@20).

## Test Types

**Unit Tests:**
- Testes de normalização e processamento de strings (ex: `normalize_name`).

**Validation Tests (ML):**
- Validação temporal em janela deslizante para o modelo Sentinela.
- Monitoramento de P@10/P@20 em tempo real via `efficiency_monitor.py`.

**Shadow Validation:**
- O modelo Challenger roda em "modo sombra" no `app.py`, sendo avaliado mas não necessariamente impactando 100% dos scores até que o blend EMA o valide.

## Common Patterns

**Async Testing:**
- Não aplicável (Uso de threads em background no Flask, validadas via logs).

**Error Testing:**
- Validação de arquivos corrompidos ou ausentes no script de promoção.

---

*Testing analysis: 2026-04-19*
