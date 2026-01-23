# ETL DE PRODUÇÃO V2 - Relatório de Execução

**Data:** 2026-01-23 10:57:47

## Resumo

- **Período:** 2022-01-01 a 2026-01-11 (1472 dias)
- **Bairros normalizados:** 121
- **Eventos CVLI:** 3,180
- **Eventos Prisões:** 3,073
- **Eventos Apreensões:** 15,209

## Tensores Gerados

- Tensor univariado (CVLI): 1472×121 = 178,112 células
- Tensor multivariado: 1472×121×3 = 534,336 células
- Sparsidade CVLI: 98.34%

## Arquivos Salvos

### Tensores
- `tensor_cvli_univariado.npy` (1472×121)
- `tensor_prisoes.npy` (1472×121)
- `tensor_apreensoes.npy` (1472×121)
- `tensor_multivariado.npy` (1472×121×3)

### CSVs
- `cvli_producao.csv` (3,180 registros)
- `operacional_producao.csv` (29,286 registros)

### Metadados
- `metadata_producao_v2.json`

## Próximos Passos

1. Usar tensores em `src/trainer.py` para retreinamento
2. Adaptar `src/model.py` para nova estrutura
3. Atualizar `src/data_loader.py` com novos paths
4. Validar com `src/predict.py`

---
**Status:** ✅ CONCLUÍDO
