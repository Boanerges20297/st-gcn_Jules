# Arquitetura Atual

## Estado atual

O sistema de produção atual usa:

- `Poisson Ranker Estadual` como backend principal
- três regiões operacionais:
  - Fortaleza
  - RMF
  - Interior
- horizonte de previsão de `14 dias`
- `VALIDATION_LOG.md` e `EfficiencyMonitor` como trilhas principais de acompanhamento

## Ponto de entrada

- App principal: [app.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/app.py:1)

Fluxo:

1. `app.py` sobe Flask e carrega dados/processados.
2. `StateOrchestrator` monta os especialistas regionais.
3. Cada região usa um artefato Poisson salvo em `models/active/production/poisson/`.
4. `EfficiencyMonitor` avalia performance recente.
5. `validation_logger` registra a janela recente em `VALIDATION_LOG.md`.

## Componentes centrais

### Orquestração

- [src/core/orchestrator.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/orchestrator.py:1)
  - carrega os modelos ativos por região
  - gera `scores_map`
  - exporta artefatos Hermes
  - mantém compatibilidade com API e health

### Backend Poisson

- [src/core/fortaleza_poisson_backend.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/fortaleza_poisson_backend.py:1)
  - apesar do nome histórico, hoje serve como backend Poisson regional
  - monta features clássicas
  - treina payloads serializáveis
  - executa inferência compatível com o orquestrador

### Processamento de dados

- [src/core/data_processing.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/data_processing.py:1)
  - reconstrói `data/processed/*.pkl`
  - é a origem dos tensores e grafos usados pelos modelos

### Monitoramento

- [src/core/efficiency_monitor.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/efficiency_monitor.py:1)
  - mede `P@10`, `P@20`, recall e cobertura por região

- [src/core/validation_logger.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/validation_logger.py:1)
  - escreve sessões em `VALIDATION_LOG.md`

- [src/core/health_monitor.py](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/src/core/health_monitor.py:1)
  - observa saúde operacional da aplicação

## Modelos ativos

Local oficial:

- [models/active/production/poisson](/C:/Users/Boanerges/Desktop/Projetos/Report%20Preview/models/active/production/poisson)

Artefatos:

- `fortaleza_poisson_regressor.pkl`
- `rmf_poisson_regressor.pkl`
- `interior_poisson_regressor.pkl`

Metadados correspondentes:

- `*.json` no mesmo diretório

## Estrutura lógica

```text
app.py
  -> src/core/orchestrator.py
      -> src/core/fortaleza_poisson_backend.py
      -> data/processed/*.pkl
      -> models/active/production/poisson/*.pkl
      -> outputs/hermes/*
  -> src/core/efficiency_monitor.py
  -> src/core/validation_logger.py
```

## O que é legado

Não são mais o caminho principal de produção:

- `models/active/legacy_torch/`
- `src/core/champion_challenger.py`
- `scripts/training/Active/train_all_specialists.py`
- checkpoints ST-GAT/ST-GCN históricos

Esses itens foram preservados para:

- auditoria
- rollback
- benchmark
- referência histórica
