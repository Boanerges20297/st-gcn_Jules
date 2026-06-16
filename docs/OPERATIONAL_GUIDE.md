# Guia Operacional

Este guia descreve apenas a operacao vigente do projeto.

## Arquitetura em operacao

- backend oficial: `Poisson Ranker Estadual`
- regioes atendidas: `fortaleza`, `rmf`, `interior`
- orquestracao: `src/core/orchestrator.py`
- artefatos promovidos: `models/active/production/poisson/`

## Inicializacao local

```powershell
.\.venv\Scripts\python.exe app.py
```

Ao iniciar, a aplicacao:

1. carrega os dados processados
2. carrega os tres artefatos Poisson ativos
3. registra a validacao de startup em `VALIDATION_LOG.md`
4. expõe o endpoint `/api/risk`

## Rotina recomendada

### Diario

- conferir se a aplicacao subiu sem erro
- validar `VALIDATION_LOG.md`
- validar resposta de `/api/risk`

### Semanal

- atualizar dados enriquecidos e processados
- retreinar/promover os modelos Poisson estaduais
- revisar benchmark ou playoff mensal se houver degradacao

## Retreino oficial

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

Quando rodar:

- `1x por semana` em rotina normal
- apos atualizacao relevante de `data/raw/*ENRIQUECIDO*.csv`
- antes, se `VALIDATION_LOG.md` piorar por `2 ou 3 ciclos`

## Sequencia operacional do retreino

1. atualizar a base bruta enriquecida
2. reprocessar os dados se necessario

```powershell
.\.venv\Scripts\python.exe src\core\data_processing.py
```

3. promover os modelos estaduais

```powershell
.\.venv\Scripts\python.exe scripts\promote_statewide_poisson_regressors.py
```

4. reiniciar a aplicacao
5. conferir `VALIDATION_LOG.md`, `/api/risk` e artefatos Hermes

## Health check minimo

- `VALIDATION_LOG.md` atualizado na inicializacao
- `/api/risk` retornando `200`
- `model_architecture` reportando `Poisson Ranker Estadual`
- `src/core/orchestrator.py` carregando `fortaleza`, `rmf` e `interior`

## O que nao esta mais valendo

Este projeto nao depende mais, como fluxo oficial de producao, de:

- ST-GCN como champion
- monitor automatico de retreino em background
- stack docker obrigatoria para operacao local
