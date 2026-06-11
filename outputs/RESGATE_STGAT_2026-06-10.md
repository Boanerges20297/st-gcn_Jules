# Resgate Cirúrgico do ST-GAT — 2026-06-10

## Objetivo
Restaurar o modelo principal de Fortaleza para uma variante `ST-GAT` confiável, usando um ponto anterior de aproximadamente 30 dias.

## Commit escolhido
- `0cd19e9d` — `2026-05-07 09:31:21`

## Motivo da escolha
- O checkpoint de Fortaleza nesse commit usa `DeepSTGAT_64`.
- Ele carrega com `41` canais no metadata legado.
- Foi mais seguro que os checkpoints de `2026-05-11`, que dependiam de configuração legada com `39` canais.

## Arquivos efetivamente resgatados
- `models/active/fortaleza_model_active.pth`

## Ajuste técnico necessário
- `src/core/orchestrator.py` foi ajustado para reconhecer checkpoints legados que salvam a arquitetura em `config.arch`, além de `model_class`.

## Arquivos preservados sem resgate
- `models/active/rmf_model.pth`
- `models/active/interior_model.pth`
- `src/core/data_processing.py`
- `src/core/architectures.py`

## Justificativa para não ampliar o resgate
- O foco operacional é `CVLI` em Fortaleza.
- `RMF` e `Interior` continuaram carregando normalmente.
- `data_processing.py` atual já é compatível em runtime porque o `orchestrator` faz padding de canais quando o checkpoint exige mais canais que o dataset atual possui.

## Backup local criado
- `models/active/fortaleza_model_active.pre_restore_2026-06-10.pth`

## Validação pós-resgate
- Fortaleza carregou como `DeepSTGAT_64`
- Janela: `120`
- Canais do modelo: `41`
- Dataset carregado: shape `109 x 1620 x 37`
- API `/api/risk?region=fortaleza` respondeu `200`

## Estado final
- **Modelo principal de Fortaleza restaurado para ST-GAT**
- `ST-GAT` segue como modo padrão no sistema
