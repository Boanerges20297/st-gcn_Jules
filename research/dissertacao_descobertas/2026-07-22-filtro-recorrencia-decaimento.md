# Filtro de Recorrencia com Decaimento Temporal

## Hipotese

Como os CVLIs de Fortaleza se concentram recorrentemente em cerca de 30 a 40 bairros, um filtro de elegibilidade deve reduzir a esparsidade e melhorar a captura futura.

## Implementacao testada

Script: `scripts/experiments/fortaleza_hybrid_capture_spike.py`

O experimento reaproveita `tests/Sentinela/freeze_total_v3.py`, usando os 40 bairros recorrentes ja derivados na plataforma.

Foram comparados:
- sem filtro;
- filtro `TOP30`;
- filtro `TOP20`;
- score historico recorrente;
- score CVLI tatico;
- score hibrido bairro;
- score hibrido recente com decaimento temporal.

Decaimento temporal usado:
- meia-vida recente: 30 dias, peso 75%;
- meia-vida antiga: 180 dias, peso 25%.

## Resultado observado

Horizonte de 14 dias, `K=20`:
- `CVLI_TATICO_TOP20`: captura media de aproximadamente 70,8%;
- `CVLI_TATICO` sem filtro: captura media de aproximadamente 65,7%.

Horizonte de 30 dias, `K=20`:
- `HIBRIDO_RECENTE_TOP30`: captura media de aproximadamente 67,5%;
- `CVLI_TATICO` sem filtro: captura media de aproximadamente 63,3%.

Para `K=10`, o `TOP30` foi mais estavel que `TOP20`, preservando melhor bairros emergentes.

## Interpretacao provisoria

O filtro agressivo reduz esparsidade e melhora captura quando a operacao seleciona um conjunto maior de areas (`K=20`).

O `TOP20` parece adequado como modo operacional concentrado. O `TOP30` parece melhor como filtro padrao, por equilibrar recorrencia e possibilidade de emergencia.

## Proxima verificacao

Antes de implementar algoritmo genetico, testar uma estrategia greedy de microzonas sobre os bairros elegiveis. Se o greedy nao superar bairro/top-K simples, o GA ainda e complexidade prematura.

