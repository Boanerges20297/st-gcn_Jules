# Resultado negativo util: raio adaptativo por tamanho do bairro

Data: 2026-07-22

## Experimento

Foi testada uma colmeia dentro dos bairros preditos, variando o raio dos hexagonos conforme o tamanho do bairro:

- bairros pequenos: hexagonos menores;
- bairros medios: hexagonos intermediarios;
- bairros grandes: hexagonos maiores.

O objetivo era preservar coerencia operacional entre bairros de tamanhos diferentes e maximizar captura por area.

## Resultado

Na rodada ampla com horizonte de 30 dias:

| Metodo | Captura | Area | Captura/100 km2 | Observacao |
|---|---:|---:|---:|---|
| Bairro top 30 | 89,04% | n/a | n/a | Melhor captura geral |
| Spatial greedy 40 zonas, raio 2 km | 85,73% | 502,65 km2 | 0,171 | Alta captura, area muito grande |
| Colmeia GA adaptativa por bairro | 18,70% | 39,21 km2 | 0,477 | Baixa captura |
| Colmeia GA adaptativa eficiente | 12,70% | 12,18 km2 | 1,042 | Melhor eficiencia, captura insuficiente |

## Interpretacao

A adaptacao do raio por tamanho do bairro melhorou a coerencia visual e pode aumentar eficiencia por km2, mas reduziu fortemente a captura. Na forma atual, nao deve ser tratada como ganho real.

A causa provavel e que o score do hexagono ainda esta pouco individualizado. A variacao geometrica por tamanho do bairro nao compensa a ausencia de predicao local forte por celula.

## Decisao

Nao usar raio adaptativo por tamanho do bairro como solucao principal neste momento. Manter como resultado negativo util e priorizar o proximo teste: score preditivo individual por hexagono, mantendo bairro como camada superior.
