# Gemini no Projeto

O Gemini nao e o motor principal de risco do sistema.

## Papel atual

Hoje o score operacional principal vem do backend:

- `Poisson Ranker Estadual`

O Gemini permanece como componente auxiliar para tarefas de apoio, como:

- interpretacao de textos
- enriquecimento narrativo
- apoio a explicacoes
- fluxos especificos de IA fora do ranking principal

## O que nao e mais verdade

Nao tratar mais o Gemini como parte da arquitetura champion de predicao CVLI.

Em especial, este projeto nao opera mais oficialmente com:

- `DeepSTGAT` como champion
- `Champion/Challenger` ST-GAT + Sentinela como verdade corrente de producao
- blending neural como caminho principal para `/api/risk`

## Fonte de verdade para o risco

Para entender a arquitetura atual, use:

- [CURRENT_ARCHITECTURE.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CURRENT_ARCHITECTURE.md)
- [CURRENT_OPERATIONS.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CURRENT_OPERATIONS.md)
- [CVLI_STOCHASTIC_BENCHMARK.md](C:/Users/Boanerges/Desktop/Projetos/Report Preview/docs/CVLI_STOCHASTIC_BENCHMARK.md)
