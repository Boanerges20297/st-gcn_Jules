# Instruções de Operação do Agente

Estas instruções valem para todo o workspace.

## Objetivo

Utilize as ferramentas MCP para obter contexto antes de analisar ou modificar código, escolhendo a ferramenta mais adequada para cada tarefa.

## Uso das ferramentas

### CodeGraph

Utilize o CodeGraph quando o objetivo for:

- compreender a arquitetura;
- localizar implementações;
- entender fluxos de execução;
- navegar entre símbolos;
- descobrir onde uma funcionalidade está implementada;
- realizar onboarding ou compreensão geral do projeto.

### Code Review Graph (CRG)

Utilize o Code Review Graph quando o objetivo envolver:

- análise de impacto (blast radius);
- revisão de código;
- avaliação de risco;
- comunidades (communities);
- fluxos (flows);
- cobertura de testes;
- planejamento de refatorações;
- identificação de dívida técnica;
- análise antes de alterações significativas.

## Estratégia

Escolha a ferramenta mais apropriada para a solicitação.

Quando necessário, utilize ambas, preferencialmente nesta ordem:

1. CodeGraph para compreender o código.
2. CRG para avaliar impacto e riscos.

Evite consultas redundantes quando apenas uma ferramenta for suficiente.

## Durante implementações

Antes de alterar código relevante:

- compreenda o contexto utilizando CodeGraph;
- avalie impactos utilizando CRG quando houver possibilidade de afetar outros módulos;
- somente então implemente a solução.

## Após alterações importantes

Sempre que concluir alterações estruturais ou funcionais relevantes:

- utilize o CRG para verificar impactos;
- reporte riscos encontrados;
- informe funções, comunidades ou fluxos afetados, quando aplicável.

## Exceções

Não utilize MCP quando:

- a solicitação não envolver código;
- o usuário pedir explicitamente para não utilizar essas ferramentas.

## Instruções de Uso

Sempre que utilizar um MCP, informe na resposta quais ferramentas MCP foram utilizadas.

## Regra de economia de tokens (obrigatória)

Objetivo principal: reduzir consumo de tokens.

- Antes de chamar MCP, avaliar custo x benefício.
- Se a tarefa for simples, direta ou localizada (por exemplo: ajustar trecho já conhecido, responder algo objetivo, validar um único ponto), não usar MCP.
- Se a consulta MCP exigir múltiplas chamadas amplas sem ganho claro de precisão, interromper e seguir com leitura direta mínima do arquivo relevante.
- Evitar rodar CodeGraph e CRG na mesma interação quando uma única ferramenta já resolver.
- Limitar exploração inicial a 1 chamada focal por ferramenta; expandir somente se houver lacuna real de contexto.
- Se perceber que a busca via MCP está ficando maior que o necessário, priorizar resposta com o contexto já suficiente e declarar a limitação.