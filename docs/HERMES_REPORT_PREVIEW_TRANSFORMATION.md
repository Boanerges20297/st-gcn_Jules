# Transformacao do Hermes no Report Preview

Data de referencia: 20/05/2026
Status: ativo em producao operacional local

## Objetivo

Este documento registra a transformacao do Hermes dentro do Report Preview: de um contexto auxiliar de prompt para uma camada operacional de resposta gerencial, com artefatos proprios, gateway de Telegram, autenticacao local e uso do Gemini CLI como motor de resposta.

O objetivo final da transformacao foi simples:

- transformar o Report Preview em uma superficie de consulta conversacional para gestores e analistas;
- responder em Portugues do Brasil, com foco operacional e analitico;
- usar primeiro os artefatos oficiais do pipeline de risco;
- nunca deixar o gestor sem resposta quando houver ao menos sinais taticos disponiveis;
- enquadrar toda resposta como previsao operacional para os proximos 7 dias.

## O que mudou

Antes da transformacao, o Hermes funcionava apenas como memoria e configuracao de contexto. A camada conversacional dependia mais do raciocinio geral do assistente do que de um contrato operacional fechado com os artefatos do projeto.

Depois da transformacao, o Hermes passou a operar como uma camada de orquestracao analitica em torno do Report Preview:

- o pipeline oficial escreve artefatos padronizados em `outputs/hermes/`;
- o Gemini CLI consome esses artefatos por meio de um wrapper PowerShell dedicado;
- o Telegram usa um gateway proprio, desacoplado da integracao nativa do Hermes;
- o acesso ao bot ficou restrito por autenticacao SQLite local;
- o fallback deixou de ser `sem dados` e passou a ser uma projecao tatico-operacional baseada nos ultimos 14 dias;
- a resposta final passou a ser orientada para previsao dos proximos 7 dias.

## Arquitetura resultante

O desenho atual da camada Hermes no Report Preview e o seguinte:

```text
src/core/orchestrator.py
  -> gera artefatos oficiais em outputs/hermes/

ask_gemini_with_hermes_memory.ps1
  -> monta prompt com SOUL.md, .hermes.md, brief, CSVs e contexto tatico 14d
  -> chama Gemini CLI
  -> salva resposta e prompt em outputs/hermes/chat/

telegram_gemini_gateway.py
  -> recebe mensagens do Telegram por polling
  -> autentica usuario via SQLite local
  -> escolhe escopo da consulta
  -> chama ask_gemini_with_hermes_memory.ps1
  -> devolve a resposta ao usuario autenticado

manage_telegram_users.py + wrappers PowerShell
  -> cadastro, troca de senha, bloqueio, auditoria e bloqueio global
```

## Componentes principais

### 1. Pipeline oficial de artefatos

O arquivo `src/core/orchestrator.py` virou o ponto central da integracao. Alem de calcular o risco oficial, ele passou a emitir uma camada Hermes pronta para consumo por chat e Telegram.

Artefatos relevantes:

- `outputs/hermes/risk_brief_latest.md`: resumo gerencial curto para resposta rapida;
- `outputs/hermes/risk_snapshot_latest.md`: snapshot textual completo;
- `outputs/hermes/risk_snapshot_latest.csv`: base estruturada consolidada;
- `outputs/hermes/risk_snapshot_latest.json`: versao estruturada para citacao de campos e drivers;
- `outputs/hermes/risk_fortaleza_latest.csv`: recorte por Fortaleza;
- `outputs/hermes/risk_rmf_latest.csv`: recorte por RMF;
- `outputs/hermes/risk_interior_latest.csv`: recorte por Interior;
- `outputs/hermes/dados_status_enriquecido_14d_latest.csv`: fallback tatico dos ultimos 14 dias;
- `outputs/hermes/dados_status_enriquecido_14d_summary_latest.md`: resumo tatico 14d pre-calculado;
- `outputs/hermes/dados_status_enriquecido_14d_summary_latest.json`: versao estruturada do resumo tatico;
- `outputs/hermes/history/`: historico por execucao dos artefatos Hermes.

Essa mudanca foi importante porque moveu a conversa para cima de artefatos verificaveis, em vez de depender apenas de descricao textual dispersa no projeto.

### 2. Contrato local do Hermes

O arquivo `.hermes.md` passou a atuar como contrato comportamental do assistente dentro do Report Preview.

Esse contrato hoje define:

- idioma obrigatorio em Portugues do Brasil;
- tom institucional para oficiais, gestores e analistas;
- prioridade de leitura dos artefatos em `outputs/hermes/`;
- proibicao de inventar ranking, score, driver ou causalidade fora do que o projeto mostra;
- uso obrigatorio do CSV tatico 14d quando os artefatos oficiais nao bastarem;
- obrigacao de responder como previsao operacional para os proximos 7 dias.

Na pratica, `.hermes.md` deixou de ser uma nota de contexto e passou a ser a regra de comportamento da camada conversacional.

### 3. Wrapper do Gemini com memoria Hermes

O arquivo `ask_gemini_with_hermes_memory.ps1` virou a porta de entrada padrao para respostas conversacionais.

Ele faz cinco coisas:

- resolve o escopo da pergunta: geral, Fortaleza, RMF ou Interior;
- carrega `SOUL.md`, `.hermes.md`, brief, CSV de escopo e resumo tatico;
- tenta localizar alvo especifico da pergunta no snapshot e no CSV tatico 14d;
- monta um prompt fechado para analise do Gemini;
- grava prompt e resposta em `outputs/hermes/chat/` e `outputs/hermes/chat/history/`.

O ponto mais importante dessa evolucao foi o fallback. Quando o artefato Hermes nao traz referencia direta para uma localidade, o wrapper nao encerra a resposta com indisponibilidade. Em vez disso, ele usa o contexto tatico dos ultimos 14 dias para sustentar uma projecao util para os proximos 7 dias.

### 4. Gateway proprio para Telegram

O arquivo `telegram_gemini_gateway.py` substituiu, na pratica, a dependencia da integracao nativa do Hermes com Telegram.

Razoes da troca:

- reduzir fragilidade operacional da rota nativa;
- controlar melhor autenticacao e auditoria local;
- acoplar a resposta diretamente aos artefatos do Report Preview;
- usar o mesmo bot como canal de consulta gerencial sem depender da pilha completa do Hermes.

O gateway atual:

- executa polling direto na API do Telegram;
- persiste estado por chat em `outputs/hermes/chat/telegram_gateway_state.json`;
- responde com feedback intermediario enquanto a consulta esta em processamento;
- escolhe o escopo com heuristica simples por texto;
- chama o wrapper PowerShell do Gemini para produzir a resposta final.

### 5. Autenticacao e autorizacao locais

A seguranca da camada Telegram foi internalizada no proprio projeto.

Banco local:

- `data/users/telegram_auth.sqlite3`

Estruturas relevantes:

- tabela `users`: usuarios autorizados;
- tabela `auth_audit`: trilha de auditoria de login e eventos correlatos;
- tabela `auth_controls`: controles como bloqueio global.

Regras atuais:

- autenticacao por `username + password`;
- senha derivada com `PBKDF2-HMAC-SHA256` e `salt` individual;
- expiracao de sessao por chat;
- limite de tentativas falhas;
- lockout temporario;
- bloqueio global opcional.

Variaveis operacionais expostas em `.env.example`:

- `TELEGRAM_AUTH_SESSION_TTL_SECONDS=28800`
- `TELEGRAM_AUTH_MAX_FAILED_ATTEMPTS=5`
- `TELEGRAM_AUTH_LOCKOUT_SECONDS=900`

### 6. Ferramentas administrativas

O gerenciamento de acesso e auditoria ficou padronizado ao redor de `manage_telegram_users.py` e wrappers mais simples na raiz e em `powershell/`.

Capacidades administrativas ja incorporadas:

- cadastrar usuario;
- atualizar senha;
- ativar ou desativar usuario;
- listar usuarios;
- aplicar bloqueio global do bot;
- consultar auditoria de autenticacao.

Isso eliminou dependencia de alteracoes manuais no banco e formalizou a operacao de acesso ao bot.

## Contrato de resposta atual

O Hermes passou a responder sob um contrato mais restritivo e util para operacao.

Principios principais:

- responder em pt-BR, de forma curta e objetiva;
- usar `outputs/hermes/` como fonte oficial do ranking e do risco;
- preferir `risk_brief_latest.md` em chat curto;
- usar snapshot, CSV e JSON quando a pergunta pedir comparacao, auditoria ou explicabilidade;
- usar o resumo tatico 14d quando ele existir, antes de cair no CSV bruto;
- nunca afirmar ranking ou bairro sem lastro em artefato;
- nunca terminar em ausencia total de resposta quando houver sinal tatico util;
- converter a evidencia disponivel em previsao operacional para os proximos 7 dias.

Estrutura preferida das respostas:

1. `Dados ate`
2. `Fonte`
3. `Leitura rapida`
4. `Previsao para os proximos 7 dias`
5. `Por que importa`
6. `Proxima acao`

## Antes e depois

### Antes

- Hermes como camada mais generica de contexto;
- Telegram nativo menos previsivel para a operacao desejada;
- fallback frequente para ausencia de resposta quando nao havia referencia direta;
- pouca separacao entre fonte oficial, leitura tatico-complementar e memoria de prompt.

### Depois

- Hermes como camada operacional acoplada ao Report Preview;
- artefatos oficiais em `outputs/hermes/` como contrato de verdade;
- Gemini CLI como motor de resposta, com prompt fechado e historico auditavel;
- gateway Telegram proprio e autenticado;
- fallback tatico apoiado em `dados_status_enriquecido_14d_latest.csv`;
- respostas explicitamente orientadas para os proximos 7 dias.

## Fluxo operacional consolidado

```text
1. O pipeline roda e atualiza outputs/hermes/
2. O gestor envia pergunta no Telegram
3. O gateway valida sessao ou credenciais no SQLite local
4. O gateway identifica o escopo da pergunta
5. O wrapper ask_gemini_with_hermes_memory.ps1 monta o prompt com os artefatos Hermes
6. O Gemini responde em formato gerencial
7. A resposta e salva em outputs/hermes/chat/history/
8. O gestor recebe a previsao operacional no Telegram
```

## Ganhos obtidos

- maior aderencia da resposta ao contexto real do projeto;
- rastreabilidade de prompt, resposta e fonte usada;
- autonomia operacional do canal Telegram;
- governanca local de acesso;
- melhor utilidade pratica em perguntas de bairro, cidade ou foco territorial;
- queda do comportamento `sem resposta` em casos com pouca referencia direta;
- alinhamento das respostas com a necessidade de previsao e nao apenas de retrospectiva.

## Limites atuais

Apesar da transformacao, alguns limites permanecem claros:

- a previsao conversacional continua dependente da qualidade e atualizacao dos artefatos de `outputs/hermes/`;
- o fallback tatico 14d e uma projecao operacional sustentada por sinais recentes, nao um novo modelo estatistico independente de forecast;
- a heuristica de escopo no Telegram ainda e simples e orientada por palavras-chave;
- a camada conversacional nao substitui validacao analitica manual quando a decisao exigir auditoria completa.

## Arquivos-chave da transformacao

- `src/core/orchestrator.py`
- `ask_gemini_with_hermes_memory.ps1`
- `.hermes.md`
- `telegram_gemini_gateway.py`
- `manage_telegram_users.py`
- `.env.example`
- `outputs/hermes/`
- `outputs/hermes/history/`
- `outputs/hermes/chat/`

## Resumo executivo

A transformacao do Hermes no Report Preview consolidou uma nova camada operacional: artefato-first, conversacional, autenticada e orientada a previsao para os proximos 7 dias. O Hermes deixou de ser apenas memoria de assistente e passou a funcionar como interface de inteligencia aplicada sobre o pipeline oficial do projeto.