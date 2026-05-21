# Pesquisa de Fase 7: Telegram + Report Preview em VPS Hostinger com Gemini CLI e MemPalace

## Objetivo da pesquisa
Mapear o estado atual do projeto para planejar uma operação em VPS Ubuntu que preserve o comportamento analítico atual no Telegram, trocando o workspace Hermes por MemPalace e adicionando sincronização de artefatos a partir da máquina local.

## Achados confirmados no código e na documentação

### 1. O runtime principal já é compatível com VPS Ubuntu
- `app.py` sobe a API Flask em `0.0.0.0:5050`.
- `CHECKLIST_DEPLOY_HOSTINGER.md` e `IMPLEMENTACAO_NUVEM_HOSTINGER.md` já documentam Ubuntu + systemd + Nginx + Certbot.
- A aplicação principal não depende do comando `gemini` para inicializar; o Gemini CLI é necessário para rotinas analíticas e para o gateway Telegram.

### 2. O gateway Telegram atual é Windows-first e Hermes-first
- `powershell/telegram_gemini_gateway.py` usa wrappers PowerShell e resolve contexto a partir de um `HermesWorkspace`.
- O script atual aponta para `ask_gemini_with_hermes_memory.ps1`, ou seja, o acoplamento com Hermes é estrutural e não apenas nominal.
- O runbook atual (`powershell/docs/telegram-gemini-runbook.md`) pressupõe start/stop/restart via PowerShell e paths Windows.

### 3. Já existe material local que pode substituir Hermes como fonte tática
- `.mempalace.md` funciona como cofre de contexto operacional/tático do projeto.
- O projeto já possui artefatos analíticos persistidos em `outputs/hermes/` que servem como base de resposta curta para Telegram.
- A mudança desejada não elimina os artefatos do projeto; ela troca a camada de memória/diretivas do operador LLM.

### 4. Há requisitos operacionais de autenticação e segurança já embutidos
- `.env.example` define controles de autenticação Telegram local: TTL de sessão, lockout e tentativas máximas.
- `powershell/telegram_gemini_gateway.py` mantém banco local SQLite de autenticação em `data/users/telegram_auth.sqlite3`.
- Isso indica que a VPS precisa persistência local de estado, não apenas execução stateless do bot.

### 5. O gargalo principal para a VPS é portabilidade operacional, não modelo
- O projeto já sabe executar Flask e Gemini CLI na Hostinger.
- O que falta é uma arquitetura operacional Linux para:
  - sincronizar `data/`, `models/`, `outputs/` a partir do ambiente local;
  - preservar a mesma lógica de resposta do Telegram;
  - trocar Hermes por MemPalace sem perder diretivas do projeto;
  - separar dados brutos, artefatos ativos e logs para rollback seguro.

## Implicações para o plano

### Arquitetura-alvo recomendada
- VPS Ubuntu 22.04/24.04 com dois serviços independentes:
  - `report-preview.service` para a API Flask.
  - `telegram-mempalace-gateway.service` para o worker Telegram/Gemini.
- Diretório raiz único do projeto em `/home/reportpreview/apps/report-preview`.
- Diretório de runtime separado para memória operacional e sincronização, por exemplo:
  - `/srv/reportpreview/runtime/mempalace/`
  - `/srv/reportpreview/sync/`
  - `/srv/reportpreview/backups/`

### Estratégia de sincronização recomendada
- Sincronização unidirecional do ambiente local para a VPS via `rsync` sobre SSH.
- Manifesto explícito do que sobe para produção:
  - `models/active/`
  - subconjuntos necessários de `data/raw/` e `data/processed/`
  - artefatos operacionais de `outputs/` necessários ao Telegram
  - arquivos de configuração e diretivas (`.mempalace.md`, templates de prompt, scripts shell)
- Exclusões obrigatórias:
  - `.venv/`, `cache/`, `.pytest_cache/`, backups temporários, logs locais ruidosos.

### Estratégia de memória/diretivas
- O comportamento atual do Telegram deve ser preservado por um wrapper Linux que:
  - injete a diretiva do projeto;
  - injete `.mempalace.md` como contexto prioritário;
  - consulte os artefatos gerados do projeto antes de chamar o Gemini CLI;
  - mantenha política de resposta curta e gerencial para chat.
- O nome Hermes deve sair da superfície operacional da VPS para evitar ambiguidade de responsabilidade.

### Riscos relevantes
- Portar só os scripts e não a disciplina de diretivas pode fazer o Gemini CLI responder diferente do ambiente atual.
- Subir `data/raw/` completo sem política de sync pode degradar deploy e rollback.
- Misturar atualização de dados, atualização de código e restart do bot em um único script aumenta risco operacional.

## Decisões propostas para a fase
1. Criar wrappers Linux dedicados para Telegram e Gemini, sem depender de PowerShell.
2. Tratar MemPalace como fonte oficial de memória operacional da VPS.
3. Implementar sincronização por manifesto, não por cópia cega do projeto inteiro.
4. Separar rollout em camadas: base da VPS, sync de artefatos, gateway Telegram, validação E2E.