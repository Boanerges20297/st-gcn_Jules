# Plano de Fase 7: Telegram + Report Preview na Hostinger VPS com Gemini CLI e MemPalace

## Objetivo
Colocar em operação uma arquitetura Ubuntu na Hostinger que execute a API do Report Preview e um gateway Telegram com Gemini CLI, preservando o comportamento analítico atual do projeto por meio de diretivas locais e MemPalace, com sincronização segura de dados brutos, artefatos e previsões a partir do ambiente local.

## Resultado esperado
- VPS Ubuntu com serviços independentes para API e gateway Telegram.
- Gemini CLI instalado e validado na VPS.
- MemPalace adotado como camada de memória operacional no lugar do workspace Hermes.
- Pipeline de sincronização local -> VPS por shell script, com manifesto, dry-run e rollback.
- Artefatos táticos e predições ST-GAT disponíveis ao gateway sem depender do ambiente Windows.

## Tarefas

### 1. Definir arquitetura operacional Linux
1. Criar o layout-alvo da VPS:
   - `/home/reportpreview/apps/report-preview` para o repositório.
   - `/srv/reportpreview/runtime/mempalace` para memória operacional e diretivas.
   - `/srv/reportpreview/releases` ou `/srv/reportpreview/backups` para rollback de artefatos.
2. Definir dois serviços `systemd`:
   - `report-preview.service` para `app.py`.
   - `telegram-mempalace-gateway.service` para o worker Telegram.
3. Definir variáveis de ambiente específicas da VPS:
   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY` ou lista equivalente suportada
   - `REPORTPREVIEW_RUNTIME_DIR`
   - `MEMPALACE_CONTEXT_PATH`
   - `APP_PORT=5050`

### 2. Portar o fluxo Telegram/Gemini para Ubuntu
1. Inventariar o que hoje é acoplado a PowerShell/Hermes:
   - `powershell/telegram_gemini_gateway.py`
   - `powershell/ask_gemini_with_hermes_memory.ps1`
   - scripts de start/stop/restart
2. Criar wrappers Linux equivalentes em shell ou Python:
   - `scripts/linux/start_telegram_mempalace_gateway.sh`
   - `scripts/linux/stop_telegram_mempalace_gateway.sh`
   - `scripts/linux/ask_gemini_with_mempalace.sh`
3. Trocar a origem de memória:
   - remover dependência de `HermesWorkspace`;
   - usar `.mempalace.md` e um diretório de runtime MemPalace na VPS;
   - preservar a ordem de leitura dos artefatos analíticos do projeto antes do prompt final ao Gemini.
4. Garantir paridade comportamental:
   - respostas curtas para Telegram;
   - leitura prioritária dos snapshots e relatórios gerados;
   - autenticação Telegram local persistida em SQLite.

### 3. Estruturar o manifesto de sincronização local -> VPS
1. Criar um manifesto explícito de sync com três grupos:
   - `core`: código, templates, configs, docs operacionais.
   - `artifacts`: `models/active/`, `outputs/` relevantes, `static_export/` quando aplicável.
   - `data`: subconjuntos necessários de `data/raw/`, `data/processed/`, bancos SQLite e caches estritamente necessários.
2. Criar script shell de sincronização com estas propriedades:
   - usa `rsync` por SSH;
   - suporta `--dry-run`;
   - suporta sync parcial por grupo;
   - escreve log datado de execução;
   - não remove destino sem confirmação explícita.
3. Separar o sync em etapas seguras:
   - sync de código;
   - sync de artefatos e previsões;
   - validação de integridade;
   - restart controlado dos serviços.
4. Definir política de exclusão obrigatória no sync:
   - `.venv/`, `cache/`, `.pytest_cache/`, `logs/` locais transitórios, backups temporários, arquivos de treino não produtivos.

### 4. Definir o contrato de artefatos usados pelo Telegram
1. Catalogar os arquivos que o gateway precisa ler para responder com precisão:
   - predições ST-GAT ativas;
   - snapshots executivos e briefs táticos;
   - contexto MemPalace.
2. Definir diretório estável na VPS para esses artefatos, sem depender de paths Windows.
3. Estabelecer regra de atualização:
   - dados e saídas são sincronizados do ambiente local;
   - o gateway nunca lê de diretório efêmero ou externo não versionado;
   - cada execução de sync pode gerar snapshot de rollback.

### 5. Hardening operacional e observabilidade
1. Persistir logs dos dois serviços em arquivos dedicados e `journalctl`.
2. Adicionar health checks mínimos:
   - API responde em `/api/model-update-status`.
   - gateway Telegram registra heartbeat e último offset processado.
3. Proteger segredos:
   - `.env` fora do Git;
   - permissões restritas;
   - sem tokens dentro de scripts shell.
4. Definir rollback operacional:
   - restaurar artefatos a partir do último backup sincronizado;
   - reiniciar apenas o serviço afetado;
   - manter histórico de manifests aplicados.

### 6. Validar ponta a ponta antes de produção plena
1. Validar na VPS:
   - `gemini --help`
   - execução de prompt mínimo com resposta controlada
   - subida do Flask e consulta local aos endpoints
2. Validar o gateway Telegram:
   - autenticação de usuário
   - consulta curta
   - consulta com ranking/explicabilidade
3. Validar a atualização via sync:
   - alterar um artefato local
   - sincronizar somente o grupo afetado
   - confirmar que a resposta do Telegram reflete a atualização
4. Validar rollback:
   - simular sync com artefato problemático
   - restaurar versão anterior sem reinstalar a VPS.

## Entregáveis
- Documento de arquitetura operacional da VPS.
- Scripts Linux para start/stop/ask do gateway Telegram com MemPalace.
- Manifesto de sincronização e shell script de deploy incremental.
- Unidades `systemd` para API e gateway.
- Runbook de operação e rollback da VPS.

## Critérios de Aceite (UAT)
- [ ] A API Flask sobe na Hostinger via `systemd` e responde localmente e via Nginx.
- [ ] O Gemini CLI está instalado e responde a um prompt mínimo na VPS.
- [ ] O gateway Telegram funciona em Ubuntu sem depender de PowerShell nem de `HermesWorkspace`.
- [ ] O contexto operacional passa a usar MemPalace como fonte primária de memória/diretiva.
- [ ] Existe script de sincronização local -> VPS com `dry-run`, manifesto e log.
- [ ] Os artefatos necessários para resposta do Telegram e para leitura das predições ST-GAT são sincronizados com sucesso.
- [ ] O processo de rollback de artefatos é testado e documentado.

## Riscos e Mitigações
- **Risco:** divergência de comportamento entre o gateway atual e o gateway Linux.
  **Mitigação:** preservar o contrato de prompt, a ordem de leitura de artefatos e a autenticação SQLite antes de mudar o runtime.
- **Risco:** sync excessivo de `data/` tornar deploy lento e frágil.
  **Mitigação:** manifesto por grupos, exclusões explícitas e sync incremental com `rsync`.
- **Risco:** mistura entre memória do projeto e memória operacional da VPS.
  **Mitigação:** diretório de runtime dedicado para MemPalace, distinto de código e de artefatos sincronizados.
- **Risco:** restart simultâneo da API e do gateway durante atualização parcial.
  **Mitigação:** rollout por serviço, com validação após cada etapa e rollback independente.

## Sequência de Execução Recomendada
1. Formalizar o layout e as variáveis de ambiente da VPS.
2. Portar o wrapper Gemini/Telegram de Hermes para MemPalace em Linux.
3. Criar manifesto e script de sync com `dry-run`.
4. Subir a API Flask na VPS.
5. Subir o gateway Telegram na VPS.
6. Validar atualização incremental de artefatos e rollback.