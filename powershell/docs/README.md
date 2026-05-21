# PowerShell Tools

Este diretório documenta os scripts PowerShell usados para operar o fluxo Hermes, o gateway Telegram/Gemini e utilitários de acesso no projeto Report Preview.

## Escopo

Os scripts em [../](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell) cobrem quatro grupos principais:

- Gateway Telegram/Gemini
- Gateway nativo do Hermes
- Controle de acesso e auditoria do bot Telegram
- Análise operacional com Gemini CLI

## Pré-requisitos

Antes de executar qualquer script:

- O projeto deve existir em `C:\Users\Boanerges\Desktop\Projetos\Report Preview` ou o parâmetro `-ProjectRoot` deve apontar para o caminho correto.
- O ambiente virtual Python deve existir em `.venv\Scripts\python.exe`.
- O Hermes Workspace deve existir no caminho configurado, por padrão `E:\Hermes_Workspace`.
- O comando `hermes` deve estar disponível no `PATH` quando o script interagir com o gateway Hermes.
- Quando aplicável, o `gemini` CLI deve estar instalado e disponível no `PATH`.

## Scripts disponíveis

### Gateway Telegram/Gemini

#### [start_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/start_telegram_gemini_gateway.ps1)
Sobe o gateway Python que integra Telegram com Gemini usando memória do Hermes.

Parâmetros:
- `-ProjectRoot`: raiz do projeto.
- `-HermesWorkspace`: workspace do Hermes.
- `-GeminiModel`: modelo passado ao gateway.

Comportamento:
- valida `python.exe`, `telegram_gemini_gateway.py` e o caminho do workspace Hermes;
- executa `hermes gateway stop` no workspace Hermes para liberar o bot Telegram;
- inicia o gateway em uma nova janela usando o Python da `.venv`;
- grava o PID em `outputs/hermes/chat/telegram_gemini_gateway.pid`.

Exemplo:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\start_telegram_gemini_gateway.ps1" -HermesWorkspace "E:\Hermes_Workspace"
```

#### [stop_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/stop_telegram_gemini_gateway.ps1)
Encerra o gateway Telegram/Gemini usando o PID salvo em disco.

Parâmetros:
- `-ProjectRoot`: raiz do projeto.

Comportamento:
- lê `outputs/hermes/chat/telegram_gemini_gateway.pid`;
- encerra o processo correspondente;
- remove o arquivo de PID.

Exemplo:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\stop_telegram_gemini_gateway.ps1"
```

#### [restart_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/restart_telegram_gemini_gateway.ps1)
Executa parada e subida do gateway Telegram/Gemini em sequência.

Parâmetros:
- `-ProjectRoot`
- `-HermesWorkspace`
- `-GeminiModel`

Exemplo:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\restart_telegram_gemini_gateway.ps1" -HermesWorkspace "E:\Hermes_Workspace"
```

### Gateway Hermes

#### [restart_hermes_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/restart_hermes_gateway.ps1)
Reinicia o gateway nativo do Hermes em uma nova janela PowerShell.

Parâmetros:
- `-HermesWorkspace`

Comportamento:
- entra no workspace Hermes;
- roda `hermes gateway stop`;
- abre nova janela com `hermes gateway run`.

Uso típico:
- quando você quer operar o gateway Hermes diretamente, sem o wrapper Telegram/Gemini;
- depois de manutenção em sessões ou configuração do workspace.

#### [reset_hermes_telegram.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/reset_hermes_telegram.ps1)
Remove sessões Telegram específicas do Hermes e opcionalmente reinicia o gateway Hermes.

Parâmetros:
- `-HermesWorkspace`
- `-TelegramUserId`
- `-NoRestart`

Comportamento:
- abre `.hermes/sessions/sessions.json` no workspace Hermes;
- remove mapeamentos e arquivos de sessão associados ao usuário ou chat informado;
- reinicia o gateway Hermes, exceto se `-NoRestart` for usado.

Uso típico:
- resetar contexto preso ou corrompido de um usuário do Telegram;
- forçar reautenticação ou reconstrução de sessão.

### Acesso e auditoria do Telegram

#### [upsert_telegram_user.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/upsert_telegram_user.ps1)
Cria um usuário do bot Telegram ou atualiza sua senha no SQLite local.

Parâmetros:
- `-ProjectRoot`
- `-Username`

Comportamento:
- verifica se o usuário já existe;
- solicita senha e confirmação via prompt seguro;
- chama o módulo Python `manage_telegram_users.py` para criar ou atualizar.

#### [telegram_access_control.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/telegram_access_control.ps1)
Aplica bloqueio por usuário ou bloqueio global do bot Telegram.

Parâmetros:
- `-Action`: `status`, `block-user`, `unblock-user`, `lock-global`, `unlock-global`
- `-ProjectRoot`
- `-Username`
- `-Reason`

Exemplos:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\telegram_access_control.ps1" -Action status
```

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\telegram_access_control.ps1" -Action block-user -Username "usuario_teste"
```

#### [auth_audit_report.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/auth_audit_report.ps1)
Consulta e formata eventos da tabela `auth_audit` do banco local do Telegram.

Parâmetros:
- `-ProjectRoot`
- `-Limit`
- `-Username`
- `-EventType`

Uso típico:
- inspecionar tentativas de login, bloqueios, desbloqueios e eventos de segurança.

### Análise com Gemini CLI

#### [analyze_risk_with_gemini.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/analyze_risk_with_gemini.ps1)
Gera uma análise gerencial usando os artefatos Hermes e o Gemini CLI.

Parâmetros:
- `-Scope`: `fortaleza`, `rmf`, `interior`, `geral`
- `-ProjectRoot`
- `-GeminiModel`
- `-RegenerateArtifacts`

Comportamento:
- lê o brief e os CSVs em `outputs/hermes`;
- opcionalmente regenera artefatos antes da análise;
- chama o `gemini` CLI;
- salva histórico e a análise gerada em `outputs/hermes/history`.

## Fluxos operacionais comuns

### Subir o gateway Telegram/Gemini

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\start_telegram_gemini_gateway.ps1" -HermesWorkspace "E:\Hermes_Workspace"
```

### Reiniciar o gateway Telegram/Gemini

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\restart_telegram_gemini_gateway.ps1" -HermesWorkspace "E:\Hermes_Workspace"
```

### Resetar sessões Telegram de um usuário

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\reset_hermes_telegram.ps1" -HermesWorkspace "E:\Hermes_Workspace" -TelegramUserId "80086019"
```

### Criar ou atualizar credenciais do bot

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\upsert_telegram_user.ps1" -Username "operador1"
```

## Diagnóstico rápido

### Erro: `HermesWorkspace nao encontrado`
Causa provável: o HD externo não está montado ou a letra do drive mudou.

Verificação:

```powershell
Test-Path "E:\Hermes_Workspace"
Get-PSDrive
```

### Erro: `Python nao encontrado`
Causa provável: a `.venv` não existe ou o `ProjectRoot` está incorreto.

Verificação:

```powershell
Test-Path "C:\Users\Boanerges\Desktop\Projetos\Report Preview\.venv\Scripts\python.exe"
```

### Gateway sobe, mas não responde no Telegram
Verifique:
- token do bot carregado em `.env` do projeto ou `.hermes/.env`;
- conectividade com a API do Telegram;
- logs em `outputs/hermes/chat/telegram_gemini_gateway.log`.
