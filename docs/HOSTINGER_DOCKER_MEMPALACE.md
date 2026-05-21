# Hostinger Docker + MemPalace

Este guia assume uma VPS Ubuntu com Docker e Docker Compose ja disponiveis.

No host atual validado em campo, a entrada HTTP/HTTPS ja esta sendo gerenciada por um Traefik em container `host-network`. Portanto, a estrategia recomendada para o Report Preview e reutilizar esse Traefik, e nao iniciar nginx local em paralelo.

## Topologia recomendada

- `app`: API Flask do Report Preview, exposta internamente ao Docker e publicada pelo Traefik.
- `telegram-gateway`: worker Telegram com Gemini CLI e MemPalace.
- Volumes persistentes para `data/`, `outputs/`, `logs/`, `models/` e `static_export/`.

## Arquivos de deploy adicionados

- `docker-compose.hostinger.yml`
- `docker/Dockerfile.telegram-gateway`
- `scripts/linux/bootstrap_hostinger_docker.sh`
- `.env.hostinger.example`
- `scripts/linux/sync_hostinger_rsync.sh`
- `scripts/linux/hostinger_rsync_excludes.txt`

## Sequencia sugerida na VPS

```bash
sudo bash scripts/linux/bootstrap_hostinger_docker.sh
cp .env.hostinger.example .env
nano .env
docker compose -f docker-compose.hostinger.yml build
docker compose -f docker-compose.hostinger.yml up -d
docker compose -f docker-compose.hostinger.yml ps
docker compose -f docker-compose.hostinger.yml logs -f app
docker compose -f docker-compose.hostinger.yml logs -f telegram-gateway
```

## Sync local -> VPS

Antes de subir ou atualizar os containers, voce pode sincronizar o projeto localmente com `rsync`:

```bash
bash scripts/linux/sync_hostinger_rsync.sh --host 76.13.121.172 --dry-run
bash scripts/linux/sync_hostinger_rsync.sh --host 76.13.121.172 --groups core,artifacts
bash scripts/linux/sync_hostinger_rsync.sh --host 76.13.121.172 --groups data
```
 
Grupos disponiveis:

- `core`: codigo, configs e arquivos de runtime do projeto.
- `artifacts`: `models/`, `outputs/` e `static_export/`.
- `data`: conteudo de `data/`.
- `all`: equivale a `core,artifacts,data`.

O arquivo `scripts/linux/hostinger_rsync_excludes.txt` impede envio de caches, venv, historicos volumosos e temporarios.

## Sync local -> VPS no Windows PowerShell

Se sua maquina local estiver em Windows e voce nao quiser depender de `bash`, use o script PowerShell abaixo. Ele empacota cada grupo em `.tar`, envia por `scp` e extrai na VPS por `ssh`.

```powershell
powershell -ExecutionPolicy Bypass -File .\powershell\sync_hostinger.ps1 -RemoteHost 76.13.121.172 -User reportpreview -DryRun
powershell -ExecutionPolicy Bypass -File .\powershell\sync_hostinger.ps1 -RemoteHost 76.13.121.172 -User reportpreview -Groups core,telegram_artifacts
powershell -ExecutionPolicy Bypass -File .\powershell\sync_hostinger.ps1 -RemoteHost 76.13.121.172 -User reportpreview -Groups data
```

Requisitos locais para esse caminho:

- `ssh` no `PATH`
- `scp` no `PATH`
- `tar` no `PATH`

Esse fluxo foi pensado para PowerShell nativo no Windows.

Grupos uteis no Windows:

- `core`: codigo, configs e wrappers.
- `telegram_artifacts`: sobe apenas `outputs/hermes`, que e o caso mais comum do CLI/Telegram.
- `outputs`: sobe `outputs/` inteiro.
- `models`: sobe `models/` inteiro.
- `static_export`: sobe `static_export/` inteiro.

Observacao pratica:

- `artifacts` continua existindo como alias, mas ele expande para `outputs,models,static_export`.
- Para o seu caso de CLI, prefira `telegram_artifacts` em vez de `artifacts`.

## Variaveis minimas no `.env`

```env
APP_PORT=5050
TRAEFIK_HOST=reportpreview.seu-dominio.com
FLASK_ENV=production
FLASK_DEBUG=0
TELEGRAM_BOT_TOKEN=seu-token
GEMINI_API_KEY=sua-chave
GOOGLE_API_KEY=sua-chave
TELEGRAM_AUTH_SESSION_TTL_SECONDS=28800
TELEGRAM_AUTH_MAX_FAILED_ATTEMPTS=5
TELEGRAM_AUTH_LOCKOUT_SECONDS=900
```

## Observacoes

- O gateway Linux usa `scripts/linux/ask_gemini_with_mempalace.py`.
- A memoria operacional primaria passa a ser `.mempalace.md`.
- Os artefatos analiticos atuais continuam sendo lidos de `outputs/hermes/`, para manter compatibilidade com o pipeline existente.
- O estado do gateway no ambiente Linux fica em `outputs/mempalace/chat/`.
- O compose novo foi preparado para Traefik com labels Docker e `TRAEFIK_HOST`.
- O container `telegram-gateway` nao e publicado externamente; ele opera apenas como worker interno.