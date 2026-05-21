# Telegram + Gemini CLI em Container

Este guia sobe apenas o fluxo `Telegram -> autenticacao SQLite -> Gemini CLI`.

Nao sobe `app.py`, nao sobe backend Flask, nao sobe `models/` e nao usa `data/processed/`.

## O que precisa existir na VPS

Raiz esperada do projeto:

`/home/reportpreview/apps/report-preview`

Arquivos e pastas minimos:

- `.env`
- `.mempalace.md`
- `.hermes.md`
- `powershell/telegram_gemini_gateway.py`
- `powershell/manage_telegram_users.py`
- `scripts/linux/ask_gemini_with_mempalace.py`
- `outputs/hermes/`
- `data/users/`
- `logs/`
- `docker/Dockerfile.telegram-gateway`
- `docker-compose.telegram-only.yml`

## Variaveis minimas no `.env`

```env
TELEGRAM_BOT_TOKEN=seu-token
GEMINI_API_KEY=sua-chave
TELEGRAM_AUTH_SESSION_TTL_SECONDS=28800
TELEGRAM_AUTH_MAX_FAILED_ATTEMPTS=5
TELEGRAM_AUTH_LOCKOUT_SECONDS=900
```

## Passo 1. Verificar o conjunto minimo

```bash
cd /home/reportpreview/apps/report-preview && \
test -f .env && \
test -f .mempalace.md && \
test -f .hermes.md && \
test -f powershell/telegram_gemini_gateway.py && \
test -f powershell/manage_telegram_users.py && \
test -f scripts/linux/ask_gemini_with_mempalace.py && \
test -f docker/Dockerfile.telegram-gateway && \
test -f docker-compose.telegram-only.yml && \
test -d outputs/hermes && \
test -d data/users && \
echo OK || echo FALTANDO_ALGO
```

## Passo 2. Build da imagem

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml build
```

## Passo 2.1. Subir ou recriar o stack

Subir normalmente:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml up -d
```

Subir forçando rebuild e recriacao:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml up -d --build --force-recreate
```

## Passo 3. Subir o container

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml up -d
```

Ver status do container:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml ps
docker ps -a --filter "name=report-preview-telegram-gateway"
```

## Passo 4. Ver logs

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml logs -f telegram-gateway
```

Ver apenas as ultimas linhas:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml logs --tail 100 telegram-gateway
```

## Passo 4.1. Entrar e sair do container

Entrar em shell interativo:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway sh
```

Sair do shell do container:

```bash
exit
```

Executar um comando sem entrar no shell:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway python --version
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway gemini --version
```

## Passo 4.2. Autenticar o Gemini CLI no container

Se voce preferir login interativo do Gemini CLI em vez de `GEMINI_API_KEY`, entre no container:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway sh
```

Depois rode:

```sh
gemini --version
gemini
```

Quando terminar, saia do shell:

```sh
exit
```

## Passo 5. Cadastrar um usuario no SQLite

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py add SEU_USUARIO --password 'SUA_SENHA'
```

Exemplo:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py add boanerges --password '90915225'
```

Listar usuarios:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py list --all
```

Trocar senha:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py set-password SEU_USUARIO --password 'NOVA_SENHA'
```

Ativar usuario:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py activate SEU_USUARIO
```

Desativar usuario:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py deactivate SEU_USUARIO
```

Excluir usuario do banco:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py delete SEU_USUARIO
```

Ver ajuda completa do gerenciador de usuarios:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway \
  python powershell/manage_telegram_users.py --help
```

## Passo 6. Reiniciar apos atualizacoes

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml up -d --build
```

Reiniciar sem rebuild:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml restart telegram-gateway
```

## Passo 7. Parar o container

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml down
```

Parar sem remover o container:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml stop telegram-gateway
```

Iniciar novamente apos stop:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml start telegram-gateway
```

## Passo 8. Teste operacional no Telegram

Abra os logs em uma aba:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml logs -f telegram-gateway
```

No Telegram:

1. envie `/start`
2. informe o usuario
3. informe a senha
4. envie uma pergunta simples

Exemplo de pergunta:

```text
qual a leitura para fortaleza nos proximos 7 dias?
```

## Passo 9. Problemas comuns

Container reiniciando em loop:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml ps
docker compose -f docker-compose.telegram-only.yml logs --tail 200 telegram-gateway
```

SQLite readonly ou `unable to open database file`:

```bash
cd /home/reportpreview/apps/report-preview
mkdir -p data/users outputs/mempalace/chat/history logs
chown -R 1001:1001 data outputs logs
docker compose -f docker-compose.telegram-only.yml up -d --force-recreate
```

Verificar variaveis essenciais no `.env`:

```bash
cd /home/reportpreview/apps/report-preview
grep -nE 'TELEGRAM_BOT_TOKEN|GEMINI_API_KEY' .env
```

Verificar se o Gemini CLI esta disponivel dentro do container:

```bash
cd /home/reportpreview/apps/report-preview
docker compose -f docker-compose.telegram-only.yml exec telegram-gateway gemini --version
```

## Observacoes operacionais

- O gateway cria `data/users/telegram_auth.sqlite3` se o banco ainda nao existir.
- O runtime de conversa fica em `outputs/mempalace/chat/`.
- Os artefatos consultados pelo prompt continuam em `outputs/hermes/`.
- Nao e necessario enviar `models/`, `data/processed/` ou o backend Flask para este modo.
- Se quiser atualizar o contexto do bot a partir da maquina local, basta substituir os arquivos em `outputs/hermes/` e reiniciar o container.
- Se voce autenticar o Gemini CLI por login interativo dentro do container, a persistencia dessa autenticacao depende de como o CLI grava suas credenciais. Para operacao previsivel em producao, prefira `GEMINI_API_KEY` no `.env`.