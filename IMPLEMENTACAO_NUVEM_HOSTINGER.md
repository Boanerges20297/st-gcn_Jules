# Implementacao em Nuvem - Report Preview

Este documento descreve a implementacao da aplicacao principal do projeto Report Preview em nuvem, com foco em Hostinger VPS Linux. O objetivo e sair do zero ate um ambiente operacional com API Flask, artefatos do modelo, atualizacao de dados, export estatico e integracoes opcionais com Gemini CLI e gateway Telegram/Gemini.

O guia foi escrito para um cenario realista de producao: VPS Ubuntu, Nginx como reverse proxy, systemd para gerenciamento do processo e opcionalmente Docker para empacotamento.

## Escopo

Este guia cobre:

- preparacao da VPS na Hostinger
- instalacao do projeto
- instalacao de dependencias Python e nativas
- configuracao de variaveis de ambiente
- instalacao e validacao do Gemini CLI
- deploy da API principal Flask
- configuracao com Nginx e HTTPS
- deploy opcional com Docker
- rotinas operacionais e troubleshooting

## Arquitetura de runtime

Hoje a aplicacao principal e iniciada por [app.py](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/app.py), que sobe um servidor Flask em `0.0.0.0:5050` e expone, entre outras, estas rotas:

- `GET /api/risk`
- `GET /api/model-update-status`
- `GET /api/anomaly_status`

O projeto tambem depende de artefatos locais em:

- `data/`
- `models/`
- `outputs/`
- `logs/`
- `static_export/`

Integracoes com Gemini CLI sao usadas principalmente pelos scripts em [powershell/](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell) e pelos fluxos que geram respostas analiticas e operacionais, especialmente:

- [powershell/analyze_risk_with_gemini.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/analyze_risk_with_gemini.ps1)
- [powershell/ask_gemini_with_hermes_memory.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/ask_gemini_with_hermes_memory.ps1)

Importante: a aplicacao principal Flask nao depende diretamente do comando `gemini` para subir. O Gemini CLI e necessario para as automacoes analiticas e para o fluxo Telegram/Gemini.

## Opcao recomendada de hospedagem

Para este projeto, a melhor opcao na Hostinger e:

- VPS Linux dedicada ou cloud VPS
- Ubuntu 22.04 LTS ou 24.04 LTS
- minimo recomendado: 4 vCPU, 8 GB RAM, 80 GB SSD
- recomendado para operacao mais confortavel: 8 vCPU, 16 GB RAM

Motivos:

- o projeto usa dependencias cientificas e geoespaciais como `torch`, `geopandas`, `shapely`, `rtree` e `scipy`
- ha leitura e escrita frequente em disco
- a aplicacao trabalha com artefatos grandes GeoJSON, KML, CSV e snapshots
- o uso do Gemini CLI e dos scripts operacionais pede um ambiente de shell completo

Hospedagem compartilhada da Hostinger nao e o modelo certo para esse projeto.

## Visao geral do deploy

Fluxo recomendado:

1. Criar e endurecer a VPS.
2. Instalar dependencias do sistema.
3. Clonar o repositorio.
4. Criar ambiente virtual Python.
5. Instalar `requirements.txt`.
6. Ajustar `.env`.
7. Validar carga de modelos e endpoint `/api/model-update-status`.
8. Configurar systemd para manter o app no ar.
9. Publicar via Nginx com HTTPS.
10. Instalar Gemini CLI se o ambiente tambem for executar analises e o gateway Telegram/Gemini.

## Preparacao da VPS

Conecte na VPS:

```bash
ssh root@SEU_IP
```

Atualize o sistema:

```bash
apt update && apt upgrade -y
```

Crie um usuario de aplicacao:

```bash
adduser reportpreview
usermod -aG sudo reportpreview
```

Opcional, mas recomendado, configure firewall:

```bash
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

Troque para o usuario da aplicacao:

```bash
su - reportpreview
```

## Dependencias de sistema

Instale o conjunto base:

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  curl \
  build-essential \
  gdal-bin \
  libgdal-dev \
  libspatialindex-dev \
  libgeos-dev \
  libproj-dev \
  proj-data \
  proj-bin \
  nginx \
  certbot \
  python3-certbot-nginx
```

Essas dependencias sao coerentes com o stack usado em [requirements.txt](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/requirements.txt) e com o [Dockerfile](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/Dockerfile), que ja indica a necessidade de `libgdal-dev` e `build-essential`.

## Estrutura recomendada na VPS

Use uma estrutura simples e previsivel:

```text
/home/reportpreview/apps/report-preview
/home/reportpreview/apps/report-preview/.venv
/home/reportpreview/apps/report-preview/logs
/home/reportpreview/apps/report-preview/outputs
```

Crie o diretorio base:

```bash
mkdir -p /home/reportpreview/apps
cd /home/reportpreview/apps
```

## Clonagem do projeto

Se o repositorio estiver no GitHub:

```bash
git clone SEU_REPOSITORIO_GIT report-preview
cd report-preview
```

Se o acesso for privado, configure chave SSH ou use token pessoal.

## Ambiente virtual Python

Dentro da pasta do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

### Observacoes sobre `torch`

Em VPS CPU-only, a instalacao padrao de `torch==2.1.2` pode funcionar, mas e pesada. Em ambiente Linux puro CPU, se houver dificuldade, use a wheel oficial CPU do PyTorch antes do restante:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Se o `requirements.txt` tentar reinstalar outra wheel, ajuste o lock localmente antes do deploy ou mantenha uma imagem Docker controlada.

## Arquivos que precisam existir em producao

Antes de subir o app, confirme que estes grupos existem:

- `models/active/` com os artefatos ativos
- `data/raw/` com os insumos necessarios
- `data/cc_state.json` quando o champion/challenger estiver em uso
- `logs/` gravavel pelo usuario da aplicacao
- `outputs/` gravavel pelo usuario da aplicacao
- `static_export/data/` se o export estatico ja tiver sido gerado

Se parte desses artefatos nao estiver versionada ou vier de pipeline externa, suba-os por `rsync`, `scp` ou pipeline de CI/CD antes do go-live.

## Configuracao de ambiente

Use [ .env.example ](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/.env.example) como base:

```bash
cp .env.example .env
```

Edite:

```bash
nano .env
```

Configuracoes minimas recomendadas:

```env
FLASK_APP=app.py
FLASK_ENV=production
LOG_LEVEL=INFO

APP_PORT=5050
SECRET_KEY=troque-por-um-segredo-forte

GOOGLE_API_KEY=sua-chave-se-usar-integracoes-google
GEMINI_API_KEYS=sua-chave-ou-lista-se-usar-rotinas-gemini

TELEGRAM_AUTH_SESSION_TTL_SECONDS=28800
TELEGRAM_AUTH_MAX_FAILED_ATTEMPTS=5
TELEGRAM_AUTH_LOCKOUT_SECONDS=900
```

Observacoes:

- o `docker-compose.yml` usa `APP_PORT`, `GOOGLE_API_KEY` e `GEMINI_API_KEYS`
- o app sobe internamente em `5050` no [app.py](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/app.py)
- em deploy sem Docker, mantenha `APP_PORT=5050` por consistencia operacional

## Teste local na VPS

Com o ambiente virtual ativo:

```bash
source .venv/bin/activate
python app.py
```

Em outro shell, teste:

```bash
curl http://127.0.0.1:5050/api/model-update-status
```

Resposta esperada:

```json
{"status":"idle"}
```

Se isso responder, o processo basico da API esta funcional.

## Deploy recomendado com systemd

Crie o arquivo de servico:

```bash
sudo nano /etc/systemd/system/report-preview.service
```

Conteudo sugerido:

```ini
[Unit]
Description=Report Preview Flask API
After=network.target

[Service]
User=reportpreview
Group=reportpreview
WorkingDirectory=/home/reportpreview/apps/report-preview
EnvironmentFile=/home/reportpreview/apps/report-preview/.env
ExecStart=/home/reportpreview/apps/report-preview/.venv/bin/python /home/reportpreview/apps/report-preview/app.py
Restart=always
RestartSec=5
StandardOutput=append:/home/reportpreview/apps/report-preview/logs/systemd_stdout.log
StandardError=append:/home/reportpreview/apps/report-preview/logs/systemd_stderr.log

[Install]
WantedBy=multi-user.target
```

Ative o servico:

```bash
sudo systemctl daemon-reload
sudo systemctl enable report-preview
sudo systemctl start report-preview
sudo systemctl status report-preview
```

Logs em tempo real:

```bash
journalctl -u report-preview -f
```

## Publicacao com Nginx

Crie um server block:

```bash
sudo nano /etc/nginx/sites-available/report-preview
```

Exemplo:

```nginx
server {
    listen 80;
    server_name seu-dominio.com www.seu-dominio.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:5050;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 60;
    }
}
```

Ative o site:

```bash
sudo ln -s /etc/nginx/sites-available/report-preview /etc/nginx/sites-enabled/report-preview
sudo nginx -t
sudo systemctl reload nginx
```

## HTTPS com Certbot

Depois que o DNS estiver apontando para a VPS:

```bash
sudo certbot --nginx -d seu-dominio.com -d www.seu-dominio.com
```

Teste renovacao:

```bash
sudo certbot renew --dry-run
```

## Deploy opcional com Docker

Se preferir empacotamento via container, o projeto ja possui [Dockerfile](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/Dockerfile) e [docker-compose.yml](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/docker-compose.yml).

### Instalar Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker reportpreview
newgrp docker
```

### Subir com Compose

Na raiz do projeto:

```bash
docker compose up -d --build
```

Validar:

```bash
docker compose ps
curl http://127.0.0.1:5000/api/model-update-status
```

Observacoes importantes:

- o `docker-compose.yml` publica `${APP_PORT:-5000}:5050`
- dentro do container o app continua ouvindo em `5050`
- o compose monta `data`, `logs`, `outputs` e `models`
- se o projeto depender de arquivos externos nao versionados, eles precisam existir no host

### Quando preferir Docker

Use Docker quando:

- quiser reproducao forte entre ambientes
- quiser reduzir drift de bibliotecas nativas
- quiser integrar com pipeline CI/CD mais previsivel

Use systemd sem Docker quando:

- quiser simplicidade operacional na VPS
- quiser inspecao direta de arquivos locais e scripts do projeto
- precisar integrar facilmente scripts Python e shell no host

## Instalacao do Gemini CLI

O Gemini CLI e necessario se a VPS tambem vai executar:

- analises por [powershell/analyze_risk_with_gemini.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/analyze_risk_with_gemini.ps1)
- respostas enriquecidas por [powershell/ask_gemini_with_hermes_memory.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/ask_gemini_with_hermes_memory.ps1)
- gateway Telegram/Gemini

Como os scripts atuais procuram o comando `gemini` no PATH e, no fluxo Windows, tambem usam `gemini.cmd`, em Linux o mais seguro e instalar a CLI oficial de forma global e validar o binario `gemini` no shell.

### Pre-requisito Node.js

Instale Node 20 LTS:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm -v
```

### Instalacao da CLI

Se a sua organizacao usa a Gemini CLI oficial via npm, a forma tipica e:

```bash
sudo npm install -g @google/gemini-cli
```

Se o pacote oficial adotado por voce tiver outro nome, ajuste este comando. O ponto importante para este projeto e que o shell resolva `gemini` com sucesso:

```bash
which gemini
gemini --help
```

### Autenticacao

A forma de autenticacao pode variar conforme a CLI instalada. Em geral voce vai usar uma destas abordagens:

- login interativo da propria CLI
- export de `GOOGLE_API_KEY`
- export de token ou credencial especifica da ferramenta

Exemplo generico via `.bashrc` ou unit file:

```bash
export GOOGLE_API_KEY="sua-chave"
```

### Teste minimo do Gemini CLI

```bash
echo "Responda somente: ok" | gemini -p "Leia stdin e responda somente ok"
```

Se a CLI retornar algo coerente, a base da integracao esta pronta.

## Importante sobre os scripts PowerShell e Linux

Os scripts atuais em [powershell/](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell) foram escritos para Windows PowerShell. Em Hostinger VPS Linux:

- a aplicacao principal Flask pode rodar normalmente
- o deploy web principal nao depende desses scripts PowerShell
- os fluxos Telegram/Gemini baseados em `.ps1` vao exigir adaptacao para Bash ou PowerShell 7 no Linux

Se voce pretende rodar tambem o gateway Telegram/Gemini na VPS Linux, existem tres caminhos:

1. manter esse gateway em uma maquina Windows separada
2. instalar PowerShell 7 na VPS e adaptar os caminhos Windows para Linux
3. portar os wrappers `.ps1` para Bash/Python

Hoje, para producao web principal, o mais limpo e separar:

- VPS Linux: API Flask principal
- maquina Windows operacional: gateway Hermes/Telegram/Gemini, se ainda dependente dos scripts atuais

## Implantacao do gateway Telegram/Gemini

Se voce quiser rodar esse gateway em nuvem no futuro, vai precisar destes blocos:

- [powershell/start_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/start_telegram_gemini_gateway.ps1)
- [powershell/telegram_gemini_gateway.py](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/telegram_gemini_gateway.py)
- [powershell/ask_gemini_with_hermes_memory.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/ask_gemini_with_hermes_memory.ps1)

E tambem:

- workspace Hermes disponivel em disco
- `TELEGRAM_BOT_TOKEN`
- banco local `data/users/telegram_auth.sqlite3`
- Gemini CLI funcional

No estado atual do projeto, isso pede mais trabalho do que o deploy da API principal. Nao trate como requisito obrigatorio para colocar o painel principal no ar.

## Export estatico e snapshots

A aplicacao usa exportadores e artefatos em:

- [scripts/export_static_snapshot.py](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/scripts/export_static_snapshot.py)
- `static_export/data/`
- `outputs/`

Depois do deploy, rode um teste operacional:

```bash
source .venv/bin/activate
python scripts/export_static_snapshot.py
```

Se o fluxo for dependente do contexto carregado por `app.py`, siga o runbook real do projeto para regeneracao dos snapshots e confira se os JSON/GeoJSON atualizados aparecem em `static_export/data/`.

## Rotina de atualizacao da aplicacao

### Atualizacao por Git

```bash
cd /home/reportpreview/apps/report-preview
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart report-preview
sudo systemctl status report-preview
```

### Validacao pos-deploy

```bash
curl http://127.0.0.1:5050/api/model-update-status
curl http://127.0.0.1:5050/api/anomaly_status
curl "http://127.0.0.1:5050/api/risk"
```

Se estiver atras do Nginx com dominio:

```bash
curl https://seu-dominio.com/api/model-update-status
```

## Observabilidade minima recomendada

Monitore pelo menos:

- status do `systemd`
- espaco em disco
- crescimento de `logs/`
- tempo de resposta das rotas principais
- integridade dos arquivos em `data/`, `models/`, `outputs/` e `static_export/data/`

Comandos uteis:

```bash
df -h
du -sh logs outputs static_export data
sudo systemctl status report-preview
journalctl -u report-preview -n 200 --no-pager
```

## Troubleshooting

### O app nao sobe

Cheque:

```bash
journalctl -u report-preview -n 200 --no-pager
```

Causas comuns:

- falta de artefato em `models/active/`
- falta de CSV ou JSON em `data/`
- erro em bibliotecas geoespaciais
- permissao insuficiente em `logs/` ou `outputs/`

### Erro de `gdal` ou `geopandas`

Reinstale bibliotecas nativas e depois as wheels Python:

```bash
sudo apt install -y gdal-bin libgdal-dev libspatialindex-dev libgeos-dev libproj-dev
source .venv/bin/activate
pip install --force-reinstall geopandas shapely rtree
```

### Rota responde no localhost, mas nao no dominio

Verifique:

- DNS
- configuracao do Nginx
- firewall
- Certbot

Testes:

```bash
curl http://127.0.0.1:5050/api/model-update-status
sudo nginx -t
sudo systemctl status nginx
```

### Gemini CLI nao encontrado

Valide:

```bash
which gemini
gemini --help
```

Se nao existir, reinstale globalmente e confira o PATH do usuario que executa o processo.

### O Gemini CLI existe, mas falha por autenticacao

Valide a estrategia adotada pela sua CLI:

- login interativo
- `GOOGLE_API_KEY`
- variavel de token especifica

E teste fora do app antes de integrar.

## Estrategia recomendada de producao

Para reduzir risco operacional, recomendo esta separacao:

### Camada 1: API principal em VPS Linux

- [app.py](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/app.py)
- `data/`, `models/`, `logs/`, `outputs/`, `static_export/`
- Nginx + systemd

### Camada 2: rotinas de IA operacional

- Gemini CLI
- scripts em [powershell/](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell)
- gateway Telegram/Gemini

Essa camada pode continuar em Windows ate voce decidir portar os wrappers para Linux.

## Checklist final de go-live

- VPS criada e atualizada
- Python e bibliotecas nativas instaladas
- repositorio clonado
- `.venv` criada
- `pip install -r requirements.txt` executado com sucesso
- `.env` preenchido
- `models/active/` e `data/` completos
- `python app.py` respondendo localmente
- `report-preview.service` ativo
- Nginx configurado
- HTTPS emitido com Certbot
- teste em `GET /api/model-update-status`
- teste em `GET /api/risk`
- Gemini CLI instalado, se as rotinas analiticas forem usar IA generativa

## Comandos resumo

### Subir manualmente para teste

```bash
cd /home/reportpreview/apps/report-preview
source .venv/bin/activate
python app.py
```

### Reiniciar servico

```bash
sudo systemctl restart report-preview
sudo systemctl status report-preview
```

### Ver logs

```bash
journalctl -u report-preview -f
tail -f /home/reportpreview/apps/report-preview/logs/systemd_stdout.log
```

### Testar saude

```bash
curl http://127.0.0.1:5050/api/model-update-status
```

### Testar Gemini CLI

```bash
which gemini
gemini --help
```

## Nota final

Se o objetivo imediato for publicar o painel e a API principal na Hostinger, faca primeiro o deploy apenas da aplicacao Flask. O gateway Hermes/Telegram/Gemini e uma segunda etapa e, no estado atual do projeto, esta mais natural em Windows por causa dos scripts PowerShell e dos caminhos ja implementados.