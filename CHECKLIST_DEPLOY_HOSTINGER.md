# Checklist de Deploy - Hostinger VPS

Este documento e a versao curta e operacional do guia [IMPLEMENTACAO_NUVEM_HOSTINGER.md](IMPLEMENTACAO_NUVEM_HOSTINGER.md). Use este checklist quando for executar o deploy da aplicacao principal em nuvem.

## Escopo

Checklist para colocar no ar:

- a API principal Flask do projeto
- Nginx com proxy reverso
- HTTPS com Certbot
- Gemini CLI, quando o ambiente tambem precisar executar rotinas analiticas

## Antes de comecar

- VPS Hostinger criada com Ubuntu 22.04 LTS ou 24.04 LTS
- acesso SSH funcionando
- dominio apontado para o IP da VPS, se for usar HTTPS
- repositorio Git acessivel
- artefatos de `models/` e `data/` disponiveis

## Checklist rapido

### 1. Preparar a VPS

- atualizar o sistema
- criar usuario de aplicacao
- liberar portas 22, 80 e 443

Comandos:

```bash
apt update && apt upgrade -y
adduser reportpreview
usermod -aG sudo reportpreview
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### 2. Instalar dependencias do sistema

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip git curl build-essential \
  gdal-bin libgdal-dev libspatialindex-dev libgeos-dev libproj-dev \
  proj-data proj-bin nginx certbot python3-certbot-nginx
```

### 3. Clonar o projeto

```bash
mkdir -p /home/reportpreview/apps
cd /home/reportpreview/apps
git clone SEU_REPOSITORIO_GIT report-preview
cd report-preview
```

### 4. Criar a venv e instalar Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Se `torch` falhar em CPU-only:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. Configurar ambiente

```bash
cp .env.example .env
nano .env
```

Minimo esperado no `.env`:

```env
FLASK_APP=app.py
FLASK_ENV=production
LOG_LEVEL=INFO
APP_PORT=5050
SECRET_KEY=troque-por-um-segredo-forte
GOOGLE_API_KEY=sua-chave-se-aplicavel
GEMINI_API_KEYS=sua-chave-se-aplicavel
```

### 6. Validar arquivos obrigatorios

Confirmar que existem:

- `models/active/`
- `data/`
- `logs/`
- `outputs/`
- `static_export/data/`, se o snapshot ja estiver pronto

### 7. Testar a aplicacao manualmente

```bash
source .venv/bin/activate
python app.py
```

Em outro terminal:

```bash
curl http://127.0.0.1:5050/api/model-update-status
```

Resposta esperada:

```json
{"status":"idle"}
```

### 8. Criar o servico systemd

Arquivo:

```bash
sudo nano /etc/systemd/system/report-preview.service
```

Conteudo:

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

Ativar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable report-preview
sudo systemctl start report-preview
sudo systemctl status report-preview
```

### 9. Configurar Nginx

```bash
sudo nano /etc/nginx/sites-available/report-preview
```

Conteudo base:

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
    }
}
```

Ativar:

```bash
sudo ln -s /etc/nginx/sites-available/report-preview /etc/nginx/sites-enabled/report-preview
sudo nginx -t
sudo systemctl reload nginx
```

### 10. Ativar HTTPS

```bash
sudo certbot --nginx -d seu-dominio.com -d www.seu-dominio.com
sudo certbot renew --dry-run
```

### 11. Testar em producao

```bash
curl https://seu-dominio.com/api/model-update-status
curl https://seu-dominio.com/api/anomaly_status
curl https://seu-dominio.com/api/risk
```

### 12. Instalar Gemini CLI, se necessario

Instale Node.js 20:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Instale a CLI:

```bash
sudo npm install -g @google/gemini-cli
```

Valide:

```bash
which gemini
gemini --help
echo "Responda somente ok" | gemini -p "Leia stdin e responda somente ok"
```

## Comandos de operacao

Reiniciar a API:

```bash
sudo systemctl restart report-preview
sudo systemctl status report-preview
```

Ver logs:

```bash
journalctl -u report-preview -f
tail -f /home/reportpreview/apps/report-preview/logs/systemd_stdout.log
```

Atualizar o codigo:

```bash
cd /home/reportpreview/apps/report-preview
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart report-preview
```

## Falhas comuns

### O app nao sobe

- checar `journalctl -u report-preview -n 200 --no-pager`
- checar se `models/active/` e `data/` estao completos
- checar permissao em `logs/` e `outputs/`

### Erro de geopandas, gdal ou rtree

```bash
sudo apt install -y gdal-bin libgdal-dev libspatialindex-dev libgeos-dev libproj-dev
source .venv/bin/activate
pip install --force-reinstall geopandas shapely rtree
```

### O dominio nao responde

- checar DNS
- checar `sudo nginx -t`
- checar `sudo systemctl status nginx`
- checar firewall

### Gemini CLI nao funciona

- checar `which gemini`
- checar autenticacao da CLI
- checar variaveis como `GOOGLE_API_KEY`

## Decisao operacional recomendada

Para subir rapido e com menos risco:

- colocar a API principal Flask na VPS Linux
- manter o gateway Hermes/Telegram/Gemini fora dessa primeira entrega, se ele ainda depender dos scripts Windows em [powershell/](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell)

Quando precisar do detalhamento completo, use [IMPLEMENTACAO_NUVEM_HOSTINGER.md](IMPLEMENTACAO_NUVEM_HOSTINGER.md).