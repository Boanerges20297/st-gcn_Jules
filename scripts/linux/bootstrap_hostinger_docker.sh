#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-reportpreview}"
APP_HOME="/home/${APP_USER}/apps/report-preview"
RUNTIME_ROOT="/srv/reportpreview"

echo "[1/6] Instalando dependencias base do host..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl git rsync

echo "[2/6] Garantindo diretorios do runtime..."
sudo mkdir -p \
  "${RUNTIME_ROOT}/runtime/mempalace" \
  "${RUNTIME_ROOT}/backups" \
  "${RUNTIME_ROOT}/sync" \
  "${RUNTIME_ROOT}/logs"

echo "[3/6] Garantindo estrutura persistente do projeto..."
sudo mkdir -p \
  "${APP_HOME}/data" \
  "${APP_HOME}/logs" \
  "${APP_HOME}/outputs" \
  "${APP_HOME}/models" \
  "${APP_HOME}/static_export"

echo "[4/6] Ajustando permissões..."
sudo chown -R "${APP_USER}:${APP_USER}" \
  "/home/${APP_USER}" \
  "${RUNTIME_ROOT}"

echo "[5/6] Validando Docker/Compose e proxy existente..."
docker --version
docker compose version
if ss -ltn '( sport = :80 or sport = :443 )' | grep -Eq ':80|:443'; then
  echo "Proxy HTTP/HTTPS ja detectado no host. O deploy recomendado usa Traefik existente em vez de nginx local."
fi

echo "[6/6] Bootstrap concluido."
echo "Projeto esperado em: ${APP_HOME}"
echo "Runtime MemPalace em: ${RUNTIME_ROOT}/runtime/mempalace"
echo "Proximo passo: copiar o repositorio e subir docker compose com docker-compose.hostinger.yml"#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-reportpreview}"
APP_HOME="/home/${APP_USER}/apps/report-preview"
RUNTIME_ROOT="/srv/reportpreview"

echo "[1/6] Instalando dependencias base do host..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl git rsync nginx

echo "[2/6] Garantindo diretorios do runtime..."
sudo mkdir -p \
  "${RUNTIME_ROOT}/runtime/mempalace" \
  "${RUNTIME_ROOT}/backups" \
  "${RUNTIME_ROOT}/sync" \
  "${RUNTIME_ROOT}/logs"

echo "[3/6] Garantindo estrutura persistente do projeto..."
sudo mkdir -p \
  "${APP_HOME}/data" \
  "${APP_HOME}/logs" \
  "${APP_HOME}/outputs" \
  "${APP_HOME}/models" \
  "${APP_HOME}/static_export"

echo "[4/6] Ajustando permissões..."
sudo chown -R "${APP_USER}:${APP_USER}" \
  "/home/${APP_USER}" \
  "${RUNTIME_ROOT}"

echo "[5/6] Validando Docker/Compose..."
docker --version
docker compose version

echo "[6/6] Bootstrap concluido."
echo "Projeto esperado em: ${APP_HOME}"
echo "Runtime MemPalace em: ${RUNTIME_ROOT}/runtime/mempalace"
echo "Proximo passo: copiar o repositorio, ajustar .env.hostinger.example -> .env e subir docker compose com docker-compose.hostinger.yml"