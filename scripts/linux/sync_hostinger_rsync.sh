#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXCLUDES_FILE="$ROOT_DIR/scripts/linux/hostinger_rsync_excludes.txt"

HOST="${HOST:-}"
USER_NAME="${USER_NAME:-reportpreview}"
SSH_PORT="${SSH_PORT:-22}"
TARGET_DIR="${TARGET_DIR:-/home/reportpreview/apps/report-preview}"
SYNC_GROUPS="${SYNC_GROUPS:-core,artifacts,data}"
DRY_RUN="false"
DELETE_MODE="false"

usage() {
  cat <<'EOF'
Uso:
  bash scripts/linux/sync_hostinger_rsync.sh --host IP_OU_HOST [opcoes]

Opcoes:
  --host HOST                Host ou IP da VPS (obrigatorio)
  --user USER                Usuario SSH. Padrao: reportpreview
  --port PORT                Porta SSH. Padrao: 22
  --target-dir DIR           Destino do projeto na VPS
  --groups LISTA             Grupos separados por virgula: core,artifacts,data,all
  --dry-run                  Simula sem enviar arquivos
  --delete                   Permite deletar no destino dentro dos paths sincronizados
  --help                     Mostra esta ajuda

Exemplos:
  bash scripts/linux/sync_hostinger_rsync.sh --host 76.13.121.172 --dry-run
  bash scripts/linux/sync_hostinger_rsync.sh --host 76.13.121.172 --groups core,artifacts
  bash scripts/linux/sync_hostinger_rsync.sh --host 76.13.121.172 --groups data
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --port)
      SSH_PORT="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --groups)
      SYNC_GROUPS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --delete)
      DELETE_MODE="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Parametro invalido: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "Erro: --host e obrigatorio." >&2
  usage >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Erro: rsync nao encontrado no ambiente local." >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "Erro: ssh nao encontrado no ambiente local." >&2
  exit 1
fi

IFS=',' read -r -a RAW_GROUPS <<< "$SYNC_GROUPS"
SELECTED_GROUPS=()
for group in "${RAW_GROUPS[@]}"; do
  trimmed="$(echo "$group" | xargs)"
  if [[ -n "$trimmed" ]]; then
    if [[ "$trimmed" == "all" ]]; then
      SELECTED_GROUPS=(core artifacts data)
      break
    fi
    SELECTED_GROUPS+=("$trimmed")
  fi
done

if [[ ${#SELECTED_GROUPS[@]} -eq 0 ]]; then
  echo "Erro: nenhum grupo valido informado em --groups." >&2
  exit 1
fi

TARGET="$USER_NAME@$HOST:$TARGET_DIR"
SSH_CMD=(ssh -p "$SSH_PORT")
RSYNC_BASE=(rsync -az --human-readable --info=progress2 --partial --mkpath -e "ssh -p $SSH_PORT" --exclude-from="$EXCLUDES_FILE")

if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_BASE+=(--dry-run --itemize-changes)
fi

if [[ "$DELETE_MODE" == "true" ]]; then
  RSYNC_BASE+=(--delete)
fi

run_sync() {
  local label="$1"
  local source_path="$2"
  local target_path="$3"

  echo "============================================================"
  echo "Grupo: $label"
  echo "Origem: $source_path"
  echo "Destino: $target_path"
  echo "============================================================"

  "${RSYNC_BASE[@]}" "$source_path" "$target_path"
}

echo "Preparando destino na VPS..."
"${SSH_CMD[@]}" "$USER_NAME@$HOST" "mkdir -p '$TARGET_DIR' '$TARGET_DIR/data' '$TARGET_DIR/models' '$TARGET_DIR/outputs' '$TARGET_DIR/logs' '$TARGET_DIR/static_export'"

for group in "${SELECTED_GROUPS[@]}"; do
  case "$group" in
    core)
      run_sync "core" "$ROOT_DIR/" "$TARGET/"
      ;;
    artifacts)
      run_sync "artifacts-models" "$ROOT_DIR/models/" "$TARGET/models/"
      run_sync "artifacts-outputs" "$ROOT_DIR/outputs/" "$TARGET/outputs/"
      run_sync "artifacts-static-export" "$ROOT_DIR/static_export/" "$TARGET/static_export/"
      ;;
    data)
      run_sync "data" "$ROOT_DIR/data/" "$TARGET/data/"
      ;;
    *)
      echo "Grupo nao suportado: $group" >&2
      exit 1
      ;;
  esac
done

echo "Sync concluido."
echo "Host: $HOST"
echo "Destino: $TARGET_DIR"
echo "Grupos: ${SELECTED_GROUPS[*]}"
echo "Dry-run: $DRY_RUN"