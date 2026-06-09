#!/usr/bin/env python3
import os
import posixpath
import sys
from pathlib import Path

try:
    import paramiko
except Exception as e:
    print('paramiko não está instalado no ambiente usado para executar este script.', file=sys.stderr)
    raise

BASE = Path.cwd()
ENV_PATH = BASE / '.env'

def load_env(path: Path):
    data = {}
    if not path.exists():
        return data
    for ln in path.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#') or '=' not in ln:
            continue
        k, v = ln.split('=', 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data

env = load_env(ENV_PATH)
host = env.get('HOSTINGER_HOST_SSH') or env.get('HOSTINGER_HOST') or env.get('HOSTINGER_SYNC_HOST')
user = env.get('HOSTINGGER_USER') or env.get('HOSTINGER_USER') or env.get('HOSTINGER_SYNC_USER')
password = env.get('VPS_HOSTINGER_PASSWORD') or env.get('HOSTINGER_SYNC_PASSWORD')

host_ssh = env.get('HOST_SSH', '').strip()
if (not host or not user) and host_ssh:
    if '@' in host_ssh:
        parsed_user, parsed_host = host_ssh.split('@', 1)
        if not user:
            user = parsed_user.strip()
        if not host:
            host = parsed_host.strip()
    elif not host:
        host = host_ssh

if not password:
    password = env.get('PASSWORD_VPS_SSH', '').strip()

target_dir = env.get('HOSTINGER_SYNC_TARGET_DIR', '/home/reportpreview/apps/report-preview')
port = int(env.get('HOSTINGER_SYNC_PORT', '22'))

if not host or not user or not password:
    print('Faltam credenciais no .env: HOST, USER ou PASSWORD nao encontrados.', file=sys.stderr)
    sys.exit(2)

FILES = [
    'powershell/telegram_gemini_gateway.py',
    'scripts/linux/ask_gemini_with_mempalace.py',
    'powershell/ask_gemini_with_hermes_memory.ps1',
    'scripts/merge_new_data.py',
    'scripts/nodes/extract_top30_sentinela_micronodes.py',
    'outputs/visible_micronodes.csv',
    'outputs/top_30_micronodes.csv',
    'outputs/top_30_micronodes_capital.csv',
    'outputs/top_30_micronodes_rmf.csv',
    'outputs/top_30_micronodes_interior.csv',
    'outputs/hermes/visible_micronodes.csv',
    'outputs/hermes/top_30_micronodes.csv',
    'outputs/hermes/top_30_micronodes_capital.csv',
    'outputs/hermes/top_30_micronodes_rmf.csv',
    'outputs/hermes/top_30_micronodes_interior.csv',
    'outputs/hermes/dados_brutos_30dias.csv',
    'outputs/hermes/dados_brutos_60dias.csv',
    'outputs/hermes/dados_brutos_90dias.csv',
    'outputs/hermes/total_cvli_rua.csv',
    'outputs/hermes/total_cvli_micronodo.csv',
    'outputs/hermes/caminho_crime.csv',
    'outputs/hermes/risk_snapshot_latest.json',
    'outputs/hermes/risk_snapshot_latest.md',
    'outputs/hermes/risk_brief_latest.md',
    'outputs/hermes/risk_snapshot_latest.csv',
    'outputs/hermes/risk_fortaleza_latest.csv',
    'outputs/hermes/risk_rmf_latest.csv',
    'outputs/hermes/risk_interior_latest.csv',
    'outputs/hermes/dados_status_enriquecido_14d_latest.csv',
    'outputs/hermes/dados_status_enriquecido_14d_summary_latest.json',
    'outputs/hermes/dados_status_enriquecido_14d_summary_latest.md',
    'outputs/hermes/ruas_criticas_latest.csv',
    'scripts/generate_pipeline_artifacts.py',
]

def ensure_remote_dir(ssh, path):
    safe = path.replace("'", "'\"'\"'")
    cmd = f"mkdir -p '{safe}'"
    ssh.exec_command(cmd)

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'Conectando {user}@{host}:{port}...')
    ssh.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    sftp = ssh.open_sftp()
    uploaded = []
    skipped = []
    try:
        for rel in FILES:
            local = BASE / rel
            if not local.exists():
                print(f'Ignorando (nao existe localmente): {rel}')
                continue
            remote = posixpath.join(target_dir, rel.replace('\\', '/'))
            remote_dir = posixpath.dirname(remote)
            
            # Check modification time and size
            local_stat = local.stat()
            local_mtime = int(local_stat.st_mtime)
            local_size = local_stat.st_size
            
            should_upload = True
            try:
                remote_stat = sftp.stat(remote)
                remote_mtime = int(remote_stat.st_mtime)
                remote_size = remote_stat.st_size
                if remote_size == local_size and abs(remote_mtime - local_mtime) <= 2:
                    skipped.append(rel)
                    should_upload = False
            except IOError:
                pass
                
            if should_upload:
                print(f'Enviando {rel} -> {remote}')
                ensure_remote_dir(ssh, remote_dir)
                sftp.put(str(local), remote)
                sftp.utime(remote, (local_mtime, local_mtime))
                uploaded.append(rel)
                print(f'  Enviado: {rel}')
    finally:
        sftp.close()
        ssh.close()

    print('\nResumo:')
    if uploaded:
        print('Enviados:')
        for u in uploaded:
            print(' -', u)
    if skipped:
        print('Ignorados (sem alteracoes):')
        for s in skipped:
            print(' -', s)

if __name__ == '__main__':
    main()
