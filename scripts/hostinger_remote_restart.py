#!/usr/bin/env python3
import os
import sys
from pathlib import Path

try:
    import paramiko
except Exception as e:
    print('paramiko não está instalado neste ambiente.', file=sys.stderr)
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


def main():
    env = load_env(ENV_PATH)
    host = env.get('HOSTINGER_HOST_SSH') or env.get('HOSTINGER_HOST') or env.get('HOSTING_HOST')
    user = env.get('HOSTINGGER_USER') or env.get('HOSTINGER_USER') or env.get('HOSTINGER_SYNC_USER') or env.get('USER')
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
        print('Faltam credenciais SSH no .env (host/user/password).', file=sys.stderr)
        sys.exit(2)

    cmd = f"cd {target_dir} && docker compose -f docker-compose.telegram-only.yml restart telegram-gateway"
    print(f'Executando remoto: {cmd} em {host}@{port}')

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=host, port=port, username=user, password=password, timeout=30)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        exit_status = stdout.channel.recv_exit_status()
        print('--- STDOUT ---')
        print(out.strip())
        print('--- STDERR ---')
        print(err.strip())
        print('--- EXIT STATUS ---')
        print(exit_status)
        if exit_status != 0:
            print('Comando remoto retornou erro.', file=sys.stderr)
            sys.exit(exit_status)
    finally:
        try:
            ssh.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
