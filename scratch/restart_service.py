import os
import sys
from pathlib import Path
import paramiko

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

port = int(env.get('HOSTINGER_SYNC_PORT', '22'))

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'Connecting to {user}@{host}:{port}...')
    ssh.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    try:
        print('Restarting crime-predict Docker container...')
        stdin, stdout, stderr = ssh.exec_command('docker restart crime-predict')
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        print('Container restarted!')
    finally:
        ssh.close()

if __name__ == '__main__':
    main()
