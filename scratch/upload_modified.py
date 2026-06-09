import os
import sys
import posixpath
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

target_dir = env.get('HOSTINGER_SYNC_TARGET_DIR', '/home/reportpreview/apps/report-preview')
port = int(env.get('HOSTINGER_SYNC_PORT', '22'))

if not host or not user or not password:
    print('Error: Missing HOST, USER or PASSWORD in .env file', file=sys.stderr)
    sys.exit(1)

FILES_TO_UPLOAD = [
    'app.py',
    'templates/index.html',
    'scripts/merge_new_data.py'
]

def ensure_remote_dir(ssh, path):
    safe = path.replace("'", "'\"'\"'")
    cmd = f"mkdir -p '{safe}'"
    ssh.exec_command(cmd)

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'Connecting to {user}@{host}:{port}...')
    ssh.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    sftp = ssh.open_sftp()
    try:
        for rel in FILES_TO_UPLOAD:
            local = BASE / rel
            if not local.exists():
                print(f'Local file does not exist: {rel}')
                continue
            remote = posixpath.join(target_dir, rel.replace('\\', '/'))
            remote_dir = posixpath.dirname(remote)
            
            print(f'Uploading {rel} -> {remote}...')
            ensure_remote_dir(ssh, remote_dir)
            sftp.put(str(local), remote)
            
            local_stat = local.stat()
            local_mtime = int(local_stat.st_mtime)
            sftp.utime(remote, (local_mtime, local_mtime))
            print(f'Successfully uploaded: {rel}')
    finally:
        sftp.close()
        ssh.close()
    print('All uploads completed successfully!')

if __name__ == '__main__':
    main()
