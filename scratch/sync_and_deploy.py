import os
import sys
import paramiko
from stat import S_ISDIR

def create_remote_dir_recursive(sftp, remote_path):
    """Creates a directory on the remote system, including parent directories if they don't exist."""
    path_parts = [p for p in remote_path.split('/') if p]
    current_path = ""
    if remote_path.startswith('/'):
        current_path = "/"
    
    for part in path_parts:
        current_path = os.path.join(current_path, part).replace('\\', '/')
        try:
            sftp.mkdir(current_path)
            print(f"Created remote directory: {current_path}")
        except IOError:
            # Directory already exists or permission error (handled by subsequent operations)
            pass

def upload_file(sftp, local_path, remote_path):
    """Uploads a single file to the remote system, creating directories if needed."""
    remote_dir = os.path.dirname(remote_path).replace('\\', '/')
    create_remote_dir_recursive(sftp, remote_dir)
    print(f"Uploading {local_path} -> {remote_path} ...")
    sftp.put(local_path, remote_path)
    print("  Done")

def upload_dir_recursive(sftp, local_dir, remote_dir):
    """Uploads a directory recursively to the remote system."""
    create_remote_dir_recursive(sftp, remote_dir)
    for root, dirs, files in os.walk(local_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, local_dir)
        if rel_path == '.':
            current_remote_dir = remote_dir
        else:
            current_remote_dir = os.path.join(remote_dir, rel_path).replace('\\', '/')
            create_remote_dir_recursive(sftp, current_remote_dir)
        
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(current_remote_dir, file).replace('\\', '/')
            upload_file(sftp, local_file, remote_file)

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    
    project_root = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    print("====================================================")
    print(f"Connecting to VPS {username}@{host}:{port} ...")
    print("====================================================")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected successfully!")
        
        sftp = ssh.open_sftp()
        print("SFTP channel opened.")
        
        # Files to upload
        changed_files = [
            (".env", ".env"),
            (".hermes.md", ".hermes.md"),
            ("docker-compose.telegram-only.yml", "docker-compose.telegram-only.yml"),
            ("docker/Dockerfile.telegram-gateway", "docker/Dockerfile.telegram-gateway"),
            ("scripts/linux/ask_gemini_with_mempalace.py", "scripts/linux/ask_gemini_with_mempalace.py"),
            ("powershell/telegram_gemini_gateway.py", "powershell/telegram_gemini_gateway.py"),
            ("scripts/nodes/extract_top30_sentinela_micronodes.py", "scripts/nodes/extract_top30_sentinela_micronodes.py"),
            ("outputs/top_30_micronodes_capital.csv", "outputs/top_30_micronodes_capital.csv"),
            ("outputs/top_30_micronodes_rmf.csv", "outputs/top_30_micronodes_rmf.csv"),
            ("outputs/top_30_micronodes_interior.csv", "outputs/top_30_micronodes_interior.csv"),
            ("outputs/hermes/visible_micronodes.csv", "outputs/hermes/visible_micronodes.csv"),
            ("outputs/hermes/top_30_micronodes.csv", "outputs/hermes/top_30_micronodes.csv"),
            ("outputs/hermes/top_30_micronodes_capital.csv", "outputs/hermes/top_30_micronodes_capital.csv"),
            ("outputs/hermes/top_30_micronodes_rmf.csv", "outputs/hermes/top_30_micronodes_rmf.csv"),
            ("outputs/hermes/top_30_micronodes_interior.csv", "outputs/hermes/top_30_micronodes_interior.csv"),
            ("outputs/hermes/dados_brutos_30dias.csv", "outputs/hermes/dados_brutos_30dias.csv"),
            ("outputs/hermes/dados_brutos_60dias.csv", "outputs/hermes/dados_brutos_60dias.csv"),
            ("outputs/hermes/dados_brutos_90dias.csv", "outputs/hermes/dados_brutos_90dias.csv"),
            ("outputs/hermes/total_cvli_rua.csv", "outputs/hermes/total_cvli_rua.csv"),
            ("outputs/hermes/total_cvli_micronodo.csv", "outputs/hermes/total_cvli_micronodo.csv"),
            ("outputs/hermes/caminho_crime.csv", "outputs/hermes/caminho_crime.csv"),
            ("outputs/hermes/risk_fortaleza_latest.csv", "outputs/hermes/risk_fortaleza_latest.csv"),
            ("outputs/hermes/risk_rmf_latest.csv", "outputs/hermes/risk_rmf_latest.csv"),
            ("outputs/hermes/risk_interior_latest.csv", "outputs/hermes/risk_interior_latest.csv"),
            ("outputs/hermes/risk_snapshot_latest.csv", "outputs/hermes/risk_snapshot_latest.csv"),
            ("outputs/hermes/dados_status_enriquecido_14d_latest.csv", "outputs/hermes/dados_status_enriquecido_14d_latest.csv"),
            ("scripts/generate_pipeline_artifacts.py", "scripts/generate_pipeline_artifacts.py"),
            ("src/hostinger_sync.py", "src/hostinger_sync.py"),
            ("scripts/hostinger_upload_selected.py", "scripts/hostinger_upload_selected.py"),
            ("scripts/ais_lookup.py", "scripts/ais_lookup.py"),
            ("data/raw/AIS_Territorios.csv", "data/raw/AIS_Territorios.csv"),
            ("data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv", "data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
        ]
        
        print("\nUploading core files...")
        for local_rel, remote_rel in changed_files:
            local_path = os.path.join(project_root, local_rel)
            remote_path = os.path.join(target_dir, remote_rel).replace('\\', '/')
            upload_file(sftp, local_path, remote_path)
            
        # Directories to upload
        changed_dirs = []
        
        print("\nUploading directories...")
        for local_rel, remote_rel in changed_dirs:
            local_path = os.path.join(project_root, local_rel)
            remote_path = os.path.join(target_dir, remote_rel).replace('\\', '/')
            upload_dir_recursive(sftp, local_path, remote_path)
            
        sftp.close()
        print("\nSFTP upload completed successfully!")
        
        # Commands to execute on VPS
        commands = [
            f"mkdir -p {target_dir}/outputs/mempalace {target_dir}/outputs/users_chat {target_dir}/data",
            f"chown -R reportpreview:reportpreview {target_dir}",
            f"chown -R 1001:1001 {target_dir}/outputs",
            f"chown -R 1001:1001 {target_dir}/logs",
            f"chown -R 1001:1001 {target_dir}/data",
            f"chmod -R 777 {target_dir}/outputs",
            f"chmod -R 777 {target_dir}/logs",
            f"chmod -R 777 {target_dir}/data",
            f"cd {target_dir} && docker compose -f docker-compose.telegram-only.yml up -d --build --force-recreate",
            f"cd {target_dir} && docker compose -f docker-compose.telegram-only.yml ps",
            f"cd {target_dir} && docker compose -f docker-compose.telegram-only.yml logs --tail 30 telegram-gateway"
        ]
        
        print("\nExecuting Docker Compose commands on VPS...")
        for cmd in commands:
            print(f"\nRunning command: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            
            # Read stdout and stderr in real-time
            out = stdout.read().decode('utf-8', errors='ignore')
            err = stderr.read().decode('utf-8', errors='ignore')
            
            if out:
                print("--- STDOUT ---")
                print(out.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            if err:
                print("--- STDERR ---")
                print(err.encode(sys.stderr.encoding or 'utf-8', errors='replace').decode(sys.stderr.encoding or 'utf-8'))
                
        print("\n====================================================")
        print("DEPLOY CONCLUDED SUCCESSFULLY!")
        print("====================================================")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
