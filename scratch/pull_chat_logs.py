import os
import sys
import paramiko
from stat import S_ISDIR

def create_local_dir_recursive(local_path):
    """Creates a local directory if it doesn't exist."""
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
        print(f"Created local directory: {local_path}")

def download_file(sftp, remote_path, local_path):
    """Downloads a single file from the remote system."""
    local_dir = os.path.dirname(local_path)
    create_local_dir_recursive(local_dir)
    
    # Check remote and local file size/existence
    remote_stat = sftp.stat(remote_path)
    if os.path.exists(local_path):
        local_stat = os.stat(local_path)
        # If sizes are the same, skip to avoid redownloading unmodified logs
        if local_stat.st_size == remote_stat.st_size:
            # print(f"Skipping (already up to date): {local_path}")
            return
            
    print(f"Downloading {remote_path} -> {local_path} ...")
    sftp.get(remote_path, local_path)
    print("  Done")

def download_dir_recursive(sftp, remote_dir, local_dir):
    """Downloads a directory recursively from the remote system."""
    create_local_dir_recursive(local_dir)
    try:
        remote_files = sftp.listdir_attr(remote_dir)
    except IOError:
        # Directory does not exist on remote
        print(f"Remote directory {remote_dir} does not exist.")
        return

    for file_attr in remote_files:
        remote_path = os.path.join(remote_dir, file_attr.filename).replace('\\', '/')
        local_path = os.path.join(local_dir, file_attr.filename)
        
        if S_ISDIR(file_attr.st_mode):
            download_dir_recursive(sftp, remote_path, local_path)
        else:
            download_file(sftp, remote_path, local_path)

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    
    project_root = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    remote_chats_dir = f"{target_dir}/outputs/users_chats"
    local_chats_dir = os.path.join(project_root, "outputs", "users_chats")
    
    print("====================================================")
    print(f"Pulling user chat logs from VPS {username}@{host}:{port} ...")
    print("====================================================")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected successfully!")
        
        sftp = ssh.open_sftp()
        print("SFTP channel opened.")
        
        print(f"\nDownloading all remote user chats from remote '{remote_chats_dir}' to local '{local_chats_dir}'...")
        download_dir_recursive(sftp, remote_chats_dir, local_chats_dir)
        
        sftp.close()
        print("\nSFTP pull completed successfully!")
        print("====================================================")
        print("CHAT LOGS SYNCED LOCALLY!")
        print("====================================================")
        
    except Exception as e:
        print(f"\nError pulling logs: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
