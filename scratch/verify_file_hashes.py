import hashlib
import paramiko
import os

def get_local_md5(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    files_to_check = [
        "scripts/linux/ask_gemini_with_mempalace.py",
        "powershell/telegram_gemini_gateway.py",
        ".hermes.md",
        ".mempalace/SOUL.md",
        "outputs/hermes/top_30_micronodes.csv",
        "outputs/hermes/visible_micronodes.csv",
    ]
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected to VPS for MD5 comparison...")
        
        for rel_path in files_to_check:
            local_path = os.path.join(r"c:\Users\Boanerges\Desktop\Projetos\Report Preview", rel_path)
            local_md5 = get_local_md5(local_path)
            
            remote_path = os.path.join(target_dir, rel_path).replace('\\', '/')
            stdin, stdout, stderr = ssh.exec_command(f"md5sum {remote_path}")
            remote_out = stdout.read().decode('utf-8').strip()
            
            if remote_out:
                remote_md5 = remote_out.split()[0]
            else:
                remote_md5 = "Error: File not found on remote"
                
            status = "MATCH" if local_md5 == remote_md5 else "MISMATCH"
            print(f"\nFile: {rel_path}")
            print(f"  Local MD5:  {local_md5}")
            print(f"  Remote MD5: {remote_md5}")
            print(f"  Status:     {status}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
