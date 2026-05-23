import paramiko
import sys

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected to VPS!")
        
        # Check files in outputs/hermes
        stdin, stdout, stderr = ssh.exec_command(f"ls -lh {target_dir}/outputs/hermes/")
        print("\n--- FILES IN REMOTE outputs/hermes ---")
        print(stdout.read().decode('utf-8'))
        
        # Check files in outputs/mempalace
        stdin, stdout, stderr = ssh.exec_command(f"ls -lh {target_dir}/outputs/mempalace/")
        print("\n--- FILES IN REMOTE outputs/mempalace ---")
        print(stdout.read().decode('utf-8'))
        
        # Check the git status or git diff if applicable
        stdin, stdout, stderr = ssh.exec_command(f"cd {target_dir} && git status")
        print("\n--- REMOTE GIT STATUS ---")
        print(stdout.read().decode('utf-8'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
