import paramiko
import sys

def run_cmd(ssh, cmd):
    print(f"\nRunning command: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    if out:
        print("--- STDOUT ---")
        print(out)
    if err:
        print("--- STDERR ---")
        print(err)

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected successfully to VPS!")
        
        run_cmd(ssh, "docker exec report-preview-telegram-gateway ls -la /app/outputs/mempalace/chat")
        run_cmd(ssh, "docker exec report-preview-telegram-gateway ls -la /app/outputs/mempalace/chat/history")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
