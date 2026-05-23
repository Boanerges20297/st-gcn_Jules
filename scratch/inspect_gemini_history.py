import paramiko
import sys

def run_cmd(ssh, cmd):
    print(f"\n========================================\nRunning: {cmd}\n========================================")
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
        
        container = "report-preview-telegram-gateway"
        
        # List files recursively
        run_cmd(ssh, f"docker exec {container} find /home/appuser/.gemini")
        
        # Read projects.json
        run_cmd(ssh, f"docker exec {container} cat /home/appuser/.gemini/projects.json")
        
        # List contents of the app directories under tmp and history
        run_cmd(ssh, f"docker exec {container} ls -la /home/appuser/.gemini/history/app")
        run_cmd(ssh, f"docker exec {container} ls -la /home/appuser/.gemini/tmp/app")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
