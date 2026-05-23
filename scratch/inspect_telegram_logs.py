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
        
        # 1. List logs directory
        run_cmd(ssh, f"docker exec {container} ls -la /app/logs")
        
        # 2. Tail the gateway log if it exists
        run_cmd(ssh, f"docker exec {container} tail -n 100 /app/logs/telegram_gateway.log")
        run_cmd(ssh, f"docker exec {container} tail -n 100 /app/logs/telegram_gemini_gateway.log")
        
        # 3. Check what's in /app/outputs/mempalace/chat/history/
        run_cmd(ssh, f"docker exec {container} ls -lat /app/outputs/mempalace/chat/history | head -n 15")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
