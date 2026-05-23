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
        
        # 1. Who is running inside the container and what is the environment?
        run_cmd(ssh, f"docker exec {container} whoami")
        run_cmd(ssh, f"docker exec {container} pwd")
        
        # 2. Check where the gemini command is located
        run_cmd(ssh, f"docker exec {container} which gemini")
        
        # 3. Check for any hidden files or folders related to gemini in /app, /root, /home/appuser
        run_cmd(ssh, f"docker exec {container} ls -la /app")
        run_cmd(ssh, f"docker exec {container} ls -la /app/.gemini")
        run_cmd(ssh, f"docker exec {container} find /app -name '.gemini' -o -name '*gemini*' -maxdepth 3")
        run_cmd(ssh, f"docker exec {container} find /home -name '.gemini' -o -name '*gemini*'")
        run_cmd(ssh, f"docker exec {container} find /root -name '.gemini' -o -name '*gemini*'")
        
        # 4. Check if we can run the gemini CLI to list sessions
        run_cmd(ssh, f"docker exec {container} gemini --list-sessions")
        run_cmd(ssh, f"docker exec {container} gemini --help")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
