import paramiko
import sys

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
        
        # 1. Check outputs/mempalace/chat contents
        print("\n=== FILES IN CHAT DIR ===")
        cmd = f"docker exec {container} ls -la /app/outputs/mempalace/chat"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        # 2. View newest 12:48:08 response
        print("\n=== gemini_chat_geral_20260522_124808.md ===")
        cmd = f"docker exec {container} cat /app/outputs/mempalace/chat/history/gemini_chat_geral_20260522_124808.md"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        # 3. View 12:44:42 response
        print("\n=== gemini_chat_geral_20260522_124442.md ===")
        cmd = f"docker exec {container} cat /app/outputs/mempalace/chat/history/gemini_chat_geral_20260522_124442.md"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))

        # 4. View tail of telegram_gemini_gateway.log
        print("\n=== TELEGRAM_GEMINI_GATEWAY.LOG ===")
        cmd = f"docker exec {container} tail -n 50 /app/outputs/mempalace/chat/telegram_gemini_gateway.log"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
