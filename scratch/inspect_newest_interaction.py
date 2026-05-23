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
        
        # View response
        print("\n=== THE NEWEST RESPONSE ===")
        cmd = f"docker exec {container} cat /app/outputs/mempalace/chat/history/gemini_chat_geral_20260522_120614.md"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        # View prompt (just a part of it, since it's very large, let's print the last 1500 characters)
        print("\n=== THE NEWEST PROMPT (TAIL) ===")
        cmd = f"docker exec {container} tail -c 4000 /app/outputs/mempalace/chat/history/gemini_chat_prompt_geral_20260522_120614.txt"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        # Also print the head of the prompt (first 1000 characters) to see the start of the prompt
        print("\n=== THE NEWEST PROMPT (HEAD) ===")
        cmd = f"docker exec {container} head -n 30 /app/outputs/mempalace/chat/history/gemini_chat_prompt_geral_20260522_120614.txt"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
