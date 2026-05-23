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
        
        # Test calling wrapper without explicit model to let it default to gemini-2.5-flash
        print("\n--- RUNNING PYTHON WRAPPER WITH INVERTED KEY ROTATION ---")
        cmd = "docker exec report-preview-telegram-gateway python /app/scripts/linux/ask_gemini_with_mempalace.py --query 'Qual a capital do Ceará? Responda em uma única linha.' --scope geral"
        print(f"Running: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print("STDOUT:", stdout.read().decode('utf-8'))
        print("STDERR:", stderr.read().decode('utf-8'))
        
        # Read the generated response
        print("\n--- VIEW LATEST GEMINI RESPONSE FROM VPS CONTAINER ---")
        cmd_cat = "docker exec report-preview-telegram-gateway cat /app/outputs/mempalace/chat/gemini_chat_geral_latest.md"
        stdin, stdout, stderr = ssh.exec_command(cmd_cat)
        print(stdout.read().decode('utf-8'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
