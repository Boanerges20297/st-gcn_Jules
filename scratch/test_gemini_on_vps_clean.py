import paramiko
import sys

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected!")
        
        # Test case 1: Query that previously timed out
        query = "Analise os homicidios em no bairro Passare, nos ultimos 90 dias."
        print(f"\n--- TESTING QUERY ON VPS: '{query}' ---")
        cmd = f"docker exec report-preview-telegram-gateway python /app/scripts/linux/ask_gemini_with_mempalace.py --query \"{query}\" --scope fortaleza"
        
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        
        if out:
            print("STDOUT:")
            print(out)
        if err:
            print("STDERR:")
            print(err)
            
        # Let's read the latest answer file to see what was written!
        print("\n--- READING LATEST ANSWER FILE ---")
        cmd = "docker exec report-preview-telegram-gateway cat /app/outputs/mempalace/chat/gemini_chat_fortaleza_latest.md"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out_answer = stdout.read().decode('utf-8', errors='ignore')
        print(out_answer)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
