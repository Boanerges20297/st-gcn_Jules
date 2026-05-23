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
        
        # Read the prompt file from VPS
        cmd = "docker exec report-preview-telegram-gateway cat /app/outputs/mempalace/chat/history/gemini_chat_prompt_geral_20260522_123249.txt"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        content_bytes = stdout.read()
        
        with open("scratch/vps_prompt.txt", "wb") as f:
            f.write(content_bytes)
        print("Downloaded to scratch/vps_prompt.txt successfully!")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
