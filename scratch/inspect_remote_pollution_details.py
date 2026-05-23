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
        print("Connected successfully!")
        
        files = [
            "/app/outputs/mempalace/chat/history/gemini_chat_geral_20260522_104746.md",
            "/app/outputs/mempalace/chat/history/gemini_chat_rmf_20260522_115913.md",
            "/app/outputs/mempalace/chat/history/gemini_chat_geral_20260522_120614.md"
        ]
        
        for f in files:
            print(f"\n====================================\nFILE: {f}\n====================================")
            cmd = f"docker exec report-preview-telegram-gateway cat {f}"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            print(stdout.read().decode('utf-8', errors='ignore'))
            print(stderr.read().decode('utf-8', errors='ignore'))
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
