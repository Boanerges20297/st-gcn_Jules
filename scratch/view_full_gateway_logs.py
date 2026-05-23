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
        print("Connected successfully!")
        
        # Read last 150 lines of the gateway log file
        log_path = "/home/reportpreview/apps/report-preview/outputs/mempalace/chat/telegram_gemini_gateway.log"
        print(f"\n--- GATEWAY LOG FILE ({log_path}) (LAST 150 LINES) ---")
        stdin, stdout, stderr = ssh.exec_command(f"tail -n 150 {log_path}")
        print(stdout.read().decode('utf-8', errors='ignore'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
