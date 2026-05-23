import paramiko
import sys

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected to VPS!")
        
        # Get last 50 lines of log of gateway
        stdin, stdout, stderr = ssh.exec_command(f"tail -n 50 {target_dir}/outputs/mempalace/chat/telegram_gemini_gateway.log")
        print("\n--- TELEGRAM GATEWAY LOGS ---")
        print(stdout.read().decode('utf-8'))
        
        err_out = stderr.read().decode('utf-8')
        if err_out:
            print("\n--- DOCKER CONTAINER STDERR ---")
            print(err_out)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
