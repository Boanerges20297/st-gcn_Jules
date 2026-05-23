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
        print("Connected to VPS successfully!")
        
        container = "report-preview-telegram-gateway"
        
        print("\n=== DOCKER CONTAINERS STATUS ===")
        cmd = "docker ps -a"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        print("\n=== GATEWAY LOGS (LATEST 30 LINES) ===")
        cmd = f"docker logs --tail 30 {container}"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
