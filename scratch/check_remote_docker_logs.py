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
        
        # Check container status
        print("\n--- CONTAINER STATUS ---")
        stdin, stdout, stderr = ssh.exec_command("docker ps -a --filter name=report-preview-telegram-gateway")
        print(stdout.read().decode('utf-8'))
        
        # Get last 50 lines of container logs
        print("\n--- CONTAINER LOGS (LAST 50 LINES) ---")
        stdin, stdout, stderr = ssh.exec_command("docker logs --tail 50 report-preview-telegram-gateway")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8')) # docker logs outputs to stderr sometimes
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
