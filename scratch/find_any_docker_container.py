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
        
        # 1. List all containers
        stdin, stdout, stderr = ssh.exec_command("docker ps -a")
        print("\n--- DOCKER PS -A ---")
        print(stdout.read().decode('utf-8'))
        
        # 2. List all directories in /var/lib/docker/containers/
        stdin, stdout, stderr = ssh.exec_command("ls -la /var/lib/docker/containers/")
        print("\n--- DOCKER CONTAINER DIRECTORIES ---")
        print(stdout.read().decode('utf-8'))
        
        # 3. Find any file in /var/lib/docker/containers/ containing '80086019' or other logs
        stdin, stdout, stderr = ssh.exec_command("grep -rn '80086019' /var/lib/docker/containers/ 2>/dev/null | head -n 30")
        print("\n--- GREP USER CHAT ID IN CONTAINER LOGS ---")
        print(stdout.read().decode('utf-8'))
        
        # 4. Search for TELEGRAM_BOT_TOKEN in the entire VPS including deleted files/deleted docker logs if possible,
        # or grep from any older json.log files that might still exist on the system.
        stdin, stdout, stderr = ssh.exec_command("find /var/lib/docker/containers/ -name '*.log' -exec grep -H 'token' {} \\; 2>/dev/null | head -n 50")
        print("\n--- ALL LOGS SEARCH ---")
        print(stdout.read().decode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
