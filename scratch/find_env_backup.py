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
        print("Connected!")
        
        # 1. Look for .env backups or history
        stdin, stdout, stderr = ssh.exec_command(f"find {target_dir} -name '.env*' -o -name '*backup*'")
        print("\n--- FIND ENV / BACKUPS ---")
        print(stdout.read().decode('utf-8'))
        
        # 2. Check shell history for TELEGRAM_BOT_TOKEN
        stdin, stdout, stderr = ssh.exec_command("history | grep -i token")
        print("\n--- SHELL HISTORY ---")
        print(stdout.read().decode('utf-8'))
        
        # 3. Check git log or git stash or git reflog
        stdin, stdout, stderr = ssh.exec_command(f"cd {target_dir} && git reflog || git log -n 5")
        print("\n--- GIT LOG ---")
        print(stdout.read().decode('utf-8'))
        
        # 4. Check docker inspect to see if the old container config has the env variable cached
        stdin, stdout, stderr = ssh.exec_command("docker inspect report-preview-telegram-gateway")
        print("\n--- DOCKER INSPECT ---")
        # Just write the env section
        inspect_out = stdout.read().decode('utf-8')
        print(inspect_out[:2000]) # first 2000 chars should show it if it's there
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
