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
        
        # 1. Search all files under /home/reportpreview for any mention of TELEGRAM_BOT_TOKEN
        # excluding .git directories and env.example
        stdin, stdout, stderr = ssh.exec_command("grep -rn 'TELEGRAM_BOT_TOKEN' /home/reportpreview/ --exclude-dir=.git --exclude='*.example'")
        print("\n--- GREP IN /home/reportpreview ---")
        print(stdout.read().decode('utf-8'))
        
        # 2. Check docker log files in /var/lib/docker/containers/ for old logs that might contain the token or start message
        # Let's search inside /var/lib/docker/containers/ for TELEGRAM_BOT_TOKEN or the old container output
        stdin, stdout, stderr = ssh.exec_command("grep -rn 'TELEGRAM_BOT_TOKEN' /var/lib/docker/containers/ 2>/dev/null | head -n 50")
        print("\n--- GREP IN DOCKER CONTAINER LOGS ---")
        print(stdout.read().decode('utf-8'))
        
        # 3. Check shell history of all users
        stdin, stdout, stderr = ssh.exec_command("cat /root/.bash_history /home/reportpreview/.bash_history 2>/dev/null | grep -E 'token|env' | tail -n 50")
        print("\n--- BASH HISTORY ---")
        print(stdout.read().decode('utf-8'))
        
        # 4. Search for the word 'TELEGRAM_BOT_TOKEN' in the system logs or auth logs
        stdin, stdout, stderr = ssh.exec_command("grep -rn 'TELEGRAM_BOT_TOKEN' /var/log/ 2>/dev/null | head -n 20")
        print("\n--- SYSTEM LOGS ---")
        print(stdout.read().decode('utf-8'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
