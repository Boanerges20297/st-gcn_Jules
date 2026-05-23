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
        
        # 1. Search for any .swp, .swo, ~ files in target_dir
        stdin, stdout, stderr = ssh.exec_command("find /home/reportpreview/apps/report-preview -name '.*.sw*' -o -name '*~'")
        print("\n--- SWAP OR TEMP FILES ---")
        print(stdout.read().decode('utf-8'))
        
        # 2. Check root or user nano/vim history files
        stdin, stdout, stderr = ssh.exec_command("cat /root/.nano_history /home/reportpreview/.nano_history /root/.viminfo /home/reportpreview/.viminfo 2>/dev/null | grep -i token")
        print("\n--- NANO/VIM HISTORY ---")
        print(stdout.read().decode('utf-8'))
        
        # 3. Check for any environment dump in /tmp or /var/tmp or logs
        stdin, stdout, stderr = ssh.exec_command("grep -rn 'TELEGRAM_BOT_TOKEN' /tmp/ /var/tmp/ 2>/dev/null")
        print("\n--- TEMP DIRECTORY SEARCH ---")
        print(stdout.read().decode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
