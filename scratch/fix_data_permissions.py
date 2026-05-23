import paramiko
import sys
import time

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    print("Connecting as root to fix data directory permissions and ownership...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected!")
        
        commands = [
            f"chown -R 1001:1001 {target_dir}/data",
            f"chmod -R 777 {target_dir}/data",
            f"ls -la {target_dir}/data",
            f"ls -la {target_dir}/data/users",
            f"cd {target_dir} && docker compose -f docker-compose.telegram-only.yml restart telegram-gateway",
            "sleep 5",
            f"cd {target_dir} && docker compose -f docker-compose.telegram-only.yml ps",
            f"tail -n 30 {target_dir}/outputs/mempalace/chat/telegram_gemini_gateway.log"
        ]
        
        for cmd in commands:
            print(f"\nRunning: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            out = stdout.read().decode('utf-8', errors='ignore')
            err = stderr.read().decode('utf-8', errors='ignore')
            
            if out:
                print("--- STDOUT ---")
                print(out)
            if err:
                print("--- STDERR ---")
                print(err)
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
