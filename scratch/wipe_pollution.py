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
        print("Connected successfully to VPS!")
        
        # Completely remove global_learnings.json in the gateway container
        cmd = "docker exec report-preview-telegram-gateway rm -f /app/outputs/mempalace/global_learnings.json"
        print(f"Running command: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        
        if out:
            print("--- STDOUT ---")
            print(out)
        if err:
            print("--- STDERR ---")
            print(err)
            
        print("\nChecking if the file was deleted successfully...")
        check_cmd = "docker exec report-preview-telegram-gateway ls -la /app/outputs/mempalace/"
        stdin, stdout, stderr = ssh.exec_command(check_cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
