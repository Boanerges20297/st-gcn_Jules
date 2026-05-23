import paramiko
import sys

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    print("Connecting as root to inspect data dir ownership...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected!")
        
        commands = [
            f"ls -ld {target_dir}/data",
            f"ls -la {target_dir}/data",
            f"ls -la {target_dir}/data/users"
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
