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
        print("Connected successfully!")
        
        # Search the entire remote container filesystem
        cmd = "docker exec report-preview-telegram-gateway grep -r -l -i \"A capital do Ceará é Fortaleza\" / --exclude-dir={proc,sys,dev,run,sys,var/lib/docker}"
        print(f"Running command: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        
        if out:
            print("Found in files:")
            print(out)
        else:
            print("Not found in the filesystem.")
            
        if err:
            print("Stderr:")
            print(err)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
