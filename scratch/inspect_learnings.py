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
        
        cmd = "ls -la /home/reportpreview/apps/report-preview/outputs /home/reportpreview/apps/report-preview/outputs/mempalace"
        print(f"Running: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print("Output:")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
