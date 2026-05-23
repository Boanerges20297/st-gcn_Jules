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
        
        # SFTP upload
        sftp = ssh.open_sftp()
        local_path = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview\scripts\linux\debug_cleaner.py"
        remote_path = "/home/reportpreview/apps/report-preview/scripts/linux/debug_cleaner.py"
        print(f"Uploading {local_path} -> {remote_path} ...")
        sftp.put(local_path, remote_path)
        sftp.close()
        
        # We run the script inside the docker container
        cmd = "docker exec report-preview-telegram-gateway python /app/scripts/linux/debug_cleaner.py"
        
        print("Running debug_cleaner.py in docker container...")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        
        print("STDOUT:")
        print(out)
        print("STDERR:")
        print(err)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
