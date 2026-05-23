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
        
        # Read remote .env
        sftp = ssh.open_sftp()
        remote_path = "/home/reportpreview/apps/report-preview/.env"
        
        with sftp.open(remote_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith("GEMINI_API_KEY="):
                new_lines.append("GEMINI_API_KEY=AIzaSyDjgJBvOyl38Tihaiim5uvDIcsWD8YGtTo\n")
                print("Updated GEMINI_API_KEY")
            elif line_strip.startswith("GOOGLE_API_KEY="):
                new_lines.append("GOOGLE_API_KEY=AIzaSyDjgJBvOyl38Tihaiim5uvDIcsWD8YGtTo\n")
                print("Updated GOOGLE_API_KEY")
            else:
                new_lines.append(line)
                
        # Write back to remote .env
        with sftp.open(remote_path, 'w') as f:
            f.writelines(new_lines)
        print("Remote .env updated successfully!")
        
        sftp.close()
        
        # Restart the container
        print("Restarting telegram-gateway on VPS...")
        stdin, stdout, stderr = ssh.exec_command("cd /home/reportpreview/apps/report-preview && docker compose -f docker-compose.telegram-only.yml up -d --build --force-recreate")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
