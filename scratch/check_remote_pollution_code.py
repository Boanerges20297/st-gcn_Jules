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
        
        # Read the file ask_gemini_with_mempalace.py on host
        cmd = "cat /home/reportpreview/apps/report-preview/scripts/linux/ask_gemini_with_mempalace.py"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        content = stdout.read().decode('utf-8', errors='ignore')
        
        # Find clean_response_pollution
        start_idx = content.find("def clean_response_pollution")
        if start_idx != -1:
            print("\n--- FOUND clean_response_pollution ON VPS HOST ---")
            print(content[start_idx:start_idx+1500])
        else:
            print("\nclean_response_pollution NOT FOUND in VPS host file!")
            
        # Read the file ask_gemini_with_mempalace.py inside docker container
        cmd = "docker exec report-preview-telegram-gateway cat /app/scripts/linux/ask_gemini_with_mempalace.py"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        content_docker = stdout.read().decode('utf-8', errors='ignore')
        
        start_idx_docker = content_docker.find("def clean_response_pollution")
        if start_idx_docker != -1:
            print("\n--- FOUND clean_response_pollution IN DOCKER CONTAINER ON VPS ---")
            print(content_docker[start_idx_docker:start_idx_docker+1500])
        else:
            print("\nclean_response_pollution NOT FOUND in docker container file!")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
