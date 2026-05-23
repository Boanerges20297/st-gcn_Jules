import paramiko
import sys
import json

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
        
        cmd = "cat /home/reportpreview/apps/report-preview/outputs/mempalace/global_learnings.json"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        content = stdout.read().decode('utf-8', errors='ignore')
        
        print("\n--- GLOBAL LEARNINGS FILE ON VPS ---")
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            print(content)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
