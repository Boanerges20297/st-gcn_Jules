import paramiko

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected!")
        
        # Run a grep on the remote VPS session files
        cmd = f"grep -rn 'futuro' {target_dir}/outputs/users_chats/boanerges/"
        print("\n--- Grep for 'futuro' in session files ---")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        print(stderr.read().decode('utf-8', errors='ignore'))
        
        cmd = f"grep -rn 'capital' {target_dir}/outputs/users_chats/boanerges/"
        print("\n--- Grep for 'capital' in session files ---")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        print(stderr.read().decode('utf-8', errors='ignore'))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
