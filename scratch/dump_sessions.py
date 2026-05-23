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
        
        # Files we want to inspect
        files_to_read = [
            "session_20260522_090532.txt",
            "session_20260522_090804.txt",
            "session_20260522_091558.txt",
            "session_20260522_092544.txt",
            "session_20260522_093555.txt"
        ]
        
        for fname in files_to_read:
            path = f"{target_dir}/outputs/users_chats/boanerges/{fname}"
            print(f"\n====================================================")
            print(f"FILE: {fname}")
            print(f"====================================================")
            stdin, stdout, stderr = ssh.exec_command(f"cat {path}")
            print(stdout.read().decode('utf-8', errors='ignore'))
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
