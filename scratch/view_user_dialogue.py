import paramiko
import sys

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    print("Connecting as root to view user chat logs on VPS...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected!")
        
        # List the directories under outputs/users_chats/
        stdin, stdout, stderr = ssh.exec_command(f"ls -la {target_dir}/outputs/users_chats")
        print("\n--- USERS CHATS DIRECTORIES ---")
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        # Check subdirectories/files for boanerges
        stdin, stdout, stderr = ssh.exec_command(f"ls -la {target_dir}/outputs/users_chats/boanerges")
        print("\n--- FILES UNDER boanerges/ ---")
        files_out = stdout.read().decode('utf-8', errors='ignore')
        print(files_out)
        
        # If there are any session files, read the most recent one
        stdin, stdout, stderr = ssh.exec_command(f"find {target_dir}/outputs/users_chats/boanerges -name 'session_*.txt' | sort | tail -n 1")
        last_file = stdout.read().decode('utf-8', errors='ignore').strip()
        if last_file:
            print(f"\n--- CONTENTS OF {last_file} ---")
            stdin, stdout, stderr = ssh.exec_command(f"cat {last_file}")
            print(stdout.read().decode('utf-8', errors='ignore'))
        else:
            print("\nNo session files found for boanerges.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
