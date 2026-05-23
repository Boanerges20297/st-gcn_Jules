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
        
        # We only print the MD file (the answer file), which is much smaller and what we care about
        fpath = f"{target_dir}/outputs/mempalace/chat/history/gemini_chat_geral_20260522_120614.md"
        print(f"\n====================================================")
        print(f"FILE: {fpath}")
        print(f"====================================================")
        stdin, stdout, stderr = ssh.exec_command(f"cat {fpath}")
        content = stdout.read().decode('utf-8', errors='ignore')
        # Print safely by replacing characters the windows terminal can't print
        print(content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    import sys
    main()
