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
        
        fpath = f"{target_dir}/outputs/mempalace/chat/history/gemini_chat_geral_20260522_120614.md"
        # Read the raw bytes of the file
        sftp = ssh.open_sftp()
        with sftp.file(fpath, "rb") as f:
            bytes_content = f.read()
        sftp.close()
        
        print("\n--- RAW BYTES OF THE FILE ---")
        print(f"Total bytes: {len(bytes_content)}")
        
        # Print a window of bytes around 'Cear'
        idx = bytes_content.find(b"Cear")
        if idx != -1:
            window = bytes_content[idx:idx+40]
            print(f"Bytes around 'Cear': {window}")
            print(f"Hex: {window.hex()}")
            for b in window:
                print(f"  {chr(b) if 32 <= b < 127 else '.'} -> {hex(b)} ({b})")
        else:
            print("'Cear' not found in bytes.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
