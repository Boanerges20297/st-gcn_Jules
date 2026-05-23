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
        
        # Search for .env files anywhere in /home or /root
        stdin, stdout, stderr = ssh.exec_command("find /home/ /root/ -name '.env*' 2>/dev/null")
        print("\n--- ALL .env FILES ON VPS ---")
        print(stdout.read().decode('utf-8'))
        
        # Search for hermes-agent configuration or other things
        stdin, stdout, stderr = ssh.exec_command("find /home/ /root/ -name '*hermes*' -o -name '*agent*' 2>/dev/null")
        print("\n--- HERMES/AGENT FILES ---")
        print(stdout.read().decode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
