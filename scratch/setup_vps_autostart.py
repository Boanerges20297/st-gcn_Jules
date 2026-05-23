import paramiko
import sys

def main():
    host = "76.13.121.172"
    port = 22
    username = "root"
    password = "T/rqgLF'9gNFXwLZc(r0"
    target_dir = "/home/reportpreview/apps/report-preview"
    
    service_content = f"""[Unit]
Description=Telegram Gemini Gateway Container
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory={target_dir}
ExecStart=/usr/bin/docker compose -f docker-compose.telegram-only.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.telegram-only.yml down

[Install]
WantedBy=multi-user.target
"""
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, password=password, timeout=15)
        print("Connected successfully!")
        
        # 1. Enable Docker service to start on boot
        print("Enabling Docker systemd service to start on boot...")
        stdin, stdout, stderr = ssh.exec_command("systemctl enable docker")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        
        # 2. Write the service file
        service_path = "/etc/systemd/system/telegram-gateway.service"
        print(f"Writing systemd service file to {service_path}...")
        
        sftp = ssh.open_sftp()
        with sftp.file(service_path, "w") as f:
            f.write(service_content)
        sftp.close()
        
        # 3. Reload systemd daemon
        print("Reloading systemd daemon...")
        stdin, stdout, stderr = ssh.exec_command("systemctl daemon-reload")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        
        # 4. Enable telegram-gateway service
        print("Enabling telegram-gateway service to start on boot...")
        stdin, stdout, stderr = ssh.exec_command("systemctl enable telegram-gateway")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        
        # 5. Start telegram-gateway service (idempotent)
        print("Starting/Validating telegram-gateway service...")
        stdin, stdout, stderr = ssh.exec_command("systemctl start telegram-gateway")
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        
        # 6. Check status of service
        print("Checking telegram-gateway service status...")
        stdin, stdout, stderr = ssh.exec_command("systemctl status telegram-gateway")
        print(stdout.read().decode('utf-8'))
        
        print("\n====================================================")
        print("AUTOSTART CONFIGURATION CONCLUDED SUCCESSFULLY!")
        print("====================================================")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
