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
        
        # We will run a python script inside the docker container that imports clean_response_pollution 
        # from /app/scripts/linux/ask_gemini_with_mempalace.py and tests it on the exact text of that file.
        python_test_code = """
import sys
sys.path.append('/app/scripts/linux')
from ask_gemini_with_mempalace import clean_response_pollution

text = "A capital do Ceará é Fortaleza, onde a Aerolândia lidera o risco crítico (81.8) e a projeção tática para os próximos 7 dias exige foco em roubos a pessoas."
query = "mas me parece que voce esta avaliando apenas o passado e buscando o futuro"

cleaned = clean_response_pollution(text, query)
print("Text:", text)
print("Cleaned:", cleaned)
print("Matched?", cleaned != text)
"""
        # Save this to a temp file in container or run it with python -c
        import shutil
        import tempfile
        
        # Let's run python -c inside the docker container
        # We need to escape quotes and newlines
        escaped_code = python_test_code.replace('"', '\\"').replace('\n', '; ')
        cmd = f'docker exec report-preview-telegram-gateway python -c "{escaped_code}"'
        
        print("Running command in docker container...")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        
        print("STDOUT:")
        print(out)
        print("STDERR:")
        print(err)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
