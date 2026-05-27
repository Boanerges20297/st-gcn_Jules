import requests
import json

print("=== VERIFICANDO ENDPOINT DO OLLAMA LOCAL ===")
try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])]
        print(f"Ollama está ONLINE!")
        print(f"Modelos instalados: {models}")
    else:
        print(f"Ollama respondeu com status {resp.status_code}: {resp.text}")
except Exception as e:
    print(f"Não foi possível conectar ao Ollama local: {e}")
