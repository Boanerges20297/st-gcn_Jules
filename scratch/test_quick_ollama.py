import requests
import time

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3:8b",
    "prompt": "Olá, responda apenas 'OK' se você estiver funcionando.",
    "stream": False
}

print("=== INICIANDO TESTE RÁPIDO DO OLLAMA ===")
start_time = time.time()
try:
    resp = requests.post(url, json=payload, timeout=45)
    duration = time.time() - start_time
    if resp.status_code == 200:
        result = resp.json().get("response", "").strip()
        print(f"Ollama respondeu em {duration:.2f}s!")
        print(f"Resposta: {result}")
    else:
        print(f"Erro {resp.status_code} em {duration:.2f}s: {resp.text}")
except Exception as e:
    duration = time.time() - start_time
    print(f"Falha após {duration:.2f}s: {e}")
