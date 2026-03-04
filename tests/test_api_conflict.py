import requests
import json

def test_conflict_simulation():
    url = "http://127.0.0.1:5050/api/simulate"
    # Coordenadas de um ponto que deve subir o risco (ex: centro de Fortaleza)
    payload = {
        "points": [
            [-3.72, -38.52]
        ],
        "type": "conflict"
    }
    
    try:
        print(f"Enviando requisição de CONFLITO para {url}...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("Sucesso!")
            top_10 = data.get('meta', {}).get('top10', [])
            print("\nTop 10 Áreas na Simulação de CONFLITO:")
            for item in top_10:
                print(f"- {item['name']}: {item['risk_score']:.2f}%")
        else:
            print(f"Erro: Status {response.status_code}")
            
    except Exception as e:
        print(f"Erro ao conectar: {e}")

if __name__ == "__main__":
    test_conflict_simulation()

