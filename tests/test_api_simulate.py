import requests
import json

def test_simulation():
    url = "http://127.0.0.1:5050/api/simulate"
    
    # Coordenadas aproximadas de um ponto em Fortaleza (ex: proximidades do Ancuri/Pedras)
    payload = {
        "points": [
            [-3.86, -38.50]
        ],
        "type": "suppression"
    }
    
    try:
        print(f"Enviando requisição para {url}...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("Sucesso!")
            print(f"Meta: {list(data.get('meta', {}).keys())}")
            print(f"Total de resultados: {len(data.get('data', []))}")
            
            # Verificar se algum bairro teve alteração ou se o retorno faz sentido
            top_10 = data.get('meta', {}).get('top10', [])
            print("\nTop 10 Áreas na Simulação:")
            for item in top_10:
                print(f"- {item['name']}: {item['risk_score']:.2f}%")
        else:
            print(f"Erro: Status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Erro ao conectar: {e}")

if __name__ == "__main__":
    test_simulation()
