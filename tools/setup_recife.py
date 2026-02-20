import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

def setup_recife_env():
    # 1. Criar Pastas
    dirs = ['data/raw/recife', 'data/processed/recife', 'models/active/recife']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Diretório criado: {d}")

    # 2. Baixar GeoJSON de Bairros
    url = "http://dados.recife.pe.gov.br/dataset/c1f100f0-f56f-4dd4-9dcc-1aa4da28798a/resource/e43bee60-9448-4d3d-92ff-2378bc3b5b00/download/bairros.geojson"
    target = 'data/raw/recife/bairros_recife.geojson'
    
    print(f"⬇️ Baixando malha de Recife de: {url}")
    try:
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            with open(target, 'wb') as f:
                f.write(r.content)
            print(f"✅ Malha salva em: {target}")
        else:
            print(f"❌ Erro ao baixar malha: HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ Falha no download: {e}")

if __name__ == "__main__":
    setup_recife_env()
