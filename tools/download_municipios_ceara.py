import requests
import os

url = 'https://servicodados.ibge.gov.br/api/v3/malhas/municipios?codigoUF=23&formato=application/vnd.geo+json'
out_path = 'data/static/municipios_ceara.geojson'

print('Downloading IBGE municipios (Ceara)…')
resp = requests.get(url, timeout=60)
if resp.status_code == 200:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    print('Saved to', out_path)
else:
    print('Failed to download:', resp.status_code, resp.text[:200])
