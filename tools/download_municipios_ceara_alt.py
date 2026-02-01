import requests
import os

# Alternative URL (IBGE older endpoint)
url = 'https://servicodados.ibge.gov.br/api/v1/localidades/municipios'
out_path = 'data/static/municipios_ceara_all.json'

print('Downloading IBGE municipios (all)')
resp = requests.get(url, timeout=60)
if resp.status_code == 200:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    print('Saved to', out_path)
else:
    print('Failed to download:', resp.status_code)
