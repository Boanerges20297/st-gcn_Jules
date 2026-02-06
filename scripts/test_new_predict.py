#!/usr/bin/env python3
import requests
import json
import time

print("Fazendo requisição ao /api/risk...")
r = requests.get('http://localhost:5000/api/risk')
print(f"Status: {r.status_code}")

if r.status_code == 200:
    print("✅ Requisição bem-sucedida!")
    time.sleep(1)  # Aguarda o arquivo ser escrito
    
    # Listar últimos predict
    from pathlib import Path
    predicts = sorted(Path('predicts').glob('predict_*.txt'), reverse=True)
    if predicts:
        latest = predicts[0]
        print(f"\n📄 Último arquivo: {latest.name}")
        print("\nMostrando top 5 nodes:\n")
        with open(latest, encoding='utf-8') as f:
            lines = f.readlines()
            in_ranking = False
            count = 0
            for line in lines:
                if 'RANKING ATUALIZADO' in line:
                    in_ranking = True
                    continue
                if in_ranking:
                    if 'CORREÇÕES' in line or '---' in line:
                        continue
                    if line.strip() and not line.startswith('Rank'):
                        print(line.rstrip())
                        count += 1
                        if count > 5:
                            break
else:
    print(f"❌ Erro: {r.status_code}")
