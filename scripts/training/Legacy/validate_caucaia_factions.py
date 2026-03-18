import json
import os

def find_caucaia_factions():
    intel_dir = 'data/raw/inteligencia/'
    factions = ['COMANDO VERMELHO', 'GDE', 'MASSA', 'PCC', 'TCP']
    results = {}
    
    for faction in factions:
        file_path = os.path.join(intel_dir, f"{faction}.geojson")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = 0
                for feature in data.get('features', []):
                    props = feature.get('properties', {})
                    # Procura por "CAUCAIA" em qualquer propriedade de texto
                    text = str(props).upper()
                    if 'CAUCAIA' in text:
                        count += 1
                results[faction] = count
                
    print("--- DOMINANCIA EM CAUCAIA (Contagem de Poligonos) ---")
    for f, c in results.items():
        print(f"{f}: {c} poligonos mapeados")

if __name__ == "__main__":
    find_caucaia_factions()
