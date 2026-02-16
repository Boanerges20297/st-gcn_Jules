import json
import os

BAIRROS_FILE = 'data/raw/bairros_centros_latlong.json'

TO_REMOVE = [
    "ALTO ALEGRE II", "CIDADE NOVA", "DIF III", "GUADALAJARA",
    "INDUSTRIAL", "IPARANA", "MARECHAL RONDON", "PARQUE ALBANO",
    "PARQUE DAS NAÇÕES", "PARQUE DAS NACOES", "PARQUE LEBLON",
    "PARQUE SOLEDADE", "PRECABURA", "RACHEL DE QUEIROZ",
    "TABAPUÁ", "URUCUTUBA"
]

MERGE_MAP = {
    "CONJUNTO CEARÁ I": "CONJUNTO CEARÁ",
    "CONJUNTO CEARÁ II": "CONJUNTO CEARÁ",
    "PRAIA DO FUTURO I": "PRAIA DO FUTURO",
    "PRAIA DO FUTURO II": "PRAIA DO FUTURO"
}

with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

regions = {}
fortaleza_nodes = set()

for name, info in data.items():
    reg = info.get('regiao', 'unknown').lower()
    regions[reg] = regions.get(reg, 0) + 1
    
    if reg == 'fortaleza':
        norm_name = name.upper().strip()
        if norm_name in TO_REMOVE:
            continue
        final_name = MERGE_MAP.get(norm_name, norm_name)
        fortaleza_nodes.add(final_name)

print(f"Raw counts: {regions}")
print(f"Fortaleza processed count: {len(fortaleza_nodes)}")
