"""Referência canônica dos bairros oficiais de Fortaleza.

O GeoJSON municipal é a autoridade para a malha.  As ocorrências históricas
podem conter grafias antigas, abreviações e erros de digitação; por isso a
normalização abaixo é propositalmente conservadora: só aliases revisados são
convertidos e valores desconhecidos continuam desconhecidos.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd


OFFICIAL_GEOJSON = Path(__file__).resolve().parents[2] / "data/static/bairros_2025.geojson"


def bairro_key(value: object) -> str:
    """Chave sem acentos/pontuação para comparação, sem inferir equivalências."""
    # O arquivo municipal chegou com alguns acentos serializados como U+FFFD
    # ("S�O", "CEAR�"). Nesses topônimos a substituição correta é "A".
    text = unicodedata.normalize("NFKD", str(value or "").replace("�", "A"))
    text = text.encode("ASCII", "ignore").decode("ASCII").upper()
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


# Somente equivalências inequívocas encontradas nas ocorrências históricas.
# Não inclua aqui localidades como IPARANA, TABAPUA ou municípios da RMF.
OFFICIAL_ALIASES = {
    "BOA VISTA": "BOA VISTA-CASTELAO",
    "CASTELAO": "BOA VISTA-CASTELAO",
    "CIRSTO REDENTOR": "CRISTO REDENTOR",
    "GENIBAU": "PARQUE GENIBAU",
    "JANGURURSSU": "JANGURUSSU",
    "JOSE WALTER": "PREFEITO JOSE WALTER",
    "PANAMERICANO": "PAN-AMERICANO",
    "PRAIA DO FURUTO II": "PRAIA DO FUTURO II",
    "SAO JOAO DO TAUAPE": "TAUAPE",
    "VILA ELLERY": "ELLERY",
    "VILA PERY": "VILA PERI",
}


def load_official_fortaleza_bairros(path: Path | str = OFFICIAL_GEOJSON) -> pd.DataFrame:
    """Carrega os 121 bairros municipais e usa o centróide do polígono como nó."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        name = properties.get("Name")
        geometry = feature.get("geometry")
        if not name or not geometry:
            continue
        # A versão municipal de 2025 traz um ponto de referência por bairro.
        # Rejeitamos outro tipo para não inventar um centróide geométrico.
        if geometry.get("type") != "Point":
            raise ValueError(f"Geometria inesperada para {name}: {geometry.get('type')}")
        longitude, latitude = geometry["coordinates"][:2]
        records.append({"name": bairro_key(name), "official_name": name, "lat": latitude, "long": longitude})

    frame = pd.DataFrame(records).drop_duplicates(subset=["name"])
    if len(frame) != 121:
        raise ValueError(f"Malha oficial inválida: esperados 121 bairros, encontrados {len(frame)} em {source}.")
    return frame.sort_values("name").reset_index(drop=True)


@lru_cache(maxsize=1)
def official_fortaleza_name_set() -> frozenset[str]:
    """Índice imutável, carregado uma vez por processo durante o rebuild."""
    return frozenset(load_official_fortaleza_bairros()["name"])


def canonicalize_fortaleza_bairro(value: object, official_names: set[str] | None = None) -> str | None:
    """Retorna o nome canônico ou ``None`` quando não há equivalência aprovada."""
    normalized = bairro_key(value)
    if not normalized:
        return None
    if official_names is None:
        official_names = official_fortaleza_name_set()
    if normalized in official_names:
        return normalized
    alias = OFFICIAL_ALIASES.get(normalized)
    if alias and bairro_key(alias) in official_names:
        return bairro_key(alias)
    return None
