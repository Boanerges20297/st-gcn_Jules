
import os
import sys
import json

# Mocking parts of llm_service
import unicodedata
import re

def _normalize_text(s: str) -> str:
    if not s:
        return ''
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9 ]+", ' ', s)
    return s.strip().upper()

_BAIRROS_CACHE = None
def busca_bairro(text: str):
    global _BAIRROS_CACHE
    if _BAIRROS_CACHE is None:
        b_path = 'data/static/fortaleza_bairros_coords.json'
        if os.path.exists(b_path):
            with open(b_path, 'r', encoding='utf-8') as fh:
                jb = json.load(fh)
                _BAIRROS_CACHE = { _normalize_text(k): k for k in jb.keys() }
        else:
            _BAIRROS_CACHE = {}

    if not text:
        return None
    t = re.sub(r"\(AIS\d+\)", ' ', text, flags=re.IGNORECASE)
    t = re.sub(r"\bAIS\d+\b", ' ', t, flags=re.IGNORECASE)
    t = re.sub(r"[()\.]", ' ', t)
    t = re.sub(r"\s+", ' ', t).strip()
    norm = _normalize_text(t)
    
    if norm in _BAIRROS_CACHE:
        return _BAIRROS_CACHE[norm]
    
    parts = [p.strip() for p in re.split(r'[,-]', t) if p.strip()]
    for p in reversed(parts):
        np = _normalize_text(p)
        if np in _BAIRROS_CACHE:
            return _BAIRROS_CACHE[np]
            
    for nm, orig in _BAIRROS_CACHE.items():
        if nm and re.search(r'\b' + re.escape(nm) + r'\b', norm):
            return orig
    return None

raw_text = "02 - M20260125399 - HOMICIDIO A BALA - VÍTIMA DO SEXO MASCULINO ( NO MESMO LOCAL UMA MULHER LESIONADA NO BRAÇO ) - VIA PÚBLICA - QUINTINO CUNHA - AIS18 - 15:40 ( ACUSADO PRESO PELA DHPP )"
descricao = "VÍTIMA DO SEXO MASCULINO ( NO MESMO LOCAL UMA MULHER LESIONADA NO BRAÇO )"
localizacao_completa = "VIA PÚBLICA - QUINTINO CUNHA"

loc_search = ' '.join([str(descricao or ''), str(localizacao_completa or ''), str(raw_text or '')])

b = busca_bairro(loc_search)
print(f"Busca Bairro result: {b}")
