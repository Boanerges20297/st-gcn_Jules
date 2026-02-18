'''LLM service for exogenous event extraction and parsing.

This module provides functions to process police log text into structured events
using Gemini LLM with deterministic fallbacks.
'''
import os
import re
import unicodedata
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Try to import Google Generative AI SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logger.warning('google-generativeai SDK not found. LLM features will be disabled.')

def _get_project_root():
    '''Get the absolute path to the project root directory.'''
    # This file is in src/llm_service.py, so root is one level up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _call_model(prompt: str, api_key: str) -> str:
    '''Call the generative model using google-generativeai SDK.'''
    if genai is None:
        raise RuntimeError('google.generativeai SDK not available')

    genai.configure(api_key=api_key)
    # Using flash model for speed and efficiency
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        prompt,
        generation_config={
            'temperature': 0.0,
            'max_output_tokens': 8192
        }
    )
    return getattr(response, 'text', str(response))

def get_gemini_api_keys() -> List[str]:
    '''Return a list of Gemini API keys found in environment variables.'''
    keys = []
    env = os.environ
    if env.get('GEMINI_API_KEYS'):
        keys = [k.strip() for k in env['GEMINI_API_KEYS'].split(',') if k.strip()]
    else:
        for name in ('GEMINI_API_KEY', 'GEMINI_API_KEY_1', 'GEMINI_API_KEY_2', 'GEMINI_API_KEY_3', 'GEMINI_API_KEY_4', 'GOOGLE_API_KEY'): 
            v = env.get(name)
            if v:
                keys.append(v)
    return keys

def _call_model_with_rotation(prompt: str, keys: List[str]) -> str:
    '''Attempt to call the model rotating through provided keys when quota is exhausted.'''
    if not keys:
        raise RuntimeError('No API keys available')
    last_exc = None
    for idx, key in enumerate(keys):
        try:
            return _call_model(prompt, key)
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # Check for quota exhaustion (429)
            if '429' in msg or 'quota' in msg or 'exhausted' in msg:
                logger.warning(f'API key {idx+1}/{len(keys)} exhausted, rotating...')
                continue
            # Re-raise critical errors (403/Auth)
            if '403' in msg or 'permission' in msg:
                raise
            logger.warning(f'API key {idx+1} failed: {msg}')
            continue
    raise last_exc

def _extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    '''Extract JSON from model response text.'''
    if not text:
        return []
    
    # Clean markdown code blocks
    s = text.strip()
    if s.startswith('```'):
        s = re.sub(r'^```(?:json)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
    
    # Try direct parse
    try:
        data = json.loads(s)
        return data if isinstance(data, list) else [data]
    except Exception:
        pass
    
    # Try finding array or object with regex
    m = re.search(r'(\[.*\]|\{.*\})', s, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data if isinstance(data, list) else [data]
        except Exception:
            pass
            
    return []

def _normalize_text(s: str) -> str:
    '''Normalize text for comparison (no accents, uppercase).'''
    if not s:
        return ''
    s = str(s)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9 ]+", ' ', s)
    return s.strip().upper()

_BAIRROS_CACHE = None
def busca_bairro(text: str):
    '''Find a Fortaleza neighborhood in text.'''
    global _BAIRROS_CACHE
    if _BAIRROS_CACHE is None:
        try:
            root = _get_project_root()
            path = os.path.join(root, 'data', 'static', 'fortaleza_bairros_coords.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _BAIRROS_CACHE = { _normalize_text(k): k for k in data.keys() }
            else:
                _BAIRROS_CACHE = {}
        except Exception:
            _BAIRROS_CACHE = {}

    if not text: return None
    t = _normalize_text(text)
    
    # Check if any cached neighborhood name is in the normalized text
    for norm_name, original in _BAIRROS_CACHE.items():
        if norm_name and re.search(r'\b' + re.escape(norm_name) + r'\b', t):
            return original
    return None

_MUNICIPIOS_CACHE = None
def busca_municipio(text: str):
    '''Find a Ceara municipality in text.'''
    global _MUNICIPIOS_CACHE
    if _MUNICIPIOS_CACHE is None:
        try:
            root = _get_project_root()
            path = os.path.join(root, 'data', 'static', 'ceara_municipios_coords.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _MUNICIPIOS_CACHE = { _normalize_text(k): k for k in data.keys() }
            else:
                _MUNICIPIOS_CACHE = {}
        except Exception:
            _MUNICIPIOS_CACHE = {}

    if not text: return None
    
    # If it's a known Fortaleza neighborhood, it's in Fortaleza
    if busca_bairro(text):
        return 'FORTALEZA'
        
    t = _normalize_text(text)
    for norm_name, original in _MUNICIPIOS_CACHE.items():
        if norm_name and re.search(r'\b' + re.escape(norm_name) + r'\b', t):
            return original
    return None

def process_exogenous_text(text: str, block_type: str = None) -> List[Dict[str, Any]]:
    '''Process police log text into structured event data.'''
    if not text or not text.strip():
        return []

    keys = get_gemini_api_keys()
    if not keys or os.environ.get('DISABLE_GENAI_FOR_TESTS') == '1':
        return _deterministic_parse(text)

    prompt = (
        "Você é um especialista em análise de segurança pública no Ceará.\n"
        "Extraia os eventos das linhas de log policial abaixo e retorne um ARRAY JSON.\n"
        "Para cada evento, use estas chaves exatamente:\n"
        "natureza, descricao, sexo, localizacao_completa, bairro, municipio, timestamp, resumo, raw_text, conflict_severity.\n\n"
        "REGRAS:\n"
        "1) 'municipio': Extraia o nome da cidade (ex: FORTALEZA, CAUCAIA, MARACANAU). Se não encontrar, deixe vazio.\n"
        "2) 'bairro': Extraia o bairro. Se for em Fortaleza e o bairro não estiver claro, use o contexto das AIS (ex: AIS18 costuma ser Quintino Cunha/Antônio Bezerra).\n"
        "3) 'conflict_severity': HIGH para homicídios, facções, execuções. MEDIUM para lesões a bala, expulsões. LOW para o resto.\n"
        "4) 'raw_text': Mantenha a linha original completa.\n\n"
        "LOGS:\n" + text
    )

    try:
        out = _call_model_with_rotation(prompt, keys)
        events = _extract_json_from_text(out)
        
        # Enrichment & Normalization
        for ev in events:
            # 1. Ensure nature/text exists
            natureza = ev.get('natureza', 'OCORRENCIA')
            raw = ev.get('raw_text') or ev.get('descricao') or ''
            loc_text = ' '.join([str(ev.get('localizacao_completa', '')), str(ev.get('bairro', '')), raw])
            
            # 2. Fix Bairro & Municipio using local static data
            b = busca_bairro(loc_text)
            if b:
                ev['bairro'] = b
                ev['municipio'] = ev.get('municipio') or 'FORTALEZA'
            else:
                m = busca_municipio(loc_text)
                if m:
                    ev['municipio'] = m
            
            # Default to Fortaleza if bairro was found but city missing
            if ev.get('bairro') and not ev.get('municipio'):
                ev['municipio'] = 'FORTALEZA'
                
            # Final touch: ensure all keys exist
            ev.setdefault('conflict_severity', 'LOW')
            ev.setdefault('timestamp', '')
            
        return events
    except Exception as e:
        logger.error(f"LLM Parse failed: {e}")
        return _deterministic_parse(text)

def _deterministic_parse(text: str) -> List[Dict[str, Any]]:
    '''Basic fallback parser for police logs.'''
    results = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        
        parts = [p.strip() for p in line.split(' - ')]
        # Typical format: ID - AIS - NATUREZA - DESC - LOCAL - BAIRRO - TIME
        natureza = "DESCONHECIDO"
        if len(parts) >= 3:
            natureza = parts[2]
            
        b = busca_bairro(line)
        m = busca_municipio(line) or ('FORTALEZA' if b else '')
        
        results.append({
            'natureza': natureza,
            'descricao': line,
            'sexo': 'MASCULINO' if 'MASCULINO' in line.upper() else ('FEMININO' if 'FEMININO' in line.upper() else ''),
            'localizacao_completa': line,
            'bairro': b or '',
            'municipio': m,
            'timestamp': '',
            'resumo': f"{natureza} em {b or 'local não identificado'}",
            'raw_text': line,
            'conflict_severity': 'HIGH' if 'HOMICIDIO' in line.upper() else 'LOW'
        })
    return results

def _mock_response(text: str) -> List[Dict[str, Any]]:
    return _deterministic_parse(text)
