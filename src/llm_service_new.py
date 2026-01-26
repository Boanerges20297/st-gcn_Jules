import os
import json
import logging
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import google.genai as genai
except Exception:
    genai = None


def _call_model(prompt: str, api_key: str) -> str:
    if genai is None:
        raise RuntimeError("google.genai not installed")
    if hasattr(genai, 'generate_text'):
        try:
            resp = genai.generate_text(model='models/gemini-1.5', input=prompt)
        except TypeError:
            resp = genai.generate_text(model='models/gemini-1.5', prompt=prompt)
        if hasattr(resp, 'text'):
            return resp.text
        if isinstance(resp, dict) and 'output' in resp:
            return str(resp['output'])
        return str(resp)
    if hasattr(genai, 'ModelsClient'):
        client = genai.ModelsClient()
        resp = client.generate(model='models/gemini-1.5', input=prompt)
        if hasattr(resp, 'text'):
            return resp.text
        if hasattr(resp, 'output'):
            return str(resp.output)
        return str(resp)
    if hasattr(genai, 'Client'):
        client = genai.Client()
        resp = client.generate_text(model='models/gemini-1.5', prompt=prompt)
        if hasattr(resp, 'text'):
            return resp.text
        return str(resp)
    raise RuntimeError('No supported genai entrypoint found')


def _extract_json_from_text(text: str):
    import json, re
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    s = re.sub(r"^```(?:json)?\n", '', s)
    s = re.sub(r"\n```$", '', s)
    m = re.search(r"(\[.*\])", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\{.*\})", s, re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            pass
    raise ValueError('No JSON found')


def process_exogenous_text(text: str) -> List[Dict[str, Any]]:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.warning('GEMINI_API_KEY not set — returning mock parse')
        return _mock_response(text)
    prompt = f"""Analise o texto e retorne uma lista JSON de ocorrências com chaves: natureza, localizacao_completa, bairro, municipio, resumo, raw_text\n\n{text}"""
    try:
        out = _call_model(prompt, api_key)
        # strip fences
        if out.startswith('```'):
            idx = out.find('\n')
            if idx != -1:
                out = out[idx+1:]
        if out.endswith('```'):
            out = out[:-3]
        events = _extract_json_from_text(out)
        normalized = []
        for evt in events:
            if not isinstance(evt, dict):
                continue
            normalized.append({
                'natureza': evt.get('natureza', evt.get('type', 'DESCONHECIDO')),
                'localizacao_completa': evt.get('localizacao_completa', evt.get('location', '')),
                'bairro': evt.get('bairro', evt.get('neighborhood', '')),
                'municipio': evt.get('municipio', 'FORTALEZA'),
                'resumo': evt.get('resumo', evt.get('summary', '')),
                'raw': evt.get('raw_text', evt.get('raw', ''))
            })
        return normalized
    except Exception:
        logger.exception('GenAI failed; using mock')
        return _mock_response(text)


def _mock_response(text: str) -> List[Dict[str, Any]]:
    events = []
    for line in text.strip().split('\n'):
        parts = line.split(' - ')
        if len(parts) >= 3:
            nature = parts[2].strip()
            loc = parts[4].strip() if len(parts) >= 5 else (parts[3].strip() if len(parts) >= 4 else 'LOCAL DESCONHECIDO')
            events.append({'natureza': nature, 'localizacao_completa': loc, 'bairro': '', 'municipio': 'FORTALEZA', 'resumo': f"{nature} - {loc}", 'raw': line})
    return events
