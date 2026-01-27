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
    prompt = ("""Analise o texto e retorne uma lista JSON de ocorrências seguindo estritamente ESTA ORDEM de extração para cada linha:

- (contador) - IGNORAR
- (codigo da ciops) - IGNORAR
- (sexo da vítima) - extrair se presente (campo: `sexo`)
- (descrição) - extrair como `descricao`
- (localização - bairro,cidade) - extrair como `localizacao_completa`, e também separá-la em `bairro` e `municipio`
- (timestamp) - extrair como `timestamp`

Se algum campo estiver incongruente ou ausente, simplesmente pule-o e continue com os próximos campos. Não tente adivinhar além do texto fornecido. Se `municipio` não for encontrado, use "FORTALEZA" como valor padrão.

Retorne uma lista JSON de objetos com chaves: `natureza`, `descricao`, `sexo`, `localizacao_completa`, `bairro`, `municipio`, `timestamp`, `resumo`, `raw_text`.

Exemplo de entrada: 02 - M20260066103 - LESAO A BALA - VITIMA DO SEXO MASCULINO - DEU ENTRADA NO HOSPITAL MUNICIPAL - SOLEDADE, CAUCAIA (AIS12) - 11:58
Exemplo de saída (JSON): [{"natureza":"LESÃO A BALA", "sexo":"MASCULINO", "descricao":"DEU ENTRADA NO HOSPITAL MUNICIPAL", "localizacao_completa":"SOLEDADE, CAUCAIA (AIS12)", "bairro":"SOLEDADE", "municipio":"CAUCAIA", "timestamp":"11:58", "resumo":"LESÃO A BALA - SOLEDADE, CAUCAIA (AIS12)", "raw_text":"..."}]

Agora processe o texto abaixo e retorne apenas o JSON requisitado:\n\n""") + "\n\n" + text
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
                'descricao': evt.get('descricao', evt.get('description', '')),
                'sexo': evt.get('sexo', evt.get('sex', '')),
                'localizacao_completa': evt.get('localizacao_completa', evt.get('location', '')),
                'bairro': evt.get('bairro', evt.get('neighborhood', '')),
                'municipio': evt.get('municipio', evt.get('city', 'FORTALEZA')),
                'timestamp': evt.get('timestamp', ''),
                'resumo': evt.get('resumo', evt.get('summary', '')),
                'raw': evt.get('raw_text', evt.get('raw', ''))
            })
        return normalized
    except Exception:
        logger.exception('GenAI failed; using mock')
        return _mock_response(text)


def _mock_response(text: str) -> List[Dict[str, Any]]:
    import re
    events = []
    for line in text.strip().split('\n'):
        parts = [p.strip() for p in line.split(' - ')]
        if len(parts) < 2:
            continue
        natureza = parts[2].strip() if len(parts) > 2 else ''
        # detect sexo
        sexo = ''
        for p in parts:
            up = p.upper()
            if 'MASCULIN' in up:
                sexo = 'MASCULINO'
                break
            if 'FEMININ' in up:
                sexo = 'FEMININO'
                break
        # timestamp candidate (last part if looks like time)
        timestamp = ''
        last = parts[-1]
        if re.search(r"\d{1,2}:\d{2}", last):
            timestamp = last
        elif re.match(r"^\d{4}-\d{2}-\d{2}", last):
            timestamp = last
        else:
            # fallback: if last token is short and contains digits
            if len(last) <= 8 and any(ch.isdigit() for ch in last):
                timestamp = last
        # find location part (prefer part containing a comma)
        loc = ''
        for p in parts:
            if ',' in p:
                loc = p
                break
        if not loc:
            if len(parts) >= 6:
                loc = parts[5]
            elif len(parts) >= 5:
                loc = parts[4]
            elif len(parts) >= 4:
                loc = parts[3]
            else:
                loc = 'LOCAL DESCONHECIDO'
        bairro = ''
        municipio = 'FORTALEZA'
        if ',' in loc:
            try:
                b, m = [s.strip() for s in loc.split(',', 1)]
                bairro = b
                municipio = re.sub(r"\s*\(.*\)$", '', m).strip() or municipio
            except Exception:
                pass
        # descricao: try typical position
        descricao = ''
        if len(parts) > 4:
            descricao = parts[4]
        elif len(parts) > 3:
            descricao = parts[3]
        resumo = f"{natureza} - {loc}" if natureza else loc
        events.append({
            'natureza': natureza,
            'descricao': descricao,
            'sexo': sexo,
            'localizacao_completa': loc,
            'bairro': bairro,
            'municipio': municipio,
            'timestamp': timestamp,
            'resumo': resumo,
            'raw': line
        })
    return events
