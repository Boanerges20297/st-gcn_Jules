import os
import re
import unicodedata
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

import google.generativeai as genai

def _call_model(prompt: str, api_key: str) -> str:
    """Call the generative model using google-generativeai SDK.
    Uses the GEMINI_API_KEY from .env file.
    """
    if genai is None:
        raise RuntimeError('google.generativeai SDK not available')

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config={
            'temperature': 0.0,
            'max_output_tokens': 8192
        }
    )
    return getattr(response, 'text', response)


def _call_model_rest(prompt: str, api_key: str, model: str = 'gemini-2.5-flash') -> str:
    """Fallback REST call to Google Generative Language API using an API key.
    Uses urllib from the standard library so no extra dependency is required.
    Returns the model output text or raises on failure.
    """
    try:
        import urllib.request, urllib.parse
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(api_key)}'
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 8192
            }
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode('utf-8')
            j = json.loads(body)
            if isinstance(j, dict) and 'candidates' in j:
                candidates = j['candidates']
                if isinstance(candidates, list) and len(candidates) > 0:
                    candidate = candidates[0]
                    if isinstance(candidate, dict) and 'content' in candidate:
                        content = candidate['content']
                        if isinstance(content, dict) and 'parts' in content:
                            parts = content['parts']
                            if isinstance(parts, list) and len(parts) > 0:
                                part = parts[0]
                                if isinstance(part, dict) and 'text' in part:
                                    return part['text']
            raise RuntimeError('No usable text in REST response')
    except Exception as e:
        logger.exception('REST call to Generative API failed')
        raise


def get_gemini_api_keys() -> List[str]:
    """Return a list of Gemini API keys found in environment variables.
    Supports a comma-separated `GEMINI_API_KEYS` or individual `GEMINI_API_KEY`,
    `GEMINI_API_KEY_1`...`GEMINI_API_KEY_4`, and a fallback `GOOGLE_API_KEY`.
    """
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
    """Attempt to call the model rotating through provided keys when quota is exhausted (HTTP 429).
    Raises the last exception if all keys fail.
    """
    if not keys:
        raise RuntimeError('No API keys available for model call')
    last_exc = None
    for idx, key in enumerate(keys):
        try:
            logger.debug('Trying Gemini API key %d/%d', idx+1, len(keys))
            return _call_model(prompt, key)
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            exhausted = False
            try:
                from google.api_core.exceptions import ResourceExhausted
                if isinstance(e, ResourceExhausted):
                    exhausted = True
            except Exception:
                if '429' in msg or 'resourceexhausted' in msg or 'quota' in msg or 'exceeded' in msg:
                    exhausted = True

            if exhausted:
                logger.warning('API key %d/%d quota exhausted (429). Rotating to next key.', idx+1, len(keys))
                continue

            # For explicit permission errors or leaked-key detection, re-raise so caller can handle.
            if '403' in msg or 'permissiondenied' in msg or 'leaked' in msg:
                logger.error('API key %d/%d failed with permission error: %s', idx+1, len(keys), msg)
                raise

            # Otherwise log and try next key (network/timeouts can be transient)
            logger.warning('API key %d/%d failed with error: %s; trying next key', idx+1, len(keys), msg)
            continue

    logger.exception('All API keys exhausted or failed; last error: %s', last_exc)
    raise last_exc


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
    """Parse exogenous event lines. Use deterministic parsing by default;
    if GEMINI_API_KEY is present, call the model and then enrich results
    using `busca_bairro` and `busca_municipio`.
    """
    def _parse_line(line: str) -> Dict[str, Any]:
        parts = [p.strip() for p in line.split(' - ') if p.strip()]
        if len(parts) < 3:
            return None
        natureza = parts[2]
        sexo = ''
        timestamp = ''
        # detect timestamp at end
        for i in range(len(parts)-1, 1, -1):
            if re.search(r"\d{1,2}:\d{2}", parts[i]):
                timestamp = parts[i]
                parts = parts[:i]
                break
        # drop AIS-like tokens at end
        while parts and re.match(r'^[A-Za-z]{2,}\d+$', parts[-1]):
            parts.pop()
        sexo_idx = None
        for idx, p in enumerate(parts[3:], start=3):
            up = p.upper()
            if 'SEXO' in up or 'VITIMA DO SEXO' in up or up in ('MASCULINO','FEMININO'):
                sexo = 'MASCULINO' if 'MASCUL' in up else ('FEMININO' if 'FEMIN' in up else '')
                sexo_idx = idx
                break
        body = parts[3:sexo_idx] if sexo_idx else parts[3:]
        bairro = ''
        municipio = ''
        localizacao_completa = ''
        if body:
            if len(body) >= 2:
                bairro = body[-2]
                municipio = body[-1]
                localizacao_completa = ', '.join(body[-3:]) if len(body) >= 3 else ', '.join(body)
            else:
                localizacao_completa = body[0]
                if ',' in body[0]:
                    parts_loc = [s.strip() for s in body[0].split(',') if s.strip()]
                    if len(parts_loc) >= 2:
                        bairro = parts_loc[-2]
                        municipio = parts_loc[-1]
        descricao = ''
        if sexo_idx:
            descricao_parts = parts[sexo_idx+1: len(parts)- (3 if len(body)>=3 else len(body))]
            descricao = ' - '.join(descricao_parts).strip()
        else:
            if len(body) > 2:
                descricao = ' - '.join(body[:-2]).strip()
        resumo = f"{natureza} - {localizacao_completa or 'LOCAL DESCONHECIDO'}"
        return {
            'natureza': natureza,
            'descricao': descricao,
            'sexo': sexo,
            'localizacao_completa': localizacao_completa,
            'bairro': bairro,
            'municipio': municipio or '',
            'timestamp': timestamp,
            'resumo': resumo,
            'raw_text': line
        }

    # Gather API keys (support rotating multiple keys)
    keys = get_gemini_api_keys()
    # Allow tests or CI to force-disable GenAI even if keys exist in .env
    if os.environ.get('DISABLE_GENAI_FOR_TESTS') == '1':
        keys = []

    # Deterministic path (no API keys): parse line-by-line and enrich
    if not keys:
        parsed = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            p = _parse_line(line)
            if not p:
                continue
            try:
                loc = p.get('localizacao_completa') or p.get('raw_text') or ''
                b = busca_bairro(loc)
                if b:
                    p['bairro'] = b
                    p['municipio'] = p.get('municipio') or 'FORTALEZA'
                else:
                    m = busca_municipio(loc)
                    if m:
                        p['municipio'] = m
            except Exception:
                pass
            parsed.append(p)
        if parsed:
            return parsed
        return _mock_response(text)

    # LLM path: call model, extract JSON, then enrich
    prompt = (
        "You will receive multiple lines of police log text. For each line return a JSON array where each element has these keys exactly: natureza, descricao, sexo, localizacao_completa, bairro, municipio, timestamp, resumo, raw_text.\n\n"
        "Important rules (follow exactly):\n"
        "1) Extract 'bairro' and 'municipio' when present. Do NOT invent municipality names. If you cannot confidently extract a municipality from the text, return an empty string for 'municipio'.\n"
        "2) If the provided text does not contain a street AND does not contain a bairro (neighborhood) — i.e. address is incomplete or missing both street and neighborhood — then set 'bairro' to the literal string 'CENTRO' of the same city. For 'municipio', extract the city if it appears in the text; if the city is not present, leave 'municipio' empty.\n"
        "3) Keep values concise: use the canonical neighborhood name when available; otherwise use 'CENTRO' as described.\n"
        "4) 'localizacao_completa' should contain whatever location text is available (may be empty).\n\n"
        + text
    )
    try:
        out = _call_model_with_rotation(prompt, keys)
    except Exception as e_call:
        msg = str(e_call)
        if 'leaked' in msg.lower() or 'permissiondenied' in msg.replace(' ', '').lower() or '403' in msg:
            logger.error('LLM API key appears invalid or leaked (403). Rotate the key and remove it from the repository.')
            raise
        logger.exception('GenAI SDK call failed after trying available keys; falling back to local mock parser')
        return _mock_response(text)

    if isinstance(out, str) and out.startswith('```'):
        idx = out.find('\n')
        if idx != -1:
            out = out[idx+1:]
    if isinstance(out, str) and out.endswith('```'):
        out = out[:-3]
    events = _extract_json_from_text(out)
    normalized = []
    for evt in events:
        if not isinstance(evt, dict):
            continue
        natureza = evt.get('natureza', evt.get('type', 'DESCONHECIDO'))
        descricao = evt.get('descricao', evt.get('description', ''))
        sexo = evt.get('sexo', evt.get('sex', ''))
        localizacao_completa = evt.get('localizacao_completa', evt.get('location', ''))
        bairro = evt.get('bairro', evt.get('neighborhood', '')) or ''
        municipio = (evt.get('municipio') or '').strip()
        timestamp = evt.get('timestamp', '')
        resumo = evt.get('resumo', evt.get('summary', ''))
        raw_text = evt.get('raw_text', evt.get('raw', ''))
        try:
            loc_search = ' '.join([str(descricao or ''), str(localizacao_completa or ''), str(raw_text or '')])
            b = busca_bairro(loc_search)
            if b:
                bairro = b
                if not municipio:
                    municipio = 'FORTALEZA'
            else:
                m = busca_municipio(loc_search)
                if m:
                    municipio = m
        except Exception:
            pass
        normalized.append({
            'natureza': natureza,
            'descricao': descricao,
            'sexo': sexo,
            'localizacao_completa': localizacao_completa,
            'bairro': bairro,
            'municipio': municipio or '',
            'timestamp': timestamp,
            'resumo': resumo,
            'raw_text': raw_text
        })
    return normalized


def _normalize_text(s: str) -> str:
    if not s:
        return ''
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9 ]+", ' ', s)
    return s.strip().upper()


_MUNICIPIOS_CACHE = None
def busca_municipio(text: str):
    """Try to find a municipality name contained in `text` using the
    `data/static/ceara_municipios_coords.json` listing. Returns the
    canonical municipality name if found, otherwise None.
    """
    global _MUNICIPIOS_CACHE
    if _MUNICIPIOS_CACHE is None:
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(base, '..'))
            mun_path = os.path.join(project_root, 'data', 'static', 'ceara_municipios_coords.json')
            if os.path.exists(mun_path):
                with open(mun_path, 'r', encoding='utf-8') as fh:
                    j = json.load(fh)
                    # map normalized -> original
                    _MUNICIPIOS_CACHE = { _normalize_text(k): k for k in j.keys() }
            else:
                _MUNICIPIOS_CACHE = {}
        except Exception:
            _MUNICIPIOS_CACHE = {}

    if not text:
        return None
    # If the text contains a known Fortaleza bairro, do NOT treat it as a municipality.
    try:
        b = busca_bairro(text)
        if b:
            return None
    except Exception:
        pass
    t = text
    # remove AIS tokens and surrounding parentheses
    t = re.sub(r"\(AIS\d+\)", ' ', t, flags=re.IGNORECASE)
    t = re.sub(r"\bAIS\d+\b", ' ', t, flags=re.IGNORECASE)
    t = re.sub(r"[()\.]", ' ', t)
    t = re.sub(r"\s+", ' ', t).strip()

    norm_full = _normalize_text(t)
    if norm_full in _MUNICIPIOS_CACHE:
        return _MUNICIPIOS_CACHE[norm_full]

    # try parts split by comma or dash
    parts = [p.strip() for p in re.split(r'[,-]', t) if p.strip()]
    for p in reversed(parts):
        np = _normalize_text(p)
        if np in _MUNICIPIOS_CACHE:
            return _MUNICIPIOS_CACHE[np]

    # try suffix n-grams (last tokens)
    tokens = norm_full.split()
    for i in range(len(tokens)):
        cand = ' '.join(tokens[i:])
        if cand in _MUNICIPIOS_CACHE:
            return _MUNICIPIOS_CACHE[cand]

    # substring/word match: check if any municipality name appears as a whole token sequence
    for nm, orig in _MUNICIPIOS_CACHE.items():
        if not nm:
            continue
        if re.search(r'\b' + re.escape(nm) + r'\b', norm_full):
            return orig

    return None


_BAIRROS_CACHE = None
def busca_bairro(text: str):
    """Try to find a Fortaleza bairro in `text` using
    `data/static/fortaleza_bairros_coords.json`. Returns the canonical
    bairro name if found, otherwise None.
    """
    global _BAIRROS_CACHE
    if _BAIRROS_CACHE is None:
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(base, '..'))
            b_path = os.path.join(project_root, 'data', 'static', 'fortaleza_bairros_coords.json')
            if os.path.exists(b_path):
                with open(b_path, 'r', encoding='utf-8') as fh:
                    jb = json.load(fh)
                    _BAIRROS_CACHE = { _normalize_text(k): k for k in jb.keys() }
            else:
                _BAIRROS_CACHE = {}
        except Exception:
            _BAIRROS_CACHE = {}

    if not text:
        return None
    t = re.sub(r"\(AIS\d+\)", ' ', text, flags=re.IGNORECASE)
    t = re.sub(r"\bAIS\d+\b", ' ', t, flags=re.IGNORECASE)
    t = re.sub(r"[()\.]", ' ', t)
    t = re.sub(r"\s+", ' ', t).strip()
    norm = _normalize_text(t)
    # exact or token matches
    if norm in _BAIRROS_CACHE:
        return _BAIRROS_CACHE[norm]
    parts = [p.strip() for p in re.split(r'[,-]', t) if p.strip()]
    for p in reversed(parts):
        np = _normalize_text(p)
        if np in _BAIRROS_CACHE:
            return _BAIRROS_CACHE[np]
    # word-boundary substring match
    for nm, orig in _BAIRROS_CACHE.items():
        if nm and re.search(r'\b' + re.escape(nm) + r'\b', norm):
            return orig
    return None

    if api_key:
        # Build a stricter prompt guiding the model to extract municipality from the location or description.
        prompt = (
            f"""You will receive multiple lines of police log text. For each line return a JSON object with these keys exactly: natureza, descricao, sexo, localizacao_completa, bairro, municipio, timestamp, resumo, raw_text.

Important rules:
- The municipio must be one of the following municipality names (case-insensitive) if it appears in the text. If none of these names appear, set municipio to an empty string. Do NOT invent municipality names.
- If the municipality appears in the description or location text, the model MUST extract it exactly as it appears in the provided list.
- The bairro should be extracted if present; if unknown, return an empty string.
- If you are not able to confidently extract municipio from the text, return an empty string for municipio rather than guessing.
- Output MUST be valid JSON array (e.g. [{...}, {...}]) and nothing else.

Municipalities sample: {', '.join(municipios_sample[:60])}
Neighborhoods sample: {', '.join(bairros_sample[:60])}

Now parse the following lines and return only JSON (no commentary):
"""
        ) + "\n\n" + text
    else:
        # No API key: fallback to deterministic parser
        parsed = []
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            p = _parse_line(line)
            if p:
                parsed.append(p)
        if parsed:
            # Enrich deterministic parses with bairro/municipio lookups
            for ev in parsed:
                try:
                    # prefer explicit localizacao_completa, then raw_text
                    loc = ev.get('localizacao_completa') or ev.get('raw_text') or ''
                    b = busca_bairro(loc)
                    if b:
                        ev['bairro'] = b
                        # when bairro is found inside Fortaleza, set municipio to Fortaleza
                        ev['municipio'] = ev.get('municipio') or 'FORTALEZA'
                    else:
                        m = busca_municipio(loc)
                        if m:
                            ev['municipio'] = m
                except Exception:
                    pass
            return parsed
        logger.warning('GEMINI_API_KEY not set and deterministic parser failed — returning mock parse')
        return _mock_response(text)

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
        # post-validate municipality extraction: if model missed a municipality
        muni_lower_set = set([m.lower() for m in municipios_sample])
        for evt in events:
            if not isinstance(evt, dict):
                continue
            natureza = evt.get('natureza', evt.get('type', 'DESCONHECIDO'))
            descricao = evt.get('descricao', evt.get('description', ''))
            sexo = evt.get('sexo', evt.get('sex', ''))
            localizacao_completa = evt.get('localizacao_completa', evt.get('location', ''))
            bairro = evt.get('bairro', evt.get('neighborhood', '')) or ''
            municipio = (evt.get('municipio') or '').strip()
            timestamp = evt.get('timestamp', '')
            resumo = evt.get('resumo', evt.get('summary', ''))
            raw_text = evt.get('raw_text', evt.get('raw', ''))

            # Enrich LLM parse with bairro/municipio lookups when missing or ambiguous
            try:
                loc_search = ' '.join([str(descricao or ''), str(localizacao_completa or ''), str(raw_text or '')])
                # try bairro first
                b = busca_bairro(loc_search)
                if b:
                    bairro = b
                    if not municipio:
                        municipio = 'FORTALEZA'
                else:
                    # try municipio
                    m = busca_municipio(loc_search)
                    if m:
                        municipio = m
            except Exception:
                pass
            normalized.append({
                'natureza': natureza,
                'descricao': descricao,
                'sexo': sexo,
                'localizacao_completa': localizacao_completa,
                'bairro': bairro,
                'municipio': municipio or '',
                'timestamp': timestamp,
                'resumo': resumo,
                'raw_text': raw_text
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
