import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_service import process_exogenous_text, busca_bairro, busca_municipio

sample_text = """EXPULSAO DE MORADORES - [19:25, 24/05/2026] Ten Braga Raio: Durante a tarde a VTR da supervisão (COPAC 90) foi acionada via Ciops informando que no local se encontravam diversas famílias que haviam sido expulsas por membros de facções criminosas meses atrás e que, em decorrência da prisão de alguns líderes da organização criminosa da área, queriam retornar para suas casas. Ao chegar no local, alguns dos indivíduos começaram a discutir, vindo a vias de fatos que nesse momento houve a intervenção dos policiais do POG, COPAC E RAIO . As partes foram conduzidas para o 13° DP para serem ouvidas  que ficou registrado um Boletim de ocorrência de N 113-4194 /2026.   Escrivão José Valdésio  Rodrigues viana , posteriormente as partes foram liberadas .
[19:25, 24/05/2026] Ten Braga Raio: Aproximadamente 50 populares querendo retornar as suas supostas casas , alguns expulsos há mais de 4 anos , outros há 30 dias. Populares deslocados anteriormente por supostamente terem mudado para ORCRIM CV , a comunidade prevalece ORCRIM MASSA, a tentativa de retorno se dá por prisões de alguns indivíduos da ORCRIM MASSA. - BARROSO - FORTALEZA - 14:09"""

print("=== TESTE DE ENRIQUECIMENTO DIRETO ===")
b = busca_bairro(sample_text)
m = busca_municipio(sample_text)
print(f"Bairro detectado: {b}")
print(f"Município detectado: {m}")

print("\n=== PARSE DO LLM SERVICE (FALLBACK DETERMINISTICO) ===")
parsed = process_exogenous_text(sample_text)
print(f"Número de itens retornados: {len(parsed)}")

# --- SIMULAÇÃO DA NOVA LÓGICA DE EXTRAÇÃO ---
import re

descricao = sample_text
ev_id = "d0c03729-af89-445a-8e51-b109221873fb"
short_time = "17:10"
now_str = "2026-05-27 09:26:36"
date_str = "2026-05-25 17:10:17"

# 1. Extração inicial via parsed ou fallbacks
natureza, municipio, bairro, severity = "OUTROS", "", "", "LOW"
if parsed:
    p = parsed[0]
    severity = p.get('conflict_severity', 'LOW')
    natureza = p.get('natureza', '').upper() or "OUTROS"
    municipio = p.get('municipio', '').upper()
    bairro = p.get('bairro', '').upper()

# 2. Geo-varredura se vier vazio
if not bairro and busca_bairro:
    bairro = (busca_bairro(descricao) or "").upper()
if not municipio and busca_municipio:
    municipio = (busca_municipio(descricao) or "").upper()
if bairro and not municipio:
    municipio = "FORTALEZA"

# 3. Natureza se vier DESCONHECIDO/OUTROS
if natureza in ("DESCONHECIDO", "OUTROS", ""):
    desc_upper = descricao.upper()
    if "EXPULSAO DE MORADORES" in desc_upper or "EXPULSÃO DE MORADORES" in desc_upper or "EXPULSAO" in desc_upper:
        natureza = "EXPULSÃO DE MORADORES"
    elif "DESLOCAMENTO FORCADO" in desc_upper or "DESLOCAMENTO FORÇADO" in desc_upper:
        natureza = "DESLOCAMENTO FORÇADO"
    elif "HOMICIDIO" in desc_upper or "HOMICÍDIO" in desc_upper:
        natureza = "HOMICÍDIO"
    elif "LESAO A BALA" in desc_upper or "LESÃO A BALA" in desc_upper:
        natureza = "LESÃO A BALA"
    elif "ACHADO DE CADAVER" in desc_upper or "ACHADO DE CADÁVER" in desc_upper:
        natureza = "ACHADO DE CADÁVER"

# Se ainda assim não detectou, tenta extrair o prefixo
if natureza in ("DESCONHECIDO", "OUTROS", ""):
    match_nature = re.match(r'^([A-ZÀ-Úa-zà-ú\s]+)\s*-\s*', descricao)
    if match_nature:
        candidate = match_nature.group(1).strip()
        if len(candidate) > 3 and len(candidate) < 40:
            natureza = candidate.upper()

# Atualização de severidade baseado na natureza real
if natureza in ("HOMICÍDIO", "LESÃO A BALA"):
    severity = "HIGH"
elif natureza in ("EXPULSÃO DE MORADORES", "DESLOCAMENTO FORÇADO"):
    severity = "MEDIUM"

# 4. Limpeza de Metadados do WhatsApp
clean_text = re.sub(r'\[\d{2}:\d{2},\s+\d{2}/\d{2}/\d{4}\]\s*[^:]+:\s*', '', descricao)
# Remover também localização do rodapé (ex: - BARROSO - FORTALEZA - 14:09)
clean_text = re.sub(r'\s*-\s*[A-ZÀ-Úa-zà-ú\s]+\s*-\s*[A-ZÀ-Úa-zà-ú\s]+\s*-\s*\d{2}:\d{2}\s*$', '', clean_text)
clean_text = clean_text.strip()

# Se clean_text começar com a natureza, remove o prefixo duplicado para localizacao_completa
def normalize_for_check(text):
    if not text: return ""
    import unicodedata
    return "".join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')

norm_clean = normalize_for_check(clean_text)
norm_nature = normalize_for_check(natureza)
if norm_nature and norm_clean.startswith(norm_nature):
    clean_text = clean_text[len(norm_nature):].strip()
    clean_text = re.sub(r'^[-\s:]+', '', clean_text)
clean_text = clean_text.strip()

# 5. Formatação do Resumo e Localização Completa
b_display = bairro.title() if bairro else "local não identificado"
m_display = municipio.title() if municipio else ""
resumo = f"{natureza} em {b_display}"

localizacao_completa = clean_text[:120]
if not localizacao_completa:
    localizacao_completa = descricao[:120]

new_event = {
    "id": ev_id,
    "bairro": bairro,
    "conflict_severity": severity,
    "descricao": descricao,
    "is_suppression": False,
    "localizacao_completa": localizacao_completa,
    "municipio": municipio,
    "natureza": natureza,
    "raw_text": descricao,
    "resumo": resumo,
    "sexo": "",
    "timestamp": short_time,
    "ingested_at": now_str,
    "date": date_str,
    "source": "google_sheets_webhook"
}

print("\n=== NOVO EVENTO EXTRAÍDO E FORMATADO ===")
import json
print(json.dumps(new_event, ensure_ascii=False, indent=2))

