import os
import csv
import json
import logging
import requests
import re
from io import StringIO
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def normalize_name(name):
    if not name: return ""
    import unicodedata
    name = str(name).lower().strip()
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]', '', name)

def sync_google_sheets(csv_url: str, exogenous_file_path: str):
    """
    Downloads CSV from Google Sheets and appends NEW events within the 7-day window.
    """
    if not csv_url:
        logger.error("No CSV URL provided for Google Sheets sync.")
        return {"status": "error", "message": "Google Sheets CSV URL not configured"}

    try:
        response = requests.get(csv_url, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        content = response.text
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        
        if len(rows) <= 1:
             return {"status": "success", "imported": 0, "message": "No data rows in spreadsheet."}

        data_rows = rows[1:]
        
        # Load local events
        existing_events = []
        if os.path.exists(exogenous_file_path):
            try:
                with open(exogenous_file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read().strip()
                    if file_text:
                        existing_events = json.loads(file_text)
            except Exception as e:
                logger.error(f"Failed to load existing events: {e}")
        
        # Cutoff: 7 days (prediction window)
        now = datetime.now()
        cutoff_date = (now - timedelta(days=7)).date()
        
        # Track existing for de-duplication
        existing_ids = {str(ev.get("id")).strip() for ev in existing_events if ev.get("id")}
        existing_texts = {normalize_name(ev.get("raw_text", "")) for ev in existing_events if ev.get("raw_text")}
        
        # Also check archives to avoid re-importing what was already archived
        archives_dir = os.path.join(os.path.dirname(exogenous_file_path), 'archives')
        if os.path.exists(archives_dir):
            for arch_file in os.listdir(archives_dir):
                if arch_file.endswith('.json'):
                    try:
                        with open(os.path.join(archives_dir, arch_file), 'r', encoding='utf-8') as af:
                            arch_data = json.load(af)
                            for aev in arch_data:
                                eid = str(aev.get("id")).strip()
                                if eid: existing_ids.add(eid)
                                txt = normalize_name(aev.get("raw_text", ""))
                                if txt: existing_texts.add(txt)
                    except: pass

        new_imported = 0
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            from src.llm_service import process_exogenous_text, busca_bairro, busca_municipio
        except ImportError:
            process_exogenous_text = None
            busca_bairro = None
            busca_municipio = None
        
        for r in data_rows:
            row = r + [''] * (7 - len(r))
            ev_id = str(row[0]).strip()
            iso_time = row[1].strip()
            descricao = row[5].strip() or ' - '.join(c.strip() for c in row[2:6] if c.strip())
            
            # 1. Parse date for cutoff check
            event_date_obj = None
            try:
                dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
                event_date_obj = dt.date()
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                short_time = dt.strftime("%H:%M")
            except Exception:
                # If no date in row, use now (treat as recent)
                event_date_obj = now.date()
                date_str = now_str
                short_time = ""

            # 2. Cutoff check: Ignore older than 7 days
            if event_date_obj < cutoff_date:
                continue

            # 3. Duplication check
            desc_norm = normalize_name(descricao)
            if (ev_id and ev_id in existing_ids) or (desc_norm and desc_norm in existing_texts):
                continue
                
            # 4. LLM Enrichment & Robust Fallback
            natureza, municipio, bairro, severity = "OUTROS", "", "", "LOW"
            parsed_items = None
            if process_exogenous_text and descricao:
                try:
                    parsed_items = process_exogenous_text(descricao)
                    if parsed_items:
                        p = parsed_items[0]
                        severity = p.get('conflict_severity', 'LOW')
                        natureza = p.get('natureza', '').upper() or "OUTROS"
                        municipio = p.get('municipio', '').upper()
                        bairro = p.get('bairro', '').upper()
                except Exception as e:
                    logger.warning(f"LLM parse failed for id {ev_id}: {e}")

            # 4.1 Geo-varredura determinística de fallback se vier vazio (processa o texto completo)
            if not bairro and busca_bairro:
                bairro = (busca_bairro(descricao) or "").upper()
            if not municipio and busca_municipio:
                municipio = (busca_municipio(descricao) or "").upper()
            if bairro and not municipio:
                municipio = "FORTALEZA"

            # 4.2 Restauração de Natureza real se vier DESCONHECIDO/OUTROS
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

            # Se ainda assim não detectou, tenta extrair o prefixo de cabeçalho
            if natureza in ("DESCONHECIDO", "OUTROS", ""):
                match_nature = re.match(r'^([A-ZÀ-Úa-zà-ú\s]+)\s*-\s*', descricao)
                if match_nature:
                    candidate = match_nature.group(1).strip()
                    if len(candidate) > 3 and len(candidate) < 40:
                        natureza = candidate.upper()

            # Forçar atualização de severidade baseada na natureza corrigida
            if natureza in ("HOMICÍDIO", "LESÃO A BALA"):
                severity = "HIGH"
            elif natureza in ("EXPULSÃO DE MORADORES", "DESLOCAMENTO FORÇADO"):
                severity = "MEDIUM"

            # 4.3 Limpeza de Metadados de WhatsApp e Horários
            # Remover padrão do WhatsApp: [HH:MM, DD/MM/YYYY] Remetente:
            clean_text = re.sub(r'\[\d{2}:\d{2},\s+\d{2}/\d{2}/\d{4}\]\s*[^:]+:\s*', '', descricao)
            # Remover metadados do rodapé da mensagem (ex: - BARROSO - FORTALEZA - 14:09)
            clean_text = re.sub(r'\s*-\s*[A-ZÀ-Úa-zà-ú\s]+\s*-\s*[A-ZÀ-Úa-zà-ú\s]+\s*-\s*\d{2}:\d{2}\s*$', '', clean_text)
            clean_text = clean_text.strip()

            # Se clean_text começar com a natureza (mesmo com acentos diferentes), remove para evitar duplicação no dashboard
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

            # 4.4 Formatação do Resumo Premium e Localização Completa
            b_display = bairro.title() if bairro else "local não identificado"
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
            
            # --- 5. ENRIQUECIMENTO DE DADOS (NOVA LÓGICA) ---
            try:
                from src.enrichment import enrich_event
                new_event = enrich_event(new_event, base_dir=os.path.dirname(os.path.dirname(exogenous_file_path)))
            except Exception as e:
                logger.warning(f"Failed to enrich event {ev_id}: {e}")
            # -----------------------------------------------

            existing_events.append(new_event)
            if ev_id: existing_ids.add(ev_id)
            if desc_norm: existing_texts.add(desc_norm)
            new_imported += 1

        if new_imported > 0:
            existing_events.sort(key=lambda x: x.get('date', ''), reverse=False)
            with open(exogenous_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_events, f, ensure_ascii=False, indent=2)

        return {"status": "success", "imported": new_imported, "message": f"Successfully imported {new_imported} new events."}
    except Exception as e:
        logger.exception(f"Error syncing from Google Sheets: {str(e)}")
        return {"status": "error", "message": str(e)}
