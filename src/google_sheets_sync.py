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
            from src.llm_service import process_exogenous_text
        except ImportError:
            process_exogenous_text = None
        
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
                
            # 4. LLM Enrichment
            natureza, municipio, bairro, severity = "OUTROS", "", "", "LOW"
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

            new_event = {
                "id": ev_id,
                "bairro": bairro,
                "conflict_severity": severity,
                "descricao": descricao,
                "is_suppression": False,
                "localizacao_completa": descricao[:120],
                "municipio": municipio,
                "natureza": natureza,
                "raw_text": descricao,
                "resumo": descricao[:80],
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
