import os
import csv
import json
import logging
import requests
from io import StringIO
from datetime import datetime

logger = logging.getLogger(__name__)

def sync_google_sheets(csv_url: str, exogenous_file_path: str):
    """
    Downloads CSV from the provided Google Sheets public URL,
    reads the rows, maps them to exogenous_events, and
    appends new ones to the local JSON file.
    
    Expected Columns (as defined in Apps Script):
    0: id
    1: timestamp (ISO date)
    2: natureza
    3: municipio
    4: bairro
    5: descricao
    ... ignored
    """
    if not csv_url:
        logger.error("No CSV URL provided for Google Sheets sync.")
        return {"status": "error", "message": "Google Sheets CSV URL not configured"}

    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        response.encoding = 'utf-8'  # Google Sheets exports as UTF-8
        
        # Read CSV
        content = response.text
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        
        if len(rows) <= 1:
             return {"status": "success", "imported": 0, "message": "No data rows in spreadsheet (or only header)."}

        # Skip header
        data_rows = rows[1:]
        
        # Load local events to check existing IDs
        existing_events = []
        if os.path.exists(exogenous_file_path):
            with open(exogenous_file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_events = json.load(f)
                except json.JSONDecodeError:
                    existing_events = []
                    
        existing_ids = {ev.get("id") for ev in existing_events if ev.get("id")}
        
        new_imported = 0
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # We only import process_exogenous_text here to avoid circular imports if any
        try:
            from src.llm_service import process_exogenous_text
        except ImportError:
            process_exogenous_text = None
            logger.warning("Could not import process_exogenous_text. Will use fallback logic.")
        
        for r in data_rows:
            # Need at least up to column 6, pad with empty if missing
            row = r + [''] * (7 - len(r))
            
            ev_id = row[0].strip()
            if not ev_id or ev_id in existing_ids:
                continue # Skip empty or already imported
                
            iso_time = row[1].strip()
            natureza = row[2].strip()
            municipio = row[3].strip()
            bairro = row[4].strip()
            descricao = row[5].strip()
            
            # Debug: log what we got from CSV
            logger.info(f"CSV row [{ev_id[:8]}]: natureza='{natureza}' municipio='{municipio}' bairro='{bairro}' descricao='{descricao[:50]}'")
            
            # Use LLM only for severity classification, not for extracting structured fields
            severity = "LOW"
            if process_exogenous_text and descricao:
                try:
                    text_for_severity = f"Natureza: {natureza}\nMunicípio: {municipio}\nBairro: {bairro}\nRelato: {descricao}"
                    parsed_items = process_exogenous_text(text_for_severity)
                    if parsed_items and len(parsed_items) > 0:
                        severity = parsed_items[0].get('conflict_severity', 'LOW')
                        # Only fill empty fields from LLM if CSV had them empty
                        if not bairro:
                            bairro = parsed_items[0].get('bairro', '')
                        if not municipio:
                            municipio = parsed_items[0].get('municipio', '')
                        if not natureza:
                            natureza = parsed_items[0].get('natureza', '') or 'OUTROS'
                except Exception as e:
                    logger.warning(f"Failed to get severity via LLM for id {ev_id}: {e}")
            
            # Try to extract bairro from description if still empty
            if not bairro and descricao:
                import re
                # Pattern: (BAIRRO NAME) — common in CIOPS format
                paren_match = re.search(r'\(([A-ZÀ-Ú\s]{3,})\)', descricao)
                if paren_match:
                    bairro = paren_match.group(1).strip()

            # Format time
            try:
                dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                short_time = dt.strftime("%H:%M")
            except Exception:
                date_str = now_str
                short_time = ""

            new_event = {
                "id": ev_id,
                "bairro": bairro.upper() if bairro else "",
                "conflict_severity": severity,
                "descricao": descricao,
                "is_suppression": False,
                "localizacao_completa": f"{bairro}, {municipio} - {descricao[:40]}..." if bairro and municipio else descricao[:60],
                "municipio": municipio.upper() if municipio else "",
                "natureza": natureza.upper() if natureza else "OUTROS",
                "raw_text": descricao,
                "resumo": f"{natureza.upper()} em {bairro or municipio or 'local'}" if natureza else descricao[:40],
                "sexo": "",
                "timestamp": short_time,
                "ingested_at": now_str,
                "date": date_str,
                "source": "google_sheets_webhook"
            }
            
            existing_events.append(new_event)
            existing_ids.add(ev_id)
            new_imported += 1

        if new_imported > 0:
            with open(exogenous_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_events, f, ensure_ascii=False, indent=2)

        return {"status": "success", "imported": new_imported, "message": f"Successfully imported {new_imported} new events."}

    except Exception as e:
        logger.exception(f"Error syncing from Google Sheets: {str(e)}")
        return {"status": "error", "message": str(e)}
