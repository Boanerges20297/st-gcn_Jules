from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))

from scripts.linux.ask_gemini_with_mempalace import get_query_specific_context

query = "Quais sao os micronodos criticos da Granja Portugal?"
scope_csv = Path("outputs/hermes/risk_fortaleza_latest.csv")
tactical_csv = Path("outputs/hermes/dados_status_enriquecido_14d_latest.csv")
micronodes_csv = Path("outputs/hermes/visible_micronodes.csv")

context = get_query_specific_context(query, scope_csv, tactical_csv, micronodes_csv, "fortaleza")
print("=== QUERY SPECIFIC CONTEXT ===")
print(context)
