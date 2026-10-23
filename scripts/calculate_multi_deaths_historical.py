"""Legacy compatibility: source-verified CVLI reconciliation only.
The old spatial GROUP BY and row-count formula overcounted mirrored victims.
The canonical replacement reads dados_status from MySQL using local .env.
By default it is read-only for files; pass --apply for audited backup+replace.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def calculate_qtd_mortes(*, apply=False):
    from scripts.repair_enriched_cvli_from_mysql import main
    old_argv = sys.argv
    try:
        sys.argv = ["repair_enriched_cvli_from_mysql.py"] + (["--apply"] if apply else [])
        return main()
    finally:
        sys.argv = old_argv
if __name__ == "__main__":
    calculate_qtd_mortes(apply="--apply" in sys.argv)
