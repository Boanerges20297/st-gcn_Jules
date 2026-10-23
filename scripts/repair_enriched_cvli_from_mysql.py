"""Repair only source-proven CVLI victim duplicates in the enriched CSV.
The MySQL source remains read-only. Never export victim names. Every source
MySQL row ID remains represented in CSV via a legacy-compatible "id | id"
alias group, avoiding repeat imports through data_processing.py.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.cvli_source_canonical import source_aliases
from src.core.occurrence_integrity import collect_mysql_ids, split_mysql_ids, deduplicate_mysql_rows
from scripts.reconcile_mysql_data import read_mysql_snapshot
CSV = ROOT / "data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"
RUNTIME = ROOT / "data/runtime/research/cvli_mysql_integrity_20261022"
FIELDS = ("bairro", "latitude", "longitude", "name", "dia_semana",
          "eh_feriado", "dia_quente_cvp", "clima", "precipitacao_mm",
          "ais_original", "regiao_risp", "area_faccao")
def build_candidate(local, source):
    original = local.copy()
    if "id" not in original or "tipo" not in original:
        raise ValueError("Required identity columns are absent")
    local_ids = defaultdict(list)
    for pos, value in enumerate(original["id"]):
        for source_id in split_mysql_ids(value):
            local_ids[source_id].append(pos)
    if any(len(positions) != 1 for positions in local_ids.values()):
        raise ValueError("Overlapping source IDs in CSV: refusing repair")
    actual_ids = set(local_ids)
    mysql_ids = {str(int(row["id"])) for row in source}
    if actual_ids != mysql_ids:
        raise ValueError("CSV/MySQL identity sets differ; refusing destructive repair")
    groups, findings = source_aliases(source)
    to_remove = set()
    modifications = {}
    retained_by_year = Counter()
    conflicts = 0
    already_grouped = 0
    for aliases in groups:
        positions = [local_ids[alias][0] for alias in aliases]
        if len(set(positions)) == 1:
            already_grouped += 1
            continue
        rows = original.iloc[positions]
        if not rows["tipo"].astype(str).str.strip().str.lower().eq("cvli").all():
            conflicts += 1
            continue
        # Existing grouped aliases may safely absorb newly imported copies
        # ONLY if every represented source ID belongs to this same verified victim.
        if set().union(*(set(split_mysql_ids(original.iloc[i]["id"])) for i in set(positions))) != set(aliases):
            conflicts += 1
            continue
        qty = pd.to_numeric(rows["qtd_mortes"], errors="coerce")
        if (qty.dropna() > 1).any() or (qty.dropna() < 0).any():
            conflicts += 1
            continue
        if (qty.dropna() % 1 != 0).any():
            conflicts += 1
            continue
        # Preserve the record with the most enrichment, preferring explicit death.
        ranked = sorted(positions, key=lambda i: (
            int(pd.to_numeric(original.iloc[i]["qtd_mortes"],errors="coerce")==1),
            sum(pd.notna(original.iloc[i].get(k)) and str(original.iloc[i].get(k)).strip()
                not in ("", "nan") for k in FIELDS),
            -i), reverse=True)
        chosen = ranked[0]
        modifications[chosen] = (aliases, int(rows.iloc[0]["data"][:4])
                                  if isinstance(rows.iloc[0]["data"],str) else 0)
        to_remove.update(i for i in positions if i != chosen)
        retained_by_year[str(modifications[chosen][1])] += len(positions) - 1
    candidate = original.drop(index=original.index[list(to_remove)]).copy()
    candidate["id"] = candidate["id"].astype("string")
    for chosen,(aliases,year) in modifications.items():
        candidate.at[original.index[chosen],"id"] = " | ".join(aliases)
        candidate.at[original.index[chosen],"qtd_mortes"] = 1.0
    candidate = candidate.reset_index(drop=True)
    # MySQL is at victim-row granularity; an unmarked CVLI survivor
    # still represents one death. Never infer deaths from repeated copies.
    is_cvli = candidate["tipo"].astype(str).str.strip().str.lower().eq("cvli")
    qty = pd.to_numeric(candidate.loc[is_cvli, "qtd_mortes"], errors="coerce")
    corrected_qty = qty.where(qty.gt(0), 1.0)
    count_missing = int((qty.isna() | qty.le(0)).sum())
    candidate.loc[is_cvli, "qtd_mortes"] = corrected_qty.to_numpy()
    if collect_mysql_ids(candidate["id"]) != actual_ids:
        raise RuntimeError("MySQL IDs lost in candidate")
    if len(deduplicate_mysql_rows(candidate)) != len(candidate):
        raise RuntimeError("Candidate contains repeated source IDs")
    non_cvli = ~original["tipo"].astype(str).str.lower().str.strip().eq("cvli")
    if len(candidate) < int(non_cvli.sum()):
        raise RuntimeError("Candidate lost non-CVLI rows")
    years = pd.to_datetime(candidate.loc[candidate.tipo.astype(str).str.lower().eq("cvli"),"data"],
                           errors="coerce").dt.year
    death = pd.to_numeric(candidate.loc[candidate.tipo.astype(str).str.lower().eq("cvli"),
                                        "qtd_mortes"],errors="coerce").fillna(0)
    by_year = {str(year):{"rows":int((years==year).sum()),
                          "deaths":int(death[years==year].sum())}
               for year in (2022,2023,2024,2025,2026)}
    report = {"source_rows":len(source),"original_rows":len(original),
      "candidate_rows":len(candidate),"verified_alias_groups":len(modifications),
      "removed_duplicate_representations":len(to_remove),
      "cvli_missing_or_zero_counts_defaulted_to_one":count_missing,
      "already_grouped":already_grouped,"unresolved_group_conflicts":conflicts,
      "ambiguous_groups_preserved":findings.get("ambiguous_groups_preserved",0),
      "removed_by_year":dict(retained_by_year),"cvli_yearly_after":by_year,
      "source_mysql_ids_preserved":len(actual_ids),
      "victim_names_in_output":False,"mysql_readonly":True}
    return candidate,report
def sha(path):
    h=sha256()
    with open(path,"rb") as fp:
        for block in iter(lambda:fp.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true",
                        help="Backup and atomically update the canonical CSV")
    args=parser.parse_args()
    before=sha(CSV)
    original=pd.read_csv(CSV,dtype={"id":"string","id_evento":"string"},low_memory=False)
    source=read_mysql_snapshot()
    candidate,report=build_candidate(original,source)
    if sha(CSV)!=before:
        raise RuntimeError("CSV changed while consulting MySQL")
    RUNTIME.mkdir(parents=True,exist_ok=True)
    report["original_sha256"]=before
    report["mode"]="apply" if args.apply else "dry-run"
    # Research candidate contains only the same non-PII columns as the current CSV.
    proposed=RUNTIME/"candidate.csv"
    candidate.to_csv(proposed,index=False,encoding="utf-8")
    report["candidate_sha256"]=sha(proposed)
    if args.apply:
        if not report["removed_duplicate_representations"]:
            print("Nothing to repair; original unchanged",flush=True)
        else:
            backups=RUNTIME/"backups"
            backups.mkdir(exist_ok=True)
            stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            backup=backups/f"enriched_before_{stamp}_{before[:12]}.csv"
            shutil.copy2(CSV,backup)
            if sha(backup)!=before:
                raise RuntimeError("Backup mismatch")
            if sha(CSV)!=before:
                raise RuntimeError("Original changed before replacement")
            # Stage in source directory: replace is atomic on the same disk.
            pending=None
            try:
                with tempfile.NamedTemporaryFile(mode="wb",dir=CSV.parent,
                     prefix=".cvli_repair_",suffix=".tmp",delete=False) as fp:
                    pending=Path(fp.name)
                    with proposed.open("rb") as input_file:
                        shutil.copyfileobj(input_file,fp)
                    fp.flush()
                    os.fsync(fp.fileno())
                if sha(CSV)!=before or sha(pending)!=report["candidate_sha256"]:
                    raise RuntimeError("Concurrent modification or temp-copy mismatch")
                os.replace(pending,CSV)
                report["backup"] = str(backup.relative_to(ROOT))
                report["official_new_sha256"] = sha(CSV)
                report["official_csv_replaced"] = True
            finally:
                if pending and pending.exists():
                    pending.unlink()
    path=RUNTIME/"report.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":
    main()
