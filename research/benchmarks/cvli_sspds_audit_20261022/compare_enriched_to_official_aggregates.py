"""Recheck corrected canonical CVLI against SSPDS daily totals (no personal data)."""
from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OFFICIAL = OUT / "fortaleza_2022_2025_pre_fix_daily.csv"
CSV = ROOT / "data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"
def main():
    source = pd.read_csv(CSV, usecols=["cidade", "tipo", "data", "qtd_mortes"], low_memory=False)
    source = source.loc[
        source["cidade"].astype(str).str.strip().str.upper().eq("FORTALEZA")
        & source["tipo"].astype(str).str.strip().str.lower().eq("cvli")
    ].copy()
    source["date"] = pd.to_datetime(source["data"], errors="coerce").dt.strftime("%Y-%m-%d")
    source["month"] = source["date"].str[:7]
    source["death"] = pd.to_numeric(source["qtd_mortes"], errors="coerce").fillna(1)
    byday = source.groupby("date").agg(repaired_cvli=("death", "sum"))
    observed = pd.read_csv(OFFICIAL, dtype={"date": "string"}).set_index("date")
    combined = observed[["official"]].join(byday, how="outer").fillna(0)
    combined = combined.loc[combined.index.str[:4].isin(["2022", "2023", "2024", "2025"])]
    combined["diff"] = combined["repaired_cvli"] - combined["official"]
    combined.reset_index().to_csv(OUT / "fortaleza_2022_2025_post_fix_daily.csv", index=False)
    combined["month"] = combined.index.str[:7]
    monthly = combined.groupby("month")[["official", "repaired_cvli", "diff"]].sum()
    monthly.to_csv(OUT / "fortaleza_2022_2025_post_fix_monthly.csv")
    summary = {}
    for year in ("2022", "2023", "2024", "2025"):
        s = combined.loc[combined.index.str.startswith(year)]
        summary[year] = {"official": int(s.official.sum()),
                         "repaired": int(s.repaired_cvli.sum()),
                         "days_equal": int(s["diff"].eq(0).sum()),
                         "days_total": len(s),
                         "days_disagree": int(s["diff"].ne(0).sum())}
    current = source.loc[source.month.str.startswith("2026")].groupby("month")["death"].sum()
    official2026 = pd.read_csv(OUT / "fortaleza_2026_pre_fix_status_monthly.csv")
    official2026["repaired_cvli"] = official2026["month"].map(current).fillna(0)
    official2026["diff"] = official2026["repaired_cvli"] - official2026["official_status_cvli"]
    official2026.to_csv(OUT / "fortaleza_2026_post_fix_status_monthly.csv", index=False)
    (OUT / "post_fix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == "__main__":
    main()
