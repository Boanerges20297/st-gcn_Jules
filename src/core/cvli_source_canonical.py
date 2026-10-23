"""CVLI source aliases: same *victim* represented by multiple MySQL row IDs.
Only strong corroboration from distinct agencies/AIS, identical victim identity,
place and time, and consistent non-agency fields allows consolidation.
No personal data is written by this module.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal, InvalidOperation
import unicodedata
from src.core.occurrence_integrity import normalize_mysql_id
def clean(value):
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in ("", "nan", "none", "nat", "null"):
        return ""
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().upper().strip()
def seconds(value):
    if isinstance(value, timedelta):
        return int(value.total_seconds())
    try:
        h, m, s = str(value).split(":")
        return 3600 * int(h) + 60 * int(m) + int(float(s))
    except (ValueError, TypeError):
        return None
def coord(value):
    try:
        f = Decimal(str(value))
        return f.quantize(Decimal(".00001")) if f.is_finite() else None
    except (TypeError, InvalidOperation):
        return None
def identity_key(row):
    if clean(row.get("tipo")) != "CVLI":
        return None
    name = clean(row.get("nome_vitima") or row.get("vitima"))
    day = str(row.get("data") or "")[:10]
    sec = seconds(row.get("hora"))
    lat, lon = coord(row.get("latitude")), coord(row.get("longitude"))
    city = clean(row.get("cidade"))
    if not (name and city and len(day) == 10 and day[4:5] == "-"
            and sec is not None and lat is not None and lon is not None):
        return None
    return (day, sec, city, lat, lon, name)
def _compatible(rows):
    # If age/sex/nature/weapon truly disagree, preserve distinct identities.
    for key in ("sexo", "idade", "tipo_evento", "arma"):
        values = {clean(row.get(key)) for row in rows}
        values.discard("")
        if len(values) > 1:
            return False
    # Deliberate evidence of two distinct source channels for same victim.
    return (len({clean(row.get("agencia")) for row in rows}) > 1
            or len({clean(row.get("ais")) for row in rows}) > 1)
def source_aliases(rows):
    indexed = defaultdict(list)
    used_ids = set()
    count = Counter()
    for source in rows:
        source_id = normalize_mysql_id(source.get("id"))
        if source_id is None or source_id in used_ids:
            raise ValueError("Missing or repeated MySQL source ID")
        used_ids.add(source_id)
        key = identity_key(source)
        if key is not None:
            indexed[key].append(source)
    alias_groups = []
    for samples in indexed.values():
        if len(samples) < 2:
            continue
        if not _compatible(samples):
            count["ambiguous_groups_preserved"] += 1
            count["ambiguous_rows_preserved"] += len(samples)
            continue
        ids = tuple(sorted((normalize_mysql_id(row["id"]) for row in samples), key=int))
        alias_groups.append(ids)
        count["groups_verified"] += 1
        count["redundant_rows_verified"] += len(ids) - 1
        year = str(samples[0].get("data"))[:4]
        if year.isdecimal():
            count["redundant_" + year] += len(ids) - 1
    return alias_groups, dict(count)

def multi_victim_event_flags(frame):
    """Identify co-located distinct CVLI victim rows after source de-duplication.
    This affects tactical classification only. CVLI target count remains
    one death per surviving victim row (or an explicit legacy quantity).
    """
    import pandas as pd
    result = pd.Series(False, index=frame.index, dtype="bool")
    required = {"tipo", "data", "hora", "cidade", "latitude", "longitude"}
    if not required.issubset(frame.columns):
        return result
    is_cvli = frame["tipo"].astype(str).str.strip().str.lower().eq("cvli")
    subset = frame.loc[is_cvli]
    if subset.empty:
        return result
    day = pd.to_datetime(subset["data"], errors="coerce").dt.strftime("%Y-%m-%d")
    sec = subset["hora"].map(seconds)
    city = subset["cidade"].map(clean)
    lat = pd.to_numeric(subset["latitude"], errors="coerce").round(5)
    lon = pd.to_numeric(subset["longitude"], errors="coerce").round(5)
    valid = day.notna() & sec.notna() & city.ne("") & lat.notna() & lon.notna()
    keys = pd.DataFrame({"day": day, "second": sec, "city": city,
                         "lat": lat, "lon": lon}, index=subset.index).loc[valid]
    if not keys.empty:
        count = keys.groupby(["day", "second", "city", "lat", "lon"])["day"].transform("size")
        result.loc[keys.index] = count.gt(1).to_numpy()
    return result
