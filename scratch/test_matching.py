import re

def normalize_name(name):
    # simple mimic of normalize_name
    import unicodedata
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return s.upper().replace('-', ' ').strip()

def _extract_city_from_props(props):
    micronodo = str(props.get('micronodo') or '').strip()
    # Split by the last occurrence of a hyphen (with optional surrounding spaces)
    parts = re.split(r'\s*-\s*', micronodo)
    if len(parts) > 1:
        city_raw = parts[-1].split('/')[0].strip()
        if city_raw:
            return city_raw

    for field in ('municipio', 'municipality', 'cidade', 'city'):
        raw = str(props.get(field) or '').strip()
        if raw:
            return raw
    return ''

def _resolve_parent_area(area_raw, micronodo_raw, risk_scores):
    candidates = []
    if area_raw:
        candidates.append(normalize_name(area_raw))

    city_candidate = normalize_name(_extract_city_from_props({'micronodo': micronodo_raw}))
    if city_candidate:
        candidates.append(city_candidate)

    micronodo_norm = normalize_name(micronodo_raw)
    if micronodo_norm:
        candidates.append(micronodo_norm)

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    print("Candidates for matching:", ordered)

    for candidate in ordered:
        if candidate in risk_scores:
            return candidate, risk_scores[candidate]

    for candidate in ordered:
        for known_name, score in risk_scores.items():
            # Use complete word boundary match to avoid false matches like CANINDE -> CANINDEZINHO
            if re.search(r'\b' + re.escape(candidate) + r'\b', known_name) or re.search(r'\b' + re.escape(known_name) + r'\b', candidate):
                return known_name, score

    return (ordered[0] if ordered else normalize_name(area_raw)), 0.0

risk_scores = {
    "CANINDEZINHO": 37.3,
    "CAUCAIA": 42.0,
    "FORTALEZA": 50.0
}

# Test cases
test_cases = [
    ("Alto da Guaramiranga- Canindé", "Alto da Guaramiranga- Canindé"),
    ("Canindezinho - AIS 09", "Canindezinho - AIS 09"),
    ("Conjunto Ceara I", "Conjunto Ceara I")
]

for area, micronodo in test_cases:
    res = _resolve_parent_area(area, micronodo, risk_scores)
    print(f"Area: {area} | Micronodo: {micronodo} => Matched: {res}\n")
