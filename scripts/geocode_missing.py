"""
Script to attempt geocoding of exogenous events when neighborhood/address lookup fails.
Generates a report `data/exogenous_events_geocoded.json` with points added where found.

Usage:
  .venv\Scripts\activate
  python scripts/geocode_missing.py

Note: Enable geocoding in `app.py` by setting `GEOCODING_ENABLED = True` and install `geopy`.
"""
import os
import json
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import app

INPUT = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
OUTPUT = os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json')
REPORT = os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded_report.json')

if not os.path.exists(INPUT):
    print('Input file not found:', INPUT)
    sys.exit(1)

with open(INPUT, 'r', encoding='utf-8') as f:
    batches = json.load(f)

# Geocoding policy (be polite with Nominatim)
MIN_DELAY_SECONDS = 1.0
MAX_RETRIES = 2
TIMEOUT = 10

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    geolocator = Nominatim(user_agent="st-gcn-geocoder", timeout=TIMEOUT)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=MIN_DELAY_SECONDS, max_retries=MAX_RETRIES, error_wait_seconds=2.0)
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

results = []
report = {'total_batches': len(batches), 'geocoded_points': 0, 'details': []}

for i, batch in enumerate(batches):
    points = batch.get('points', [])
    new_points = []
    for evt in batch.get('events', []) if isinstance(batch.get('events'), list) else []:
        # Try heuristics like app.parse_exogenous pipeline would: full location, bairro, municipio
        latlng = None
        reasons = []
        if evt.get('localizacao_completa'):
            res = app.find_node_coordinates(evt['localizacao_completa'])
            if res:
                latlng = res[:2]
                reasons.append(('localizacao_completa', res[2]))
        if latlng is None and evt.get('bairro'):
            res = app.find_node_coordinates(evt['bairro'])
            if res:
                latlng = res[:2]
                reasons.append(('bairro', res[2]))
        if latlng is None and evt.get('municipio'):
            res = app.find_node_coordinates(evt['municipio'])
            if res:
                latlng = res[:2]
                reasons.append(('municipio', res[2]))

        if latlng is None and app.GEOCODING_ENABLED and GEOPY_AVAILABLE:
            # Try rate-limited geocoding (may be slow / rate-limited). Prefer full location then bairro/mun.
            q = evt.get('localizacao_completa') or evt.get('bairro') or evt.get('municipio') or ''
            if q:
                try:
                    loc = geocode(q)
                    if loc:
                        latlng = (loc.latitude, loc.longitude)
                        reasons.append(('geocode', 'nominatim'))
                except Exception as e:
                    report['details'].append({'batch_index': i, 'event_sample': evt, 'geocode_error': str(e)})
        elif latlng is None and app.GEOCODING_ENABLED and not GEOPY_AVAILABLE:
            report['details'].append({'batch_index': i, 'event_sample': evt, 'geocode_error': 'geopy missing'})

        if latlng:
            new_points.append({'lat': latlng[0], 'lng': latlng[1], 'raw_event': evt, 'reasons': reasons})
            report['geocoded_points'] += 1
        else:
            report['details'].append({'batch_index': i, 'event_sample': evt, 'found': False})

    # preserve existing points and append new ones
    batch_out = dict(batch)
    batch_out['points'] = points + new_points
    results.append(batch_out)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(REPORT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print('Wrote:', OUTPUT)
print('Report:', REPORT)
print('Geocoded points:', report['geocoded_points'])
