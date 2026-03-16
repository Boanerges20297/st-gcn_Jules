"""
Gera o arquivo data/streets_by_municipio.json com as localidades/ruas críticas
de cada município fora de Fortaleza, combinando três fontes:

  1. Campo 'bairro' do CSV oficial (imediato, sem API)
  2. Reverse geocoding via Nominatim para registros sem bairro mas com lat/lng
     (incremental — salva progresso a cada lote; respeita rate limit do Nominatim
      com backoff exponencial ao receber 429)
  3. Campo 'bairro' + texto dos eventos exógenos

Execução:
  python scripts/gerar_streets_municipios.py              # últimos 30 dias, com geocoding
  python scripts/gerar_streets_municipios.py --fast       # só bairros + exógenos, sem geocoding
  python scripts/gerar_streets_municipios.py --days 60    # ampliar janela temporal
"""

import os
import re
import sys
import json
import time
import unicodedata
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH        = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
EXO_PATH        = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
OUTPUT_PATH     = os.path.join(BASE_DIR, 'data', 'streets_by_municipio.json')
GEO_CACHE_PATH  = os.path.join(BASE_DIR, 'data', 'geo_reverse_cache_municipios.json')

INVALID_TERMS = ['HOMICIDIO', 'BALA', 'FOGO', 'LESAO', 'MORTE', 'CADAVER',
                 'LATROCINIO', 'TIRO', 'EXECUCAO', 'ACHADO', 'CVLI', 'CVP',
                 'PASSAGEM', 'VIOLENCIA', 'ESFAQUEAMENTO']

# Bairros exclusivos de Fortaleza que não devem aparecer em outros municípios.
# Erros de atribuição no CSV (ex: bairro de FOR registrado com cidade=Caucaia).
FORTALEZA_ONLY = {
    'CONJUNTO CEARA', 'CONJUNTO CEARA I', 'CONJUNTO CEARA II', 'CONJUNTO CEARA III',
    'PRAIA DE IRACEMA', 'MEIRELES', 'ALDEOTA', 'VARJOTA', 'COCÓ', 'COCO',
    'AGUA FRIA', 'PARANGABA', 'PARQUELÂNDIA', 'PARQUELANDIA', 'FARIAS BRITO',
    'MONTESE', 'DAMAS', 'BENFICA', 'FÁTIMA', 'FATIMA', 'JOSE BONIFÁCIO',
    'JOSE BONIFACIO', 'DIONÍSIO TORRES', 'DIONISIO TORRES',
}

def normalize(s: str) -> str:
    if not s: return ''
    s = unicodedata.normalize('NFKD', str(s).upper())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^A-Z0-9 ]+', ' ', s).strip()


def _load_reverse_cache() -> dict:
    if os.path.exists(GEO_CACHE_PATH):
        try:
            with open(GEO_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_reverse_cache(cache: dict):
    tmp = GEO_CACHE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, GEO_CACHE_PATH)


def _reverse_batch(clusters: list, cache: dict, save_interval: int = 25) -> dict:
    """
    Faz reverse geocoding de uma lista de (key, lat, lng) usando Nominatim.
    - Delay mínimo de 2.5s entre requests (respeita Usage Policy Nominatim)
    - Backoff exponencial em caso de 429: espera 60s → 120s → 240s
    - Persiste progresso a cada save_interval requisições
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderRateLimited, GeocoderTimedOut, GeocoderServiceError
    except ImportError:
        print("⚠️  geopy não encontrado. Instale: pip install geopy")
        return cache

    MIN_DELAY    = 2.5   # segundos entre requests normais
    MAX_RETRIES  = 5     # tentativas por cluster
    BACKOFF_BASE = 60    # segundos de espera inicial ao receber 429

    geolocator = Nominatim(user_agent="report_preview_municipios_v2", timeout=12)

    new_requests = [(k, lat, lng) for k, lat, lng in clusters if k not in cache]
    total = len(new_requests)
    print(f"   🌐 {total} clusters a geocodificar (já em cache: {len(clusters) - total})...")

    def _do_reverse(lat, lng):
        """Chama reverse com retry+backoff manual para 429."""
        wait = BACKOFF_BASE
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(MIN_DELAY)
                return geolocator.reverse(f"{lat}, {lng}", language='pt')
            except GeocoderRateLimited:
                print(f"   ⏳ Rate limit 429 — aguardando {wait}s antes de tentar novamente...")
                time.sleep(wait)
                wait = min(wait * 2, 300)  # backoff exponencial, máx 5min
            except GeocoderTimedOut:
                print(f"   ⚠️  Timeout — tentativa {attempt+1}/{MAX_RETRIES}")
                time.sleep(10)
            except GeocoderServiceError as e:
                print(f"   ⚠️  Erro de serviço: {e}")
                time.sleep(30)
            except Exception as e:
                print(f"   ⚠️  Erro inesperado: {e}")
                return None
        return None

    for i, (key, lat, lng) in enumerate(new_requests):
        location = _do_reverse(lat, lng)
        if location:
            addr = location.raw.get('address', {})
            street = (addr.get('road') or addr.get('pedestrian') or
                      addr.get('footway') or addr.get('residential') or
                      addr.get('suburb') or '')
            cache[key] = street.upper().strip() if street else ''
        else:
            cache[key] = ''

        if (i + 1) % save_interval == 0:
            _save_reverse_cache(cache)
            pct = (i + 1) / total * 100
            remaining = total - (i + 1)
            eta_min = remaining * MIN_DELAY / 60
            print(f"   💾 Progresso salvo: {i + 1}/{total} ({pct:.0f}%) — ETA ~{eta_min:.0f}min")

    _save_reverse_cache(cache)
    return cache


def build(fast_mode: bool = False, days: int = 30):
    print("=" * 60)
    print(f"🏙️  GERAÇÃO DE LOCALIDADES CRÍTICAS POR MUNICÍPIO")
    print(f"   Modo: {'rápido (sem geocoding)' if fast_mode else 'completo (com geocoding)'}")
    print(f"   Janela: últimos {days} dias")
    print(f"   Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    mun_locs: dict = defaultdict(Counter)   # peso ponderado (para desempate)
    mun_cvli:  dict = defaultdict(Counter)   # contagem bruta de CVLI

    cutoff_date = datetime.now() - timedelta(days=days)

    # ── FONTE 1: Campo 'bairro' do CSV oficial ──────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV não encontrado: {CSV_PATH}")
    else:
        print(f"📖 Lendo CSV oficial (últimos {days} dias)...")
        df = pd.read_csv(
            CSV_PATH,
            usecols=['cidade', 'bairro', 'tipo', 'latitude', 'longitude', 'data'],
            low_memory=False
        )

        # Filtrar últimos N dias
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        df = df[df['data'] >= cutoff_date]
        print(f"   📅 Registros após filtro de {days} dias: {len(df):,}")

        df_nf = df[
            df['cidade'].notna() &
            ~df['cidade'].str.upper().str.contains('FORTALEZA', na=True) &
            df['latitude'].notna() & df['longitude'].notna()
        ].copy()

        # 1a. Registros com bairro: contagem direta
        df_with_bairro = df_nf[df_nf['bairro'].notna() & (df_nf['bairro'].str.len() > 2)]
        for _, rw in df_with_bairro.iterrows():
            ck = normalize(rw['cidade'])
            bairro = normalize(str(rw['bairro']))
            if bairro and not any(t in bairro for t in INVALID_TERMS) \
                    and bairro not in FORTALEZA_ONLY:
                is_cvli = str(rw.get('tipo', '')).lower() == 'cvli'
                peso = 3 if is_cvli else 1
                mun_locs[ck][bairro] += peso
                if is_cvli:
                    mun_cvli[ck][bairro] += 1

        print(f"   ✅ {len(df_with_bairro)} registros com bairro processados.")

        # 1b. Registros SEM bairro: geocoding por clusters lat/lng (modo completo)
        if not fast_mode:
            df_no_bairro = df_nf[
                df_nf['bairro'].isna() | (df_nf['bairro'].str.len() <= 2)
            ].copy()
            df_no_bairro['lat2'] = df_no_bairro['latitude'].round(2)
            df_no_bairro['lng2'] = df_no_bairro['longitude'].round(2)

            # Agrupar por (cidade, lat2, lng2) → centróide e contagem
            clusters_df = df_no_bairro.groupby(['cidade', 'lat2', 'lng2']).agg(
                lat_mean=('latitude', 'mean'),
                lng_mean=('longitude', 'mean'),
                count=('latitude', 'size'),
                is_cvli=('tipo', lambda x: (x.str.lower() == 'cvli').sum())
            ).reset_index()

            # Priorizar clusters com mais CVLI (geocodificar os mais relevantes primeiro)
            clusters_df = clusters_df.sort_values('is_cvli', ascending=False)

            geo_cache = _load_reverse_cache()
            cluster_list = [
                (f"{row['lat2']}_{row['lng2']}", row['lat_mean'], row['lng_mean'])
                for _, row in clusters_df.iterrows()
            ]
            print(f"   🗺️  {len(cluster_list)} clusters sem bairro a geocodificar...")
            geo_cache = _reverse_batch(cluster_list, geo_cache)

            # Aplicar resultado ao mun_locs
            for _, row in clusters_df.iterrows():
                ck = normalize(row['cidade'])
                key = f"{row['lat2']}_{row['lng2']}"
                street = geo_cache.get(key, '')
                if street and len(street) > 4 and not any(t in street for t in INVALID_TERMS):
                    cvli_count = int(row['is_cvli'])
                    peso = cvli_count * 3 + int(row['count'])
                    mun_locs[ck][street] += peso
                    if cvli_count > 0:
                        mun_cvli[ck][street] += cvli_count

            print(f"   ✅ Geocoding concluído: {len(cluster_list)} clusters.")

    # ── FONTE 2: Eventos Exógenos ────────────────────────────────────────
    if os.path.exists(EXO_PATH):
        print("📖 Indexando eventos exógenos...")
        with open(EXO_PATH, 'r', encoding='utf-8') as f:
            exo = json.load(f)
        street_re = re.compile(
            r'\b(?:RUA|AV(?:ENIDA)?\.?\s*|TRAVESSA|ALAMEDA|PRA[CÇ]A|RODOVIA|BR[-\s]\d+|CE[-\s]\d+)'
            r'[A-Z0-9\u00C0-\u00FF][A-Z0-9\u00C0-\u00FF\s,\.]{2,50}',
            re.IGNORECASE | re.UNICODE
        )
        count_exo = 0
        for ev in exo:
            mun = normalize(ev.get('municipio') or '')
            if not mun or mun == 'FORTALEZA':
                continue
            bairro = normalize(ev.get('bairro') or '')
            if bairro and len(bairro) > 2 and not any(t in bairro for t in INVALID_TERMS):
                mun_locs[mun][bairro] += 5  # alta relevância: evento confirmado
            txt = str(ev.get('raw_text') or ev.get('descricao') or '')
            for m in street_re.findall(txt.upper()):
                st = ' '.join(m.split())[:60].strip()
                if len(st) > 5 and not any(t in st for t in INVALID_TERMS):
                    mun_locs[mun][st] += 3
            count_exo += 1
        print(f"   ✅ {count_exo} eventos exógenos indexados.")

    # ── FONTE 3: KML de Inteligência de Facções ──────────────────────────
    # Arquivos: ORCRIMS 2026.kml, COMANDO VERMELHO.kml, TCP GDE.kml
    # Cada placemark tem campo MUNICIPIO/CIDADE + NOME (ou nome codificado no title)
    KML_FILES = [
        os.path.join(BASE_DIR, 'data', 'static', 'ORCRIMS 2026.kml'),
        os.path.join(BASE_DIR, 'data', 'static', 'COMANDO VERMELHO.kml'),
        os.path.join(BASE_DIR, 'data', 'static',
                     'TCP  GDE - TERCEIRO COMANDO PURO E GUARDIÕES DO ESTADO.kml'),
    ]
    try:
        import xml.etree.ElementTree as ET
        kml_ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        total_kml = 0
        for kml_path in KML_FILES:
            if not os.path.exists(kml_path):
                continue
            tree = ET.parse(kml_path)
            root_kml = tree.getroot()
            placemarks = root_kml.findall('.//kml:Placemark', kml_ns)
            count_kml = 0
            for pm in placemarks:
                ext = pm.find('kml:ExtendedData', kml_ns)
                ext_data = {}
                if ext is not None:
                    for sd in ext.findall('kml:SchemaData/kml:SimpleData', kml_ns):
                        ext_data[sd.get('name')] = sd.text
                    for data in ext.findall('kml:Data', kml_ns):
                        ext_data[data.get('name')] = data.findtext('kml:value',
                                                                    namespaces=kml_ns)

                municipio = normalize(ext_data.get('MUNICIPIO') or ext_data.get('CIDADE') or '')
                nome = normalize(ext_data.get('NOME') or '')
                pm_title = normalize(pm.findtext('kml:name', default='', namespaces=kml_ns) or '')

                # Para placemarks sem campo MUNICIPIO: extrair do título
                # Formato típico: "BAIRRO  MUNICIPIO  FACCAO" ou "BAIRRO - MUNICIPIO - FACCAO"
                if not municipio or municipio == 'FORTALEZA':
                    # Separadores: múltiplos espaços ou " - "
                    parts = [p.strip() for p in re.split(r'\s{2,}|\s+-\s+', pm_title) if len(p.strip()) > 2]
                    for part in parts[1:]:  # municipio nunca é o primeiro token
                        part_clean = re.sub(r'\s*AIS\s*\d+.*$', '', part).strip()
                        if part_clean and part_clean != 'FORTALEZA':
                            # Verificar se é um município conhecido (pelo menos 5 chars)
                            if len(part_clean) >= 5 and not any(
                                t in part_clean for t in INVALID_TERMS + ['CV', 'TCP', 'PCC', 'GDE', 'MASSA']
                            ):
                                municipio = part_clean
                                if not nome and parts[0]:
                                    nome_raw = re.sub(r'\s*-\s*AIS\s*\d+.*$', '', parts[0]).strip()
                                    nome = nome_raw
                                break

                if municipio and municipio != 'FORTALEZA' and nome:
                    if not any(t in nome for t in INVALID_TERMS) and len(nome) > 3:
                        mun_locs[municipio][nome] += 8  # peso alto: intel de facções confirmada
                        count_kml += 1

            total_kml += count_kml
            print(f"   🗺️  {os.path.basename(kml_path)}: {count_kml} localidades indexadas.")
        print(f"   ✅ KML de facções: {total_kml} localidades não-Fortaleza indexadas.")
    except Exception as kml_err:
        print(f"   ⚠️  Erro ao processar KMLs: {kml_err}")

    # ── Consolidar e salvar ──────────────────────────────────────────────
    # Formato: { "MUNICIPIO": [ {"loc": "BAIRRO", "cvli": 4, "score": 20}, ... ] }
    # cvli  = contagem bruta de registros CVLI (ordenação primária)
    # score = peso ponderado total (desempate: CVLI=3pts, crimes=1pt, exó=5pt, KML=8pt)
    result = {}
    for mun_key, counter in mun_locs.items():
        cvli_counter = mun_cvli.get(mun_key, {})
        candidates = [
            (loc, score)
            for loc, score in counter.most_common(30)
            if len(loc) > 3 and loc not in ('NAN', 'NONE', 'N A', 'SEM BAIRRO')
        ]
        # Ordenar: CVLI bruto desc → score ponderado desc
        candidates.sort(key=lambda x: (cvli_counter.get(x[0], 0), x[1]), reverse=True)
        top = [
            {'loc': loc, 'cvli': cvli_counter.get(loc, 0), 'score': score}
            for loc, score in candidates[:10]
        ]
        if top:
            result[mun_key] = top

    tmp = OUTPUT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)

    print(f"\n✅ Concluído: {len(result)} municípios → {OUTPUT_PATH}")
    for mun in ['caucaia', 'maracanau', 'sobral', 'juazeiro do norte']:
        key = normalize(mun)
        entries = result.get(key, [])[:4]
        print(f"   {mun.title():25s}: {[e['loc'] + '(cvli=' + str(e['cvli']) + ')' for e in entries]}")


if __name__ == '__main__':
    fast = '--fast' in sys.argv
    days = 30
    for arg in sys.argv:
        if arg.startswith('--days='):
            try:
                days = int(arg.split('=')[1])
            except ValueError:
                pass
        elif arg == '--days' and sys.argv.index(arg) + 1 < len(sys.argv):
            try:
                days = int(sys.argv[sys.argv.index(arg) + 1])
            except (ValueError, IndexError):
                pass
    build(fast_mode=fast, days=days)
