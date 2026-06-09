import geopandas as gpd
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import zipfile
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
INTEL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')
DICT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
DOWNLOADS_DIR = r'C:\Users\Boanerges\Downloads'
CURRENT_KML_PATH = os.path.join(INTEL_DIR, 'current_orcrim.kml')
STATIC_KML_PATH = os.path.join(BASE_DIR, 'data', 'static', 'ORCRIMS 2026.kml')
LOCAL_KMZ_PATH = os.path.join(INTEL_DIR, 'ORCRIMS 2026.kmz')
UPDATE_STATUS_PATH = os.path.join(INTEL_DIR, 'orcrim_update_status.json')
REQUEST_TIMEOUT = 60
CHROME_DOWNLOAD_TIMEOUT = 45
CHROME_PATH_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
]
INTELLIGENCE_OUTPUTS = [
    os.path.join(INTEL_DIR, 'micronodos_faccoes_2026.csv'),
    os.path.join(INTEL_DIR, 'bairros_faccoes_2026.csv'),
    os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv'),
    os.path.join(INTEL_DIR, 'micronodos_faccoes_2026.geojson'),
]


class GoogleMapsAuthError(RuntimeError):
    """Falha de autenticação/autorização ao exportar o mapa do Google My Maps."""


class ChromeProfileDownloadTimeout(RuntimeError):
    """Chrome foi aberto, mas o download autenticado não apareceu a tempo."""


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371


def normalize_text(text):
    if not text:
        return ''
    normalized = ''.join(
        char for char in unicodedata.normalize('NFKD', str(text))
        if unicodedata.category(char) != 'Mn'
    )
    normalized = normalized.upper().strip()
    normalized = re.sub(r'\s*-\s*AIS.*$', '', normalized)
    normalized = re.sub(r'\s*-\s*(CV|PCC|GDE|TCP|MASSA|OKAIDA).*', '', normalized)
    return normalized


def _iso_now():
    return datetime.now().isoformat(timespec='seconds')


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _detect_faction_from_folder(folder_name: str) -> str:
    folder_name = (folder_name or '').upper()
    if 'COMANDO VERMELHO' in folder_name or ' CV ' in folder_name:
        return 'CV'
    if 'TCP' in folder_name or 'GDE' in folder_name:
        return 'TCP/GDE'
    if 'PCC' in folder_name:
        return 'PCC'
    if 'MASSA' in folder_name:
        return 'MASSA'
    if 'OKAIDA' in folder_name:
        return 'OKAIDA'
    if 'DISPUTA' in folder_name:
        return 'DISPUTA'
    return 'NEUTRO'


def _canonicalize_coordinate_sequence(raw_coordinates: str):
    coordinates = []
    for lon, lat in _parse_kml_coordinates(raw_coordinates):
        coordinates.append((round(float(lon), 6), round(float(lat), 6)))
    return coordinates


def _build_semantic_signature_entries(kml_bytes: bytes):
    """
    Gera uma assinatura semântica baseada apenas nos territórios efetivamente
    extraídos para facções, ignorando estilos, ids e outros metadados do Google.
    """
    try:
        root = ET.fromstring(kml_bytes.decode('utf-8', errors='ignore'))
    except Exception:
        return []

    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    signature_entries = []

    for folder in root.findall('.//kml:Folder', namespaces):
        folder_name = folder.findtext('kml:name', default='', namespaces=namespaces)
        faction = _detect_faction_from_folder(folder_name)
        if faction == 'NEUTRO':
            continue

        for placemark in folder.findall('./kml:Placemark', namespaces):
            placemark_name = placemark.findtext('kml:name', default='S/N', namespaces=namespaces)
            polygon_signatures = []
            point_signature = None

            for polygon_elem in placemark.findall('.//kml:Polygon', namespaces):
                outer_text = polygon_elem.findtext(
                    './/kml:outerBoundaryIs//kml:LinearRing//kml:coordinates',
                    default='',
                    namespaces=namespaces,
                )
                outer_coords = _canonicalize_coordinate_sequence(outer_text)
                inner_signatures = []
                for inner_elem in polygon_elem.findall('.//kml:innerBoundaryIs', namespaces):
                    inner_text = inner_elem.findtext(
                        './/kml:LinearRing//kml:coordinates',
                        default='',
                        namespaces=namespaces,
                    )
                    inner_coords = _canonicalize_coordinate_sequence(inner_text)
                    if inner_coords:
                        inner_signatures.append(inner_coords)

                if outer_coords:
                    polygon_signatures.append({
                        'outer': outer_coords,
                        'inners': sorted(inner_signatures, key=lambda ring: json.dumps(ring, ensure_ascii=False)),
                    })

            if not polygon_signatures:
                coords_elem = placemark.find('.//kml:Point//kml:coordinates', namespaces)
                if coords_elem is not None and coords_elem.text:
                    point_coords = _canonicalize_coordinate_sequence(coords_elem.text)
                    if point_coords:
                        point_signature = point_coords[0]

            signature_entries.append({
                'faction': faction,
                'name': normalize_text(placemark_name),
                'geometry': sorted(polygon_signatures, key=lambda item: json.dumps(item, ensure_ascii=False)) if polygon_signatures else point_signature,
            })

    signature_entries.sort(key=lambda item: (item['faction'], item['name'], json.dumps(item['geometry'], ensure_ascii=False)))
    return signature_entries


def _get_semantic_content_hash(kml_bytes: bytes) -> str:
    signature_entries = _build_semantic_signature_entries(kml_bytes)
    if not signature_entries:
        return _get_content_hash(kml_bytes)
    canonical_payload = json.dumps(signature_entries, ensure_ascii=False, separators=(',', ':'))
    return _sha256_bytes(canonical_payload.encode('utf-8'))


def _get_content_hash(kml_bytes: bytes) -> str:
    """
    Gera um hash baseado apenas no conteúdo estrutural do KML,
    ignorando metadados dinâmicos da Google (timestamps, IDs de sessão, etc).
    """
    try:
        content = kml_bytes.decode('utf-8', errors='ignore')
        # Remover tags de TimeStamp que mudam a cada exportação
        content = re.sub(r'<gx:TimeStamp>.*?</gx:TimeStamp>', '', content, flags=re.DOTALL)
        content = re.sub(r'<TimeStamp>.*?</TimeStamp>', '', content, flags=re.DOTALL)
        # Remover metadados de visualização/sessão da Google
        content = re.sub(r'<LookAt>.*?</LookAt>', '', content, flags=re.DOTALL)
        # Remover espaços em branco extras para estabilidade
        content = re.sub(r'\s+', ' ', content).strip()
        return _sha256_bytes(content.encode('utf-8'))
    except Exception:
        return _sha256_bytes(kml_bytes)


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as file_obj:
        return _sha256_bytes(file_obj.read())


def _file_info(path: str):
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        'path': path,
        'size_bytes': stat.st_size,
        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds'),
        'sha256': _sha256_file(path),
    }


def _read_update_status():
    if not os.path.exists(UPDATE_STATUS_PATH):
        return {}
    try:
        with open(UPDATE_STATUS_PATH, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj) or {}
    except Exception:
        return {}


def _write_update_status(payload: dict):
    os.makedirs(INTEL_DIR, exist_ok=True)
    with open(UPDATE_STATUS_PATH, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def get_orcrim_update_status():
    status = _read_update_status()
    status['paths'] = {
        'kmz_local': _file_info(LOCAL_KMZ_PATH),
        'kml_working': _file_info(CURRENT_KML_PATH),
        'kml_static': _file_info(STATIC_KML_PATH),
        'intelligence_csv': _file_info(os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv')),
    }
    return status


def _log_existing_state():
    print('🧭 [ORCRIMS] Estado anterior detectado:')
    for label, path in [
        ('KMZ local', LOCAL_KMZ_PATH),
        ('KML de trabalho', CURRENT_KML_PATH),
        ('KML estático', STATIC_KML_PATH),
        ('CSV inteligência', os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv')),
    ]:
        info = _file_info(path)
        if not info:
            print(f'   - {label}: inexistente')
            continue
        print(f"   - {label}: {info['modified_at']} | {info['size_bytes']} bytes | sha256={info['sha256'][:12]}...")

    status = _read_update_status()
    if status:
        print(
            '   - Última atualização registrada: '
            f"checked_at={status.get('last_checked_at', 'N/A')} | "
            f"updated_at={status.get('last_updated_at', 'N/A')} | "
            f"source_url={status.get('source_url', 'N/A')} | "
            f"status={status.get('status', 'N/A')}"
        )


def _extract_network_link_from_kml(kml_content: str) -> str:
    try:
        root = ET.fromstring(kml_content)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        href = root.findtext('.//kml:NetworkLink//kml:href', default='', namespaces=namespaces)
        return href.strip()
    except Exception:
        match = re.search(r'<href>(https?://[^<]+)</href>', kml_content, re.IGNORECASE)
        return match.group(1).strip() if match else ''


def _extract_network_link_from_kmz(kmz_path: str) -> str:
    if not os.path.exists(kmz_path):
        return ''
    with zipfile.ZipFile(kmz_path, 'r') as zip_ref:
        kml_names = [name for name in zip_ref.namelist() if name.lower().endswith('.kml')]
        if not kml_names:
            return ''
        kml_content = zip_ref.read(kml_names[0]).decode('utf-8', errors='ignore')
    return _extract_network_link_from_kml(kml_content)


def _resolve_official_url():
    candidates = [LOCAL_KMZ_PATH]
    if os.path.exists(DOWNLOADS_DIR):
        download_candidates = sorted(
            [
                os.path.join(DOWNLOADS_DIR, name)
                for name in os.listdir(DOWNLOADS_DIR)
                if 'ORCRIM' in name.upper() and name.lower().endswith('.kmz')
            ],
            reverse=True,
        )
        candidates.extend(download_candidates)

    for kmz_path in candidates:
        url = _extract_network_link_from_kmz(kmz_path)
        if url:
            print(f'🌐 [ORCRIMS] Link oficial identificado em {kmz_path}: {url}')
            return url
    return ''


def _load_env_for_cookies():
    env_path = os.path.join(BASE_DIR, '.env')
    data = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        data[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass
    return data


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        value = _load_env_for_cookies().get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _find_chrome_executable() -> str:
    for candidate in CHROME_PATH_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


def _chrome_download_candidates():
    if not os.path.exists(DOWNLOADS_DIR):
        return []
    candidates = []
    for name in os.listdir(DOWNLOADS_DIR):
        lower_name = name.lower()
        if lower_name.endswith(('.kmz', '.kml', '.crdownload')):
            candidates.append(os.path.join(DOWNLOADS_DIR, name))
    return candidates


def _wait_for_downloaded_kml_via_chrome(started_at: float):
    deadline = time.time() + CHROME_DOWNLOAD_TIMEOUT
    while time.time() < deadline:
        for path in sorted(_chrome_download_candidates(), key=lambda item: os.path.getmtime(item), reverse=True):
            if path.lower().endswith('.crdownload'):
                continue
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            if stat.st_mtime + 1 < started_at:
                continue
            if stat.st_size <= 0:
                continue
            return path
        time.sleep(1)
    raise ChromeProfileDownloadTimeout(
        'Chrome foi aberto com o perfil logado, mas nenhum KML/KMZ novo apareceu em Downloads '
        f'em até {CHROME_DOWNLOAD_TIMEOUT}s.'
    )


def _download_payload_via_logged_in_chrome(source_url: str):
    chrome_executable = _find_chrome_executable()
    if not chrome_executable:
        raise RuntimeError('Google Chrome não encontrado para fallback autenticado.')

    profile_name = os.environ.get('ORCRIMS_CHROME_PROFILE') or _load_env_for_cookies().get('ORCRIMS_CHROME_PROFILE') or 'Default'
    started_at = time.time()
    print(f'🌐 [ORCRIMS] Tentando exportar via Chrome logado (perfil={profile_name})...')
    subprocess.Popen(
        [
            chrome_executable,
            f'--profile-directory={profile_name}',
            '--new-window',
            source_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    downloaded_path = _wait_for_downloaded_kml_via_chrome(started_at)
    print(f'📥 [ORCRIMS] Download detectado via Chrome: {downloaded_path}')
    with open(downloaded_path, 'rb') as file_obj:
        return file_obj.read(), {
            'content_type': 'application/vnd.google-earth.kmz' if downloaded_path.lower().endswith('.kmz') else 'application/vnd.google-earth.kml+xml',
            'downloaded_via': 'chrome_profile',
            'download_path': downloaded_path,
            'profile': profile_name,
        }


def _build_google_session(source_url: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': source_url.replace('/kml?', '/viewer?'),
    })

    try:
        env = _load_env_for_cookies()
        cookie_header = env.get('GOOGLE_MAPS_COOKIE', '')
        if cookie_header:
            loaded = 0
            for item in cookie_header.split(';'):
                item = item.strip()
                if not item or '=' not in item:
                    continue
                name, value = item.split('=', 1)
                name = name.strip()
                value = value.strip()
                if not name:
                    continue
                session.cookies.set(name, value, domain='.google.com')
                session.cookies.set(name, value, domain='www.google.com')
                loaded += 1
            print(f"🔑 [ORCRIMS] Cookies de autenticação carregados do .env ({loaded} entradas)")
        else:
            print("⚠️ [ORCRIMS] GOOGLE_MAPS_COOKIE não definido no .env")
    except Exception as e:
        print(f"⚠️ [ORCRIMS] Erro ao carregar .env para cookies: {e}")

    return session


def _raise_for_google_auth_failure(response: requests.Response, source_url: str):
    final_url = response.url or ''
    history_urls = [item.headers.get('Location', '') for item in response.history]
    content_type = (response.headers.get('Content-Type') or '').lower()
    body_text = ''

    if 'text/' in content_type or 'html' in content_type:
        try:
            body_text = response.text[:4000]
        except Exception:
            body_text = ''

    auth_markers = (
        'accounts.google.com',
        'servicelogin',
        'cookiemismatch',
        'signin/identifier',
    )
    combined_targets = ' '.join([final_url] + history_urls + [body_text]).lower()
    if any(marker in combined_targets for marker in auth_markers):
        raise GoogleMapsAuthError(
            'Google não aceitou a sessão do GOOGLE_MAPS_COOKIE para exportar o My Maps '
            f'(url final: {final_url or source_url}). Atualize o cookie autenticado no .env.'
        )

    if response.status_code == 403 and 'google.com/maps/d' in final_url:
        raise GoogleMapsAuthError(
            'Google bloqueou a exportação autenticada do My Maps. '
            'Verifique se o GOOGLE_MAPS_COOKIE ainda representa uma sessão válida no navegador.'
        )


def _download_official_payload(source_url: str):
    print(f'⬇️ [ORCRIMS] Baixando KML oficial: {source_url}')
    session = _build_google_session(source_url)
    response = session.get(source_url, timeout=REQUEST_TIMEOUT)
    _raise_for_google_auth_failure(response, source_url)
    response.raise_for_status()
    content = response.content
    print(
        f"✅ [ORCRIMS] Download concluído: status={response.status_code} | bytes={len(content)} | "
        f"etag={response.headers.get('ETag', 'N/A')} | last-modified={response.headers.get('Last-Modified', 'N/A')}"
    )
    return content, {
        'etag': response.headers.get('ETag', ''),
        'last_modified': response.headers.get('Last-Modified', ''),
        'content_length': response.headers.get('Content-Length', ''),
        'content_type': response.headers.get('Content-Type', ''),
        'final_url': response.url or source_url,
        'redirect_chain': [item.headers.get('Location', '') for item in response.history],
    }


def _extract_kml_bytes_from_payload(payload: bytes) -> bytes:
    if payload[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(payload), 'r') as zip_ref:
            kml_names = [name for name in zip_ref.namelist() if name.lower().endswith('.kml')]
            if not kml_names:
                raise ValueError('KMZ recebido sem arquivo KML interno.')
            print(f'📦 [ORCRIMS] Payload remoto é KMZ. Extraindo {kml_names[0]}...')
            return zip_ref.read(kml_names[0])
    return payload


def _persist_kml_bytes(kml_bytes: bytes):
    os.makedirs(INTEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATIC_KML_PATH), exist_ok=True)
    with open(CURRENT_KML_PATH, 'wb') as file_obj:
        file_obj.write(kml_bytes)
    with open(STATIC_KML_PATH, 'wb') as file_obj:
        file_obj.write(kml_bytes)


def _capture_backups(paths):
    backups = {}
    for path in paths:
        if os.path.exists(path):
            with open(path, 'rb') as file_obj:
                backups[path] = file_obj.read()
        else:
            backups[path] = None
    return backups


def _restore_backups(backups):
    for path, content in backups.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if content is None:
            if os.path.exists(path):
                os.remove(path)
            continue
        with open(path, 'wb') as file_obj:
            file_obj.write(content)


def _parse_kml_coordinates(raw_coordinates: str):
    coordinates = []
    for chunk in (raw_coordinates or '').strip().split():
        parts = chunk.split(',')
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except Exception:
            continue
        coordinates.append((lon, lat))
    if coordinates and coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    return coordinates


def _build_polygon_geometry(placemark, namespaces):
    polygon_geometries = []
    for polygon_elem in placemark.findall('.//kml:Polygon', namespaces):
        outer_text = polygon_elem.findtext(
            './/kml:outerBoundaryIs//kml:LinearRing//kml:coordinates',
            default='',
            namespaces=namespaces,
        )
        shell = _parse_kml_coordinates(outer_text)
        if len(shell) < 4:
            continue

        holes = []
        for inner_elem in polygon_elem.findall('.//kml:innerBoundaryIs', namespaces):
            inner_text = inner_elem.findtext('.//kml:LinearRing//kml:coordinates', default='', namespaces=namespaces)
            hole = _parse_kml_coordinates(inner_text)
            if len(hole) >= 4:
                holes.append(hole)

        polygon = Polygon(shell=shell, holes=holes or None)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        if isinstance(polygon, MultiPolygon):
            polygon_geometries.extend([geom for geom in polygon.geoms if not geom.is_empty])
        else:
            polygon_geometries.append(polygon)

    if not polygon_geometries:
        return None
    if len(polygon_geometries) == 1:
        return polygon_geometries[0]
    return MultiPolygon(polygon_geometries)


def _generate_intelligence_from_kml(kml_working_path: str):
    print(f'🧠 [ORCRIMS] Reprocessando inteligência territorial a partir de {kml_working_path}')
    with open(DICT_PATH, 'r', encoding='utf-8') as file_obj:
        official_dict = json.load(file_obj)
    official_names = {normalize_text(name): name for name in official_dict.keys()}
    centers = [{'name': name, 'lat': coords['lat'], 'long': coords['long']} for name, coords in official_dict.items()]

    tree = ET.parse(kml_working_path)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    micronodes = []
    micronode_geometries = []
    for folder in root.findall('.//kml:Folder', namespaces):
        name_elem = folder.find('kml:name', namespaces)
        folder_name = name_elem.text.upper() if name_elem is not None and name_elem.text else ''
        faction = _detect_faction_from_folder(folder_name)

        if faction == 'NEUTRO':
            continue

        for placemark in folder.findall('.//kml:Placemark', namespaces):
            name_raw = placemark.find('kml:name', namespaces).text if placemark.find('kml:name', namespaces) is not None else 'S/N'
            geometry = _build_polygon_geometry(placemark, namespaces)
            lat, lon = None, None
            if geometry is not None and not geometry.is_empty:
                anchor = geometry.representative_point()
                lon, lat = float(anchor.x), float(anchor.y)
            else:
                coords_elem = placemark.find('.//kml:coordinates', namespaces)
                if coords_elem is not None and coords_elem.text:
                    try:
                        coords = coords_elem.text.strip().split()[0].split(',')
                        lon, lat = float(coords[0]), float(coords[1])
                        geometry = Point(lon, lat)
                    except Exception:
                        geometry = None

            area_id = 'DESCONHECIDO'
            norm_name = normalize_text(name_raw)
            if norm_name in official_names:
                area_id = official_names[norm_name]
            elif lat and lon:
                min_dist = float('inf')
                for center in centers:
                    distance = haversine(lon, lat, center['long'], center['lat'])
                    if distance < min_dist:
                        min_dist = distance
                        area_id = center['name']

            micronode_record = {
                'micronodo': name_raw,
                'area_oficial': area_id,
                'faction': faction,
                'lat': lat,
                'long': lon,
            }
            micronodes.append(micronode_record)
            micronode_geometries.append({**micronode_record, 'geometry': geometry})

    df_micro = pd.DataFrame(micronodes)
    counts = df_micro.groupby(['area_oficial', 'faction']).size().reset_index(name='n')
    df_aggregated = counts.sort_values('n', ascending=False).drop_duplicates('area_oficial')

    final_rows = []
    for official_area in official_dict.keys():
        row = df_aggregated[df_aggregated['area_oficial'] == official_area]
        faction = row.iloc[0]['faction'] if not row.empty else 'NEUTRO'
        final_rows.append({'local': official_area, 'faccao_predominante': faction, 'grau_dominio': 0.85 if not row.empty else 0.0})

    df_final = pd.DataFrame(final_rows)
    
    # --- OTIMIZAÇÃO: ESCREVER APENAS SE HOUVER MUDANÇA REAL ---
    def _write_if_changed(df, path, is_geojson=False):
        if os.path.exists(path):
            try:
                if is_geojson:
                    # Para GeoJSON, comparamos o conteúdo JSON carregado
                    with open(path, 'r', encoding='utf-8') as f:
                        old_data = json.load(f)
                    new_data = json.loads(df.to_json())
                    if old_data == new_data: return
                else:
                    # Para CSV, comparamos dataframes (ignorando pequenas variações de float se necessário)
                    old_df = pd.read_csv(path)
                    if old_df.equals(df): return
            except: pass
            
        if is_geojson:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(df.to_json())
        else:
            df.to_csv(path, index=False)
        print(f"💾 [ORCRIMS] Arquivo atualizado: {os.path.basename(path)}")

    _write_if_changed(df_micro, INTELLIGENCE_OUTPUTS[0])
    _write_if_changed(df_final, INTELLIGENCE_OUTPUTS[1])
    _write_if_changed(df_final, INTELLIGENCE_OUTPUTS[2])
    gdf_records = [record for record in micronode_geometries if record.get('geometry') is not None]
    gdf = gpd.GeoDataFrame(gdf_records, geometry='geometry', crs='EPSG:4326')
    _write_if_changed(gdf, INTELLIGENCE_OUTPUTS[3], is_geojson=True)

    print(f"✅ [ORCRIMS] Inteligência territorial processada: micronodos={len(df_micro)} | áreas oficiais={len(df_final)}")


def refresh_orcrim_from_official(force: bool = False):
    print('🔄 [ORCRIMS] Iniciando atualização no startup...')
    _log_existing_state()

    current_status = _read_update_status()
    
    # Cooldown de 1 hora para evitar downloads sucessivos a cada inicialização (Google My Maps não envia ETag).
    if not force:
        last_checked_str = current_status.get('last_checked_at')
        if last_checked_str:
            try:
                last_checked = datetime.fromisoformat(last_checked_str)
                from datetime import datetime as dt
                if (dt.now() - last_checked).total_seconds() < 3600:
                    print(f'⏱️ [ORCRIMS] Cooldown ativo. Última checagem foi recente ({last_checked_str}). Pulando download.')
                    return {'updated': False, 'reason': 'cooldown_active'}
            except Exception:
                pass

    source_url = _resolve_official_url()
    fallback_available = bool(os.path.exists(CURRENT_KML_PATH) and os.path.exists(STATIC_KML_PATH))
    if not source_url:
        print('⚠️ [ORCRIMS] Nenhum NetworkLink oficial foi encontrado no KMZ existente. Mantendo base atual.')
        current_status.update({
            'last_checked_at': _iso_now(),
            'source_url': current_status.get('source_url', ''),
            'status': 'no_official_link_found',
            'fallback_used': fallback_available,
        })
        _write_update_status(current_status)
        return {'updated': False, 'reason': 'no_official_link_found', 'fallback_used': fallback_available}

    try:
        payload_bytes, headers = _download_official_payload(source_url)
        kml_bytes = _extract_kml_bytes_from_payload(payload_bytes)
    except Exception as error:
        download_failures = [str(error)]
        if isinstance(error, GoogleMapsAuthError) and _env_flag('ORCRIMS_USE_CHROME_FALLBACK', default=True):
            print(f'⚠️ [ORCRIMS] Sessão por cookie rejeitada. Tentando Chrome logado: {error}')
            try:
                payload_bytes, headers = _download_payload_via_logged_in_chrome(source_url)
                kml_bytes = _extract_kml_bytes_from_payload(payload_bytes)
            except Exception as chrome_error:
                download_failures.append(f'chrome_profile: {chrome_error}')
                error = chrome_error
            else:
                error = None
        if error is None:
            pass
        else:
            combined_error = ' | '.join(download_failures)
            print(f'❌ [ORCRIMS] Falha ao baixar KML oficial: {combined_error}')
            if fallback_available:
                status = 'fallback_active'
            elif isinstance(error, GoogleMapsAuthError):
                status = 'auth_cookie_invalid'
            else:
                status = 'download_failed'
            current_status.update({
                'last_checked_at': _iso_now(),
                'last_error': combined_error,
                'source_url': source_url,
                'status': status,
                'fallback_used': fallback_available,
            })
            _write_update_status(current_status)
            return {'updated': False, 'reason': status, 'fallback_used': fallback_available, 'error': combined_error}

    downloaded_raw_hash = _get_content_hash(kml_bytes)
    downloaded_semantic_hash = _get_semantic_content_hash(kml_bytes)
    
    # Para comparação local, geramos hashes equivalentes sobre os arquivos persistidos.
    def _get_file_hashes(path):
        if not os.path.exists(path):
            return {'raw': '', 'semantic': ''}
        with open(path, 'rb') as f:
            file_bytes = f.read()
        return {
            'raw': _get_content_hash(file_bytes),
            'semantic': _get_semantic_content_hash(file_bytes),
        }

    current_hashes = _get_file_hashes(CURRENT_KML_PATH)
    static_hashes = _get_file_hashes(STATIC_KML_PATH)
    
    semantic_changed = force or (
        downloaded_semantic_hash != current_hashes['semantic']
        or downloaded_semantic_hash != static_hashes['semantic']
    )
    raw_changed = force or (
        downloaded_raw_hash != current_hashes['raw']
        or downloaded_raw_hash != static_hashes['raw']
    )
    print(
        f"🔎 [ORCRIMS] Comparação semântica: novo={downloaded_semantic_hash[:12]}... | "
        f"atual={current_hashes['semantic'][:12] if current_hashes['semantic'] else 'N/A'}... | "
        f"estático={static_hashes['semantic'][:12] if static_hashes['semantic'] else 'N/A'}..."
    )
    if raw_changed and not semantic_changed:
        print(
            f"ℹ️ [ORCRIMS] Diferença detectada apenas no XML bruto/metadados: bruto={downloaded_raw_hash[:12]}... | "
            f"atual={current_hashes['raw'][:12] if current_hashes['raw'] else 'N/A'}... | "
            f"estático={static_hashes['raw'][:12] if static_hashes['raw'] else 'N/A'}..."
        )

    if not semantic_changed:
        print('ℹ️ [ORCRIMS] Nenhuma mudança detectada em relação à última base local. Reprocessamento dispensado.')
        current_status.update({
            'last_checked_at': _iso_now(),
            'source_url': source_url,
            'status': 'no_change',
            'fallback_used': False,
            'download_sha256': downloaded_raw_hash,
            'download_semantic_sha256': downloaded_semantic_hash,
            'headers': headers,
            'last_error': '',
        })
        _write_update_status(current_status)
        return {'updated': False, 'reason': 'no_change', 'source_url': source_url, 'fallback_used': False}

    print('📝 [ORCRIMS] Mudança detectada. Persistindo novo KML e reprocessando inteligência...')
    backup_paths = [CURRENT_KML_PATH, STATIC_KML_PATH] + INTELLIGENCE_OUTPUTS
    backups = _capture_backups(backup_paths)
    try:
        _persist_kml_bytes(kml_bytes)
        _generate_intelligence_from_kml(CURRENT_KML_PATH)
    except Exception as error:
        print(f'⚠️ [ORCRIMS] Falha ao aplicar atualização nova. Restaurando última base válida: {error}')
        _restore_backups(backups)
        current_status.update({
            'last_checked_at': _iso_now(),
            'source_url': source_url,
            'status': 'fallback_restored',
            'fallback_used': True,
            'download_sha256': downloaded_raw_hash,
            'download_semantic_sha256': downloaded_semantic_hash,
            'headers': headers,
            'last_error': str(error),
        })
        _write_update_status(current_status)
        return {'updated': False, 'reason': 'fallback_restored', 'source_url': source_url, 'fallback_used': True, 'error': str(error)}

    current_status.update({
        'last_checked_at': _iso_now(),
        'last_updated_at': _iso_now(),
        'source_url': source_url,
        'status': 'updated',
        'fallback_used': False,
        'download_sha256': downloaded_raw_hash,
        'download_semantic_sha256': downloaded_semantic_hash,
        'headers': headers,
        'last_error': '',
    })
    _write_update_status(current_status)
    print('✅ [ORCRIMS] Atualização oficial concluída com sucesso no startup.')
    return {'updated': True, 'reason': 'content_changed', 'source_url': source_url, 'fallback_used': False}


def import_kml():
    files = [name for name in os.listdir(DOWNLOADS_DIR) if 'ORCRIM' in name.upper()]
    if not files:
        print('❌ Nenhum arquivo ORCRIM encontrado em Downloads.')
        return

    latest_file = os.path.join(DOWNLOADS_DIR, sorted(files)[-1])
    print(f'📂 Processando: {latest_file}')

    if latest_file.lower().endswith('.kmz'):
        with zipfile.ZipFile(latest_file, 'r') as zip_ref:
            zip_ref.extractall(INTEL_DIR)
            doc_kml = os.path.join(INTEL_DIR, 'doc.kml')
            if os.path.exists(doc_kml):
                if os.path.exists(CURRENT_KML_PATH):
                    os.remove(CURRENT_KML_PATH)
                os.rename(doc_kml, CURRENT_KML_PATH)
    else:
        shutil.copy(latest_file, CURRENT_KML_PATH)

    _persist_kml_bytes(open(CURRENT_KML_PATH, 'rb').read())
    _generate_intelligence_from_kml(CURRENT_KML_PATH)
    _write_update_status({
        'last_checked_at': _iso_now(),
        'last_updated_at': _iso_now(),
        'source_url': 'manual_downloads_import',
        'status': 'manual_import',
        'fallback_used': False,
        'download_sha256': _sha256_file(CURRENT_KML_PATH),
        'headers': {},
        'last_error': '',
    })


if __name__ == '__main__':
    import_kml()
