import geopandas as gpd
import base64
import ctypes
import hashlib
import io
import json
import os
import re
import sqlite3
import socket
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.request
import zipfile
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from xml.etree import ElementTree as ET

import pandas as pd
import requests
import websocket
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
try:
    import browser_cookie3
except Exception:
    browser_cookie3 = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
INTEL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')
DICT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
DOWNLOADS_DIR = r'C:\Users\Boanerges\Downloads'
CURRENT_KML_PATH = os.path.join(INTEL_DIR, 'current_orcrim.kml')
STATIC_KML_PATH = os.path.join(BASE_DIR, 'data', 'static', 'ORCRIMS 2026.kml')
LOCAL_KMZ_PATH = os.path.join(INTEL_DIR, 'ORCRIMS 2026.kmz')
LOCAL_KML_PATH = os.path.join(INTEL_DIR, 'ORCRIMS 2026.kml')
UPDATE_STATUS_PATH = os.path.join(INTEL_DIR, 'orcrim_update_status.json')
REQUEST_TIMEOUT = 60
CHROME_DOWNLOAD_TIMEOUT = 45
CHROME_PATH_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
]
CHROME_USER_DATA_DIR_CANDIDATES = [
    os.environ.get('ORCRIMS_CHROME_USER_DATA_DIR', ''),
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'User Data'),
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


class ChromeProfileCookieError(RuntimeError):
    """NÃ£o foi possÃ­vel reaproveitar os cookies do perfil logado do Chrome."""


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


def _find_chrome_user_data_dir() -> str:
    env = _load_env_for_cookies()
    explicit_dir = os.environ.get('ORCRIMS_CHROME_USER_DATA_DIR') or env.get('ORCRIMS_CHROME_USER_DATA_DIR', '')
    candidates = [explicit_dir, *CHROME_USER_DATA_DIR_CANDIDATES]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


def _update_nested_dict(target: dict, path, value):
    current = target
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value


def _prepare_chrome_profile_for_auto_download(profile_name: str):
    user_data_dir = _find_chrome_user_data_dir()
    if not user_data_dir:
        return

    profile_dir = os.path.join(user_data_dir, profile_name)
    preferences_path = os.path.join(profile_dir, 'Preferences')
    if not os.path.exists(preferences_path):
        return

    try:
        with open(preferences_path, 'r', encoding='utf-8') as file_obj:
            preferences = json.load(file_obj)
    except Exception as error:
        print(f'⚠️ [ORCRIMS] Não foi possível ler Preferences do Chrome ({preferences_path}): {error}')
        return

    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    desired_values = {
        ('download', 'default_directory'): DOWNLOADS_DIR,
        ('download', 'prompt_for_download'): False,
        ('download', 'directory_upgrade'): True,
        ('download', 'extensions_to_open'): '',
        ('safebrowsing', 'enabled'): True,
    }

    changed = False
    for path, expected_value in desired_values.items():
        current = preferences
        for key in path[:-1]:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        current_value = current.get(path[-1]) if isinstance(current, dict) else None
        if current_value != expected_value:
            _update_nested_dict(preferences, path, expected_value)
            changed = True

    if not changed:
        return

    try:
        with open(preferences_path, 'w', encoding='utf-8') as file_obj:
            json.dump(preferences, file_obj, ensure_ascii=False, separators=(',', ':'))
        print(f'✅ [ORCRIMS] Preferências do Chrome ajustadas para download automático ({profile_name}).')
    except Exception as error:
        print(f'⚠️ [ORCRIMS] Não foi possível atualizar Preferences do Chrome ({preferences_path}): {error}')


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


def _dpapi_unprotect(ciphertext: bytes) -> bytes:
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [('cbData', ctypes.c_uint32), ('pbData', ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    in_blob = DATA_BLOB(len(ciphertext), ctypes.create_string_buffer(ciphertext))
    out_blob = DATA_BLOB()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ChromeProfileCookieError('Falha ao descriptografar dado protegido via DPAPI.')

    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _get_chrome_master_key(user_data_dir: str) -> bytes:
    local_state_path = os.path.join(user_data_dir, 'Local State')
    if not os.path.exists(local_state_path):
        raise ChromeProfileCookieError(f'Arquivo Local State nÃ£o encontrado em {local_state_path}.')

    with open(local_state_path, 'r', encoding='utf-8') as file_obj:
        local_state = json.load(file_obj)

    encrypted_key_b64 = (((local_state.get('os_crypt') or {}).get('encrypted_key')) or '').strip()
    if not encrypted_key_b64:
        raise ChromeProfileCookieError('Chrome Local State sem os_crypt.encrypted_key.')

    encrypted_key = base64.b64decode(encrypted_key_b64)
    if encrypted_key.startswith(b'DPAPI'):
        encrypted_key = encrypted_key[5:]
    return _dpapi_unprotect(encrypted_key)


def _decrypt_chrome_cookie_value(encrypted_value: bytes, master_key: bytes) -> str:
    if not encrypted_value:
        return ''

    if encrypted_value.startswith((b'v10', b'v11', b'v20')):
        nonce = encrypted_value[3:15]
        cipherbytes = encrypted_value[15:]
        plaintext = AESGCM(master_key).decrypt(nonce, cipherbytes, None)
        return plaintext.decode('utf-8', errors='ignore')

    try:
        return _dpapi_unprotect(encrypted_value).decode('utf-8', errors='ignore')
    except Exception:
        return encrypted_value.decode('utf-8', errors='ignore')


def _load_google_cookies_from_chrome_profile(profile_name: str):
    user_data_dir = _find_chrome_user_data_dir()
    if not user_data_dir:
        raise ChromeProfileCookieError('DiretÃ³rio User Data do Chrome nÃ£o encontrado.')

    cookie_db_candidates = [
        os.path.join(user_data_dir, profile_name, 'Network', 'Cookies'),
        os.path.join(user_data_dir, profile_name, 'Cookies'),
    ]
    cookie_db_path = next((path for path in cookie_db_candidates if os.path.exists(path)), '')
    if not cookie_db_path:
        raise ChromeProfileCookieError(f'Banco de cookies do perfil {profile_name} nÃ£o encontrado.')

    master_key = _get_chrome_master_key(user_data_dir)

    with tempfile.NamedTemporaryFile(prefix='orcrims_cookies_', suffix='.sqlite', delete=False) as temp_file:
        temp_cookie_db = temp_file.name
    try:
        _copy_file_locked(cookie_db_path, temp_cookie_db)
        conn = sqlite3.connect(temp_cookie_db)
        try:
            cursor = conn.execute(
                """
                SELECT host_key, name, value, encrypted_value, path, is_secure
                FROM cookies
                WHERE host_key LIKE '%google.com%'
                """
            )
            cookies = []
            for host_key, name, value, encrypted_value, path, is_secure in cursor.fetchall():
                cookie_value = value or _decrypt_chrome_cookie_value(encrypted_value or b'', master_key)
                if not cookie_value:
                    continue
                cookies.append({
                    'host_key': host_key,
                    'name': name,
                    'value': cookie_value,
                    'path': path or '/',
                    'secure': bool(is_secure),
                })
        finally:
            conn.close()
    finally:
        try:
            os.remove(temp_cookie_db)
        except OSError:
            pass

    if not cookies:
        raise ChromeProfileCookieError(f'Nenhum cookie Google utilizÃ¡vel foi lido do perfil {profile_name}.')
    return cookies


def _copy_file_with_robocopy(src: str, dst: str) -> bool:
    """Copia um arquivo em uso pelo Chrome usando robocopy (ignora locks do Windows)."""
    src_dir = os.path.dirname(src)
    src_file = os.path.basename(src)
    dst_dir = os.path.dirname(dst)
    dst_file = os.path.basename(dst)
    os.makedirs(dst_dir, exist_ok=True)
    if dst_file != src_file:
        tmp_dst = os.path.join(dst_dir, src_file)
    else:
        tmp_dst = dst
    try:
        result = subprocess.run(
            ['robocopy', src_dir, dst_dir, src_file, '/R:1', '/W:0', '/NFL', '/NDL', '/NJH', '/NJS', '/NP'],
            capture_output=True,
            timeout=15,
        )
        # robocopy retorna 1 quando copia com sucesso (não é erro)
        if result.returncode <= 1 and os.path.exists(tmp_dst):
            if tmp_dst != dst:
                shutil.move(tmp_dst, dst)
            return True
    except Exception:
        pass
    return False


def _copy_file_locked(src: str, dst: str):
    """Copia src para dst, usando robocopy como fallback se o arquivo estiver travado."""
    try:
        shutil.copy2(src, dst)
    except (PermissionError, OSError):
        if not _copy_file_with_robocopy(src, dst):
            raise


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _copy_minimal_chrome_profile(profile_name: str):
    user_data_dir = _find_chrome_user_data_dir()
    if not user_data_dir:
        raise ChromeProfileCookieError('DiretÃ³rio User Data do Chrome nÃ£o encontrado.')

    temp_root = tempfile.mkdtemp(prefix='orcrims_chrome_headless_')
    temp_user_data_dir = os.path.join(temp_root, 'User Data')
    temp_profile_dir = os.path.join(temp_user_data_dir, profile_name)
    source_profile_dir = os.path.join(user_data_dir, profile_name)
    if not os.path.exists(source_profile_dir):
        raise ChromeProfileCookieError(f'Perfil do Chrome nÃ£o encontrado: {source_profile_dir}.')

    ignored_dir_names = {
        'Cache', 'Code Cache', 'GPUCache', 'ShaderCache', 'GrShaderCache',
        'DawnCache', 'DawnGraphiteCache', 'DawnWebGPUCache', 'Crashpad',
        'OptimizationHints', 'AutofillStates', 'VideoDecodeStats',
    }
    ignored_path_parts = {
        os.path.join('Service Worker', 'CacheStorage'),
        os.path.join('Service Worker', 'ScriptCache'),
    }

    os.makedirs(temp_profile_dir, exist_ok=True)
    copied_files = 0
    skipped_files = 0
    skipped_errors = []
    for root, dirnames, filenames in os.walk(source_profile_dir):
        relative_root = os.path.relpath(root, source_profile_dir)
        if relative_root == '.':
            relative_root = ''
        if relative_root and any(
            relative_root == ignored_part or relative_root.startswith(ignored_part + os.sep)
            for ignored_part in ignored_path_parts
        ):
            dirnames[:] = []
            continue

        dirnames[:] = [name for name in dirnames if name not in ignored_dir_names]
        destination_root = os.path.join(temp_profile_dir, relative_root)
        os.makedirs(destination_root, exist_ok=True)

        for filename in filenames:
            lower_filename = filename.lower()
            if lower_filename in {'lock', 'cookies-journal', 'safe browsing cookies', 'safe browsing cookies-journal'}:
                skipped_files += 1
                continue
            if relative_root.startswith('Sessions'):
                skipped_files += 1
                continue
            source_path = os.path.join(root, filename)
            destination_path = os.path.join(destination_root, filename)
            try:
                _copy_file_locked(source_path, destination_path)
                copied_files += 1
            except (PermissionError, OSError) as error:
                skipped_files += 1
                if len(skipped_errors) < 10:
                    skipped_errors.append(f'{os.path.relpath(source_path, source_profile_dir)} ({error})')

    local_state_source = os.path.join(user_data_dir, 'Local State')
    if os.path.exists(local_state_source):
        os.makedirs(temp_user_data_dir, exist_ok=True)
        shutil.copy2(local_state_source, os.path.join(temp_user_data_dir, 'Local State'))

    print(
        f'🧪 [ORCRIMS] Perfil temporário do Chrome preparado: copiados={copied_files} | '
        f'ignorados={skipped_files} | perfil={profile_name}'
    )
    if skipped_errors:
        print(f"⚠️ [ORCRIMS] Arquivos do perfil ignorados por trava: {'; '.join(skipped_errors)}")

    return temp_root, temp_user_data_dir


def _wait_for_devtools_endpoint(port: int, timeout_seconds: int = 20):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version', timeout=2) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as error:
            last_error = error
            time.sleep(0.5)
    raise ChromeProfileCookieError(f'DevTools do Chrome headless nÃ£o respondeu na porta {port}: {last_error}')


def _list_devtools_targets(port: int):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=5) as response:
        return json.loads(response.read().decode('utf-8'))


def _cdp_call(ws, method: str, params: dict | None = None, call_id: int = 1):
    ws.send(json.dumps({'id': call_id, 'method': method, 'params': params or {}}))
    while True:
        payload = json.loads(ws.recv())
        if payload.get('id') == call_id:
            if payload.get('error'):
                raise ChromeProfileCookieError(f'CDP {method} falhou: {payload["error"]}')
            return payload.get('result', {})


def _extract_google_cookies_from_cdp_target(ws, warmup_url: str | None = None):
    try:
        _cdp_call(ws, 'Page.enable')
    except Exception:
        pass

    if warmup_url:
        try:
            _cdp_call(ws, 'Page.navigate', {'url': warmup_url})
            time.sleep(2)
        except Exception:
            pass

    try:
        cookie_payload = _cdp_call(ws, 'Storage.getCookies')
    except Exception:
        cookie_payload = _cdp_call(ws, 'Network.getCookies')

    cookies = [
        {
            'host_key': item.get('domain', ''),
            'name': item.get('name', ''),
            'value': item.get('value', ''),
            'path': item.get('path', '/') or '/',
            'secure': bool(item.get('secure', False)),
        }
        for item in cookie_payload.get('cookies', [])
        if 'google.' in (item.get('domain', '') or '')
    ]
    return cookies


def _load_google_cookies_via_browser_cookie3(profile_name: str):
    if browser_cookie3 is None:
        raise ChromeProfileCookieError('browser-cookie3 não está disponível neste ambiente.')

    user_data_dir = _find_chrome_user_data_dir()
    if not user_data_dir:
        raise ChromeProfileCookieError('Diretório User Data do Chrome não encontrado.')

    cookie_file = os.path.join(user_data_dir, profile_name, 'Network', 'Cookies')
    if not os.path.exists(cookie_file):
        cookie_file = os.path.join(user_data_dir, profile_name, 'Cookies')
    if not os.path.exists(cookie_file):
        raise ChromeProfileCookieError(f'Banco de cookies do perfil {profile_name} não encontrado.')

    try:
        cookie_jar = browser_cookie3.chrome(cookie_file=cookie_file, domain_name='google.com')
    except Exception as error:
        raise ChromeProfileCookieError(f'browser-cookie3 falhou ao ler o perfil {profile_name}: {error}') from error

    cookies = []
    for item in cookie_jar:
        if 'google.' not in (item.domain or '') or not item.value:
            continue
        cookies.append({
            'host_key': item.domain,
            'name': item.name,
            'value': item.value,
            'path': item.path or '/',
            'secure': bool(item.secure),
        })
    if not cookies:
        raise ChromeProfileCookieError(f'browser-cookie3 não expôs cookies Google do perfil {profile_name}.')
    print(f"🔐 [ORCRIMS] Cookies Google lidos via browser-cookie3 ({profile_name}): {len(cookies)} entradas.")
    return cookies


def _load_google_cookies_via_headless_chrome(profile_name: str, warmup_url: str | None = None):
    chrome_executable = _find_chrome_executable()
    if not chrome_executable:
        raise ChromeProfileCookieError('Google Chrome nÃ£o encontrado para fallback headless.')

    temp_root, temp_user_data_dir = _copy_minimal_chrome_profile(profile_name)
    port = _get_free_tcp_port()
    process = None
    ws = None
    try:
        command = [
            chrome_executable,
            f'--user-data-dir={temp_user_data_dir}',
            f'--profile-directory={profile_name}',
            f'--remote-debugging-port={port}',
            '--remote-allow-origins=*',
            '--headless=new',
            '--disable-gpu',
            '--no-first-run',
            '--no-default-browser-check',
            'about:blank',
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        version_payload = _wait_for_devtools_endpoint(port)
        ws = websocket.create_connection(version_payload['webSocketDebuggerUrl'], timeout=10)
        cookies = _extract_google_cookies_from_cdp_target(ws, warmup_url=warmup_url or 'https://www.google.com/maps/')
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        shutil.rmtree(temp_root, ignore_errors=True)

    if not cookies:
        raise ChromeProfileCookieError(f'Chrome headless nÃ£o expÃ´s cookies Google do perfil {profile_name}.')
    print(f"🔐 [ORCRIMS] Cookies Google expostos via Chrome headless ({profile_name}): {len(cookies)} entradas.")
    return cookies


def _chrome_debug_port() -> int:
    raw = os.environ.get('ORCRIMS_CHROME_DEBUG_PORT') or _load_env_for_cookies().get('ORCRIMS_CHROME_DEBUG_PORT') or ''
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return 9222


def _probe_devtools_endpoint(port: int):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version', timeout=2) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return None


def _pick_cdp_target_ws(port: int) -> str:
    try:
        targets = _list_devtools_targets(port)
    except Exception:
        targets = []
    page_target = next(
        (item for item in targets if item.get('type') == 'page' and item.get('webSocketDebuggerUrl')),
        None,
    )
    if page_target:
        return page_target['webSocketDebuggerUrl']
    any_target = next((item for item in targets if item.get('webSocketDebuggerUrl')), None)
    if any_target:
        return any_target['webSocketDebuggerUrl']
    version_payload = _probe_devtools_endpoint(port) or {}
    if version_payload.get('webSocketDebuggerUrl'):
        return version_payload['webSocketDebuggerUrl']
    raise ChromeProfileCookieError(f'Nenhum alvo CDP com WebSocket exposto na porta {port}.')


def _get_chrome_pids() -> list:
    """Retorna lista de PIDs do Chrome em execução."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq chrome.exe', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=10,
        )
        pids = []
        for line in result.stdout.splitlines():
            parts = line.strip().strip('"').split('","')
            if len(parts) >= 2:
                try:
                    pids.append(int(parts[1]))
                except ValueError:
                    pass
        return pids
    except Exception:
        return []


def _close_chrome_gracefully(timeout_seconds: int = 10) -> list:
    """Fecha todas as instâncias do Chrome graciosamente. Retorna PIDs que foram fechados."""
    pids = _get_chrome_pids()
    if not pids:
        return []
    print(f'🔄 [ORCRIMS] Fechando Chrome temporariamente para captura de cookies ({len(pids)} processos)...')
    # TASKKILL /IM envia WM_CLOSE (gracioso), não força
    subprocess.run(['taskkill', '/IM', 'chrome.exe'], capture_output=True, timeout=10)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _get_chrome_pids():
            break
        time.sleep(0.5)
    # Se ainda restam, força
    remaining = _get_chrome_pids()
    if remaining:
        subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], capture_output=True, timeout=10)
        time.sleep(1)
    print('✅ [ORCRIMS] Chrome fechado.')
    return pids


def _reopen_chrome(chrome_executable: str, user_data_dir: str, profile_name: str):
    """Reabre o Chrome normalmente (restaura abas anteriores)."""
    try:
        subprocess.Popen(
            [chrome_executable, f'--user-data-dir={user_data_dir}', f'--profile-directory={profile_name}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print('🌐 [ORCRIMS] Chrome reaberto normalmente.')
    except Exception as e:
        print(f'⚠️ [ORCRIMS] Não foi possível reabrir o Chrome: {e}')


def _load_google_cookies_via_live_chrome_session(source_url: str, profile_name: str):
    port = _chrome_debug_port()

    # 1) Sessão viva já exposta com --remote-debugging-port: lemos direto.
    if _probe_devtools_endpoint(port):
        ws = None
        try:
            ws = websocket.create_connection(_pick_cdp_target_ws(port), timeout=10)
            cookies = _extract_google_cookies_from_cdp_target(ws)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        if cookies:
            print(f"🔐 [ORCRIMS] Cookies capturados da sessão viva já ativa (porta {port}): {len(cookies)} entradas.")
            return cookies
        print(f'⚠️ [ORCRIMS] Porta de debug {port} ativa, mas sem cookies Google expostos.')

    chrome_executable = _find_chrome_executable()
    user_data_dir = _find_chrome_user_data_dir()
    if not chrome_executable or not user_data_dir:
        raise ChromeProfileCookieError('Chrome ou User Data não disponível para captura de sessão viva.')

    # 2) Chrome está aberto sem debug port: fechar graciosamente, subir com debug,
    #    capturar cookies, fechar e reabrir normalmente.
    chrome_was_open = bool(_get_chrome_pids())
    if chrome_was_open:
        _close_chrome_gracefully()
        time.sleep(1)

    process = None
    ws = None
    try:
        command = [
            chrome_executable,
            f'--user-data-dir={user_data_dir}',
            f'--profile-directory={profile_name}',
            f'--remote-debugging-port={port}',
            '--remote-allow-origins=*',
            '--headless=new',
            '--disable-gpu',
            '--no-first-run',
            '--no-default-browser-check',
            'about:blank',
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        version_payload = _wait_for_devtools_endpoint(port, timeout_seconds=20)
        ws = websocket.create_connection(version_payload['webSocketDebuggerUrl'], timeout=10)
        cookies = _extract_google_cookies_from_cdp_target(ws, warmup_url='https://www.google.com/maps/')
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        if chrome_was_open:
            _reopen_chrome(chrome_executable, user_data_dir, profile_name)

    if not cookies:
        raise ChromeProfileCookieError(f'Chrome não expôs cookies Google do perfil {profile_name}.')
    print(f"🔐 [ORCRIMS] Cookies Google capturados via Chrome com debug port ({profile_name}): {len(cookies)} entradas.")
    return cookies


def _build_google_session_from_chrome_profile(source_url: str, profile_name: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': source_url.replace('/kml?', '/viewer?'),
    })

    try:
        cookies = _load_google_cookies_via_live_chrome_session(source_url, profile_name)
    except Exception as live_error:
        print(f'⚠️ [ORCRIMS] Captura via Chrome falhou ({live_error}). Tentando leitura direta do perfil...')
        try:
            cookies = _load_google_cookies_from_chrome_profile(profile_name)
        except Exception as cookie_error:
            print(f'⚠️ [ORCRIMS] Leitura direta do banco de cookies falhou ({cookie_error}). Tentando browser-cookie3...')
            cookies = _load_google_cookies_via_browser_cookie3(profile_name)
    for cookie in cookies:
        domain = cookie['host_key'] if cookie['host_key'].startswith('.') else cookie['host_key']
        session.cookies.set(
            cookie['name'],
            cookie['value'],
            domain=domain,
            path=cookie['path'],
            secure=cookie['secure'],
        )

    print(f"🔐 [ORCRIMS] Cookies Google carregados do perfil do Chrome ({profile_name}): {len(cookies)} entradas.")
    return session


def _download_official_payload_with_session(source_url: str, session: requests.Session, download_label: str):
    print(f'⬇️ [ORCRIMS] Baixando KML oficial via {download_label}: {source_url}')
    response = session.get(source_url, timeout=REQUEST_TIMEOUT)
    _raise_for_google_auth_failure(response, source_url)
    response.raise_for_status()
    content = response.content
    print(
        f"✅ [ORCRIMS] Download concluÃ­do via {download_label}: status={response.status_code} | bytes={len(content)} | "
        f"etag={response.headers.get('ETag', 'N/A')} | last-modified={response.headers.get('Last-Modified', 'N/A')}"
    )
    return content, {
        'etag': response.headers.get('ETag', ''),
        'last_modified': response.headers.get('Last-Modified', ''),
        'content_length': response.headers.get('Content-Length', ''),
        'content_type': response.headers.get('Content-Type', ''),
        'final_url': response.url or source_url,
        'redirect_chain': [item.headers.get('Location', '') for item in response.history],
        'downloaded_via': download_label,
    }


def _download_payload_via_chrome_profile_cookies(source_url: str):
    profile_name = os.environ.get('ORCRIMS_CHROME_PROFILE') or _load_env_for_cookies().get('ORCRIMS_CHROME_PROFILE') or 'Default'
    session = _build_google_session_from_chrome_profile(source_url, profile_name)
    payload, headers = _download_official_payload_with_session(source_url, session, f'chrome_profile_cookies:{profile_name}')
    headers['profile'] = profile_name
    return payload, headers


def _download_payload_via_logged_in_chrome(source_url: str):
    chrome_executable = _find_chrome_executable()
    if not chrome_executable:
        raise RuntimeError('Google Chrome não encontrado para fallback autenticado.')

    profile_name = os.environ.get('ORCRIMS_CHROME_PROFILE') or _load_env_for_cookies().get('ORCRIMS_CHROME_PROFILE') or 'Default'
    user_data_dir = _find_chrome_user_data_dir()
    _prepare_chrome_profile_for_auto_download(profile_name)
    started_at = time.time()
    print(f'🌐 [ORCRIMS] Tentando exportar via Chrome logado (perfil={profile_name})...')
    command = [
        chrome_executable,
        f'--profile-directory={profile_name}',
        '--new-window',
        source_url,
    ]
    if user_data_dir:
        command.insert(1, f'--user-data-dir={user_data_dir}')
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    session = _build_google_session(source_url)
    return _download_official_payload_with_session(source_url, session, 'env_cookie')


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


def _load_kml_bytes_from_local_file() -> bytes | None:
    """
    Retorna os bytes KML do arquivo copiado manualmente em INTEL_DIR,
    se for mais recente que current_orcrim.kml. Tenta KML direto e depois KMZ.
    """
    candidates = [
        (LOCAL_KML_PATH, False),
        (LOCAL_KMZ_PATH, True),
    ]
    current_mtime = os.path.getmtime(CURRENT_KML_PATH) if os.path.exists(CURRENT_KML_PATH) else 0.0

    for path, is_kmz in candidates:
        if not os.path.exists(path):
            continue
        if os.path.getmtime(path) <= current_mtime:
            print(f'ℹ️ [ORCRIMS] Arquivo local {os.path.basename(path)} não é mais recente que current_orcrim.kml — ignorado.')
            continue
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            if is_kmz:
                with zipfile.ZipFile(io.BytesIO(raw), 'r') as zf:
                    kml_names = [n for n in zf.namelist() if n.lower().endswith('.kml')]
                    if not kml_names:
                        continue
                    kml_bytes = zf.read(kml_names[0])
            else:
                kml_bytes = raw
            print('\U0001f4c2 [ORCRIMS] Fallback local: usando {} ({} bytes)'.format(os.path.basename(path), len(kml_bytes)))
            return kml_bytes
        except Exception as e:
            print('⚠️ [ORCRIMS] Falha ao ler arquivo local {}: {}'.format(os.path.basename(path), e))
    return None


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

    # Estratégia: a fonte de verdade é a sessão viva do Chrome logado. Pegamos os
    # cookies daquela sessão, montamos um requests.Session() e baixamos o KMZ por HTTP.
    # O cookie estático do .env vira fallback; o download visual do navegador é o
    # último recurso e fica desligado por padrão.
    payload_bytes = headers = kml_bytes = None
    download_failures = []
    saw_auth_error = False

    if _env_flag('ORCRIMS_USE_CHROME_FALLBACK', default=True):
        try:
            payload_bytes, headers = _download_payload_via_chrome_profile_cookies(source_url)
            kml_bytes = _extract_kml_bytes_from_payload(payload_bytes)
        except Exception as chrome_error:
            saw_auth_error = saw_auth_error or isinstance(chrome_error, GoogleMapsAuthError)
            download_failures.append(f'chrome_profile_cookies: {chrome_error}')
            print(f'⚠️ [ORCRIMS] Sessão viva do Chrome não resolveu ({chrome_error}). Tentando cookie do .env...')

    if kml_bytes is None:
        try:
            payload_bytes, headers = _download_official_payload(source_url)
            kml_bytes = _extract_kml_bytes_from_payload(payload_bytes)
        except Exception as env_error:
            saw_auth_error = saw_auth_error or isinstance(env_error, GoogleMapsAuthError)
            download_failures.append(f'env_cookie: {env_error}')

    if kml_bytes is None and _env_flag('ORCRIMS_USE_CHROME_UI_FALLBACK', default=False):
        print('⚠️ [ORCRIMS] Cookies não resolveram. Último recurso: abrir Chrome logado para download visual.')
        try:
            payload_bytes, headers = _download_payload_via_logged_in_chrome(source_url)
            kml_bytes = _extract_kml_bytes_from_payload(payload_bytes)
        except Exception as ui_error:
            download_failures.append(f'chrome_profile_ui: {ui_error}')

    if kml_bytes is None:
        combined_error = ' | '.join(download_failures)
        print(f'❌ [ORCRIMS] Falha ao baixar KML oficial: {combined_error}')
        # Tenta arquivo copiado manualmente na pasta inteligencia
        kml_bytes = _load_kml_bytes_from_local_file()
        if kml_bytes is not None:
            headers = {'downloaded_via': 'local_file_fallback'}
        else:
            if fallback_available:
                status = 'fallback_active'
            elif saw_auth_error:
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
