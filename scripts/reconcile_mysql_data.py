"""Create a source-verified reconciliation candidate without replacing the CSV.

An occurrence can have different victims with different MySQL IDs and even
different id_evento values. Never deduplicate by location, event or victim name.
Only representations verified against the current MySQL snapshot are combined.
Names stay in memory for checks and are not written to the candidate or report.
"""

import argparse
from collections import Counter, defaultdict
import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unicodedata

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.core.occurrence_integrity import (
    InvalidOccurrenceId, deduplicate_mysql_rows, normalize_mysql_id,
    sanitize_death_count, split_mysql_ids,
)

PRIVATE_COLUMNS = {'nome_vitima', 'vitima'}
ENRICHMENT_COLUMNS = {
    'bairro', 'ais', 'name', 'dia_semana', 'eh_feriado', 'dia_quente_cvp',
    'clima', 'precipitacao_mm', 'ais_original', 'regiao_risp',
}


def text(value):
    if value is None or pd.isna(value):
        return ''
    return unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().upper().strip()


def row_ids(value):
    try:
        return split_mysql_ids(value)
    except InvalidOccurrenceId:
        return ()


def event_id(value):
    try:
        return normalize_mysql_id(value)
    except InvalidOccurrenceId:
        return None


def seconds(value):
    if isinstance(value, datetime.timedelta):
        return int(value.total_seconds())
    try:
        parts = str(value).strip().split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    except (ValueError, IndexError):
        return None


def event_signature(row):
    return (event_id(row.get('id_evento')), str(row.get('data'))[:10],
            seconds(row.get('hora')), text(row.get('tipo')))


def coordinate(value):
    try:
        number = Decimal(str(value))
        return number.quantize(Decimal('.00001')) if number.is_finite() else None
    except InvalidOperation:
        return None


def physical_signature(row):
    return (str(row.get('data'))[:10], seconds(row.get('hora')),
            text(row.get('tipo')), text(row.get('cidade')),
            coordinate(row.get('latitude')), coordinate(row.get('longitude')))


def serial(value):
    if isinstance(value, datetime.timedelta):
        total = int(value.total_seconds())
        return f'{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}'
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def reconcile(local, source_rows):
    """Return candidate and audit; ambiguous local rows remain independent.

    Grouped rows are expanded only when every original ID and their common
    place/time agree with the source. For correcting a lost multi-victim count,
    distinct nonempty names also corroborate the victims. Equal names alone
    are never grounds for removing a MySQL row.
    """
    by_event = defaultdict(list)
    by_id = {}
    for source in source_rows:
        source_id = normalize_mysql_id(source['id'])
        if not source_id or source_id in by_id:
            raise ValueError('Source snapshot must have unique, nonempty MySQL IDs')
        by_id[source_id] = source
        by_event[event_signature(source)].append(source)

    stats = Counter()
    unresolved = []
    candidates = {}
    independent = []
    private = [column for column in local.columns if column in PRIVATE_COLUMNS]
    columns = [column for column in local.columns if column not in PRIVATE_COLUMNS]
    if 'qtd_mortes' not in columns:
        columns.append('qtd_mortes')

    def keep_unresolved(position, row, reason):
        stats[reason] += 1
        unresolved.append({'row_position': position, 'mysql_ids': list(row_ids(row.get('id'))),
                           'id_evento': event_id(row.get('id_evento')), 'reason': reason})
        independent.append({key: value for key, value in row.items() if key not in PRIVATE_COLUMNS})

    for position, row in enumerate(local.to_dict('records')):
        matches = by_event.get(event_signature(row), [])
        if len(matches) != 1:
            keep_unresolved(position, row, 'no_unique_source_signature')
            continue
        anchor = matches[0]
        ids = row_ids(row.get('id'))
        if not ids:
            keep_unresolved(position, row, 'missing_or_invalid_local_id')
            continue
        deaths = sanitize_death_count(row.get('qtd_mortes'), row.get('tipo'))
        represented = [anchor]
        if len(ids) > 1:
            represented = [by_id.get(row_id) for row_id in ids]
            if (
                any(item is None for item in represented)
                or normalize_mysql_id(anchor['id']) not in ids
                or any(physical_signature(item) != physical_signature(anchor) for item in represented)
            ):
                keep_unresolved(position, row, 'grouped_ids_not_verified_by_source')
                continue
            names = [text(item.get('nome_vitima')) for item in represented]
            if text(row.get('tipo')) != 'CVLI' or not all(names) or len(set(names)) != len(names):
                keep_unresolved(position, row, 'grouped_victims_require_individual_review')
                continue
            if deaths not in (1, len(represented)):
                keep_unresolved(position, row, 'explicit_quantity_conflicts_with_source_victims')
                continue
            stats['verified_grouped_rows_expanded'] += 1
            if deaths == 1:
                stats['lost_multi_victim_counts_recovered'] += 1

        for source in represented:
            source_id = normalize_mysql_id(source['id'])
            result = {key: value for key, value in row.items() if key not in PRIVATE_COLUMNS}
            for column, value in source.items():
                if column in columns and column not in ENRICHMENT_COLUMNS and column not in PRIVATE_COLUMNS:
                    result[column] = serial(value)
            result['id'] = source_id
            result['id_evento'] = event_id(source.get('id_evento'))
            result['qtd_mortes'] = 1.0 if len(represented) > 1 else deaths
            if source_id not in ids:
                stats['verified_id_associations_corrected'] += 1
            quality = sum(bool(text(result.get(column))) for column in ENRICHMENT_COLUMNS)
            if source_id in candidates:
                previous, previous_quality = candidates[source_id]
                # One confirmed source row, represented several times locally.
                # Never sum its copies. Preserve the strongest explicit quantity.
                quantity = max(previous['qtd_mortes'], result['qtd_mortes'])
                if quality > previous_quality:
                    result['qtd_mortes'] = quantity
                    candidates[source_id] = (result, quality)
                else:
                    previous['qtd_mortes'] = quantity
                stats['verified_duplicate_representations_removed'] += 1
            else:
                candidates[source_id] = (result, quality)

    candidate = pd.DataFrame(
        [value[0] for value in candidates.values()] + independent, columns=columns,
    )
    report = {
        'source_rows': len(source_rows), 'local_rows': len(local),
        'candidate_rows': len(candidate), 'verified_source_ids': len(candidates),
        'unresolved_rows_preserved': len(independent), 'private_columns_removed': private,
        'actions': dict(stats), 'unresolved': unresolved,
        'official_csv_replaced': False,
    }
    return candidate, report


def read_mysql_snapshot():
    import pymysql
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
    options = {key: os.getenv(env, '').strip().strip('"') for key, env in [
        ('host', 'MYSQL_HOST'), ('user', 'MYSQL_USER'),
        ('password', 'MYSQL_PASSWORD'), ('database', 'MYSQL_DATABASE'),
    ]}
    options.update(port=int(os.getenv('MYSQL_PORT', '3306').strip().strip('"')),
                   charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
                   connect_timeout=8, read_timeout=60, write_timeout=10)
    for attempt in range(3):
        try:
            connection = pymysql.connect(**options)
            try:
                with connection.cursor() as cursor:
                    cursor.execute('START TRANSACTION READ ONLY')
                    cursor.execute('SELECT * FROM dados_status ORDER BY id')
                    source = cursor.fetchall()
                connection.rollback()
                return source
            finally:
                connection.close()
        except pymysql.OperationalError as exc:
            code = exc.args[0] if exc.args else None
            if code not in (2003, 2006, 2013) or attempt == 2:
                raise RuntimeError(f'MySQL snapshot unavailable (code {code})') from None
            print(f'MySQL retry {attempt + 1}: code {code}', flush=True)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    official = BASE_DIR / 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    original_hash = hashlib.sha256(official.read_bytes()).hexdigest()
    local = pd.read_csv(official, dtype={'id': 'string', 'id_evento': 'string'}, low_memory=False)
    candidate, report = reconcile(local, read_mysql_snapshot())
    if hashlib.sha256(official.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError('Official CSV changed during reconciliation; rerun with a fresh snapshot')
    report['official_sha256'] = original_hash
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = args.output_dir / 'candidate.csv'
    candidate.to_csv(candidate_path, index=False, encoding='utf-8')
    verified = candidate.iloc[:report['verified_source_ids']].copy()
    if len(deduplicate_mysql_rows(verified)) != len(verified):
        raise RuntimeError('Verified candidate unexpectedly contains repeated MySQL IDs')
    verified_path = args.output_dir / 'verified.csv'
    verified.to_csv(verified_path, index=False, encoding='utf-8')
    unresolved_path = args.output_dir / 'pending.csv'
    candidate.iloc[report['verified_source_ids']:].to_csv(
        unresolved_path, index=False, encoding='utf-8',
    )
    report['verified_sha256'] = hashlib.sha256(verified_path.read_bytes()).hexdigest()
    report['pending_sha256'] = hashlib.sha256(unresolved_path.read_bytes()).hexdigest()
    report['candidate_is_ready_to_replace_official'] = not report['unresolved_rows_preserved']
    report['candidate_sha256'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    report_path = args.output_dir / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'unresolved'}, indent=2))


if __name__ == '__main__':
    main()
