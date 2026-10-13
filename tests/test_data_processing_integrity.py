import datetime
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.core import data_processing as processing
from src.core.occurrence_integrity import OccurrenceIdentityConflict


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / 'data' / 'raw'
    raw.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, 'dotenv', SimpleNamespace(load_dotenv=lambda: None))
    for name, value in {
        'MYSQL_HOST': 'fake.test', 'MYSQL_PORT': '3306', 'MYSQL_USER': 'test',
        'MYSQL_PASSWORD': 'test', 'MYSQL_DATABASE': 'test',
    }.items():
        monkeypatch.setenv(name, value)
    return raw


def fake_database(monkeypatch, ids, rows):
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    cursor.fetchall.side_effect = [[{'id': value} for value in ids], rows]
    connection = Mock()
    connection.cursor.return_value = cursor
    connect = Mock(return_value=connection)
    monkeypatch.setitem(sys.modules, 'pymysql', SimpleNamespace(
        connect=connect, cursors=SimpleNamespace(DictCursor=object),
    ))
    return cursor, connection, connect


def official(raw, rows):
    target = raw / 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def batch_rows(raw):
    return json.loads((raw / 'dados_status.json').read_text(encoding='utf-8'))[-1]['data']


def test_download_recovers_real_gap_below_grouped_max(workspace, monkeypatch):
    official(workspace, [{'id': '2 | 3', 'qtd_mortes': 2}])
    cursor, connection, _ = fake_database(monkeypatch, [1, 2, 3], [{'id': 1}])
    assert processing.check_and_download_mysql_data() is True
    assert batch_rows(workspace) == [{'id': '1'}]
    assert cursor.execute.call_args_list[1].args[1] == (1,)
    connection.close.assert_called_once()


def test_grouped_ids_are_committed_and_deleted_ids_are_not_gaps(workspace, monkeypatch):
    official(workspace, [{'id': '2.0 | 4', 'qtd_mortes': 2}])
    cursor, _, _ = fake_database(monkeypatch, [2, 4], [])
    assert processing.check_and_download_mysql_data() is False
    assert cursor.execute.call_count == 1
    assert not (workspace / 'dados_status.json').exists()


def test_pending_download_is_not_treated_as_committed_history(workspace, monkeypatch):
    pending = workspace / 'dados_status.json'
    pending.write_text(json.dumps([{'type': 'table', 'data': [{'id': '10'}]}]))
    cursor, _, _ = fake_database(monkeypatch, [9, 10], [{'id': 10}, {'id': 9}])
    assert processing.check_and_download_mysql_data() is True
    assert cursor.execute.call_args_list[1].args[1] == (10, 9)
    assert len(batch_rows(workspace)) == 2


def test_failed_merge_batch_is_downloaded_again(workspace, monkeypatch):
    official(workspace, [{'id': '1'}])
    for _ in range(2):
        cursor, _, _ = fake_database(monkeypatch, [1, 2], [{'id': 2}])
        assert processing.check_and_download_mysql_data() is True
        assert cursor.execute.call_args_list[1].args[1] == (2,)


def test_conflicting_official_ids_stop_before_connecting(workspace, monkeypatch):
    official(workspace, [
        {'id': '42', 'id_evento': '900'}, {'id': '42.0', 'id_evento': '901'},
    ])
    _, _, connect = fake_database(monkeypatch, [], [])
    with pytest.raises(OccurrenceIdentityConflict):
        processing.check_and_download_mysql_data()
    connect.assert_not_called()


def test_incomplete_download_preserves_previous_batch(workspace, monkeypatch):
    pending = workspace / 'dados_status.json'
    pending.write_text('previous pending batch')
    _, connection, _ = fake_database(monkeypatch, [1, 2], [{'id': 2}])
    with pytest.raises(RuntimeError, match='incompleto'):
        processing.check_and_download_mysql_data()
    assert pending.read_text() == 'previous pending batch'
    connection.close.assert_called_once()


def test_mysql_equivalent_copies_are_removed_and_time_is_serialized(workspace, monkeypatch):
    row = {'id': 1, 'hora': datetime.timedelta(hours=13, minutes=4),
           'data': datetime.date(2026, 1, 2), 'idade': None}
    fake_database(monkeypatch, [1], [row, dict(row)])
    assert processing.check_and_download_mysql_data() is True
    assert batch_rows(workspace) == [
        {'id': '1', 'hora': '13:04:00', 'data': '2026-01-02', 'idade': None},
    ]


def test_atomic_write_failure_keeps_previous_batch_and_cleans_temp(workspace, monkeypatch):
    pending = workspace / 'dados_status.json'
    pending.write_text('previous')
    monkeypatch.setattr(processing.os, 'replace', Mock(side_effect=OSError('locked')))
    with pytest.raises(OSError, match='locked'):
        processing._write_mysql_batch(str(pending), [{'id': '2'}])
    assert pending.read_text() == 'previous'
    assert not list(workspace.glob('*.tmp'))


def test_merge_failure_stops_processing(workspace, monkeypatch):
    scripts = workspace.parents[1] / 'scripts'
    scripts.mkdir()
    (scripts / 'merge_new_data.py').touch()
    monkeypatch.setattr(processing, 'check_and_download_mysql_data', lambda: True)
    run = Mock(side_effect=subprocess.CalledProcessError(1, ['merge']))
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        processing.process_ism_data(sync_mysql=True)
    assert run.call_args.kwargs['check'] is True
    assert not (workspace.parents[1] / 'data' / 'processed').exists()


def test_missing_merge_script_stops_processing(workspace, monkeypatch):
    monkeypatch.setattr(processing, 'check_and_download_mysql_data', lambda: True)
    with pytest.raises(FileNotFoundError, match='merge_new_data'):
        processing.process_ism_data(sync_mysql=True)


def test_successful_merge_returns_after_checked_subprocess(workspace, monkeypatch):
    scripts = workspace.parents[1] / 'scripts'
    scripts.mkdir()
    (scripts / 'merge_new_data.py').touch()
    monkeypatch.setattr(processing, 'check_and_download_mysql_data', lambda: True)
    run = Mock(return_value=subprocess.CompletedProcess(['merge'], 0))
    monkeypatch.setattr(subprocess, 'run', run)
    processing.process_ism_data(sync_mysql=True)
    run.assert_called_once()
    assert run.call_args.kwargs['check'] is True


def test_rebuild_csv_conflict_stops_before_tensor_or_cache_updates(workspace, monkeypatch):
    path = official(workspace, [
        {'id': '42', 'id_evento': '900', 'data': '2026-01-01', 'tipo': 'cvli'},
        {'id': '42.0', 'id_evento': '901', 'data': '2026-01-02', 'tipo': 'cvli'},
    ])
    monkeypatch.setattr(processing, 'OCORRENCIAS_FILE', str(path))
    cache = Mock()
    monkeypatch.setattr(processing, 'update_geo_streets_cache', cache)
    with pytest.raises(OccurrenceIdentityConflict):
        processing.process_ism_data(update_geo_cache=True)
    cache.assert_not_called()
    assert not (workspace.parents[1] / 'data' / 'processed').exists()


def test_rebuild_json_conflict_stops_before_tensor_updates(workspace, monkeypatch):
    path = workspace / 'enriched.json'
    path.write_text(json.dumps([
        {'id': '42', 'id_evento': '900', 'data': '2026-01-01', 'tipo': 'cvli'},
        {'id': '42.0', 'id_evento': '901', 'data': '2026-01-02', 'tipo': 'cvli'},
    ]))
    monkeypatch.setattr(processing, 'OCORRENCIAS_FILE', str(path))
    with pytest.raises(OccurrenceIdentityConflict):
        processing.process_ism_data()


def test_existing_cvli_quantity_is_preserved():
    assert processing.sanitize_qtd_mortes(3, 'cvli') == 3
    assert processing.sanitize_qtd_mortes(None, 'cvli') == 1


def test_rebuild_removes_copy_before_counting_cvli(workspace, monkeypatch):
    row = {'id': '42', 'id_evento': '900', 'data': '2026-01-01',
           'tipo': 'cvli', 'cidade': 'FORTALEZA', 'bairro': 'CENTRO',
           'qtd_mortes': 3}
    path = official(workspace, [row, dict(row, id='42.0')])
    monkeypatch.setattr(processing, 'OCORRENCIAS_FILE', str(path))
    captured = []

    class StopBeforeBuildingTensors(Exception):
        pass

    def capture(frame):
        captured.append(frame)
        raise StopBeforeBuildingTensors

    monkeypatch.setattr(processing, 'update_geo_streets_cache', capture)
    with pytest.raises(StopBeforeBuildingTensors):
        processing.process_ism_data(update_geo_cache=True)
    assert len(captured[0]) == 1
    assert captured[0]['qtd_mortes'].sum() == 3


def test_missing_ids_are_fetched_in_bounded_batches(workspace, monkeypatch):
    ids = list(range(1, 1002))
    cursor, connection, _ = fake_database(monkeypatch, ids, [])
    cursor.fetchall.side_effect = [
        [{'id': value} for value in ids],
        [{'id': value} for value in range(1001, 1, -1)],
        [{'id': 1}],
    ]
    assert processing.check_and_download_mysql_data() is True
    assert len(cursor.execute.call_args_list[1].args[1]) == 1000
    assert cursor.execute.call_args_list[2].args[1] == (1,)
    assert len(batch_rows(workspace)) == 1001
    connection.close.assert_called_once()
