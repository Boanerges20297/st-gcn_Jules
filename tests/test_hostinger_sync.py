import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.hostinger_sync import HostingerSyncManager


class HostingerSyncTests(unittest.TestCase):
    def test_risk_fingerprint_ignores_runtime_metadata(self):
        artifact_a = {
            'generated_at': '2026-05-21T10:00:00',
            'artifacts': {
                'latest_json': 'outputs/hermes/risk_snapshot_latest.json',
            },
            'data_limit': '2026-05-20',
            'status_enriquecido_14d': {
                'reference_date': '2026-05-20',
                'row_count': 42,
                'window_days': 14,
            },
            'rankings': {
                'fortaleza_bairros_top30': [
                    {'rank': 1, 'name': 'ALDEOTA', 'risk_score': 91.2},
                ],
            },
        }
        artifact_b = {
            **artifact_a,
            'generated_at': '2026-05-21T10:05:00',
            'artifacts': {
                'history_json': 'outputs/hermes/history/risk_snapshot_20260521_100500.json',
            },
        }

        fingerprint_a = HostingerSyncManager.build_risk_fingerprint(artifact_a)
        fingerprint_b = HostingerSyncManager.build_risk_fingerprint(artifact_b)

        self.assertEqual(fingerprint_a, fingerprint_b)

    def test_data_merge_sync_skips_when_files_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / 'logs').mkdir(parents=True, exist_ok=True)
            (project_root / 'data' / 'raw').mkdir(parents=True, exist_ok=True)

            (project_root / 'data' / 'raw' / 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv').write_text(
                'id,data\n1,2026-05-20\n',
                encoding='utf-8',
            )
            (project_root / 'data' / 'geo_streets_cache.json').write_text(
                json.dumps([{'rua': 'RUA TESTE'}]),
                encoding='utf-8',
            )
            (project_root / 'VALIDATION_LOG.md').write_text('ok\n', encoding='utf-8')

            env = {
                'HOSTINGER_SYNC_ENABLED': 'true',
                'HOSTINGER_SYNC_HOST': 'example.com',
                'HOSTINGER_SYNC_USER': 'reportpreview',
                'HOSTINGER_SYNC_PASSWORD': 'secret',
            }
            with patch.dict('os.environ', env, clear=False):
                manager = HostingerSyncManager(str(project_root))
                with patch.object(manager, '_upload_relative_files', return_value=['data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv']) as upload_mock:
                    first_result = manager.sync_data_merge_artifacts()
                    second_result = manager.sync_data_merge_artifacts()

            self.assertEqual(first_result['status'], 'synced')
            self.assertEqual(second_result['status'], 'skipped')
            self.assertEqual(upload_mock.call_count, 1)


if __name__ == '__main__':
    unittest.main()