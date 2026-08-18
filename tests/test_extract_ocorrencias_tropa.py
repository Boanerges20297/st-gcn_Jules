import json
from pathlib import Path
from unittest.mock import patch

from data.raw.scripts.extract_ocorrencias_tropa import ensure_input_file


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload.encode("utf-8")


def test_ensure_input_file_downloads_remote_payload_when_forced(tmp_path, monkeypatch):
    target_path = tmp_path / "ocorrencias_tropa.json"

    monkeypatch.setenv("TROPA_SOURCE_URL", "https://example.test/ocorrencias")
    monkeypatch.setenv("TROPA_DOWNLOAD_LIMIT", "50")

    with patch("data.raw.scripts.extract_ocorrencias_tropa.urlopen", return_value=FakeResponse('{"records": [1]}')) as mocked_urlopen:
        downloaded = ensure_input_file(str(target_path), force_download=True)

    assert downloaded is True
    assert target_path.exists()
    assert json.loads(target_path.read_text(encoding="utf-8"))["records"] == [1]
    assert mocked_urlopen.called


def test_ensure_input_file_downloads_remote_payload_when_env_forces_it(tmp_path, monkeypatch):
    target_path = tmp_path / "ocorrencias_tropa.json"

    monkeypatch.setenv("TROPA_SOURCE_URL", "https://example.test/ocorrencias")
    monkeypatch.setenv("TROPA_FORCE_DOWNLOAD", "1")
    monkeypatch.setenv("TROPA_DOWNLOAD_LIMIT", "50")

    with patch("data.raw.scripts.extract_ocorrencias_tropa.urlopen", return_value=FakeResponse('{"records": [2]}')) as mocked_urlopen:
        downloaded = ensure_input_file(str(target_path))

    assert downloaded is True
    assert target_path.exists()
    assert json.loads(target_path.read_text(encoding="utf-8"))["records"] == [2]
    assert mocked_urlopen.called
