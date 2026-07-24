import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_static_snapshot import _write_json


class StaticSnapshotWriterTest(unittest.TestCase):
    def test_replaces_json_without_exposing_partial_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'explainability.json'
            _write_json(path, {'version': 1})
            _write_json(path, {'version': 2, 'items': ['fortaleza']})
            self.assertEqual({'version': 2, 'items': ['fortaleza']}, json.loads(path.read_text(encoding='utf-8')))
            self.assertEqual([], list(Path(directory).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
