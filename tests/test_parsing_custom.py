import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.llm_service import _mock_response

class TestParsingCustom(unittest.TestCase):
    def test_mock_parsing_heuristic(self):
        # User example: M18838999-(tipo ocorrência)- descrição-local-data hora
        # Actually usually starts with Index "01 - "
        # Let's test the format the user likely sees: "01 - M... - TYPE - DESC - LOC"

        raw_text = "01 - M18838999 - ROUBO DE VEICULO - TENTATIVA DE ROUBO NA RUA A - RUA A, FORTALEZA - 10/10/2023"
        events = _mock_response(raw_text)

        self.assertEqual(len(events), 1)
        evt = events[0]

        # We expect Nature to be the 3rd element (index 2)
        self.assertEqual(evt['natureza'], "ROUBO DE VEICULO")
        # Location might be 5th element (index 4) if present
        self.assertEqual(evt['localizacao_completa'], "RUA A, FORTALEZA")

    def test_mock_parsing_variations(self):
        # Shorter format
        raw_text = "02 - M999 - HOMICIDIO - LOCAL X"
        events = _mock_response(raw_text)
        self.assertEqual(events[0]['natureza'], "HOMICIDIO")
        self.assertEqual(events[0]['localizacao_completa'], "LOCAL X")

if __name__ == '__main__':
    unittest.main()
