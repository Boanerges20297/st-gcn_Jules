import os
import sys
import logging
from pathlib import Path

# Mock project root and setup logging
project_root = Path(r"c:\Users\Boanerges\Desktop\Projetos\Report Preview")
logging.basicConfig(level=logging.INFO)

# We will instantiate a minimal mock Gateway class to test our methods
class MockGateway:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

    def _normalize_location(self, name: str) -> str:
        import unicodedata
        # Strip common quotation marks, whitespace, and brackets
        cleaned = str(name).strip("'`\"[](){}<> \t\r\n")
        nfkd_form = unicodedata.normalize('NFKD', cleaned)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).strip().upper()

    def _load_valid_locations(self) -> None:
        if hasattr(self, "_valid_cities_set"):
            return
        
        self._valid_cities_set = set()
        self._valid_neighborhoods_set = set()
        
        path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
        if not path.exists():
            path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
            
        if not path.exists():
            logging.warning("CSV dados_brutos_90dias.csv nao encontrado para carregar locais.")
            return
            
        try:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cidade = r.get("cidade") or ""
                    bairro = r.get("bairro") or ""
                    
                    norm_c = self._normalize_location(cidade)
                    norm_b = self._normalize_location(bairro)
                    
                    if norm_c:
                        self._valid_cities_set.add(norm_c)
                    if norm_b:
                        self._valid_neighborhoods_set.add(norm_b)
            logging.info("Carregados %d cidades e %d bairros validos do CSV 90d.", len(self._valid_cities_set), len(self._valid_neighborhoods_set))
        except Exception as e:
            logging.error("Erro ao carregar locais validos: %s", e)

    def _is_valid_location(self, name: str) -> tuple[bool, str, str]:
        self._load_valid_locations()
        norm = self._normalize_location(name)
        if not norm:
            return False, "", ""
            
        # 1. Exact match city
        if norm in self._valid_cities_set:
            return True, "cidade", norm
        # 2. Exact match neighborhood
        if norm in self._valid_neighborhoods_set:
            return True, "bairro", norm
            
        # 3. User input is a substring of target (e.g. "aldeot" -> "ALDEOTA", "barros" -> "BARROSO")
        if len(norm) >= 3:
            matched_bairros = [b for b in self._valid_neighborhoods_set if norm in b]
            matched_cities = [c for c in self._valid_cities_set if norm in c]
            
            if matched_bairros:
                matched_bairros.sort(key=lambda x: len(x) - len(norm))
                return True, "bairro", matched_bairros[0]
            if matched_cities:
                matched_cities.sort(key=lambda x: len(x) - len(norm))
                return True, "cidade", matched_cities[0]
                
        # 4. Target is a full word in user input (e.g. "bairro do barroso" -> "BARROSO")
        if len(norm) >= 3:
            import re
            for b in sorted(self._valid_neighborhoods_set, key=len, reverse=True):
                pattern = r'\b' + re.escape(b) + r'\b'
                if re.search(pattern, norm):
                    return True, "bairro", b
                    
            for c in sorted(self._valid_cities_set, key=len, reverse=True):
                pattern = r'\b' + re.escape(c) + r'\b'
                if re.search(pattern, norm):
                    return True, "cidade", c
                    
        return False, "", ""

def main():
    gateway = MockGateway(project_root)
    
    # Test cases to run
    test_cases = [
        # User input, expected type, expected matched name
        ("'Barroso'", "bairro", "BARROSO"),
        ("Barroso", "bairro", "BARROSO"),
        ("'Barro'", "cidade", "BARRO"),
        ("Barro", "cidade", "BARRO"),
        ("Aldeota", "bairro", "ALDEOTA"),
        ("Aldeot", "bairro", "ALDEOTA"),
        ("Fortaleza", "cidade", "FORTALEZA"),
        ("Municipio de Caucaia", "cidade", "CAUCAIA"),
        ("Bairro do Barroso", "bairro", "BARROSO"),
        ("Invalido123", "", "")
    ]
    
    print("\n--- INICIANDO TESTES DE VALIDAÇÃO DE LOCALIZAÇÃO ---\n")
    
    success_count = 0
    for text, expected_type, expected_name in test_cases:
        is_valid, m_type, m_name = gateway._is_valid_location(text)
        
        # If expected is invalid
        if expected_type == "":
            passed = (not is_valid)
        else:
            passed = is_valid and (m_type == expected_type) and (m_name == expected_name)
            
        status = "PASSED" if passed else "FAILED"
        if passed:
            success_count += 1
            print(f"[OK] [{status}] Input: {text:25} => Matched: {m_type:8} - {m_name}")
        else:
            print(f"[FAIL] [{status}] Input: {text:25} => Expected: {expected_type} - {expected_name} | Got: {m_type} - {m_name}")
            
    print(f"\nResultado final: {success_count}/{len(test_cases)} testes aprovados.")

if __name__ == "__main__":
    main()
