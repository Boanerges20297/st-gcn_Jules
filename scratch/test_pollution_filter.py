import sys
import unicodedata
from pathlib import Path

# Adjust path to find ask_gemini_with_mempalace modules/functions
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "linux"))
from ask_gemini_with_mempalace import is_polluted_or_trivial

def test_filter():
    test_cases = [
        # Polluted / Trivial entries (should return True)
        ("A capital é Fortaleza", True),
        ("A capital e Fortaleza", True),
        ("capital do ceará", True),
        ("Qual a capital do Ceará?", True),
        ("Fortaleza é a capital.", True),
        ("Fortaleza e a capital do Ceara", True),
        ("A capital", True),
        ("capital", True),
        ("abc", True), # Too short
        
        # Valid tactical entries (should return False)
        ("Bairro Mondubim apresentou queda de CVLI contrariando score alto.", False),
        ("Facção Massa unificada no Grande Bom Jardim mudou dinâmica de risco territorial.", False),
        ("Previsão de chuva forte para os próximos 3 dias em Sobral pode impactar deslocamento das equipes.", False),
        ("Bairro Aldeota registra trégua temporária nas ocorrências de roubo a pedestres.", False)
    ]
    
    success = True
    print("Executing pollution filter tests...")
    print("----------------------------------------------------")
    for text, expected in test_cases:
        result = is_polluted_or_trivial(text)
        status = "PASSED" if result == expected else "FAILED"
        print(f"[{status}] Text: '{text}' | Expected: {expected} | Got: {result}")
        if result != expected:
            success = False
            
    print("----------------------------------------------------")
    if success:
        print("ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_filter()
