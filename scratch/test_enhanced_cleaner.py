import re
import unicodedata
import sys

def clean_response_pollution(text: str, query: str) -> str:
    if not text:
        return text
        
    query_lower = query.lower()
    if "capital" in query_lower:
        return text
        
    norm_chars = []
    orig_indices = []
    
    for orig_idx, char in enumerate(text):
        decomp = unicodedata.normalize('NFKD', char)
        stripped_decomp = "".join(c for c in decomp if not unicodedata.combining(c))
        for c in stripped_decomp:
            norm_chars.append(c)
            orig_indices.append(orig_idx)
            
    norm_text = "".join(norm_chars).lower()
    
    suffix_pattern = r"([,;.]?\s*(onde\b|que\b|e\s+o\s+nucleo\b|de\s+modo\s+que\b)|[,;.:]?\s*)\s*"
    pattern_std = r"(como\s+)?a\s+capital\s+(do\s+ceara\s+)?(e\s+|eh\s+)?fortaleza\b" + suffix_pattern
    pattern_inv = r"fortaleza\s+(e\s+|eh\s+)?(a\s+)?capital\s*(do\s+ceara\s*)?\b" + suffix_pattern
    
    match = re.search(pattern_std, norm_text)
    if not match:
        match = re.search(pattern_inv, norm_text)
        
    if match:
        norm_start_idx = match.start()
        norm_end_idx = match.end()
        
        orig_start_idx = orig_indices[norm_start_idx]
        
        if norm_end_idx < len(orig_indices):
            orig_end_idx = orig_indices[norm_end_idx]
        else:
            orig_end_idx = len(text)
            
        left_part = text[:orig_start_idx].rstrip()
        right_part = text[orig_end_idx:].lstrip()
        
        if left_part and right_part:
            if left_part.endswith("|") or left_part.endswith(":") or left_part.endswith("—") or left_part.endswith("-"):
                cleaned_right = right_part[0].upper() + right_part[1:] if right_part else ""
                return f"{left_part} {cleaned_right}".strip()
            else:
                cleaned_right = right_part[0].upper() + right_part[1:] if right_part else ""
                return left_part + " " + cleaned_right
        elif right_part:
            return right_part[0].upper() + right_part[1:]
        elif left_part:
            return left_part
            
    return text

def test_cleaner():
    test_cases = [
        # (original_text, query, expected_cleaned_text)
        (
            "A capital do Ceará é Fortaleza, onde a Aerolândia lidera o risco crítico (81.8) e a projeção tática para os próximos 7 dias exige foco em roubos a pessoas.",
            "mas me parece que voce esta avaliando apenas o passado e buscando o futuro",
            "A Aerolândia lidera o risco crítico (81.8) e a projeção tática para os próximos 7 dias exige foco em roubos a pessoas."
        ),
        (
            "A capital do Ceará é Fortaleza; o monitoramento Hermes (15/05/2026) mantém a Aerolândia sob risco crítico (81.8) com 65 ocorrências recentes.",
            "Ranking da região metropolitana por número de cvl, previsão de estouro?",
            "O monitoramento Hermes (15/05/2026) mantém a Aerolândia sob risco crítico (81.8) com 65 ocorrências recentes."
        ),
        (
            "como a capital do ceará é fortaleza, onde a Aerolândia...",
            "qualquer pergunta conceitual",
            "A Aerolândia..."
        ),
        (
            "A capital é Fortaleza. onde Caucaia lidera...",
            "Outra pergunta",
            "Caucaia lidera..."
        ),
        (
            "A capital e Fortaleza, onde Caucaia...",
            "Pergunta",
            "Caucaia..."
        ),
        # Inverted Order
        (
            "Fortaleza é a capital do Ceará, onde a Aerolândia lidera o risco...",
            "Pergunta conceitual",
            "A Aerolândia lidera o risco..."
        ),
        (
            "Fortaleza é a capital, onde Caucaia lidera...",
            "Pergunta",
            "Caucaia lidera..."
        ),
        # Prefix Header
        (
            "Dados ate 15/05/2026 | Fonte: Report Preview | Fortaleza é a capital, onde a Aerolândia lidera...",
            "Pergunta conceitual",
            "Dados ate 15/05/2026 | Fonte: Report Preview | A Aerolândia lidera..."
        ),
        # Prefix with arbitrary text
        (
            "Por que importa/Crítica ao Modelo: A capital do Ceará é Fortaleza, onde a Aerolândia lidera...",
            "Pergunta conceitual",
            "Por que importa/Crítica ao Modelo: A Aerolândia lidera..."
        ),
        # If the user explicitly asks about the capital, do NOT clean!
        (
            "A capital do Ceará é Fortaleza.",
            "Qual a capital do Ceará?",
            "A capital do Ceará é Fortaleza."
        ),
        # Non-matching texts should remain untouched
        (
            "O monitoramento indica risco alto na Caucaia.",
            "Como está o risco?",
            "O monitoramento indica risco alto na Caucaia."
        )
    ]
    
    success = True
    print("Executing ENHANCED clean_response_pollution tests...")
    print("----------------------------------------------------")
    for text, query, expected in test_cases:
        result = clean_response_pollution(text, query)
        status = "PASSED" if result == expected else "FAILED"
        print(f"[{status}] Query: '{query}'")
        print(f"  Raw: '{text}'")
        print(f"  Got: '{result}'")
        print(f"  Exp: '{expected}'")
        if result != expected:
            success = False
            
    print("----------------------------------------------------")
    if success:
        print("ALL ENHANCED CLEANER TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME ENHANCED CLEANER TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_cleaner()
