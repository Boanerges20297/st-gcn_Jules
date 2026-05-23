import re
import unicodedata

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
    
    # Flexible pattern covering all known boilerplate starts in normalized form
    pattern = r"^(como\s+)?a\s+capital\s+(do\s+ceara\s+)?(e\s+|eh\s+)?fortaleza\b([,;.]?\s*onde\b|[,;.]?\s*que\b|[,;.:]?\s*)\s*"
    
    match = re.match(pattern, norm_text)
    if match:
        norm_end_idx = match.end()
        if norm_end_idx > 0:
            if norm_end_idx < len(orig_indices):
                orig_end_idx = orig_indices[norm_end_idx]
            else:
                orig_end_idx = len(text)
                
            stripped = text[orig_end_idx:].strip()
            if stripped:
                return stripped[0].upper() + stripped[1:]
                
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
    print("Executing clean_response_pollution tests...")
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
        print("ALL CLEANER TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME CLEANER TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_cleaner()
