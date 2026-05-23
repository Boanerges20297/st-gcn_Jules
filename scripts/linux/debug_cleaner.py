import sys
from pathlib import Path

# Add the current directory to sys.path to be able to import
sys.path.append(str(Path(__file__).parent))
from ask_gemini_with_mempalace import clean_response_pollution

text = "A capital do Ceará é Fortaleza, onde a Aerolândia lidera o risco crítico (81.8) e a projeção tática para os próximos 7 dias exige foco em roubos a pessoas."
query = "mas me parece que voce esta avaliando apenas o passado e buscando o futuro"

cleaned = clean_response_pollution(text, query)
print("Text:", text)
print("Cleaned:", cleaned)
print("Matched?", cleaned != text)
