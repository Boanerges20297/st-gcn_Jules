
import os
import time
import logging
from dotenv import load_dotenv

# Configuração de log para ver o que acontece no terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-Test")

load_dotenv()

try:
    import google.generativeai as genai
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ ERRO: GEMINI_API_KEY não encontrada no ambiente.")
    else:
        print(f"🔑 Chave encontrada (início): {api_key[:5]}...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        print("🚀 Enviando prompt de teste para o Gemini...")
        start = time.time()
        
        response = model.generate_content("Diga 'Conectado' se você estiver funcionando.")
        
        end = time.time()
        print(f"✅ Resposta recebida: {response.text}")
        print(f"⏱️ Tempo de resposta: {end - start:.2f} segundos")

except Exception as e:
    print(f"❌ Falha crítica no teste de LLM: {e}")
