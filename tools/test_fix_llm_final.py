
import os
import sys
import json

# Adiciona o diretório raiz ao sys.path para importar src
sys.path.append(os.getcwd())

from src.llm_service import process_exogenous_text, busca_bairro, busca_municipio

raw_text = "02 - M20260125399 - HOMICIDIO A BALA - VÍTIMA DO SEXO MASCULINO ( NO MESMO LOCAL UMA MULHER LESIONADA NO BRAÇO ) - VIA PÚBLICA - QUINTINO CUNHA - AIS18 - 15:40 ( ACUSADO PRESO PELA DHPP )"

print("--- Teste de Busca Direta ---")
b = busca_bairro(raw_text)
m = busca_municipio(raw_text)
print(f"Bairro detectado: {b}")
print(f"Municipio detectado: {m}")

print("\n--- Teste de Processamento Completo (Simulado/Determinístico) ---")
# Forçamos o modo determinístico para testar a lógica de enriquecimento local
os.environ['DISABLE_GENAI_FOR_TESTS'] = '1'
events = process_exogenous_text(raw_text)

if events:
    ev = events[0]
    print(f"Evento Natureza: {ev.get('natureza')}")
    print(f"Evento Bairro: {ev.get('bairro')}")
    print(f"Evento Municipio: {ev.get('municipio')}")
    print(f"Evento Severidade: {ev.get('conflict_severity')}")
else:
    print("Nenhum evento processado.")
