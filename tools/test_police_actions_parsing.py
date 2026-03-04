
import os
import sys
import json
import re

# Adiciona o diretório raiz ao path para importar src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service import process_exogenous_text, _deterministic_parse

def test_parsing():
    test_text = """Ações Policiais em 27/02/2026:
01 - M20260179445 - RAIO01 - ST JEFFERSON - ABANDONO DE MATERIAL (UMA PISTOLA CALIBRE 9mm, MUNIÇÕES, MACONHA E UMA BALANÇA DE PRECISÃO) - BARROSO - 09:03 - SEM PRESO - 6º DP

02 - M20260181459 - VTRA105 - ST FARIAS - VEÍCULO LOCALIZADO - S/PRESO - BR020 - TUCUNDUBA, CAUCAIA (AIS26) - 23ºDP - 11:36
"""
    print("Testing with Deterministic Parse (Fallback)...")
    results = _deterministic_parse(test_text)
    
    print(f"Total events found: {len(results)}")
    for idx, ev in enumerate(results):
        print(f"\nEvent {idx+1}:")
        print(f"  Natureza: {ev.get('natureza')}")
        print(f"  Bairro: {ev.get('bairro')}")
        print(f"  Municipio: {ev.get('municipio')}")
        print(f"  Is Suppression: {ev.get('is_suppression')}")
        print(f"  Date: {ev.get('date')}")
        
        # Validações básicas
        assert ev.get('is_suppression') is True, f"Event {idx+1} should be suppression"
        assert ev.get('date') == '2026-02-27', f"Event {idx+1} date should be 2026-02-27"

    print("\n✅ Deterministic Test Passed!")

    # Se houver chaves de API, testamos o fluxo completo (opcional/debug)
    from src.llm_service import get_gemini_api_keys
    if get_gemini_api_keys():
        print("\nTesting with LLM Parse...")
        try:
            results_llm = process_exogenous_text(test_text)
            print(f"Total events found (LLM): {len(results_llm)}")
            for idx, ev in enumerate(results_llm):
                print(f"Event {idx+1} (LLM) - Is Suppression: {ev.get('is_suppression')}")
                assert ev.get('is_suppression') is True
        except Exception as e:
            print(f"LLM test skipped/failed: {e}")

if __name__ == "__main__":
    try:
        test_parsing()
    except AssertionError as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
