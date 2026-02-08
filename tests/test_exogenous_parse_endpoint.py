#!/usr/bin/env python
"""
test_exogenous_parse_endpoint.py

Script para diagnosticar erros em POST /api/exogenous/parse

Testa vários payloads para ver qual está falhando
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:5050"
ENDPOINT = "/api/exogenous/parse"

# Test payloads
TEST_CASES = [
    {
        "name": "Payload válido - Evento com localização completa",
        "payload": {
            "text": """
            01 - M20260051350 - LESÃO A BALA - VITIMA DO SEXO MASCULINO - DEU ENTRADA NA UPA DA PAJUÇARA
            - LESIONADO NAS COSTAS - RUA JOÃO HENRIQUE DA SILVA SN - PAJUÇARA, MARACANAÚ - 00:25
            """
        }
    },
    {
        "name": "Payload válido - Evento simples",
        "payload": {
            "text": "HOMICÍDIO A BALA - FORTALEZA - 18:50"
        }
    },
    {
        "name": "Payload vazio",
        "payload": {
            "text": ""
        }
    },
    {
        "name": "Payload null",
        "payload": {
            "text": None
        }
    },
    {
        "name": "Payload JSON inválido",
        "payload": None  # Este causará erro de serialização
    },
    {
        "name": "Sem campo 'text'",
        "payload": {
            "evento": "HOMICÍDIO A BALA"
        }
    },
]

def test_endpoint():
    """Testa endpoint /api/exogenous/parse com vários payloads"""
    
    print("="*80)
    print("TESTANDO ENDPOINT: POST /api/exogenous/parse")
    print("="*80)
    print(f"URL base: {BASE_URL}\n")
    
    results = []
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*80}")
        
        payload = test['payload']
        
        # Display payload
        if payload is not None:
            print(f"\n[PAYLOAD]:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
        else:
            print(f"\n[PAYLOAD]: INVALID JSON")
        
        try:
            # Make request
            response = requests.post(
                f"{BASE_URL}{ENDPOINT}",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            # Log response
            print(f"\n[RESPONSE]:")
            print(f"  Status: {response.status_code} {response.reason}")
            print(f"  Content-Type: {response.headers.get('Content-Type')}")
            print(f"  Body: {response.text[:500]}")
            
            if response.status_code == 200:
                print(f"\n✅ SUCCESS")
                resp_json = response.json()
                if 'points' in resp_json:
                    print(f"   Pontos encontrados: {len(resp_json['points'])}")
            else:
                print(f"\n❌ ERROR ({response.status_code})")
                try:
                    error_body = response.json()
                    print(f"   Error: {error_body.get('error', 'N/A')}")
                    if 'missing_city' in error_body:
                        print(f"   Missing cities: {len(error_body['missing_city'])} ocorrências")
                except:
                    print(f"   (Response is not JSON)")
            
            results.append({
                'test': test['name'],
                'status': response.status_code,
                'success': response.status_code == 200
            })
            
        except requests.exceptions.ConnectionError:
            print(f"\n❌ CONNECTION ERROR")
            print(f"   Servidor não está respondendo em {BASE_URL}")
            results.append({
                'test': test['name'],
                'status': 'CONNECTION_ERROR',
                'success': False
            })
        
        except Exception as e:
            print(f"\n❌ EXCEPTION: {type(e).__name__}")
            print(f"   {str(e)[:200]}")
            results.append({
                'test': test['name'],
                'status': str(type(e).__name__),
                'success': False
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMÁRIO DE TESTES")
    print(f"{'='*80}\n")
    
    successes = sum(1 for r in results if r['success'])
    print(f"Total: {len(results)} testes")
    print(f"✅ Sucesso: {successes}")
    print(f"❌ Falha: {len(results) - successes}\n")
    
    for r in results:
        status_icon = "✅" if r['success'] else "❌"
        print(f"{status_icon} {r['test']}")
        print(f"   └─ HTTP {r['status']}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMENDAÇÕES")
    print(f"{'='*80}\n")
    
    if all(r['status'] == 'CONNECTION_ERROR' for r in results):
        print("[❌] Servidor NÃO está rodando!")
        print("     Inicie com: python app.py")
    else:
        print("[✅] Servidor respondendo")
        
        failures = [r for r in results if not r['success']]
        if failures:
            print(f"\n[⚠️] {len(failures)} teste(s) falhando:")
            for f in failures:
                print(f"     • {f['test']} → HTTP {f['status']}")
        
        print("\n[Debug] Para mais detalhes, verifique:")
        print("        1. python -u app.py (rode com flush para ver logs em tempo real)")
        print("        2. Console do navegador (F12 → Network → exogenous/parse)")
        print("        3. Este script com: python tests/test_exogenous_parse_endpoint.py")

if __name__ == "__main__":
    test_endpoint()
