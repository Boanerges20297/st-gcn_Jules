#!/usr/bin/env python3
"""
Script para testar a funcionalidade de simulação de equipes
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_simulation():
    """Testa a simulação de equipes"""
    
    print("=" * 70)
    print("🧪 TESTE DE SIMULAÇÃO DE EQUIPES - ST-GCN Jules")
    print("=" * 70)
    
    # Coordenadas de teste (Fortaleza - Centro)
    # Latitude: -3.731862, Longitude: -38.526669
    test_points = [
        [-3.731862, -38.526669],  # Centro
    ]
    
    print("\n📍 Pontos de teste:")
    for i, pt in enumerate(test_points, 1):
        print(f"   {i}. Lat: {pt[0]}, Lon: {pt[1]}")
    
    # Teste 1: Simulação de Equipe (Suppression)
    print("\n" + "=" * 70)
    print("TEST 1: Simulação de Equipe Tática (Suppression)")
    print("=" * 70)
    
    payload = {
        "points": test_points,
        "type": "suppression"
    }
    
    print("\n📤 Enviando payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print("\n⏳ Aguardando resposta...")
        response = requests.post(
            f"{BASE_URL}/api/simulate",
            json=payload,
            timeout=30
        )
        
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Resposta recebida:")
            print(f"   - Áreas retornadas: {len(data.get('data', []))}")
            print(f"   - Timestamp: {data.get('timestamp')}")
            
            # Mostrar top 5 áreas com maior risco
            areas = data.get('data', [])
            if areas:
                print("\n🔴 Top 5 Áreas de ALTO RISCO:")
                high_risk = [a for a in areas if a.get('risk_score', 0) >= 80][:5]
                for i, area in enumerate(high_risk, 1):
                    print(f"   {i}. {area.get('node_name', 'N/A')} - " +
                          f"Risk: {area.get('risk_score', 0):.1f}")
            
            print("\n✅ Simulação de SUPRESSÃO funcionando!")
            return True
        else:
            print(f"\n❌ Erro: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERRO: Não consegui conectar ao servidor.")
        print("   Certifique-se de que o app está rodando em http://localhost:5000")
        return False
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        return False

def test_simulation_exogenous():
    """Testa simulação de conflitos exógenos"""
    
    print("\n" + "=" * 70)
    print("TEST 2: Simulação de Conflito Exógeno")
    print("=" * 70)
    
    test_points = [
        [-3.731862, -38.526669],  # Centro
    ]
    
    payload = {
        "points": test_points,
        "type": "exogenous"
    }
    
    print("\n📤 Enviando payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print("\n⏳ Aguardando resposta...")
        response = requests.post(
            f"{BASE_URL}/api/simulate",
            json=payload,
            timeout=30
        )
        
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            areas = data.get('data', [])
            
            if areas:
                print("\n🔴 Top 5 Áreas com CONFLITO:")
                high_risk = [a for a in areas if a.get('risk_score', 0) >= 80][:5]
                for i, area in enumerate(high_risk, 1):
                    print(f"   {i}. {area.get('node_name', 'N/A')} - " +
                          f"Risk: {area.get('risk_score', 0):.1f}")
            
            print("\n✅ Simulação de CONFLITO funcionando!")
            return True
        else:
            print(f"\n❌ Erro: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    try:
        result1 = test_simulation()
        time.sleep(2)
        result2 = test_simulation_exogenous()
        
        print("\n" + "=" * 70)
        if result1 and result2:
            print("✅ TODOS OS TESTES PASSARAM - Simulação funcionando!")
        else:
            print("❌ Alguns testes falharam")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Teste interrompido pelo usuário")
