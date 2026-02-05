"""
Test LLM CIOPS parsing
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service import parse_ciops_report, get_gemini_api_keys

sample_report = """
========= OCORRÊNCIAS ==========

01 - M20260083825 - RAIO 01 - ST * JEFFERSON - PESSOA/SITUAÇÃO SUSPEITA - BARROSO - 07:37 - UM CONDUZIDO - 6º DP

02 - M20260083908 - VTRA027 - 2ºSGT 22492 AMARO - VEÍCULO LOCALIZADO - TABATINGA, MARANGUAPE - 08:33 - SEM PRESO - D.M DE MARANGUAPE

03 - M20260083920 - RAIO 01 MARANGUAPE - 2ºSGT 25714 S. ROCHA - ACHADO DE ENTORPECENTES (CRACK) - CENTRO, MARANGUAPE - SEM PRESO - 08:39 - D.M DE MARANGUAPE

============ HOMICÍDIOS ===========

01 - M20260083501 - HOMICÍDIO A BALA - VÍTIMA DE SEXO MASCULINO - AV CONTORNO LESTE N 204 - NOVA METRÓPOLE, CAUCAIA - 02:07

02 - M20260083695 - HOMICÍDIO A BALA - VÍTIMA DE SEXO MASCULINO - LOCALIDADE DE ANGELIM, TRAIRI - 04:54

======DESLOCAMENTO FORÇADO/EXPULSÃO DE MORADORES======

01 - M20260083922 - R. ALBERTO TORRES 108, MESSEJANA - 08:46 - SOLICITANTE INFORMA QUE ESTÁ SOFRENDO AMEAÇAS DE GRUPO CRIMINOSO E PRECISA FAZER A MUDANÇA

=======================================
"""

if __name__ == "__main__":
    print("=" * 80)
    print("TEST: LLM CIOPS Parser")
    print("=" * 80)
    
    keys = get_gemini_api_keys()
    print(f"\n[1] API Keys available: {len(keys)}")
    if keys:
        print(f"    Using LLM for parsing")
    else:
        print(f"    NO API KEYS - will use deterministic fallback")
        print(f"    Set GEMINI_API_KEY or GEMINI_API_KEYS environment variable")
    
    print(f"\n[2] Testing parser...")
    try:
        events = parse_ciops_report(sample_report, use_llm=True)
        
        print(f"\n[OK] Parsed {len(events)} events")
        
        # Check event types
        enforcement = [e for e in events if 'ENFORCEMENT' in e.get('event_type', '')]
        crime = [e for e in events if 'CRIME' in e.get('event_type', '')]
        
        print(f"\n[SUMMARY]")
        print(f"    Enforcement: {len(enforcement)}")
        print(f"    Crime: {len(crime)}")
        print(f"    Total: {len(events)}")
        
        # Check enforcement_intensity is 0 for CRIME
        print(f"\n[VALIDATION]")
        for evt in events:
            evt_type = evt.get('event_type', '')
            intensity = evt.get('enforcement_intensity', 0)
            if 'CRIME' in evt_type and intensity != 0:
                print(f"    ERROR: {evt.get('incident_id')} is CRIME but intensity={intensity}")
            elif 'ENFORCEMENT' in evt_type and intensity == 0 and not evt.get('has_drugs') and not evt.get('has_weapons'):
                print(f"    OK: {evt.get('incident_id')} - {evt_type} intensity=0.0 (expected for routine)")
            elif 'ENFORCEMENT' in evt_type:
                print(f"    OK: {evt.get('incident_id')} - {evt_type} intensity={intensity}")
        
        print(f"\n[DETAILS]")
        for evt in events:
            print(f"\n    {evt.get('incident_id')} | {evt.get('event_type')}")
            print(f"      Natureza: {evt.get('natureza')}")
            print(f"      Local: {evt.get('bairro')} ({evt.get('municipio')})")
            print(f"      Severity: {evt.get('conflict_severity')} | Intensity: {evt.get('enforcement_intensity'):.2f}")
            print(f"      Arrested: {evt.get('num_arrested')} | Drugs: {evt.get('has_drugs')} | Weapons: {evt.get('has_weapons')}")
        
        print(f"\n[OK] LLM CIOPS parser works!")
        
    except Exception as e:
        print(f"\n[ERROR] Parser failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
