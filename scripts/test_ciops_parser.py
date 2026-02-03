"""
Test CIOPS report parser
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service import parse_ciops_report
import json

sample_report = """
========= OCORRÊNCIAS ==========

01 - M20260083825 - RAIO 01 - ST * JEFFERSON - PESSOA/SITUAÇÃO SUSPEITA - BARROSO - 07:37 - UM CONDUZIDO - 6º DP

02 - M20260083908 - VTRA027 - 2ºSGT 22492 AMARO - VEÍCULO LOCALIZADO - TABATINGA, MARANGUAPE - 08:33 - SEM PRESO - D.M DE MARANGUAPE

03 - M20260083920 - RAIO 01 MARANGUAPE - 2ºSGT 25714 S. ROCHA - ACHADO DE ENTORPECENTES (CRACK) - CENTRO, MARANGUAPE - SEM PRESO - 08:39 - D.M DE MARANGUAPE

04 - M20260084202 - RAIO 01 CASCAVEL - 3ºSGT 25363 SILVA ALENCAR - VEÍCULO LOCALIZADO - (01 MOTO) - 02 PRESOS -  CASCAVEL - D.M. CASCAVEL - 13:32

05 - M20260084272 - RAIO 05 - ST FREITAS - RECEPTAÇÃO - 01 PRESO - QUINTINO CUNHA - 10ºDP - 14:33 

06 - M20260084579 - RAIO 02 MARANGUAPE - SGT F.SOUSA - ACHADO DE ENTORPECENTES - MARANGUAPE - DMM - 18:28

07 - M20260084656 - RAIO 02 PACAJUS - SGT MAURÍCIO - ABANDONO DE MATERIAL - ESPINGARDA E PISTOLAS ARTESANAIS, DROGA - PACAJUS - DMH - 19:07

============ HOMICÍDIOS ===========

01 - M20260083501 - HOMICÍDIO A BALA - VÍTIMA DE SEXO MASCULINO - AV CONTORNO LESTE N 204 - NOVA METRÓPOLE, CAUCAIA - 02:07

02 - M20260083695 - HOMICÍDIO A BALA - VÍTIMA DE SEXO MASCULINO - LOCALIDADE DE ANGELIM, TRAIRI - 04:54

========== LESÃO À BALA ==========

S/A 

========= ACHADO DE CADÁVER ===========

S/A

====== OCORRÊNCIAS COM POLICIAIS/AGENTES DE SEGURANÇA ======

S/A

======DESLOCAMENTO FORÇADO/EXPULSÃO DE MORADORES======

01 - M20260083922 - R. ALBERTO TORRES 108, MESSEJANA - 08:46 - SOLICITANTE INFORMA QUE ESTÁ SOFRENDO AMEAÇAS DE GRUPO CRIMINOSO E PRECISA FAZER A MUDANÇA

=======================================
"""

if __name__ == "__main__":
    print("=" * 80)
    print("TEST: CIOPS Report Parser")
    print("=" * 80)
    
    events = parse_ciops_report(sample_report)
    
    print(f"\n[OK] Parsed {len(events)} events\n")
    
    # Group by type
    enforcement = [e for e in events if 'ENFORCEMENT' in e.get('event_type', '')]
    crime = [e for e in events if 'CRIME' in e.get('event_type', '')]
    
    print(f"SUMMARY:")
    print(f"   Enforcement operations: {len(enforcement)}")
    print(f"   Crime events: {len(crime)}")
    print(f"   Total arrests: {sum(e.get('num_arrested', 0) for e in events)}")
    print(f"   Drugs seized: {sum(1 for e in events if e.get('has_drugs'))}")
    print(f"   Weapons seized: {sum(1 for e in events if e.get('has_weapons'))}")
    
    print(f"\n{'ENFORCEMENT OPERATIONS':^80}")
    print("-" * 80)
    for evt in enforcement:
        print(f"[ENFORCEMENT] {evt.get('incident_id')} | {evt.get('block_type', '')}")
        print(f"   Natureza: {evt.get('natureza', '')}")
        print(f"   Local: {evt.get('localizacao_completa', '')} ({evt.get('municipio', '')})")
        print(f"   Arrested: {evt.get('num_arrested', 0)} | Drugs: {evt.get('has_drugs')} | Weapons: {evt.get('has_weapons')}")
        print(f"   Severity: {evt.get('conflict_severity')} | Intensity: {evt.get('enforcement_intensity'):.2f}")
        print(f"   -> Canal 9 intensity: {evt.get('enforcement_intensity', 0):.2f}")
        print()
    
    print(f"{'CRIME EVENTS':^80}")
    print("-" * 80)
    for evt in crime:
        print(f"[CRIME] {evt.get('incident_id')} | {evt.get('event_type', '')}")
        print(f"   Natureza: {evt.get('natureza', '')}")
        print(f"   Local: {evt.get('localizacao_completa', '')} ({evt.get('municipio', '')})")
        print(f"   Severity: {evt.get('conflict_severity')}")
        print()
    
    print("=" * 80)
    print("[OK] TEST COMPLETED")
    print("=" * 80)
