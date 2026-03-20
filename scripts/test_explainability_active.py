import os
import sys
import json

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR))

from src.explanation_generator import ExplanationGenerator

def test_explain():
    print("🧠 Testando Explicabilidade (Tentativa 49 - Ativa)...")
    
    gen = ExplanationGenerator()
    
    # Simular contexto de Barroso (Top 1 no Monitor)
    context = {
        'name': 'BARROSO',
        'node_id': 12,
        'score': 8.7,
        'rank': 1,
        'tier': 'top_5',
        'confidence': 0.92,
        'cvli_count_recent': 4,
        'cvli_count_prev': 1,
        'nearby_impact_names': ['ANCURI', 'MESSEJANA'],
        'events': [
            {'natureza': 'Homicídio', 'descricao': 'Conflito de facções na Travessa Unidos', 'is_suppression': False},
            {'natureza': 'Ação Policial', 'descricao': 'Operação no Coqueirinho', 'is_suppression': True, 'is_qualified_suppression': True}
        ]
    }
    
    print("\n1. Testando Explicação Heurística (Elite Fallback):")
    explanation = gen.explain_node_ranking(12, 1, context)
    
    print(f"📍 LOCAL: {explanation['name']} (#{explanation['rank']})")
    print(f"📝 RESUMO: {explanation['summary']}")
    print("🔍 FATORES:")
    for f in explanation['factors']:
        print(f"   [{f['name']}] ({f['importance']}): {f['explanation']}")
    print(f"💡 INTERPRETAÇÃO: {explanation['interpretation']}")

    # Tentar LLM se houver chave (Simulado)
    print("\n2. Tentando Análise Estratégica via Gemini (se disponível)...")
    llm_expl = gen._get_llm_explanation('BARROSO', 1, 8.7, context)
    if llm_expl:
        print(f"🤖 RESUMO LLM: {llm_expl['summary']}")
        for f in llm_expl['factors']:
            print(f"   - {f['explanation']}")
    else:
        print("⚠️ Gemini não disponível ou desativado. Usando apenas a Lógica Elite.")

if __name__ == "__main__":
    test_explain()
