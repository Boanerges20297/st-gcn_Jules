import sys, os, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.explanation_generator import ExplanationGenerator, create_sample_context
import src.llm_service as llmsvc

gen = ExplanationGenerator()
# Build an explanation using the generator
gen = ExplanationGenerator()
# Use a numeric node id for the generator but map it to a human area name for managers
node_id = 42
area_name = 'Aldeota, Fortaleza'
ctx = create_sample_context(node_id)
expl = gen.explain_node_ranking(node_id, 3, ctx)

# Post-process the generated explanation to replace technical 'Node 42' mentions
# with the real area name so the manager text refers to the bairro/cidade.
if isinstance(expl.get('summary'), str):
    expl['summary'] = expl['summary'].replace(f'Node {node_id}', area_name).replace(f'Node {node_id}', area_name)
# Replace nearby node mentions in factor explanations (Node X -> Localidade X)
for f in expl.get('factors', []):
    if isinstance(f.get('explanation'), str):
        f['explanation'] = f['explanation'].replace(f'Node {node_id-1}', 'localidade vizinha 1').replace(f'Node {node_id+1}', 'localidade vizinha 2')
        f['explanation'] = f['explanation'].replace(f'Node {node_id}', area_name)

print('\n=== Generated explanation object (truncated) ===\n')
print(json.dumps({k: expl[k] for k in ('node_id','summary','factors','confidence')}, indent=2, ensure_ascii=False))

# Prepare a manager-facing prompt in Portuguese
prompt = (
    "Você é um assistente que reescreve explicações técnicas para um gestor municipal.\n"
    "Recebe a explicação estruturada em JSON abaixo. Produza um parágrafo curto (3-4 frases) em português claro, "
    "destacando os fatores principais, o nível de confiança e recomendação de ação. Seja objetivo e não repita chaves JSON.\n\n"
    "EXPLICAÇÃO JSON:\n" + json.dumps(expl, ensure_ascii=False, indent=2)
)

# Try to call the legacy model if keys are available; otherwise produce a deterministic mock
try:
    legacy = llmsvc._legacy
    # prefer the helper keys function if present
    keys = []
    try:
        keys = legacy.get_gemini_api_keys()
    except Exception:
        keys = llmsvc._legacy.get_gemini_api_keys() if hasattr(llmsvc._legacy, 'get_gemini_api_keys') else []

    if keys:
        print('\nCalling remote generative model (using available GEMINI keys)...')
        out = legacy._call_model_with_rotation(prompt, keys)
        print('\n=== Model output ===\n')
        print(out)
    else:
        print('\nNo Gemini API keys found in environment; returning deterministic harmonized text (mock).')
        # Build a simple harmonized paragraph from explanation
        top_factors = [f['name'] for f in expl.get('factors', [])[:3]]
        factors_str = ', '.join(top_factors)
        conf_pct = int(round(expl.get('confidence', 0.0) * 100))
        harmonized = (
            f"Resumo gerencial: {expl.get('summary')} Principais fatores: {factors_str}. "
            f"Confiança estimada em {conf_pct}%. Recomendação: monitorar ações prioritárias nesta localidade."
        )
        print('\n=== Mock harmonized text ===\n')
        print(harmonized)

except Exception as e:
    print('ERROR during LLM test:', repr(e))
    # fallback deterministic output
    top_factors = [f['name'] for f in expl.get('factors', [])[:3]]
    factors_str = ', '.join(top_factors)
    conf_pct = int(round(expl.get('confidence', 0.0) * 100))
    harmonized = (
        f"Resumo gerencial: {expl.get('summary')} Principais fatores: {factors_str}. "
        f"Confiança estimada em {conf_pct}%. Recomendação: monitorar ações prioritárias nesta localidade."
    )
    print('\n=== Fallback harmonized text ===\n')
    print(harmonized)
