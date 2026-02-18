import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
print('PYTHONPATH set to:', sys.path[0])

try:
    from src.explanation_generator import ExplanationGenerator, create_sample_context
    print('OK: imported src.explanation_generator')
    gen = ExplanationGenerator()
    ctx = create_sample_context(146)
    expl = gen.explain_node_ranking(146, 1, ctx)
    print('Generated summary:', expl.get('summary')[:120])
    print('Factors:', [f['name'] for f in expl.get('factors', [])])
except Exception as e:
    print('IMPORT ERROR:', repr(e))
    # Try legacy backup path directly
    try:
        import importlib.util
        legacy = os.path.join(BASE, 'backup', 'src_legacy', 'explanation_generator.py')
        spec = importlib.util.spec_from_file_location('legacy_expl', legacy)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print('OK: loaded legacy explanation_generator from', legacy)
        gen = mod.ExplanationGenerator()
        ctx = mod.create_sample_context(146)
        expl = gen.explain_node_ranking(146, 1, ctx)
        print('Legacy summary:', expl.get('summary')[:120])
        print('Legacy factors:', [f['name'] for f in expl.get('factors', [])])
    except Exception as e2:
        print('LEGACY LOAD ERROR:', repr(e2))
