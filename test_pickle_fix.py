#!/usr/bin/env python
"""Teste rápido de carregamento de pickle com StringDtype fix."""
import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Unpickler customizado
class StringDtypeFixedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas.core.arrays.string_' and name == 'StringDtype':
            import numpy as np
            return lambda *args, **kwargs: np.dtype('object')
        return super().find_class(module, name)

def test_load(region):
    path = os.path.join(BASE_DIR, 'data', 'processed', f'processed_{region}.pkl')
    print(f"\n🔍 Testando {region}...")
    try:
        with open(path, 'rb') as f:
            data = StringDtypeFixedUnpickler(f).load()
        print(f"✅ {region}: Carregado com sucesso")
        if 'nodes_gdf' in data:
            gdf = data['nodes_gdf']
            print(f"   - {len(gdf)} nós")
            print(f"   - Colunas: {list(gdf.columns)[:5]}...")
        return True
    except Exception as e:
        print(f"❌ {region}: {e}")
        return False

if __name__ == '__main__':
    success = all(test_load(r) for r in ['fortaleza', 'rmf', 'interior'])
    sys.exit(0 if success else 1)
