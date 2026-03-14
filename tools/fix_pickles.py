import pandas as pd
import pickle
import os
import numpy as np

def fix_pickle_for_pandas2(filepath):
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    print(f"Lendo {filepath} (com pandas {pd.__version__})...")
    try:
        # Lê com o Pandas 3.0.0
        data = pd.read_pickle(filepath)
        
        if isinstance(data, dict) and 'nodes_gdf' in data:
            gdf = data['nodes_gdf']
            
            # Converte todas as colunas de texto (StringDtype) para object puro do Numpy
            for col in gdf.columns:
                if col != 'geometry':
                    try:
                        # Força a conversão para lista de strings puras e depois array de objetos
                        gdf[col] = np.array(gdf[col].astype(str).tolist(), dtype=object)
                    except Exception as e:
                        print(f"Aviso ao converter coluna {col}: {e}")
            
            # Repassa o GeoDataFrame limpo para o dict
            data['nodes_gdf'] = gdf
            
        if isinstance(data, dict) and 'dates' in data:
             data['dates'] = np.array([str(d) for d in data['dates']], dtype=object)
             
        # Salva usando o pickle nativo com protocolo antigo (mais seguro)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=4)
            
        print(f"✅ Arquivo corrigido e salvo: {filepath}")
    except Exception as e:
        print(f"❌ Erro ao converter {filepath}: {e}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    regions = ['fortaleza', 'rmf', 'interior']
    for reg in regions:
        path = os.path.join(base_dir, 'data', 'processed', f'processed_{reg}.pkl')
        fix_pickle_for_pandas2(path)
