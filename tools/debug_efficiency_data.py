import pickle
import os
import sys
import numpy as np

# Mock geopandas
class MockGeoPandas:
    class GeoDataFrame: pass
    def __getattr__(self, name): return None
sys.modules['geopandas'] = MockGeoPandas()

def debug_efficiency():
    path = 'data/processed/processed_fortaleza.pkl'
    if not os.path.exists(path):
        print("File not found.")
        return
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
        nf = data.get('node_features')
        dates = data.get('dates', [])
        
        print(f"Total dates in .pkl: {len(dates)}")
        print(f"Last 3 dates in dataset: {dates[-3:] if len(dates) >= 3 else dates}")
        
        # Check CVLI channel (0) for the last 14 days (Monitor standard)
        window = 14
        recent_cvli = nf[:, -window:, 0]
        total_recent = recent_cvli.sum()
        print(f"Total CVLI in last {window} days of dataset: {total_recent}")
        
        # Checking alignment with raw CSV
        csv_path = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'
        if os.path.exists(csv_path):
            df_raw = pd.read_csv(csv_path, usecols=['data', 'tipo'], low_memory=False)
            df_raw['data'] = pd.to_datetime(df_raw['data'], errors='coerce')
            last_csv_date = df_raw['data'].max()
            print(f"Last date in Raw CSV: {last_csv_date}")
            
            if last_csv_date < dates[-1]:
                print(f"🚨 ALERTA: O arquivo .pkl está à frente do CSV em {(dates[-1].date() - last_csv_date.date()).days} dias.")
                print(f"Isto causa o SKIP no Monitor pois a janela de 14 dias termina em zeros.")

if __name__ == '__main__':
    debug_efficiency()
