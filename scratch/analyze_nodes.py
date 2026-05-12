import pandas as pd
import unicodedata
import re
import os

def normalize_name(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
    return re.sub(r'\s*-\s*AIS.*$', '', text).strip()

def analyze():
    path = 'data/raw/nodes_gdf.csv'
    if not os.path.exists(path):
        print(f"File {path} not found")
        return
        
    df = pd.read_csv(path)
    # Filter Fortaleza
    df_fort = df[df['regiao'].str.lower().str.contains('capital|fortaleza', na=False)].copy()
    df_fort['norm'] = df_fort['name'].apply(normalize_name)
    
    print(f"Total nodes in Fortaleza: {len(df_fort)}")
    print(f"Unique names in Fortaleza: {df_fort['norm'].nunique()}")
    
    dupes = df_fort.groupby('norm').size()
    dupes = dupes[dupes > 1].sort_values(ascending=False)
    print("\nDuplicate Bairros:")
    print(dupes)

if __name__ == "__main__":
    analyze()
