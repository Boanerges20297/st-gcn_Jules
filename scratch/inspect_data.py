import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    print("\n--- TIPO VALUE COUNTS ---")
    print(df['tipo'].value_counts(dropna=False))
    
    print("\n--- DATE RANGE ---")
    df['parsed_date'] = pd.to_datetime(df['data'], errors='coerce')
    print(f"Min Date: {df['parsed_date'].min()}")
    print(f"Max Date: {df['parsed_date'].max()}")
    print(f"Invalid Dates: {df['parsed_date'].isna().sum()}")

if __name__ == "__main__":
    main()
