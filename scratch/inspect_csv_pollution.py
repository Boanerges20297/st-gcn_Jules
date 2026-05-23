import pandas as pd
from pathlib import Path

def main():
    hermes_dir = Path("outputs/hermes")
    if not hermes_dir.exists():
        print("Hermes dir does not exist locally.")
        return
        
    print("Checking local CSV files for the substring 'A capital' or 'Fortaleza'...")
    for path in hermes_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path)
            # Search for any cell containing 'A capital' or 'Fortaleza'
            for col in df.columns:
                matches = df[df[col].astype(str).str.contains("A capital|Fortaleza|Ceará", case=False, na=False)]
                if not matches.empty:
                    print(f"\n--- MATCH IN FILE {path.name}, COLUMN {col} ---")
                    for idx, row in matches.head(5).iterrows():
                        print(f"Row {idx}: Name={row.get('name', 'N/A')} | Content={row[col]}")
        except Exception as e:
            print(f"Error reading {path.name}: {e}")

if __name__ == "__main__":
    main()
