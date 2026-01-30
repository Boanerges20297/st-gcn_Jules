import pandas as pd
import geopandas as gpd
import json
import os
import sys

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.data_processing import load_occurrences

def main():
    file1 = 'data/raw/dados_status_ocorrencias_gerais.json'
    file2 = 'data/raw/dados_status_1201_2701.json'
    output_file = 'data/raw/dados_merged.json'

    print(f"Loading {file1}...")
    # data_processing.load_occurrences handles parsing and normalization
    df1 = load_occurrences(file1)

    print(f"Loading {file2}...")
    df2 = load_occurrences(file2)

    print(f"Merging {len(df1)} and {len(df2)} records...")
    # concat works for GeoDataFrames too, returns GeoDataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)

    print(f"Total before dedupe: {len(merged_df)}")

    # Deduplicate based on core fields
    # data is datetime64[ns], convert to string for robust comparison if needed,
    # but pandas handles datetime comparison well.
    subset_cols = ['data', 'latitude', 'longitude', 'tipo']

    # Check if we have other columns we want to preserve or use for dedupe?
    # Usually address, etc. But lat/lon/time/type is unique enough for an event.

    merged_df = merged_df.drop_duplicates(subset=subset_cols)
    print(f"Total after dedupe: {len(merged_df)}")

    # Sort by date
    merged_df = merged_df.sort_values('data')

    # Check date range
    print(f"Date Range: {merged_df['data'].min()} to {merged_df['data'].max()}")

    # Convert back to simple DataFrame for JSON export (drop geometry)
    df_export = pd.DataFrame(merged_df.drop(columns='geometry'))

    # Convert datetime objects to string format YYYY-MM-DD
    # load_occurrences converts to datetime, so we reverse it.
    df_export['data'] = df_export['data'].dt.strftime('%Y-%m-%d')

    # Convert to list of dicts
    records = df_export.to_dict('records')

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Merge complete.")

if __name__ == "__main__":
    main()
