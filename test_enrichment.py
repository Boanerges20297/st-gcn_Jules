from app import load_data_and_models, enrich_regions, nodes_gdf
import pandas as pd

# Load data
load_data_and_models()

# Check before
from app import nodes_gdf
print("Before Enrichment:")
if 'CIDADE' in nodes_gdf.columns:
    print(nodes_gdf['CIDADE'].value_counts().head())
    print("Empty:", len(nodes_gdf[nodes_gdf['CIDADE'].isna() | (nodes_gdf['CIDADE'] == '')]))
else:
    print("No CIDADE column.")

# Run Enrichment
enrich_regions()

# Check after
print("\nAfter Enrichment:")
print(nodes_gdf['CIDADE'].value_counts().head())
print("Empty:", len(nodes_gdf[nodes_gdf['CIDADE'].isna() | (nodes_gdf['CIDADE'] == '')]))
