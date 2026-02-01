import pickle
import os
import geopandas as gpd

def check():
    path = 'data/processed/processed_graph_data.pkl'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        nodes_gdf = data['nodes_gdf']
        print(f"Total Nodes: {len(nodes_gdf)}")

        geom_types = nodes_gdf.geometry.type.value_counts()
        print("\nGeometry Types:")
        print(geom_types)

        polygons = nodes_gdf[nodes_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        print(f"\nNodes with Polygon Geometry: {len(polygons)}")

        if len(polygons) == 0:
            print("\nCRITICAL: No polygons found. All nodes are Points.")
            print("Action Required: Upload 'ceara_municipios.geojson' and 'fortaleza_bairros.geojson' to 'data/raw/'")
            print("Then run: python src/data_processing.py")
        else:
            print(f"SUCCESS: {len(polygons)} nodes have Polygon geometries.")

    except Exception as e:
        print(f"Error inspecting data: {e}")

if __name__ == "__main__":
    check()
