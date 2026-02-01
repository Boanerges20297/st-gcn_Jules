import zipfile
import os
import geopandas as gpd

kmz_path = 'data/static/bairros_2025.kmz'
out_dir = 'data/static'
out_kml = os.path.join(out_dir, 'bairros_2025.kml')
out_geojson = os.path.join(out_dir, 'bairros_2025.geojson')

if not os.path.exists(kmz_path):
    print('KMZ not found:', kmz_path)
    raise SystemExit(1)

with zipfile.ZipFile(kmz_path, 'r') as z:
    # find .kml file inside
    kml_name = None
    for name in z.namelist():
        if name.lower().endswith('.kml'):
            kml_name = name
            break
    if kml_name is None:
        print('No KML found inside KMZ')
        raise SystemExit(1)
    # extract
    z.extract(kml_name, out_dir)
    extracted = os.path.join(out_dir, kml_name)
    # move to out_kml root if nested
    if extracted != out_kml:
        os.replace(extracted, out_kml)

# Try reading KML with geopandas
try:
    gdf = gpd.read_file(out_kml)
    print('KML layers read, features:', len(gdf))
except Exception as e:
    print('Failed to read KML with geopandas:', e)
    raise

# Normalize column names and ensure geometry valid
gdf['geometry'] = gdf.geometry.apply(lambda x: x.buffer(0) if not x.is_valid else x)

# Save as GeoJSON
gdf.to_file(out_geojson, driver='GeoJSON', encoding='utf-8')
print('Saved GeoJSON to', out_geojson)
