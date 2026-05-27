import json

with open("C:/Users/Boanerges/Desktop/Projetos/screenshot-report_preview/public/data/cvli_points.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

interior_in_rmf_or_fortaleza = []
for feat in data.get("features", []):
    props = feat.get("properties", {})
    region = props.get("region", "unknown").lower()
    
    lng, lat = feat["geometry"]["coordinates"]
    
    if region == "interior":
        # Check if inside my rough Fortaleza+RMF bbox:
        # A rough bounding box for Fortaleza+RMF is lat between -4.25 and -3.45, lng between -39.05 and -38.05
        if -4.25 <= lat <= -3.45 and -39.05 <= lng <= -38.05:
            interior_in_rmf_or_fortaleza.append(props)

for p in interior_in_rmf_or_fortaleza:
    print(f"City: {p.get('cidade')} | Lat: {feat['geometry']['coordinates'][1]}, Lng: {feat['geometry']['coordinates'][0]}")
    
print(f"Total Interior in RMF/Fortaleza: {len(interior_in_rmf_or_fortaleza)}")
