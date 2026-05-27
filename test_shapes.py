import json
import sys
sys.path.append('C:/Users/Boanerges/Desktop/Projetos/Report Preview')
import app as report_app

report_app.load_data_and_models()

shapes = report_app._load_municipality_shapes_for_street_foci()
print(f"Loaded {len(shapes)} shapes.")
if "SAO GONCALO DO AMARANTE" in shapes:
    print("SAO GONCALO DO AMARANTE is present.")
else:
    print("SAO GONCALO DO AMARANTE is MISSING!")
    
# check the exact name
for k in shapes.keys():
    if "GON" in k:
        print(f"Found: {k}")

