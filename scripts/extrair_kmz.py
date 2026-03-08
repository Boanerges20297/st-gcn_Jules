import zipfile
import os

kmz_path = r'C:\Users\Boanerges\Downloads\ORCRIMS 2026.kmz'
dest_dir = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\inteligencia'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

try:
    with zipfile.ZipFile(kmz_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        print(f"Arquivos extraídos para: {dest_dir}")
        
        # Renomear o doc.kml se existir para ORCRIMS_2026.kml
        old_kml = os.path.join(dest_dir, 'doc.kml')
        new_kml = os.path.join(dest_dir, 'ORCRIMS_2026.kml')
        if os.path.exists(old_kml):
            if os.path.exists(new_kml):
                os.remove(new_kml)
            os.rename(old_kml, new_kml)
            print(f"Renomeado doc.kml para {new_kml}")
except Exception as e:
    print(f"Erro ao extrair KMZ: {e}")
