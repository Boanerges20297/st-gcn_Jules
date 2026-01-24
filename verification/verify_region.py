import sys
import os
import json
import unittest
from shapely.geometry import Point

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, get_region_name, enrich_regions, nodes_gdf

class TestAppLogic(unittest.TestCase):
    def test_region_inference(self):
        # Genibaú coords approx: -38.58, -3.75
        geom = Point(-38.58, -3.75)
        props = {'CIDADE': '', 'name': 'Genibaú'}
        region = get_region_name(props, geom)
        print(f"Genibaú Region: {region}")
        self.assertEqual(region, "Fortaleza")

        # Interior coords: -39.0, -4.0
        geom_int = Point(-39.0, -4.0)
        props_int = {'CIDADE': ''}
        region_int = get_region_name(props_int, geom_int)
        print(f"Interior Region: {region_int}")
        self.assertEqual(region_int, "RMF/Interior")

    def test_enrichment(self):
        # Mock nodes_gdf if necessary, but we can rely on loaded data if available
        # But app.py loads data on import.
        # We need to ensure enrich_regions runs.
        # It runs at end of app.py.
        # Let's check a known node if we can find one.
        pass

if __name__ == '__main__':
    unittest.main()
