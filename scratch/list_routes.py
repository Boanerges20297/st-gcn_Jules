import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

print("=== LISTA DE ROTAS REGISTRADAS NO FLASK ===")
for rule in app.url_map.iter_rules():
    print(f"Route: {rule.rule} -> Endpoint: {rule.endpoint} -> Methods: {list(rule.methods)}")
