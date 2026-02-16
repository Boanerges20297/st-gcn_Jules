import os
import sys
import numpy as np
import pickle

# Set project root
ROOT = os.getcwd()
sys.path.append(ROOT)

from Phase4.orchestrator import StateOrchestrator

def test():
    print("Testing Orchestrator with Regional Subsets...")
    try:
        orch = StateOrchestrator(ROOT)
        risks = orch.get_combined_risk()
        
        # Load global nodes to check coverage
        with open('data/processed/processed_graph_data_global.pkl', 'rb') as f:
            global_data = pickle.load(f)
        nodes_gdf = global_data['nodes_gdf']
        
        print(f"Total global nodes: {len(nodes_gdf)}")
        print(f"Computed risks: {len(risks)}")
        
        # Check by region
        for reg in ['fortaleza', 'rmf', 'interior']:
            # Handle regiao vs region_type
            reg_mask = (nodes_gdf['regiao'] == reg) | (nodes_gdf.get('region_type', '') == reg)
            if reg == 'fortaleza': # fallback for 'capital'
                 reg_mask |= (nodes_gdf.get('region_type', '') == 'capital')
            
            reg_nodes = nodes_gdf[reg_mask]['name'].tolist()
            from Phase4.orchestrator import normalize_name
            reg_norms = [normalize_name(n) for n in reg_nodes]
            
            reg_risks = [risks.get(n) for n in reg_norms if n in risks]
            if reg_risks:
                print(f"Region {reg.upper()}:")
                print(f"  - Count: {len(reg_risks)}/{len(reg_nodes)}")
                print(f"  - Risk Range: {min(reg_risks):.2f} to {max(reg_risks):.2f}")
                print(f"  - Average Risk: {np.mean(reg_risks):.2f}")
            else:
                print(f"❌ Region {reg.upper()} HAS NO RISKS COMPUTED!")

    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
