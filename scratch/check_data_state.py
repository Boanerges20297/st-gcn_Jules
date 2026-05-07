import pandas as pd
import os
from datetime import datetime

path = r'c:\Users\Boanerges\Desktop\Projetos\Report Preview\data\processed\processed_fortaleza.pkl'
if os.path.exists(path):
    data = pd.read_pickle(path)
    dates = data.get('dates')
    if dates is not None:
        print(f"Total dates: {len(dates)}")
        print(f"First date: {dates[0]}")
        print(f"Last date: {dates[-1]}")
        
        # Check node_features for the last 30 steps
        nf = data.get('node_features')
        if nf is not None:
            # Canal 0 is CVLI
            last_cvli = nf[:, -30:, 0].sum()
            print(f"Sum of CVLI in last 30 steps: {last_cvli}")
            
            # Check shape
            print(f"Node features shape: {nf.shape}")
    else:
        print("No 'dates' key found in pickle.")
else:
    print(f"File not found: {path}")
