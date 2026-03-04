#!/usr/bin/env python3
import subprocess
import json

print("=" * 60)
print("COMMAND 1: git log --oneline -5")
print("=" * 60)
r = subprocess.run(['git', 'log', '--oneline', '-5', '--', 'logs/efficiency_history.json'], capture_output=True, text=True, cwd=r'C:\Users\Boanerges\Desktop\Projetos\st-gcn_jules')
print(r.stdout or r.stderr)

print("\n" + "=" * 60)
print("COMMAND 2: JSON efficiency history")
print("=" * 60)
try:
    d = json.load(open('logs/efficiency_history.json'))
    print(f'Entries: {len(d)}')
    for e in d:
        if 'global' in e and 'p20' in e['global']:
            print(f"  {e['date']}: global_p20={e['global']['p20']:.3f}")
except Exception as ex:
    print(f"Error: {ex}")
