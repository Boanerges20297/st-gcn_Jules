from pathlib import Path
root=Path(__file__).resolve().parent
count=0
for p in root.iterdir():
    if not p.is_file() or p.suffix.lower() not in ('.csv','.json','.md','.py'):
        continue
    raw=p.read_bytes()
    fixed=raw.replace(b'\r\r\n',b'\n').replace(b'\r\n',b'\n')
    if fixed!=raw:
        p.write_bytes(fixed)
        count+=1
print('normalized_line_endings',count)
