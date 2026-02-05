#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from collections import Counter

proj = Path(__file__).parents[1]
top = proj / 'plots' / 'top10_bairros_prison_bairro_summary_180d.json'
raw = proj / 'data' / 'raw' / 'View_Ocorrencias_Operacionais_Modelo_NORMALIZADO.csv'
eval_in = proj / 'plots' / 'eval_top10_top10_bairros_prison_bairro_summary_180d_cvli_180d.json'

def find_bairro_for_id(id_val):
    id_str = str(id_val)
    candidates = []
    with raw.open('r', encoding='utf-8', errors='ignore') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row: continue
            # check any field equals id_str or id_str + .0
            if any(f.strip()==id_str or f.strip()==(id_str+'.0') for f in row):
                # BairroOcor column at index 11 if header as expected
                if len(row) > 11:
                    b = row[11].strip()
                    if b:
                        candidates.append(b)
    if not candidates:
        return None
    # return most common
    return Counter(candidates).most_common(1)[0][0]

def main():
    if not top.exists():
        print('top file missing:', top)
        return
    with top.open('r', encoding='utf-8') as fh:
        top10 = json.load(fh)

    # try to map unmapped ids
    mapped = []
    for item in top10:
        if item.get('bairro_name'):
            mapped.append(item)
            continue
        bid = item.get('bairro_id')
        name = find_bairro_for_id(bid)
        item['bairro_name'] = name
        mapped.append(item)

    out = proj / 'plots' / 'top10_bairros_prison_bairro_summary_180d_mapped.json'
    with out.open('w', encoding='utf-8') as fh:
        json.dump(mapped, fh, ensure_ascii=False, indent=2)
    print('Wrote mapped top10 to', out)

    # also update evaluation JSON if exists
    if eval_in.exists():
        with eval_in.open('r', encoding='utf-8') as fh:
            ej = json.load(fh)
        # replace bairro_name where idx matches bairro_id
        for r in ej.get('results', []):
            for t in mapped:
                if r.get('bairro_id') == t.get('bairro_id'):
                    r['bairro_name'] = t.get('bairro_name')
        out_eval = eval_in.parent / (eval_in.stem + '_mapped.json')
        with out_eval.open('w', encoding='utf-8') as fh:
            json.dump(ej, fh, ensure_ascii=False, indent=2)
        print('Wrote mapped evaluation to', out_eval)

if __name__ == '__main__':
    main()
