#!/usr/bin/env python3
import json, unicodedata
import json, unicodedata, argparse
from pathlib import Path
import numpy as np

proj = Path(__file__).parents[1]
meta = proj / 'data' / 'processed' / 'metadata_producao_v2.json'
nf = proj / 'data' / 'processed' / 'graph_data' / 'node_features.npy'

def norm(s):
    if s is None: return None
    s = str(s).upper()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.replace('\n',' ').strip()

def main():
    p = argparse.ArgumentParser(description='Evaluate top10 bairros against model features')
    p.add_argument('--top', help='Top-10 JSON file produced by top10_bairros.py', default=str(proj / 'plots' / 'top10_bairros_prison_bairro_summary_30d.json'))
    p.add_argument('--window', type=int, default=30, help='historical window in days to sum model features')
    p.add_argument('--feature', choices=['cvli','cvp','both'], default='both')
    p.add_argument('--out', help='Output JSON path', default=None)
    args = p.parse_args()

    top_path = Path(args.top)
    if not top_path.exists():
        raise SystemExit(f'top file not found: {top_path}')

    with top_path.open('r', encoding='utf-8') as f:
        top10 = json.load(f)
    with meta.open('r', encoding='utf-8') as f:
        m = json.load(f)
    names = m.get('bairros_normalizados', [])
    name_map = {norm(n): i for i,n in enumerate(names)}

    arr = np.load(nf, allow_pickle=True)
    if arr.ndim != 3:
        raise SystemExit('unexpected node_features shape')
    N,T,F = arr.shape
    w = args.window if T >= args.window else T

    sums_cvli = np.sum(arr[:, -w:, 0], axis=1)
    sums_cvp = np.sum(arr[:, -w:, 1], axis=1) if F>1 else np.zeros(N)

    detected = []
    for item in top10:
        bid = item.get('bairro_id')
        bname = item.get('bairro_name')
        matched_idx = None
        if bname:
            n = norm(bname)
            if n in name_map:
                matched_idx = name_map[n]
        try:
            if matched_idx is None and isinstance(bid, int) and 0 <= bid < N:
                matched_idx = int(bid)
        except Exception:
            pass
        if matched_idx is not None:
            if args.feature == 'cvli':
                score = float(sums_cvli[matched_idx])
            elif args.feature == 'cvp':
                score = float(sums_cvp[matched_idx])
            else:
                score = float(sums_cvli[matched_idx] + sums_cvp[matched_idx])

            name_for_output = bname
            if name_for_output is None and isinstance(matched_idx, int) and matched_idx < len(names):
                name_for_output = names[matched_idx]
            detected.append({
                'bairro_id': bid,
                'bairro_name': name_for_output,
                'idx': matched_idx,
                'model_score': score,
                'cvli': float(sums_cvli[matched_idx]),
                'cvp': float(sums_cvp[matched_idx])
            })

    detected = sorted(detected, key=lambda x: x['model_score'], reverse=True)
    out_json = args.out or (top_path.parents[0] / f'eval_top10_{top_path.stem}_{args.feature}_{w}d.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump({'generated_at':__import__('datetime').datetime.utcnow().isoformat(), 'window':w, 'feature':args.feature, 'results':detected}, fh, ensure_ascii=False, indent=2)
    print('Wrote evaluation JSON:', out_json)
    print(json.dumps(detected, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
