#!/usr/bin/env python3
"""Imprime os N bairros mais críticos a partir de um CSV de resumo por bairro.

Por padrão usa `plots/prison_bairro_summary_30d.csv` e calcula uma pontuação
simples: score = 3*cvli_sum + 2*cvp_sum + 1*prison_sum. Aceita arquivo de nomes
para mapear `bairro_id` -> nome.
"""
import argparse
import json
from pathlib import Path
import sys

try:
    import pandas as pd
except Exception:
    print("Por favor instale pandas: pip install pandas", file=sys.stderr)
    raise


def load_mapping(path: Path):
    if not path.exists():
        return None
    if path.suffix.lower() in ('.json',):
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        # Accept both id->name and name->coords formats.
        # If keys are numeric strings, convert to int.
        mapping = {}
        for k, v in data.items():
            try:
                ik = int(k)
                mapping[ik] = v
            except Exception:
                # If values are coords (list), skip — not an id map.
                if isinstance(v, (str,)):
                    mapping[k] = v
        return mapping
    return None


def load_name_lists(project_root: Path):
    # Try production metadata list first
    meta = project_root / 'data' / 'processed' / 'metadata_producao_v2.json'
    if meta.exists():
        try:
            with meta.open('r', encoding='utf-8') as fh:
                jb = json.load(fh)
            names = jb.get('bairros_normalizados')
            if isinstance(names, list):
                return {i: names[i] for i in range(len(names))}
        except Exception:
            pass

    # Fallback: use bairros_centros_latlong.json to build name set (no indices)
    bpath = project_root / 'data' / 'raw' / 'bairros_centros_latlong.json'
    if bpath.exists():
        try:
            with bpath.open('r', encoding='utf-8') as fh:
                jb = json.load(fh)
            # return mapping of normalized name -> canonical
            return {k: k for k in jb.keys()}
        except Exception:
            pass

    return {}


def main():
    p = argparse.ArgumentParser(description='Top bairros críticos')
    p.add_argument('--csv', default=str(Path(__file__).parents[1] / 'plots' / 'prison_bairro_summary_30d.csv'))
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--names', help='JSON file mapping bairro_id -> nome (opcional)')
    p.add_argument('--weights', help='Pesos como three comma-separated numbers for cvli,cvp,prison', default='3,2,1')
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'CSV não encontrado: {csv_path}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    for c in ('bairro_id', 'prison_sum', 'cvli_sum', 'cvp_sum'):
        if c not in df.columns:
            print(f'Coluna esperada ausente: {c}', file=sys.stderr)
            sys.exit(3)

    try:
        w_cvli, w_cvp, w_prison = [float(x) for x in args.weights.split(',')]
    except Exception:
        print('Formato de --weights inválido; use por exemplo 3,2,1', file=sys.stderr)
        sys.exit(4)

    df['score'] = df['cvli_sum'] * w_cvli + df['cvp_sum'] * w_cvp + df['prison_sum'] * w_prison

    mapping = None
    if args.names:
        mapping = load_mapping(Path(args.names))

    topn = df.sort_values('score', ascending=False).head(args.top)
    # build name map from project metadata/raw lists
    project_root = Path(__file__).parents[1]
    idx_name = load_name_lists(project_root)

    out_rows = []
    for rank, (_, row) in enumerate(topn.iterrows(), start=1):
        try:
            bid = int(row['bairro_id'])
        except Exception:
            bid = row['bairro_id']
        name = None
        if mapping:
            name = mapping.get(bid) or mapping.get(str(bid))
        if name is None and isinstance(bid, int) and bid in idx_name:
            name = idx_name.get(bid)

        out_rows.append({
            'rank': rank,
            'bairro_id': bid,
            'bairro_name': name,
            'score': float(row['score']),
            'prison_sum': int(row['prison_sum']),
            'cvli_sum': int(row['cvli_sum']),
            'cvp_sum': int(row['cvp_sum'])
        })

    # print human-readable
    print(f'Top {args.top} bairros por score (weights cvli,cvp,prison = {w_cvli},{w_cvp},{w_prison})')
    print('-' * 60)
    for r in out_rows:
        label = f"{r['bairro_name']} ({r['bairro_id']})" if r['bairro_name'] else str(r['bairro_id'])
        print(f"{label}: score={r['score']:.2f} prison={r['prison_sum']} cvli={r['cvli_sum']} cvp={r['cvp_sum']}")

    # write JSON if requested
    if args.names or args.csv:
        out_path = Path(args.csv).parents[0] / f"top10_bairros_{Path(args.csv).stem}.json"
    else:
        out_path = project_root / 'plots' / f'top10_bairros_{args.top}.json'
    try:
        with out_path.open('w', encoding='utf-8') as fh:
            json.dump(out_rows, fh, ensure_ascii=False, indent=2)
        print('Wrote JSON:', out_path)
    except Exception as e:
        print('Falha ao escrever JSON:', e, file=sys.stderr)


if __name__ == '__main__':
    main()
