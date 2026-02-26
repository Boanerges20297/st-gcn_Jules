import json
import re
import unicodedata
from pathlib import Path


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'[^\w\s]', ' ', s)
    return s.strip()


def simple_norm_upper(s: str) -> str:
    if not s:
        return ''
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.upper()


def choose_natureza_from_line(line: str) -> str:
    parts = [p.strip() for p in line.split(' - ')]
    natureza = 'DESCONHECIDO'
    norm_upper = simple_norm_upper(line)

    action_keywords = [
        'ABANDON', 'ACHAD', 'APREENS', 'PORTE', 'VEICUL', 'MANDAD', 'PRESO', 'CONDUZ',
        'ESTUPR', 'HOMICID', 'LESAO', 'ACHADO', 'TRAFIC', 'MORTE', 'ROUBO', 'RECEPTA',
        'VEICULO', 'RECEPTAÇÃO', 'TRÁFICO', 'ACHADO DE ENTORPECENTES'
    ]

    chosen = None
    if len(parts) >= 3:
        for p in parts[2:]:
            np = simple_norm_upper(p)
            for kw in action_keywords:
                if kw in np:
                    chosen = p
                    break
            if chosen:
                break

    if chosen:
        natureza = chosen
    elif len(parts) >= 3 and parts[2]:
        natureza = parts[2]

    # overrides for homicide / lesão a bala
    if 'HOMICIDIO' in norm_upper or 'HOMICÍDIO' in norm_upper:
        natureza = 'HOMICÍDIO'
    elif (('LESAO' in norm_upper or 'LESÃO' in norm_upper) and 'BALA' in norm_upper) or 'LESÃO A BALA' in norm_upper:
        natureza = 'LESÃO A BALA'

    return natureza


def main():
    src = Path('data/exogenous_events.json')
    backup = src.with_suffix('.json.bak2')
    if not src.exists():
        print('File not found:', src)
        return

    data = json.loads(src.read_text(encoding='utf-8'))
    changed = []

    for i, item in enumerate(data):
        line = item.get('raw_text') or item.get('descricao') or ''
        new_nat = choose_natureza_from_line(line)
        old_nat = item.get('natureza') or ''
        if new_nat and new_nat != old_nat:
            changed.append((i, old_nat, new_nat, line))
            item['natureza'] = new_nat
            bairro = item.get('bairro') or ''
            item['resumo'] = f"{new_nat} em {bairro or 'local não identificado'}"

    # backup
    src.replace(backup)
    src.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Processed {len(data)} entries. Changed: {len(changed)}')
    for idx, old, new, line in changed[:200]:
        print(f'- idx {idx}: "{old}" -> "{new}"')


if __name__ == '__main__':
    main()
