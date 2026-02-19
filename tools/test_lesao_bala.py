import os
import sys
import json

# ensure repo root in path
sys.path.append(os.getcwd())

from src.llm_service import process_exogenous_text


def main():
    # Load the last event from the events file
    with open('data/exogenous_events.json', 'r', encoding='utf-8') as f:
        events = json.load(f)

    last = events[-1]
    raw = last.get('raw_text') or last.get('descricao') or ''

    print('--- Raw text to parse ---')
    print(raw)
    print('')

    # Force deterministic parsing for reproducible local test
    os.environ['DISABLE_GENAI_FOR_TESTS'] = '1'
    parsed = process_exogenous_text(raw)

    if not parsed:
        print('No events produced by parser')
        return

    print('--- Parsed events ---')
    for i, ev in enumerate(parsed):
        print(f'Event #{i}')
        for k in ('natureza', 'conflict_severity', 'resumo', 'bairro', 'municipio', 'timestamp'):
            print(f'  {k}: {ev.get(k)}')


if __name__ == '__main__':
    main()
