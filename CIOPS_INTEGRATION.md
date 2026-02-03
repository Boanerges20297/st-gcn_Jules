# CIOPS Daily Report Parser - Integration Guide

## Overview
Parseador de relatórios CIOPS (Centro Integrado de Operações de Segurança) que extrai eventos operacionais e criminais com classificação automática para canal 9 (enforcement activity).

## Features

### 1. **Block Detection**
- Detecta automaticamente blocos separados por headers (===)
- Suporta múltiplas seções: OCORRÊNCIAS, HOMICÍDIOS, LESÃO À BALA, DESLOCAMENTO FORÇADO, etc.
- Ignora seções vazias (S/A)

### 2. **Line Parsing (Deterministic)**
Formato suportado:
```
NN - INCIDENT_ID - CONTEXT - NATUREZA - LOCATION - TIMESTAMP - ADDITIONAL
01 - M20260083825 - RAIO 01 - ST * JEFFERSON - PESSOA/SITUAÇÃO SUSPEITA - BARROSO - 07:37 - UM CONDUZIDO
```

Extrai:
- `incident_id`: M20260083825
- `block_type`: OCORRÊNCIAS, HOMICÍDIOS, etc
- `natureza`: PESSOA/SITUAÇÃO SUSPEITA
- `localizacao_completa`: BARROSO
- `municipio`: FORTALEZA (default)
- `timestamp`: 07:37
- `num_arrested`: 0-N (extrai números antes de "PRESO")
- `has_drugs`: boolean (detecta DROGA, ENTORPECENTE, CRACK, COCAÍNA)
- `has_weapons`: boolean (detecta ARMA, PISTOLA, ESPINGARDA, FUZIL)

### 3. **Event Classification**
```
event_type = ENFORCEMENT_OPERATION | ENFORCEMENT_DISPLACEMENT | CRIME_HOMICIDE | CRIME_INJURY | CRIME_BODY
```

### 4. **Enforcement Intensity (Canal 9 Value)**
Calcula intensidade 0.0-1.0 para canal 9:
- **Presos**: intensity = min(1.0, num_arrested / 5.0)
  - 1 preso = 0.20
  - 2 presos = 0.40
  - 5+ presos = 1.00
- **Drogas/Armas**: intensity = max(previous, 0.70)
- **Deslocamento Forçado**: intensity = max(0.50, previous)

Exemplo:
```
M20260084656: DROGA + ARMA + 0 PRESOS → intensity = 0.70
M20260084202: 2 PRESOS → intensity = 0.40
M20260083922: DESLOCAMENTO + AMEAÇAS → intensity = 0.50
```

### 5. **Conflict Severity**
```
HIGH:   HOMICÍDIO, LESÃO À BALA, CADÁVER
MEDIUM: DESLOCAMENTO FORÇADO, EXPULSÃO, AMEAÇAS
LOW:    Operações de rotina, achado de droga sem confronto
```

## Usage

### API Endpoint
```bash
POST /api/ciops/parse-report

{
  "report": "========= OCORRÊNCIAS ==========\n01 - M20260083825 - ..."
}

Response:
{
  "status": "success",
  "events": [
    {
      "incident_id": "M20260083825",
      "block_type": "OCORRÊNCIAS",
      "event_type": "ENFORCEMENT_OPERATION",
      "natureza": "PESSOA/SITUAÇÃO SUSPEITA",
      "localizacao_completa": "BARROSO",
      "bairro": "BARROSO",
      "municipio": "FORTALEZA",
      "timestamp": "07:37",
      "conflict_severity": "LOW",
      "enforcement_intensity": 0.0,
      "num_arrested": 0,
      "has_drugs": false,
      "has_weapons": false,
      "lat": -3.75,
      "lng": -38.52,
      "match_quality": "specific"
    },
    ...
  ],
  "summary": {
    "total": 10,
    "enforcement": 8,
    "crime": 2,
    "with_arrests": 3,
    "high_severity": 2,
    "medium_severity": 1,
    "low_severity": 7
  }
}
```

### Python Usage
```python
from src.llm_service import parse_ciops_report

report = """========= OCORRÊNCIAS ==========
01 - M20260083825 - RAIO 01 - ST * JEFFERSON - PESSOA/SITUAÇÃO SUSPEITA - BARROSO - 07:37 - UM CONDUZIDO
"""

events = parse_ciops_report(report)
for evt in events:
    print(f"{evt['incident_id']}: Intensity={evt['enforcement_intensity']:.2f}")
```

## Integration with Model (Canal 9)

### Current 8-channel pipeline:
1. CVLI (Homicídios) - count/day/node
2. CVP (Crimes Patrimoniais) - count/day/node
3. TENSION_INDEX - formula-based
4. DOW_SIN, DOW_COS - temporal
5. MONTH_SIN, MONTH_COS - temporal
6. IS_WEEKEND - binary

### New Canal 9:
**ENFORCEMENT_ACTIVITY** - Daily aggregated intensity from CIOPS:
- Type: continuous 0.0-1.0
- Aggregation: sum/day/node of enforcement_intensity values
- Lag effect: Applied during data preprocessing (3-7 day forward lag)
- Impact: Model learns police activity → crime reduction patterns

## Testing
```bash
python scripts/test_ciops_parser.py
```

Output shows:
- 10 events parsed (8 enforcement, 2 crime)
- Intensity values correctly calculated
- Location mapping verified
- Severity classification confirmed

## Next Steps

1. **Save CIOPS events to database**
   - Endpoint: `/api/ciops/save-events`
   - Store: data/processed/ciops_daily_events.json

2. **Aggregate for canal 9**
   - Sum enforcement_intensity by date/bairro
   - Store in processed_graph_data.pkl as channel 9

3. **Retrain model**
   - New shape: (319, 1491, 9) instead of (319, 1491, 8)
   - Update train.py: MODEL_INPUT_CHANNELS = 9
   - Update app.py: WINDOW_CHANNELS = 9

4. **Evaluate impact**
   - Expected gain: +1 to +3% in P@5
   - Monitor false positives (should decrease)
   - Track where enforcement activity has most impact

## Known Issues

- [ ] LLM parsing not yet enabled (GEMINI_API_KEY validation)
- [ ] Geographic encoding (lat/lng) fallback to city-level if bairro not found
- [ ] Hourly timestamps not captured (only HH:MM available)
- [ ] Some bairro names may need normalization (accents, CAPS)
