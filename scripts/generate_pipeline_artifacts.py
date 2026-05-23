#!/usr/bin/env python3
"""
Pipeline script to generate optimized analytics artifacts for the Report Preview gateway.
Produces pre-computed summaries to conserve Gemini API tokens and RPM.

Artifacts:
1. outputs/hermes/dados_brutos_30dias.csv
2. outputs/hermes/dados_brutos_60dias.csv
3. outputs/hermes/dados_brutos_90dias.csv
4. outputs/hermes/total_cvli_rua.csv
5. outputs/hermes/total_cvli_micronodo.csv
6. outputs/hermes/caminho_crime.csv
"""

import os
import sys
import math
from datetime import timedelta
from pathlib import Path
import pandas as pd
from ais_lookup import AISLookup
import numpy as np

# Setup path resolution
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Standardize Output Directory
OUT_DIR = BASE_DIR / 'outputs' / 'hermes'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Standardize Input Paths
ENRIQUECIDO_CSV_CANDIDATES = [
    BASE_DIR / 'data' / 'raw' / 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv',
    BASE_DIR / 'data' / 'raw' / 'dados_status_enriquecido.csv'
]
MICRONODES_CSV = BASE_DIR / 'data' / 'raw' / 'inteligencia' / 'micronodos_faccoes_2026.csv'

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates geodetic distance in kilometers between coordinates."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

def load_enriched_dataset():
    """Tries candidate files to load the main enriched dataset."""
    for path in ENRIQUECIDO_CSV_CANDIDATES:
        if path.exists():
            print(f"[DATASET] Lendo base de dados enriquecida de: {path.name} ({path.stat().st_size / 1024**2:.2f} MB)...")
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"Erro: Dataset não encontrado em nenhuma das rotas candidatas: {[p.name for p in ENRIQUECIDO_CSV_CANDIDATES]}")

def build_temporal_slices(df):
    """Generates 30, 60, and 90 days raw occurrences relative to max_date."""
    print("[TEMPORAL] Processando recortes temporais (30, 60, 90 dias)...")
    # Parse dates
    df['parsed_date'] = pd.to_datetime(df['data'], errors='coerce')
    max_date = df['parsed_date'].max()
    if pd.isnull(max_date):
        print("  [WARN] Erro: Não foi possível inferir a data máxima do dataset. Usando fallback 2026-05-15.")
        max_date = pd.to_datetime('2026-05-15')
    
    print(f"  Data de referência máxima inferida: {max_date.strftime('%Y-%m-%d')}")
    
    # Save slices
    for days in [30, 60, 90]:
        cutoff = max_date - timedelta(days=days)
        slice_df = df[df['parsed_date'] >= cutoff].copy()
        
        # Drop temporary parsing column before saving
        slice_df = slice_df.drop(columns=['parsed_date'])
        
        out_path = OUT_DIR / f"dados_brutos_{days}dias.csv"
        slice_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"  [OK] Salvo: {out_path.name} | Ocorrencias: {len(slice_df)}")

def build_historical_street_ranking(df):
    """Aggregates historical CVLI counts grouped by street and neighborhood."""
    print("[STREETS] Consolidando ranking de homicidios historicos por ruas...")
    cvli = df[df['tipo'] == 'cvli'].copy()
    
    # Drop rows without street name or with placeholders
    cvli = cvli.dropna(subset=['name'])
    cvli = cvli[cvli['name'].astype(str).str.strip().str.upper().replace('', np.nan).notnull()]
    
    # Standardize values to upper case for clean normalization
    cvli['cidade'] = cvli['cidade'].fillna('DESCONHECIDO').astype(str).str.strip().str.upper()
    cvli['bairro'] = cvli['bairro'].fillna('DESCONHECIDO').astype(str).str.strip().str.upper()
    cvli['name'] = cvli['name'].astype(str).str.strip().str.upper()
    
    # Group and count
    street_counts = cvli.groupby(['cidade', 'bairro', 'name']).size().reset_index(name='cvli_count')
    street_counts = street_counts.rename(columns={'name': 'rua'})
    street_counts = street_counts.sort_values(by='cvli_count', ascending=False).reset_index(drop=True)
    
    out_path = OUT_DIR / "total_cvli_rua.csv"
    street_counts.to_csv(out_path, index=False, encoding='utf-8')
    print(f"  [OK] Salvo: {out_path.name} | Ruas mapeadas: {len(street_counts)}")

def build_micronode_intensity_mapping(df):
    """Maps historical CVLIs to active micronodes with multi-distance buffer counts."""
    print("[MICRONODES] Calculando intensidade de homicidios por micronodos ativos...")
    if not MICRONODES_CSV.exists():
        print(f"  [WARN] Alerta: Micronodos ativos nao encontrados em {MICRONODES_CSV.name}. Pulando geracao do artefato.")
        return
        
    micronodes = pd.read_csv(MICRONODES_CSV)
    cvli = df[(df['tipo'] == 'cvli') & df['latitude'].notnull() & df['longitude'].notnull()].copy()
    
    cvli_lats = cvli['latitude'].values
    cvli_lons = cvli['longitude'].values
    
    micro_lats = micronodes['lat'].values
    micro_lons = micronodes['long'].values
    
    # Fast vectorized distance calculation to assign nearest micronode
    closest_idx = []
    closest_dist = []
    
    for clat, clon in zip(cvli_lats, cvli_lons):
        dists = haversine_distance(clat, clon, micro_lats, micro_lons)
        idx = np.argmin(dists)
        closest_idx.append(idx)
        closest_dist.append(dists[idx])
        
    cvli['closest_micro_idx'] = closest_idx
    cvli['dist_km'] = closest_dist
    
    # Pre-calculate counts for concentric buffers
    count_500m = np.zeros(len(micronodes), dtype=int)
    count_1km = np.zeros(len(micronodes), dtype=int)
    count_2km = np.zeros(len(micronodes), dtype=int)
    count_total = np.zeros(len(micronodes), dtype=int)
    
    for idx, dist in zip(closest_idx, closest_dist):
        count_total[idx] += 1
        if dist <= 0.5:
            count_500m[idx] += 1
        if dist <= 1.0:
            count_1km[idx] += 1
        if dist <= 2.0:
            count_2km[idx] += 1
            
    # Enrich micronodes dataframe
    micronodes['cvli_count_500m'] = count_500m
    micronodes['cvli_count_1km'] = count_1km
    micronodes['cvli_count_2km'] = count_2km
    micronodes['cvli_count_total'] = count_total
    
    # Sort descending by primary standard (1km pressure)
    micronodes = micronodes.sort_values(by='cvli_count_1km', ascending=False).reset_index(drop=True)
    
    out_path = OUT_DIR / "total_cvli_micronodo.csv"
    micronodes.to_csv(out_path, index=False, encoding='utf-8')
    print(f"  [OK] Salvo: {out_path.name} | Micronodos ativos analisados: {len(micronodes)}")

def build_crime_path_transitions(df):
    """Generates the chronological crime migration pathways (caminho_crime) grouped by AIS."""
    print("[PATHWAY] Tracando rota cronologica de homicidios e roubos (caminho_crime) por tipo e AIS...")
    cvli = df[(df['tipo'].isin(['cvli', 'cvp'])) & df['latitude'].notnull() & df['longitude'].notnull()].copy()
    
    # Vectorized precise datetime assembly
    cvli['data'] = cvli['data'].astype(str)
    cvli['hora'] = cvli['hora'].fillna('00:00:00').astype(str)
    datetime_str = cvli['data'] + ' ' + cvli['hora']
    cvli['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
    
    # Drop rows with invalid parsed datetimes
    cvli = cvli.dropna(subset=['datetime'])
    
    # Sort globally before groupings
    cvli = cvli.sort_values(by='datetime').reset_index(drop=True)
    
    transitions = []
    
    # Group by tipo and AIS
    for (crime_type, ais_name), group in cvli.groupby(['tipo', 'ais']):
        group = group.sort_values(by='datetime').copy()
        if len(group) < 2:
            # Single occurrence has no next transition
            group['prox_bairro'] = np.nan
            group['prox_rua'] = np.nan
            group['prox_lat'] = np.nan
            group['prox_lon'] = np.nan
            group['dias_para_prox'] = np.nan
            group['distancia_para_prox_km'] = np.nan
            transitions.append(group)
            continue
            
        group['prox_bairro'] = group['bairro'].shift(-1)
        group['prox_rua'] = group['name'].shift(-1)
        group['prox_lat'] = group['latitude'].shift(-1)
        group['prox_lon'] = group['longitude'].shift(-1)
        
        # Calculate time delta in days
        group['prox_datetime'] = group['datetime'].shift(-1)
        group['dias_para_prox'] = (group['prox_datetime'] - group['datetime']).dt.total_seconds() / (24.0 * 3600.0)
        
        # Calculate spatial displacement in km
        group['distancia_para_prox_km'] = haversine_distance(
            group['latitude'].values, group['longitude'].values,
            group['prox_lat'].values, group['prox_lon'].values
        )
        
        # Drop temporary parsing datetime
        group = group.drop(columns=['prox_datetime'])
        transitions.append(group)
        
    if transitions:
        cvli_transitions = pd.concat(transitions).sort_values(by='datetime').reset_index(drop=True)
        
        # Select and format outputs
        output_cols = [
            'datetime', 'tipo', 'cidade', 'ais', 'regiao_risp', 'bairro', 'name', 'latitude', 'longitude',
            'prox_bairro', 'prox_rua', 'dias_para_prox', 'distancia_para_prox_km'
        ]
        # Keep only existing columns
        output_cols = [c for c in output_cols if c in cvli_transitions.columns]
        output_df = cvli_transitions[output_cols].rename(columns={'name': 'rua'})
        
        # Round numerical metrics
        if 'dias_para_prox' in output_df.columns:
            output_df['dias_para_prox'] = output_df['dias_para_prox'].round(3)
        if 'distancia_para_prox_km' in output_df.columns:
            output_df['distancia_para_prox_km'] = output_df['distancia_para_prox_km'].round(3)
            
        out_path = OUT_DIR / "caminho_crime.csv"
        output_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"  [OK] Salvo: {out_path.name} | Relacoes sequenciais: {len(output_df)}")
    else:
        print("  [WARN] Alerta: Nenhuma transicao cronologica pode ser computada.")

def enrich_ais_column(df):
    """Remaps the 'ais' column using the official AIS_Territorios.csv (34 AIS)."""
    print("[AIS] Enriquecendo coluna 'ais' com mapeamento oficial (AIS_Territorios.csv)...")
    try:
        lookup = AISLookup(str(BASE_DIR))
        ais_series, risp_series = lookup.resolve_series(
            df['cidade'] if 'cidade' in df.columns else pd.Series([''] * len(df)),
            df['bairro'] if 'bairro' in df.columns else pd.Series([''] * len(df))
        )
        # Preserve original AIS for reference
        if 'ais' in df.columns:
            df['ais_original'] = df['ais']
        df['ais'] = ais_series.values
        df['regiao_risp'] = risp_series.values

        matched = (df['ais'] != '').sum()
        total = len(df)
        print(f"  [OK] Mapeamento AIS aplicado: {matched}/{total} ocorrencias mapeadas ({matched/total*100:.1f}%)")
        unmatched = df[df['ais'] == '']
        if len(unmatched) > 0:
            unmatched_cities = unmatched['cidade'].fillna('').str.upper().value_counts().head(5)
            print(f"  [INFO] Top 5 cidades sem match AIS: {dict(unmatched_cities)}")
    except FileNotFoundError as e:
        print(f"  [WARN] AIS_Territorios.csv nao encontrado: {e}. Mantendo coluna 'ais' original.")
    except Exception as e:
        print(f"  [WARN] Erro ao enriquecer AIS: {e}. Mantendo coluna 'ais' original.")
    return df

def main():
    print("==================================================")
    print("INICIANDO GERACAO DE ARTEFATOS DO PIPELINE DE DADOS")
    print("==================================================")
    
    try:
        df = load_enriched_dataset()
        print(f"  Dataset carregado com {len(df)} ocorrencias totais.")
        
        # Enrich AIS column with official mapping before generating artifacts
        df = enrich_ais_column(df)
        
        # Generate the 6 requested artifacts
        build_temporal_slices(df)
        build_historical_street_ranking(df)
        build_micronode_intensity_mapping(df)
        build_crime_path_transitions(df)
        
        print("\nTODOS OS 6 ARTEFATOS GERADOS COM SUCESSO EM outputs/hermes/")
        print("==================================================")
    except Exception as e:
        print(f"\nERRO CRITICO NO PIPELINE: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
