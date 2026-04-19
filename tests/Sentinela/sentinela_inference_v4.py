import os, sys, json, warnings, pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configurações
warnings.filterwarnings("ignore")

# Caminhos base
BASE_PATH = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
OUT_SENTINELA = os.path.join(BASE_PATH, "tests", "Sentinela")

def run_inference_v4():
    print("[Sentinela V4] Gerando Ranking Granular por Zonas...")
    
    # 1. Carregar Modelo e Metadados
    model_path = os.path.join(OUT_SENTINELA, "sentinela_v4_model.pkl")
    if not os.path.exists(model_path):
        print("Erro: Modelo V4 nao encontrado. Rode train_sentinela_v4.py.")
        return
    
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    
    model = payload["model"]
    feat_names = payload["feature_names"]
    zones_meta = payload["zones_meta"]
    
    # 2. Preparar Dados de Entrada (Inferencia)
    df_inf = pd.DataFrame(zones_meta)
    X = df_inf[feat_names]
    
    # 3. Predizer Probabilidades (V4 - Hotspots)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X))
    
    # 4. Cálculo de Score Bruto e Calibragem (0-100)
    # Combinamos Probabilidade, GAPS (Vácuo Policial) e Infraestrutura (Vial)
    raw_scores = (probs * 0.4 + 
                 (df_inf["gap_index"] / 100).clip(0, 1) * 0.4 + 
                 (df_inf["vial_criticality"] / 10).clip(0, 1) * 0.2)
    
    # Calibragem Global: Mapear o maior risco do estado para ~95%
    max_raw = raw_scores.max() if len(raw_scores) > 0 else 1
    # Multiplicador para atingir a escala 0-100 baseada no teto do estado
    df_inf["risk_index"] = (raw_scores / max_raw * 95).clip(0, 100)
    
    # 5. Ordenar e Classificar conforme Tabela do Usuario
    df_inf = df_inf.sort_values(by="risk_index", ascending=False)
    
    ranking_v4 = []
    for rank, (_, row) in enumerate(df_inf.iterrows(), 1):
        idx = row["risk_index"]
        if idx >= 80:
            status = "CRITICO"
        elif idx >= 60:
            status = "ALTO"
        elif idx >= 30:
            status = "MODERADO"
        else:
            status = "BAIXO"
            
        ranking_v4.append({
            "rank": rank,
            "zone_id": row["zone_id"],
            "bairro": row["bairro"],
            "indice_risco": round(float(idx), 1),
            "status": status,
            "gap_index": round(float(row["gap_index"]), 2),
            "natureza_critica": row["faction"],
            "centroide": {"lat": row["lat"], "lng": row["lng"]}
        })
    
    # 5. Exportar JSON de Transmissao
    output_json = os.path.join(OUT_SENTINELA, "ranking_sentinela_v4.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "tipo": "ZONAS_500M_V4",
            "janela_historica": "60 dias",
            "horizonte_predicao": "14 dias",
            "total_zonas": len(ranking_v4),
            "ranking": ranking_v4
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Sucesso: Ranking V4 exportado para {output_json}")
    print("Predicao para os proximos 14 dias com base nos ultimos 60 dias.")
    # Preview Top 5
    for r in ranking_v4[:5]:
        print(f"#{r['rank']} {r['zone_id']} - Risco: {r['indice_risco']}% - GAP: {r['gap_index']}")

if __name__ == "__main__":
    run_inference_v4()
