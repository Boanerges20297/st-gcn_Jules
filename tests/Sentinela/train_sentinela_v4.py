import os, sys, json, warnings, pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

# Configurações
warnings.filterwarnings("ignore")

# Caminhos base
BASE_PATH = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
DATA_RAW = os.path.join(BASE_PATH, "data", "raw")
OUT_SENTINELA = os.path.join(BASE_PATH, "tests", "Sentinela")

def train_v4():
    print("[Sentinela V4] Iniciando Pipeline de Treinamento por Zona...")
    
    # 1. Carregar Inteligência Geográfica
    intel_path = os.path.join(OUT_SENTINELA, "sentinela_v4_intelligence.csv")
    if not os.path.exists(intel_path):
        print("Erro: Inteligencia V4 nao encontrada. Rode engine_intentionality.py.")
        return
    
    df_intel = pd.read_csv(intel_path)
    
    # 2. Engenharia de Features de Conectividade e Ruas Criticas
    # Requisito 7: Cruzar com ruas criticas ao longo do tempo
    streets_cache = os.path.join(BASE_PATH, "data", "geo_streets_cache.json")
    critical_weights = {}
    if os.path.exists(streets_cache):
        with open(streets_cache, "r", encoding="utf-8") as f:
            streets = json.load(f)
            for s in streets:
                b = s.get("bairro", "").upper()
                critical_weights[b] = critical_weights.get(b, 0) + s.get("ocorrencias", 0)
    
    df_intel["bairro_norm"] = df_intel["bairro"].str.upper()
    df_intel["vial_criticality"] = df_intel["bairro_norm"].map(lambda x: critical_weights.get(x, 0))
    
    # 3. Preparar Dataset
    # Como temos apenas 5 zonas (para teste), vamos criar um pequeno dataset sintético de treino
    # baseado nas proporções históricas para demonstrar o aprendizado.
    # No pipeline real, isso usaria o histórico temporal de cada zona.
    
    X = df_intel[["cvli_total", "score_intel", "gap_index", "vial_criticality"]].copy()
    
    # Target sintético: Alta probabilidade onde gap_index > 50 e vial_criticality > 0
    y = ((df_intel["gap_index"] > 50) | (df_intel["vial_criticality"] > 10)).astype(int)
    
    # Adicionar encoding de facção
    df_intel["faction_code"] = pd.factorize(df_intel["faction"])[0]
    X["faction_code"] = df_intel["faction_code"]
    
    print(f"Dataset de Treino: {len(X)} zonas processadas.")
    
    # 4. Treinar Modelos (LGBM Lean V4)
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        importance_type='gain'
    )
    
    model.fit(X, y)
    
    # 5. Salvar Modelo e Metadados
    payload = {
        "model": model,
        "feature_names": X.columns.tolist(),
        "zones_meta": df_intel.to_dict(orient="records"),
        "trained_at": str(datetime.now()),
        "janela_historica": "60 dias",
        "horizonte_predicao": "14 dias"
    }
    
    model_out = os.path.join(OUT_SENTINELA, "sentinela_v4_model.pkl")
    with open(model_out, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"Sucesso: Modelo V4 treinado e salvo em {model_out}")
    print("Importancia de Features:")
    for f, imp in zip(X.columns, model.feature_importances_):
        print(f" - {f}: {imp:.2f}")

if __name__ == "__main__":
    train_v4()
