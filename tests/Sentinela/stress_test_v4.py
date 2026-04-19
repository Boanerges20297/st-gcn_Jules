import os, sys, json, pickle
import pandas as pd
import numpy as np

# Caminhos
BASE_PATH = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
OUT_SENTINELA = os.path.join(BASE_PATH, "tests", "Sentinela")

def run_stress_test():
    print("=== [Sentinela V4] STRESS TEST DE SENSIBILIDADE ===")
    
    # 1. Carregar Modelo e Metadados originais
    model_path = os.path.join(OUT_SENTINELA, "sentinela_v4_model.pkl")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    
    df_meta = pd.DataFrame(payload["zones_meta"])
    model = payload["model"]
    feat_names = payload["feature_names"]
    
    # 2. Isolar Carire_Z4 antes da mudanca
    target_zone = "CARIRE_Z4"
    z_before = df_meta[df_meta["zone_id"] == target_zone].copy()
    
    # 3. Simular Inferencia Original (Ranking atual)
    def calibrate(scores, max_val):
        return (scores / max_val * 95).clip(0, 100)

    raw_before = (model.predict_proba(z_before[feat_names])[:, 1] * 0.4 + 
                  (z_before["gap_index"] / 100).clip(0, 1) * 0.4 + 
                  (z_before["vial_criticality"] / 10).clip(0, 1) * 0.2).iloc[0]
    
    # Usar o max_raw do estado para manter a proporcao global
    all_raw = (model.predict_proba(df_meta[feat_names])[:, 1] * 0.4 + 
               (df_meta["gap_index"] / 100).clip(0, 1) * 0.4 + 
               (df_meta["vial_criticality"] / 10).clip(0, 1) * 0.2)
    max_global = all_raw.max()
    
    risk_before = (raw_before / max_global * 95)
    
    print(f"\n[ZONA: {target_zone}]")
    print(f"ESTADO ATUAL:")
    print(f" - CVLI (60d): {z_before['cvli_total'].iloc[0]}")
    print(f" - Intel (Score): {z_before['score_intel'].iloc[0]}")
    print(f" - GAP Index: {z_before['gap_index'].iloc[0]}")
    print(f" - RISCO CALIBRADO: {risk_before:.1f}%")
    
    # 4. INJETAR 5 FUZIS (Stress Test)
    # 5 fuzis = 5 * 15 (Peso nature) = 75 pontos de Intel
    print(f"\n>>> INJETANDO INTELIGENCIA: Apreensao de 5 fuzis no Carire...")
    z_after = z_before.copy()
    z_after["score_intel"] = 75.0
    # Recalcular GAP: cvli / (intel + 0.1)
    z_after["gap_index"] = z_after["cvli_total"] / (z_after["score_intel"] + 0.1)
    
    # 5. Simular Inferencia de Stress
    raw_after = (model.predict_proba(z_after[feat_names])[:, 1] * 0.4 + 
                 (z_after["gap_index"] / 100).clip(0, 1) * 0.4 + 
                 (z_after["vial_criticality"] / 10).clip(0, 1) * 0.2).iloc[0]
    
    risk_after = (raw_after / max_global * 95)
    
    print(f"\nESTADO APOS APREENSAO:")
    print(f" - Intel (Score): {z_after['score_intel'].iloc[0]} (Subiu!)")
    print(f" - GAP Index: {z_after['gap_index'].iloc[0]:.2f} (Caiu drasticamente!)")
    print(f" - NOVO RISCO CALIBRADO: {risk_after:.1f}%")
    
    delta = risk_before - risk_after
    print(f"\nSENSIBILIDADE: O Risco caiu {delta:.1f}% com a acao policial.")
    if risk_after < 80:
        print("RESULTADO: A zona deixaria de ser 'CRITICA' no Dashboard.")

if __name__ == "__main__":
    run_stress_test()
