import torch
import os
import sys
import numpy as np

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64
except ImportError:
    from src.core.architectures import DeepSTGAT_64

def analyze_weights(region_key):
    model_path = f'models/active/{region_key}_model.pth'
    if not os.path.exists(model_path):
        print(f"Modelo {region_key} não encontrado.")
        return

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Pegar pesos da primeira camada: layer1.time_conv.weight
    # Shape esperado: [32, 29, 1, 3] -> (out_channels, in_channels, k_h, k_w)
    weights = state_dict['layer1.time_conv.weight'].numpy()
    
    # Calcular a magnitude média (L1 norm) por canal de entrada (in_channels = 29)
    # Somamos os valores absolutos sobre as dimensões de saída e kernel
    channel_importance = np.mean(np.abs(weights), axis=(0, 2, 3))
    
    # Mapeamento de Canais
    channel_names = [
        "00: CVLI (Homicídios)",
        "01: VEHICLE (Roubos/Furtos)",
        "02: Tensão (Estrutural/Facção)",
        "03: DOW (Segunda)", "04: DOW (Terça)", "05: DOW (Quarta)", "06: DOW (Quinta)", 
        "07: DOW (Sexta)", "08: DOW (Sábado)", "09: DOW (Domingo)",
        "10: MONTH (Jan)", "11: MONTH (Fev)", "12: MONTH (Mar)", "13: MONTH (Abr)", 
        "14: MONTH (Mai)", "15: MONTH (Jun)", "16: MONTH (Jul)", "17: MONTH (Ago)", 
        "18: MONTH (Set)", "19: MONTH (Out)", "20: MONTH (Nov)", "21: MONTH (Dez)",
        "22: Weekend (Fim de Semana)",
        "23: Supressão (Ação Policial - Alívio)",
        "24: Exógeno (Tensão/Ameaça)",
        "25: Crítico (Alertas/Chacinas)",
        "26: Incursion (Invasões Território)",
        "27: Disponível/Infraestrutura",
        "28: Pulso Global (Crime no Estado)"
    ]

    # Ordenar por importância
    results = sorted(zip(channel_names, channel_importance), key=lambda x: x[1], reverse=True)

    print(f"\n--- ANÁLISE DE PESOS DOS CANAIS: {region_key.upper()} ---")
    print(f"{'Canal':<40} | {'Magnitude Média':<15}")
    print("-" * 60)
    for name, imp in results:
        print(f"{name:<40} | {imp:.6f}")

if __name__ == "__main__":
    analyze_weights('fortaleza')
    analyze_weights('rmf')
    analyze_weights('interior')
