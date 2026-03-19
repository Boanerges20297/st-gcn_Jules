import torch
import os

model_path = 'models/active/fortaleza_model_active.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    print("--- METADADOS DO MODELO SALVO ---")
    p10 = ckpt.get('p10')
    if p10 is None:
        # Tentar chaves alternativas como 'p5' ou buscar no dicionário
        for k in ckpt.keys():
            if k.startswith('p'):
                p10 = ckpt[k]
                print(f"Métrica encontrada ({k}): {p10*100:.2f}%")
    else:
        print(f"P@10: {p10*100:.2f}%")
    
    print(f"Configuração: {ckpt.get('config', 'Não encontrada')}")
else:
    print(f"Modelo não encontrado em: {model_path}")
