import json
import os
import matplotlib.pyplot as plt

def generate_dashboard(region_key):
    path = f'logs/metrics_{region_key}_ELITE.json'
    if not os.path.exists(path):
        print(f"Aguardando dados de {region_key}...")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    epochs = [d['epoch'] for d in data]
    loss = [d['loss'] for d in data]
    p10 = [d['p10'] * 100 for d in data]
    grad = [d['grad'] for d in data]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(epochs, loss, color=color_loss, marker='o', label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color_p10 = 'tab:blue'
    ax2.set_ylabel('P@10 (%)', color=color_p10)
    ax2.plot(epochs, p10, color=color_p10, marker='s', label='P@10', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_p10)

    ax3 = ax1.twinx()
    # Deslocar o terceiro eixo para a direita
    ax3.spines['right'].set_position(('outward', 60))
    color_grad = 'tab:green'
    ax3.set_ylabel('Grad Norm (L2)', color=color_grad)
    ax3.plot(epochs, grad, color=color_grad, marker='^', label='Grad Norm', linewidth=1, linestyle=':')
    ax3.tick_params(axis='y', labelcolor=color_grad)

    plt.title(f'Dashboard de Treinamento ELITE P10 - {region_key.upper()}')
    fig.tight_layout()
    
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig(f'outputs/plots/dashboard_{region_key}_ELITE.png', dpi=150)
    plt.close()
    print(f"✅ Dashboard gerado para {region_key}")

if __name__ == "__main__":
    for reg in ['fortaleza', 'rmf', 'interior']:
        generate_dashboard(reg)
