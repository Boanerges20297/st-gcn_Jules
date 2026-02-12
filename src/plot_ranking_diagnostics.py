import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DIAG_PATH = Path('outputs/diagnostics_ranking_integration.json')
OUT_DIR = Path('outputs/diagnostic_plots')
OUT_DIR.mkdir(exist_ok=True)

def load_diag():
    with open(DIAG_PATH, 'r') as f:
        return json.load(f)

def plot_score_distributions(diag):
    # Plot distribution of ranking scores (mean, std, min, max per window)
    r_means = [w['r_stats']['mean'] for w in diag['per_window']]
    r_stds = [w['r_stats']['std'] for w in diag['per_window']]
    r_maxs = [w['r_stats']['max'] for w in diag['per_window']]
    r_mins = [w['r_stats']['min'] for w in diag['per_window']]
    plt.figure(figsize=(8,4))
    plt.plot(r_means, label='mean')
    plt.plot(r_stds, label='std')
    plt.plot(r_maxs, label='max')
    plt.plot(r_mins, label='min')
    plt.title('Ranking score stats per window')
    plt.xlabel('Window')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/'ranking_score_stats.png')
    plt.close()

def plot_p5_curves(diag):
    # Plot P@5 for each method per window
    p5s = [w['p5_stgcn'] for w in diag['per_window']]
    p5r = [w['p5_rank'] for w in diag['per_window']]
    p5c = [w['p5_comb'] for w in diag['per_window']]
    plt.figure(figsize=(10,4))
    plt.plot(p5s, label='ST-GCN')
    plt.plot(p5r, label='Ranking')
    plt.plot(p5c, label='Hybrid (0.6/0.4)')
    plt.title('P@5 per window')
    plt.xlabel('Window')
    plt.ylabel('P@5')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/'p5_per_window.png')
    plt.close()

def plot_correlation_curves(diag):
    # Plot Pearson and Spearman correlations per window
    pear = [w['pearson'] for w in diag['per_window']]
    spear = [w['spearman'] for w in diag['per_window']]
    plt.figure(figsize=(10,4))
    plt.plot(pear, label='Pearson')
    plt.plot(spear, label='Spearman')
    plt.title('Correlation ST-GCN vs Ranking per window')
    plt.xlabel('Window')
    plt.ylabel('Correlation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/'correlation_per_window.png')
    plt.close()

def plot_p5_boxplots(diag):
    # Boxplot of P@5 by method
    p5s = [w['p5_stgcn'] for w in diag['per_window']]
    p5r = [w['p5_rank'] for w in diag['per_window']]
    p5c = [w['p5_comb'] for w in diag['per_window']]
    plt.figure(figsize=(6,4))
    plt.boxplot([p5s, p5r, p5c], labels=['ST-GCN','Ranking','Hybrid'])
    plt.title('P@5 distribution by method')
    plt.ylabel('P@5')
    plt.tight_layout()
    plt.savefig(OUT_DIR/'p5_boxplot.png')
    plt.close()

def main():
    diag = load_diag()
    plot_score_distributions(diag)
    plot_p5_curves(diag)
    plot_correlation_curves(diag)
    plot_p5_boxplots(diag)
    print('Plots saved to', OUT_DIR)

if __name__ == '__main__':
    main()