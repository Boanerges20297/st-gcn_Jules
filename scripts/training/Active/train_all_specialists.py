"""
train_all_specialists.py — Script oficial de retreino dos 3 especialistas regionais.

Incorpora os melhores achados do TRAINING_LOG.md (até Tentativa 45):

  FORTALEZA  — 33ch | window=120 | lr=0.01  | dropout=0.3 | margin=1.0 | K=10
               (Tent.45: PReLU + 4 canais momentum + Cold Streak negativo → 55.3% P@10)

  RMF        — 29ch | window=90  | lr=0.018 | dropout=0.5 | margin=1.5 | K=5
               (Tent.35: GAcc=8 steps → 74.1% P@5)

  INTERIOR   — 33ch | window=120 | lr=0.005 | dropout=0.3 | margin=1.0 | K=10
               (Nova config 2026-03-15: mesmo pipeline momentum do Fortaleza,
                adaptado para 50 nós com 98.8% de CVLI zero)

Cold Streak: negativo (0 a -30) — nós ativos quebram a sequência, nós frios
ficam em -30. Consistente com inferência em orchestrator.py.

Estratégia ativa de retreino (2026-03-16):
- Canal 0 (CVLI bruto) preservado sem normalização.
- Canal 24 reconstruído em memória como soma móvel 7d do CVLI bruto,
  substituindo a média móvel legada dos arquivos processados.
- Horizonte alvo de previsão ajustado para 14 dias.
- Objetivo: não suprimir picos/outliers que podem anteceder conflito iminente.
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import random
import gc
import subprocess

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64, DeepSTGAT_80, ShallowGAT
    from training_vault import TrainingVault
except ImportError:
    from src.core.architectures import DeepSTGAT_64, DeepSTGAT_80, ShallowGAT
    from src.core.training_vault import TrainingVault

import re
from datetime import datetime

# Configuração de Log
log_file = os.path.join(ROOT_DIR, 'logs', 'training_ALL_SPECIALISTS.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREDICT_HORIZON = 14

# ─────────────────────────────────────────────
# Configuração por região
# ─────────────────────────────────────────────
# Campos: window, lr, epochs, dropout, margin, k_eval
#         use_momentum (True → 33ch; False → 29ch)
#         grad_accum   (gradient accumulation steps)
#         output_name  (nome do .pth salvo em models/active/)
REGION_CONFIGS = {
    'fortaleza': dict(
        # T89: OneCycleLR → CosineAnnealingWarmRestarts (T0=10, Tmult=2)
        #      O modelo sempre pica na Época 1-3 com OneCycle — o LR já passou
        #      do ótimo antes da validação. Com Cosine+Restarts, o modelo tem
        #      múltiplos ciclos para buscar novos mínimos sem explodir.
        #      lr_base: 1e-4 → 3e-4 (começa mais "esperto", sem warmup lento)
        #      grad_accum mantido em 6 (ruído estocástico)
        #      dropout 0.35, wd 0.005 mantidos (T88 sem problema nesses)
        window=120, lr=1e-3, epochs=200, patience=40, dropout=0.35, margin=1.0,
        k_eval=10, use_momentum=True, grad_accum=6,
        raw_cvli_context=True,
        output_name='fortaleza_model_active.pth',
        focal_alpha=0.70, focal_gamma=2.5, ranking_weight=7.0,
        weight_decay=0.005,
        # Cosine scheduler params
        scheduler='cosine_restarts', cosine_T0=10, cosine_Tmult=2, cosine_eta_min=1e-6
    ),
    'rmf': dict(
        window=90,  lr=0.018, epochs=120, patience=20, dropout=0.5, margin=1.5,
        k_eval=5,  use_momentum=True, grad_accum=8,
        raw_cvli_context=True,
        output_name='rmf_model.pth',
        focal_alpha=0.50, focal_gamma=2.0, ranking_weight=7.0
    ),
    'interior': dict(
        window=120, lr=0.005, epochs=120, patience=20, dropout=0.3, margin=1.0,
        k_eval=10, use_momentum=True,  grad_accum=32,
        raw_cvli_context=True,
        output_name='interior_model.pth',
        focal_alpha=0.40, focal_gamma=2.0, ranking_weight=4.0
    ),
}

# ─────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────
def autosave_training_log(region_key, config):
    """Grava as configurações de treinamento no TRAINING_LOG.md de forma permanente e incremental."""
    log_path = os.path.join(ROOT_DIR, 'TRAINING_LOG.md')
    if not os.path.exists(log_path):
        return

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Encontra o número máximo de Tentativa atual
        attempts = re.findall(r'## Tentativa (\d+)', content)
        next_attempt = max([int(x) for x in attempts]) + 1 if attempts else 1

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

        new_entry = f"\n\n## Tentativa {next_attempt} (Autolog - {region_key.upper()}) — {now_str}\n"
        new_entry += f"**Arquivo de Origem:** `train_all_specialists.py`\n\n"
        new_entry += f"### 1. Hiperparâmetros (Carga Automática)\n"
        new_entry += f"- **Target (Horizonte)**: {PREDICT_HORIZON} dias\n"
        new_entry += f"- **Janela (Window)**: {config.get('window', 'N/A')} dias\n"
        new_entry += f"- **Learning Rate**: {config.get('lr', 'N/A')}\n"
        new_entry += f"- **Dropout**: {config.get('dropout', 'N/A')}\n"
        new_entry += f"- **Épocas**: {config.get('epochs', 'N/A')}\n"
        new_entry += f"- **Patience**: {config.get('patience', 'N/A')}\n"
        new_entry += f"- **Grad Accumulation**: {config.get('grad_accum', 'N/A')}\n\n"
        new_entry += f"### 2. Loss & Ranking\n"
        new_entry += f"- **Focal Alpha**: {config.get('focal_alpha', 'N/A')}\n"
        new_entry += f"- **Focal Gamma**: {config.get('focal_gamma', 'N/A')}\n"
        new_entry += f"- **Ranking Weight**: {config.get('ranking_weight', 'N/A')}\n"
        new_entry += f"- **Métrica de Avaliação**: P@{config.get('k_eval', 'N/A')}\n\n"
        new_entry += f"### 3. Resultados\n"
        new_entry += f"- *(A preencher após a conclusão)*\n\n"
        new_entry += f"---\n"

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(new_entry)

        logging.info(f"📝 Auto-log incremental: Tentativa {next_attempt} registrada no TRAINING_LOG.md")
    except Exception as e:
        logging.error(f"❌ Erro ao escrever log auto-incremental em TRAINING_LOG.md: {e}")

def normalize_adj(adj):
    """
    V5 ROW NORMALIZATION (Random Walk):
    Em vez de esmagar simetricamente (o que mata a tática) ou passar bruto (o que abafa a 
    história do próprio nó com um tsunami de sinal dos vizinhos), fazemos a normalização 
    direcional D^-1 A.
    Isso faz com que a soma das "atenções" aos vizinhos seja 1.0, mantendo a escala equilibrada 
    com h_self, MAS preserva a proporção exata: o vizinho frágil continua recebendo 15x mais 
    foco que o vizinho comum!
    """
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(adj)


def build_momentum_features(features):
    """
    Computa 4 canais de Multi-Scale Momentum + Cold Streak.
      Ch+0 : Δ7d   (recent_7  − past_7)
      Ch+1 : Δ14d  (recent_14 − past_14)
      Ch+2 : Δ30d  (recent_30 − past_30)
      Ch+3 : Cold Streak negativo (0 a -30) — negativo para que nós
             ativos (streak=0) estejam ACIMA dos frios (streak=-30).
    """
    N, T, _ = features.shape
    momentum_feat = np.zeros((N, T, 4))
    cold_streak = np.zeros(N)
    for t in range(60, T):
        r7  = features[:, t-7:t,   0].sum(axis=1)
        p7  = features[:, t-14:t-7, 0].sum(axis=1)
        momentum_feat[:, t, 0] = r7 - p7

        r14 = features[:, t-14:t,   0].sum(axis=1)
        p14 = features[:, t-28:t-14, 0].sum(axis=1)
        momentum_feat[:, t, 1] = r14 - p14

        r30 = features[:, t-30:t,   0].sum(axis=1)
        p30 = features[:, t-60:t-30, 0].sum(axis=1)
        momentum_feat[:, t, 2] = r30 - p30

        crimes = features[:, t, 0]
        cold_streak = np.where(crimes > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)   # negativo
    return momentum_feat


def rebuild_raw_cvli_context(features):
    """
    Reconstrói canais derivados de CVLI diretamente do sinal bruto.

    Hoje corrigimos explicitamente o canal 24, que nos artefatos legados pode vir
    como média móvel 7d. Para retreino operacional, usamos soma móvel 7d para não
    amortecer picos pequenos porém relevantes.
    """
    rebuilt = features.copy()
    num_nodes = rebuilt.shape[0]
    for node_idx in range(num_nodes):
        rebuilt[node_idx, :, 24] = (
            np.convolve(rebuilt[node_idx, :, 0], np.ones(7, dtype=np.float32), mode='full')[:rebuilt.shape[1]]
        )
    return rebuilt


def run_repo_git_command(args):
    return subprocess.run(
        ['git', *args],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=180,
        check=False,
        env={
            **os.environ,
            'PYTHONIOENCODING': 'utf-8',
        },
    )


def publish_training_artifacts(trained_regions, failed_regions):
    logging.info("📤 Verificando alterações git do repositório principal após o treino...")

    status_result = run_repo_git_command(['status', '--porcelain'])
    if status_result.returncode != 0:
        raise RuntimeError(
            f"Falha ao consultar status git do repositório principal: "
            f"{status_result.stderr.strip() or status_result.stdout.strip()}"
        )

    changed_entries = [line for line in status_result.stdout.splitlines() if line.strip()]
    if not changed_entries:
        logging.info("ℹ️ Nenhuma alteração detectada para publicar no repositório principal.")
        return

    add_result = run_repo_git_command(['add', '-A'])
    if add_result.returncode != 0:
        raise RuntimeError(f"Falha no git add do repositório principal: {add_result.stderr.strip() or add_result.stdout.strip()}")

    trained_label = '-'.join(trained_regions) if trained_regions else 'sem-regioes'
    status_label = 'partial' if failed_regions else 'complete'
    commit_message = (
        f"chore: finalize specialists training {status_label} "
        f"[{trained_label}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    commit_result = run_repo_git_command(['commit', '-m', commit_message])
    if commit_result.returncode != 0:
        combined_output = (commit_result.stderr or commit_result.stdout or '').strip()
        if 'nothing to commit' not in combined_output.lower():
            raise RuntimeError(f"Falha no git commit do repositório principal: {combined_output}")

    pull_result = run_repo_git_command(['pull', '--rebase', 'origin', 'main'])
    if pull_result.returncode != 0:
        logging.warning(
            f"⚠️ Falha no git pull --rebase antes do push: "
            f"{pull_result.stderr.strip() or pull_result.stdout.strip()}"
        )

    push_result = run_repo_git_command(['push', 'origin', 'main'])
    if push_result.returncode != 0:
        raise RuntimeError(f"Falha no git push do repositório principal: {push_result.stderr.strip() or push_result.stdout.strip()}")

    logging.info(f"✅ Alterações do treino publicadas no repositório principal com commit: {commit_message}")


class BinaryFocalRankingLoss(nn.Module):
    """
    Focal Loss adaptada para Ranking. 
    Foca nos hotspots (classe minoritária) e penaliza erros em áreas críticas.
    """
    def __init__(self, alpha=0.25, gamma=2.0, ranking_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ranking_weight = ranking_weight

    def forward(self, pred, target):
        # Target binário para Focal (houve ou não crime no horizonte)
        target_bin = (target > 0).float()
        
        # Binary Focal Loss
        probs = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_bin, reduction='none')
        p_t = probs * target_bin + (1 - probs) * (1 - target_bin)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # Ranking Loss (MSE apenas nos positivos para forçar intensidade correta)
        if target_bin.sum() > 0:
            pos_mask = target_bin > 0
            rank_loss = F.mse_loss(pred[pos_mask], target[pos_mask])
        else:
            rank_loss = 0.0

        return focal_loss + self.ranking_weight * rank_loss + 0.01 * torch.norm(pred, 2)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class SpecialistTrainer:
    def __init__(self, region_key):
        cfg = REGION_CONFIGS[region_key]
        self.region_key   = region_key
        self.window       = cfg['window']
        self.lr           = cfg['lr']
        self.epochs       = cfg['epochs']
        self.patience     = cfg.get('patience', 20)
        self.dropout      = cfg['dropout']
        self.k_eval       = cfg['k_eval']
        self.use_momentum = cfg['use_momentum']
        self.raw_cvli_context = cfg.get('raw_cvli_context', True)
        self.grad_accum   = cfg['grad_accum']
        self.output_name  = cfg['output_name']
        self.margin       = cfg['margin']
        self.best_pk      = 0.0
        self.vault        = None
        self.cvp_ratio_data = None  # ⭐ V4: Novo canal de pressão tática

    def train(self):
        cfg = REGION_CONFIGS[self.region_key]
        focal_alpha = cfg.get('focal_alpha', 0.25)
        focal_gamma = cfg.get('focal_gamma', 2.0)
        ranking_w   = cfg.get('ranking_weight', 1.0)

        logging.info("\n" + "═"*80)
        logging.info(f"🚀 ESPECIALISTA: {self.region_key.upper()} (TENTATIVA 78 - SENTINELA V4 SURPRISE FOCUS)")
        logging.info(f"📊 METODOLOGIA: Split Temporal (85/15) | Blindagem de Seleção (Cutoff 2025)")
        logging.info(f"📉 LOSS: Regional Focal Ranking (alpha={focal_alpha}, gamma={focal_gamma}, rank_w={ranking_w})")
        logging.info(
            f"⚙️ ARQ: DeepSTGAT_64 | window={self.window} | lr={self.lr} | dropout={self.dropout} | "
            f"K={self.k_eval} | {self.vault.get_memory_vector().shape[0] if self.vault else 37}ch ({'MemPalace Gated' if self.vault else 'Elite Base'}) | "
            f"target_horizon={PREDICT_HORIZON}d | grad_accum={self.grad_accum} | device={DEVICE}"
        )
        logging.info("═"*80)

        # Regista automaticamente no Markdown (LOG INCREMENTAL)
        autosave_training_log(self.region_key, cfg)

        path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        nf           = data['node_features'].copy()   # (N, T, 29)
        adj_geo_np   = data['adj_geo']
        adj_conf_np  = data['adj_conflict']
        N, T, C_base = nf.shape

        if self.raw_cvli_context:
            logging.info("🧱 Reconstruindo canal 24 com soma móvel 7d do CVLI bruto (sem média)...")
            nf = rebuild_raw_cvli_context(nf)

        # Adjacências normalizadas
        adj_geo  = torch.tensor(normalize_adj(adj_geo_np),  dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(normalize_adj(adj_conf_np), dtype=torch.float32).to(DEVICE)

        # Engenharia de features (momentum opcional)
        if self.use_momentum:
            logging.info("🔧 Calculando canais de Multi-Scale Momentum + Cold Streak (negativo)...")
            momentum_feat = build_momentum_features(nf)
            features = nf.copy()
            if features.shape[2] >= 37:
                features[:, :, 33:37] = momentum_feat
            else:
                features = np.concatenate([nf, momentum_feat], axis=2)
        else:
            features = nf.copy()

        C_ext = features.shape[2]
        # Inicia o Vault de Memória para Fortaleza
        if self.region_key == 'fortaleza':
            self.vault = TrainingVault(N, ROOT_DIR)
            C_ext += 2 # Adiciona slots para o Canal 38 (MemPalace) e Canal 39 (CVLI Ratio)
            logging.info(f"🧠 MemPalace Treino Ativado: Canal 38 preparado para {N} nós.")
            
            # Carregar CVP para o Canal 39
            csv_path = os.path.join(ROOT_DIR, "data", "raw", "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
            if os.path.exists(csv_path):
                logging.info("📈 Calculando Canal 39 (CVP/CVLI Ratio) para Fortaleza...")
                df_c = pd.read_csv(csv_path, low_memory=False)
                df_c = df_c[df_c["cidade"].str.upper() == "FORTALEZA"].copy()
                df_c["data"] = pd.to_datetime(df_c["data"], errors="coerce")
                df_c = df_c.dropna(subset=["data", "bairro"])
                df_c["bairro"] = df_c["bairro"].str.upper().str.strip()
                
                # Mapa de datas e bairros do dataset original (Sincronizado com Rebuild ISM 2022)
                dates_dt = [pd.Timestamp("2022-01-01") + pd.Timedelta(days=i) for i in range(T)]
                dm = {d: i for i, d in enumerate(dates_dt)}
                # Note: node_features assume uma ordem de nós. Precisamos do mapeamento de bairros.
                # Como não temos o mapa direto aqui, vamos inferir que nf[:, :, 0] é o CVLI base.
                
                # Construção do ratio dinâmico
                cvp_df = df_c[df_c["tipo"].str.lower() == "cvp"].groupby(["data", "bairro"]).size().reset_index(name="v")
                self.cvp_ratio_data = np.zeros((N, T), dtype=np.float32)
                
                # Mapeamento de Bairros (Ordem do Tensor)
                if 'nodes_gdf' in data and 'name' in data['nodes_gdf'].columns:
                    node_names = data['nodes_gdf']['name'].tolist()
                elif 'nodes' in data:
                    node_names = data['nodes']
                else:
                    node_names = []
                
                if node_names:
                    nm = {str(b).upper().strip(): i for i, b in enumerate(node_names)}
                    for _, r in cvp_df.iterrows():
                        ni, ti = nm.get(str(r["bairro"]).upper().strip()), dm.get(r["data"])
                        if ni is not None and ti is not None:
                            # Razão CVP / (CVLI + 1)
                            cvli_val = nf[ni, ti, 0]
                            self.cvp_ratio_data[ni, ti] = r["v"] / (cvli_val + 1.0)
                    
                    logging.info(f"✅ Canal 39 preenchido para {len(node_names)} nós via nodes_gdf['name'].")
                else:
                    logging.warning("⚠️ 'nodes_gdf' ou 'name' não encontrado no .pkl. Usando média global.")
                    self.cvp_ratio_data = np.full((N, T), 0.5, dtype=np.float32)

            # --- VALIDAÇÃO DE SANIDADE V4 ---
            logging.info("🔍 SANITY CHECK (Canais Elite):")
            if self.vault:
                v_mem = self.vault.get_memory_vector()
                logging.info(f"   [Ch38 - MemPalace] Max: {v_mem.max():.4f} | Ativos: {(v_mem>0).sum()}/{N}")
            if self.cvp_ratio_data is not None:
                logging.info(f"   [Ch39 - CVLI Ratio] Max: {self.cvp_ratio_data.max():.4f} | Média: {self.cvp_ratio_data.mean():.4f}")
            logging.info("────────────────────────────────────────────────────────────────")

        # Diagnóstico CVLI
        cvli_flat = features[:, :, 0].flatten()
        nz = (cvli_flat > 0).sum()
        logging.info(f"📊 CVLI não-zero (Canal 0): {nz}/{len(cvli_flat)} ({nz/len(cvli_flat)*100:.2f}%)")

        # Construção dos pares (X, Y) com SPLIT TEMPORAL (sem shuffle)
        logging.info("🔄 Gerando janelas com Normalização Z-Score Local (per-window)...")
        X_list, Y_list = [], []
        for t in range(self.window, T - PREDICT_HORIZON):
            # Janela de entrada
            x_window = features[:, t-self.window:t, :].copy()
            
            # 🔄 NORMALIZAÇÃO LOCAL POR JANELA (Z-Score)
            # Para cada canal, normaliza com base na média/std da própria janela (exceto binários/sazonais)
            for c in range(C_ext):
                if c in [0, 1, 2, 24, 27, 28, 31, 33, 34, 35, 36]:
                    m = x_window[:, :, c].mean()
                    s = x_window[:, :, c].std() + 1e-6
                    x_window[:, :, c] = (x_window[:, :, c] - m) / s
            
            x = torch.tensor(x_window, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            y = torch.tensor(
                nf[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32
            )
            
            # Slot vazio para o Canal 38 (MemPalace) e 39 (CVLI Ratio)
            if self.region_key == 'fortaleza':
                # Adiciona dois canais extras de zeros (N, 2, window)
                x = torch.cat([x, torch.zeros((1, 2, N, self.window))], dim=1)
                
            X_list.append(x)
            Y_list.append(y)

        logging.info(f"📐 Janelas construídas: {len(X_list)}")

        # ⏳ SPLIT TEMPORAL ESTRITO (Sem shuffle)
        # 85% passado para treino, 15% futuro recente para validação
        split = int(len(X_list) * 0.85)
        train_X = X_list[:split]
        train_Y = Y_list[:split]
        val_X   = X_list[split:]
        val_Y   = Y_list[split:]
        logging.info(f"⏳ Split Temporal: {len(train_X)} treino (Passado) | {len(val_X)} validação (Futuro)")

        # Modelo e otimizador
        # ⭐ ATUALIZAÇÃO V5: Substituindo DeepSTGAT por ShallowGAT para evitar overthinking na Matriz Tática
        model = ShallowGAT(num_nodes=N, in_channels=C_ext, time_steps=self.window, dropout=self.dropout).to(DEVICE)
            
        # T89: weight_decay e lr_base lidos do config por região
        wd      = REGION_CONFIGS[self.region_key].get('weight_decay', 0.01)
        lr_base = self.lr  # Para Fortaleza: 3e-4 (T89); outros: mantém config
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=wd)
        criterion = BinaryFocalRankingLoss(alpha=focal_alpha, gamma=focal_gamma, ranking_weight=ranking_w)

        steps_per_epoch = (len(train_X) // self.grad_accum) + 1
        sched_type = REGION_CONFIGS[self.region_key].get('scheduler', 'onecycle')

        if sched_type == 'cosine_restarts':
            # T89: CosineAnnealingWarmRestarts — múltiplos ciclos, sem pico agressivo
            T0      = REGION_CONFIGS[self.region_key].get('cosine_T0', 10)
            Tmult   = REGION_CONFIGS[self.region_key].get('cosine_Tmult', 2)
            eta_min = REGION_CONFIGS[self.region_key].get('cosine_eta_min', 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T0, T_mult=Tmult, eta_min=eta_min
            )
            logging.info(f"📐 Scheduler: CosineAnnealingWarmRestarts (T0={T0}, Tmult={Tmult}, eta_min={eta_min})")
        else:
            # Legado: OneCycleLR (RMF e Interior mantêm)
            pct_start = REGION_CONFIGS[self.region_key].get('pct_start', 0.2)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs, pct_start=pct_start
            )
            logging.info(f"📐 Scheduler: OneCycleLR (max_lr={self.lr}, pct_start={pct_start})")
        output_path = os.path.join(ROOT_DIR, 'models', 'active', self.output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        no_improve_epochs = 0
        for epoch in range(self.epochs):
            model.train()
            if self.vault:
                self.vault.clear_epoch()
            epoch_loss = 0.0
            epoch_grads = []
            sample_idx = list(range(len(train_X)))
            random.shuffle(sample_idx)
            optimizer.zero_grad()

            for step, idx in enumerate(sample_idx):
                x_input = train_X[idx].to(DEVICE)
                
                # Injeta a memória atual do Vault no Canal 37 e Ratio no Canal 38
                if self.vault:
                    # ⭐ ATUALIZAÇÃO V4: Canal Dropout de 50% para evitar dependência viciada na memória
                    if random.random() > 0.2:
                        mem_vec = torch.tensor(self.vault.get_memory_vector(), dtype=torch.float32).to(DEVICE)
                        # Expande mem_vec para (1, 1, N, window)
                        x_input[:, 37, :, :] = mem_vec.view(1, N, 1).expand(-1, -1, self.window)
                    else:
                        # Drop: Zera o canal de memória para forçar o modelo a aprender com os outros canais
                        x_input[:, 37, :, :] = 0.0
                    
                    # Injeta o Canal 39 (CVLI Ratio) dinâmico
                    if self.cvp_ratio_data is not None:
                        # Pega o ratio da janela correspondente
                        ratio_win = self.cvp_ratio_data[:, idx:idx+self.window]
                        x_input[:, 38, :, :] = torch.tensor(ratio_win, dtype=torch.float32).to(DEVICE)

                # ⭐ ATUALIZAÇÃO V5.1: Context Sensing Estabilizado
                # Tau aumentado para 730 (suavização) e normalização de peso para manter escala de gradiente
                tau = 730
                w_raw = np.exp(idx / tau)
                # Normalização: Garante que o gradiente médio não mude, apenas a distribuição de importância
                # Média teórica dos pesos para normalização (baseada no range de treino)
                w_norm = w_raw / np.mean([np.exp(i/tau) for i in range(len(train_X))])

                pred = model(x_input, [adj_geo, adj_conf]).squeeze()
                
                # Aplicação do peso normalizado no gradiente
                raw_loss = criterion(pred, train_Y[idx].to(DEVICE))
                loss = (raw_loss * w_norm) / self.grad_accum
                loss.backward()

                if (step + 1) % self.grad_accum == 0:
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                    epoch_grads.append(grad_norm)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    # T89: CosineRestarts faz step por época, não por batch
                    if sched_type != 'cosine_restarts':
                        scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer.zero_grad()

                    step_loss = loss.item() * self.grad_accum
                    epoch_loss += step_loss
                    batch_num  = (step + 1) // self.grad_accum

                    with torch.no_grad():
                        k_eff = min(self.k_eval, (train_Y[idx] > 0).sum().item())
                        if k_eff > 0:
                            _, t_idx = torch.topk(train_Y[idx].to(DEVICE), k_eff)
                            _, p_idx = torch.topk(pred, k_eff)
                            bp = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k_eff
                            logging.info(
                                f"E{epoch+1:03d} | B{batch_num:03d} | "
                                f"LR: {current_lr:.5f} | Loss: {step_loss:.4f} | "
                                f"Grad: {grad_norm:.4f} | P@{k_eff}: {bp*100:.1f}%"
                            )

            # ── Validação ──────────────────────────────────────
            model.eval()
            pk10_list = []
            pk20_list = []
            with torch.no_grad():
                for vx, vy in zip(val_X, val_Y):
                    n_real = (vy > 0).sum().item()
                    if n_real > 0:
                        # Injeta memória na validação também
                        if self.vault:
                            mem_vec = torch.tensor(self.vault.get_memory_vector(), dtype=torch.float32).to(DEVICE)
                            vx[:, 37, :, :] = mem_vec.view(1, N, 1).expand(-1, -1, self.window)

                        vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                        
                        # P@10
                        k10 = min(10, n_real)
                        _, t_idx10 = torch.topk(vy, k10)
                        _, p_idx10 = torch.topk(vpred, 10)
                        pk10_list.append(len(set(t_idx10.cpu().numpy()) & set(p_idx10.cpu().numpy())) / k10)
                        
                        # P@20
                        k20 = min(20, n_real)
                        _, t_idx20 = torch.topk(vy, k20)
                        _, p_idx20 = torch.topk(vpred, 20)
                        score20 = len(set(t_idx20.cpu().numpy()) & set(p_idx20.cpu().numpy())) / k20
                        pk20_list.append(score20)
                        
                        # ⭐ ATUALIZAÇÃO V4: Registro de surpresas movido exclusivamente para o FINAL da época
                        # (Removido do loop de validação para evitar Logical Leak / Data Leakage)
                        # O Vault agora só aprende com o erro real após a validação ser concluída
                        pass

            if self.vault:
                # ⭐ ATUALIZAÇÃO V4: Gravação das surpresas consolidada no fim da época baseada no erro total
                # Isso garante que a próxima época tenha a memória de erro, mas sem leak durante a validação atual.
                with torch.no_grad():
                    for vx, vy in zip(val_X, val_Y):
                        vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                        _, t_idx20 = torch.topk(vy, min(20, (vy>0).sum().item()))
                        _, p_idx20 = torch.topk(vpred, 20)
                        top20_set = set(p_idx20.cpu().numpy())
                        for node_idx in t_idx20.cpu().numpy():
                            if node_idx not in top20_set:
                                self.vault.record_surprise(node_idx, intensity=vy[node_idx].item())
                
                self.vault.save(epoch + 1)

            avg_p10  = np.mean(pk10_list) if pk10_list else 0.0
            avg_p20  = np.mean(pk20_list) if pk20_list else 0.0
            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            avg_grad = np.mean(epoch_grads) if epoch_grads else 0.0
            
            # Alvo estratégico: P@20 para Fortaleza, P@k original para outros
            current_metric = avg_p20 if self.region_key == 'fortaleza' else avg_p10
            metric_label   = "P@20" if self.region_key == 'fortaleza' else f"P@{self.k_eval}"

            logging.info(
                f"\n---> ÉPOCA {epoch+1:03d} [{self.region_key.upper()}] | "
                f"Val P@10: {avg_p10*100:.2f}% | Val P@20: {avg_p20*100:.2f}% | "
                f"Loss: {avg_loss:.4f} | Grad: {avg_grad:.4f} | "
                f"Recorde ({metric_label}): {max(self.best_pk, current_metric)*100:.2f}% <---\n"
            )

            if current_metric > self.best_pk:
                self.best_pk = current_metric
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'p10': avg_p10,
                    'p20': avg_p20,
                    'best_metric': metric_label,
                    'config': {
                        'window':      self.window,
                        'nodes':       N,
                        'arch':        'DeepSTGAT_64',
                        'in_channels': C_ext,
                        'region':      self.region_key,
                        'predict_horizon_days': PREDICT_HORIZON,
                        'raw_cvli_context': self.raw_cvli_context,
                        'channel24_mode': 'rolling_sum_7d',
                    }
                }, output_path)
                logging.info(f"💎 NOVO RECORDE [{self.region_key.upper()}]: {metric_label}={self.best_pk*100:.2f}% → {self.output_name}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    logging.info(f"🛑 Early stopping na época {epoch+1} para {self.region_key.upper()} (patience={self.patience})")
                    break

            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            # T89: CosineRestarts — step por época (não por batch)
            if sched_type == 'cosine_restarts':
                scheduler.step()

        logging.info(f"\n✅ {self.region_key.upper()} concluído. Melhor {metric_label}: {self.best_pk*100:.2f}% | Melhor P@10: {max(pk10_list)*100:.2f}% | Melhor P@20: {max(pk20_list)*100:.2f}%")


# ─────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────
def main():
    # Uso: py train_all_specialists.py [região1 região2 ...] [--publish]
    # --publish : publica no git após o treino (desativado por padrão — achar o ponto ótimo antes)
    args = sys.argv[1:]
    do_publish = '--publish' in args
    regions_raw = [a for a in args if not a.startswith('--')]
    regions = regions_raw if regions_raw else list(REGION_CONFIGS.keys())

    if not do_publish:
        logging.info("🔒 Modo LOCAL: publish automático DESATIVADO. Use --publish para enviar ao git.")

    trained_regions = []
    failed_regions  = []
    for region in regions:
        if region not in REGION_CONFIGS:
            logging.error(f"❌ Região desconhecida: {region}. Opções: {list(REGION_CONFIGS.keys())}")
            failed_regions.append(region)
            continue
        try:
            SpecialistTrainer(region).train()
            trained_regions.append(region)
        except Exception as exc:
            failed_regions.append(region)
            logging.error(f"❌ ERRO CRÍTICO em {region.upper()}: {exc}")
            import traceback
            traceback.print_exc()

    if trained_regions and do_publish:
        try:
            publish_training_artifacts(trained_regions, failed_regions)
        except Exception as exc:
            logging.error(f"❌ ERRO AO PUBLICAR: {exc}")
    elif trained_regions and not do_publish:
        logging.info(
            f"✅ Treino local concluído: {trained_regions}. "
            f"Quando satisfeito com as métricas, publique com:\n"
            f"   py scripts/training/Active/train_all_specialists.py {' '.join(trained_regions)} --publish"
        )

if __name__ == "__main__":
    main()
