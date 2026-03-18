"""
RETREINO DO MODELO INTERIOR — 33 CANAIS (Multi-Scale Momentum + Cold Streak)
=============================================================================
Motivação:
  O interior possui 50 nós com 98.8% de CVLI zero. O modelo atual (29ch, window=90)
  não consegue diferenciar hotspots persistentes (Juazeiro, Sobral) de cidades frias.

Solução:
  Adicionar 4 canais sintéticos computados a partir do sinal CVLI bruto (canal 0):
    Ch 29: Δ CVLI 7 dias  (recent_7 - past_7)
    Ch 30: Δ CVLI 14 dias (recent_14 - past_14)
    Ch 31: Δ CVLI 30 dias (recent_30 - past_30)
    Ch 32: Cold Streak     (dias consecutivos sem CVLI, cap 30)

  O Cold Streak é o canal mais importante para o interior: cidades como Juazeiro
  (max 8 CVLIs em 1526 dias) terão streak=0 após cada evento, enquanto nós 
  verdadeiramente frios terão streak permanentemente em 30.

Resultado esperado:
  Modelo salvo em models/active/interior_retrain_64.pth
  Orquestrador detectará o arquivo e usará in_channels=33, window=120.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import random
import gc

sys.path.append(os.getcwd())
try:
    from src.core.architectures import DeepSTGAT_64
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src', 'core'))
    from architectures import DeepSTGAT_64

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_INTERIOR_33CH.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- HIPERPARÂMETROS ---
EPOCHS = 120
LR_MAX = 0.005          # Mais conservador que Fortaleza (sinal muito esparso)
WINDOW = 120            # Mesma janela temporal que Fortaleza
DROPOUT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_ACCUMULATION_STEPS = 32
REGION = 'interior'
# P@K para interior: top-10 de 50 nós (20%)
K_EVAL = 10


def load_processed_data(region_key):
    path = f'data/processed/processed_{region_key}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class ContrastiveTopKLoss(nn.Module):
    """
    Loss contrastiva: força os top-K nós (por CVLI verdadeiro) a ter score
    maior que a média dos nós de fundo por uma margem.
    """
    def __init__(self, k=10, margin=1.0):
        super().__init__()
        self.k = k
        self.margin = margin

    def forward(self, pred, target):
        k_eff = min(self.k, (target > 0).sum().item())
        if k_eff == 0:
            # Nenhum crime nesta janela: penaliza dispersão de scores
            return 0.01 * torch.norm(pred, 2)

        _, topk_indices = torch.topk(target, k_eff)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[topk_indices] = True

        hotspot_scores = pred[mask]
        background_scores = pred[~mask]
        bg_mean = background_scores.mean()

        loss = F.relu(self.margin - (hotspot_scores - bg_mean)).mean()
        reg_penalty = 0.01 * torch.norm(pred, 2)
        return loss + reg_penalty


def build_momentum_features(features):
    """
    Computa 4 canais sintéticos de momentum CVLI:
      Canal 0 (Ch29): Δ 7d  — aceleração recente
      Canal 1 (Ch30): Δ 14d — tendência quinzenal
      Canal 2 (Ch31): Δ 30d — tendência mensal
      Canal 3 (Ch32): Cold Streak (invertido) — dias sem CVLI, cap 30
    """
    N, T, _ = features.shape
    momentum_feat = np.zeros((N, T, 4))
    cold_streak = np.zeros(N)

    for t in range(60, T):
        # Δ 7 dias
        recent_7 = features[:, t-7:t, 0].sum(axis=1)
        past_7   = features[:, t-14:t-7, 0].sum(axis=1)
        momentum_feat[:, t, 0] = recent_7 - past_7

        # Δ 14 dias
        recent_14 = features[:, t-14:t, 0].sum(axis=1)
        past_14   = features[:, t-28:t-14, 0].sum(axis=1)
        momentum_feat[:, t, 1] = recent_14 - past_14

        # Δ 30 dias
        recent_30 = features[:, t-30:t, 0].sum(axis=1)
        past_30   = features[:, t-60:t-30, 0].sum(axis=1)
        momentum_feat[:, t, 2] = recent_30 - past_30

        # Cold Streak (Invertido): dias consecutivos sem crime, negativo (cap 30)
        crimes_today = features[:, t, 0]
        cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)

    return momentum_feat


def train_interior_33ch():
    logging.info("\n" + "="*80)
    logging.info("🚀 RETREINO INTERIOR — 33 CANAIS (Multi-Scale Momentum + Cold Streak)")
    logging.info(f"⚙️ PARAMS: LR={LR_MAX} | WINDOW={WINDOW} | DROPOUT={DROPOUT} | K={K_EVAL} | DEVICE={DEVICE}")
    logging.info("="*80)

    data = load_processed_data(REGION)
    features = data['node_features']   # shape (50, 1526, 29)
    N, T_total, C = features.shape
    logging.info(f"📦 Interior: {N} nós, {T_total} timesteps, {C} canais originais")

    # Diagnóstico da densidade CVLI
    cvli_flat = features[:, :, 0].flatten()
    nz = (cvli_flat > 0).sum()
    logging.info(f"📊 CVLI canal 0: {nz}/{len(cvli_flat)} não-zeros ({nz/len(cvli_flat)*100:.2f}%)")
    logging.info(f"📊 Max CVLI por nó (top 10): {sorted(features[:,:,0].max(axis=1).tolist(), reverse=True)[:10]}")

    # Matrizes de adjacência
    adj_geo  = torch.tensor(normalize_adj(data['adj_geo']),      dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)

    # --- ENGENHARIA DE FEATURES: +4 canais de momentum ---
    logging.info("🔧 Calculando canais de Multi-Scale Momentum + Cold Streak...")
    momentum_feat = build_momentum_features(features)
    features_extended = np.concatenate([features, momentum_feat], axis=2)  # (50, 1526, 33)
    C_ext = features_extended.shape[2]
    logging.info(f"✅ Features estendidas: shape={features_extended.shape} ({C_ext} canais)")

    # Dados brutos — sem normalização para preservar picos de criminalidade

    # --- CONSTRUÇÃO DOS PARES (X, Y) ---
    # Y = CVLI total nos próximos 7 dias (janela de previsão)
    X, Y = [], []
    for t in range(WINDOW, T_total - 7):
        x = torch.tensor(
            features_extended[:, t-WINDOW:t, :], dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor(features[:, t:t+7, 0].sum(axis=1), dtype=torch.float32)
        X.append(x)
        Y.append(y)

    logging.info(f"📐 Total de janelas construídas: {len(X)}")

    # Split aleatório 85/15
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)
    split = int(len(X) * 0.85)
    train_X = [X[i] for i in indices[:split]]
    train_Y = [Y[i] for i in indices[:split]]
    val_X   = [X[i] for i in indices[split:]]
    val_Y   = [Y[i] for i in indices[split:]]
    logging.info(f"🔀 Split: {len(train_X)} treino | {len(val_X)} validação")

    # --- MODELO ---
    model = DeepSTGAT_64(
        num_nodes=N, in_channels=C_ext, time_steps=WINDOW, dropout=DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    criterion = ContrastiveTopKLoss(k=K_EVAL, margin=1.0)

    steps_per_epoch = (len(train_X) // GRADIENT_ACCUMULATION_STEPS) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS, pct_start=0.2
    )

    best_p10 = 0.0
    os.makedirs('models/active', exist_ok=True)
    output_path = 'models/active/interior_retrain_64.pth'

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_grads = []

        idxs = list(range(len(train_X)))
        random.shuffle(idxs)

        optimizer.zero_grad()
        for i, idx in enumerate(idxs):
            pred = model(train_X[idx].to(DEVICE), [adj_geo, adj_conf]).squeeze()
            loss = criterion(pred, train_Y[idx].to(DEVICE)) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                epoch_grads.append(grad_norm)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                optimizer.zero_grad()

                step_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                epoch_loss += step_loss

                current_step = (i + 1) // GRADIENT_ACCUMULATION_STEPS
                with torch.no_grad():
                    k_eff = min(K_EVAL, (train_Y[idx] > 0).sum().item())
                    if k_eff > 0:
                        _, t_idx = torch.topk(train_Y[idx].to(DEVICE), k_eff)
                        _, p_idx = torch.topk(pred, k_eff)
                        batch_p = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k_eff
                        logging.info(
                            f"E{epoch+1:03d} | Batch {current_step:03d} | LR: {current_lr:.5f} | "
                            f"Loss: {step_loss:.4f} | Grad: {grad_norm:.4f} | P@{k_eff}: {batch_p*100:.1f}%"
                        )

        # Validação
        model.eval()
        p_list = []
        with torch.no_grad():
            for vx, vy in zip(val_X, val_Y):
                k_eff = min(K_EVAL, (vy > 0).sum().item())
                if k_eff > 0:
                    vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                    _, t_idx = torch.topk(vy, k_eff)
                    _, p_idx = torch.topk(vpred, K_EVAL)  # sempre avalia top-10
                    p_score = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k_eff
                    p_list.append(p_score)

        avg_p    = np.mean(p_list) if p_list else 0.0
        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        avg_grad = np.mean(epoch_grads) if epoch_grads else 0.0

        logging.info(
            f"\n---> ÉPOCA {epoch+1:03d} | Val P@{K_EVAL}: {avg_p*100:.2f}% | "
            f"Loss: {avg_loss:.4f} | Grad: {avg_grad:.4f} | "
            f"Recorde: {max(best_p10, avg_p)*100:.2f}% <---\n"
        )

        if avg_p > best_p10:
            best_p10 = avg_p
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'window':      WINDOW,
                    'nodes':       N,
                    'arch':        'DeepSTGAT_64',
                    'in_channels': C_ext,
                    'region':      REGION,
                }
            }, output_path)
            logging.info(f"💎 NOVO RECORDE: P@{K_EVAL}={best_p10*100:.2f}% → {output_path}")

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    logging.info(f"\n{'='*80}")
    logging.info(f"✅ Retreino concluído. Melhor P@{K_EVAL}: {best_p10*100:.2f}%")
    logging.info(f"📁 Modelo salvo em: {output_path}")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    train_interior_33ch()
