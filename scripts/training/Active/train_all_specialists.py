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

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64
except ImportError:
    from src.core.architectures import DeepSTGAT_64

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
PREDICT_HORIZON = 7

# ─────────────────────────────────────────────
# Configuração por região
# ─────────────────────────────────────────────
# Campos: window, lr, epochs, dropout, margin, k_eval
#         use_momentum (True → 33ch; False → 29ch)
#         grad_accum   (gradient accumulation steps)
#         output_name  (nome do .pth salvo em models/active/)
REGION_CONFIGS = {
    'fortaleza': dict(
        window=120, lr=0.01,  epochs=120, dropout=0.3, margin=1.0,
        k_eval=10, use_momentum=True,  grad_accum=32,
        output_name='fortaleza_retrain_64.pth',
    ),
    'rmf': dict(
        window=90,  lr=0.018, epochs=120, dropout=0.5, margin=1.5,
        k_eval=5,  use_momentum=False, grad_accum=8,
        output_name='rmf_model.pth',
    ),
    'interior': dict(
        window=120, lr=0.005, epochs=120, dropout=0.3, margin=1.0,
        k_eval=10, use_momentum=True,  grad_accum=32,
        output_name='interior_retrain_64.pth',
    ),
}

# ─────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────
def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


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


class ContrastiveTopKLoss(nn.Module):
    """
    Força o score dos top-K nós (por CVLI verdadeiro) a superar
    a média de fundo por `margin`. Penaliza frouxidão nos hotspots.
    """
    def __init__(self, k=10, margin=1.0):
        super().__init__()
        self.k = k
        self.margin = margin

    def forward(self, pred, target):
        k_eff = min(self.k, (target > 0).sum().item())
        if k_eff == 0:
            return 0.01 * torch.norm(pred, 2)
        _, topk_idx = torch.topk(target, k_eff)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[topk_idx] = True
        hotspot_scores = pred[mask]
        bg_mean = pred[~mask].mean()
        loss = F.relu(self.margin - (hotspot_scores - bg_mean)).mean()
        return loss + 0.01 * torch.norm(pred, 2)


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
        self.dropout      = cfg['dropout']
        self.k_eval       = cfg['k_eval']
        self.use_momentum = cfg['use_momentum']
        self.grad_accum   = cfg['grad_accum']
        self.output_name  = cfg['output_name']
        self.margin       = cfg['margin']
        self.best_pk      = 0.0

    def train(self):
        logging.info("\n" + "="*80)
        logging.info(
            f"🚀 {self.region_key.upper()} | window={self.window} | "
            f"lr={self.lr} | epochs={self.epochs} | dropout={self.dropout} | "
            f"margin={self.margin} | K={self.k_eval} | "
            f"{'33ch+momentum' if self.use_momentum else '29ch'} | "
            f"grad_accum={self.grad_accum} | device={DEVICE}"
        )
        logging.info("="*80)

        path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        nf           = data['node_features']          # (N, T, 29)
        adj_geo_np   = data['adj_geo']
        adj_conf_np  = data['adj_conflict']
        N, T, C_base = nf.shape

        # Adjacências normalizadas
        adj_geo  = torch.tensor(normalize_adj(adj_geo_np),  dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(normalize_adj(adj_conf_np), dtype=torch.float32).to(DEVICE)

        # Engenharia de features (momentum opcional)
        if self.use_momentum:
            logging.info("🔧 Calculando canais de Multi-Scale Momentum + Cold Streak (negativo)...")
            momentum_feat = build_momentum_features(nf)
            features = np.concatenate([nf, momentum_feat], axis=2)   # (N, T, 33)
        else:
            features = nf.copy()

        C_ext = features.shape[2]
        logging.info(f"📦 Shape final: ({N}, {T}, {C_ext})")

        # Diagnóstico CVLI
        cvli_flat = features[:, :, 0].flatten()
        nz = (cvli_flat > 0).sum()
        logging.info(f"📊 CVLI não-zero: {nz}/{len(cvli_flat)} ({nz/len(cvli_flat)*100:.2f}%)")

        # Dados brutos — sem normalização para preservar picos de criminalidade

        # Construção dos pares (X, Y)
        # Y = soma de CVLI bruto nos próximos 7 dias
        X_list, Y_list = [], []
        for t in range(self.window, T - PREDICT_HORIZON):
            x = torch.tensor(
                features[:, t-self.window:t, :], dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0)
            y = torch.tensor(
                nf[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32
            )
            X_list.append(x)
            Y_list.append(y)

        logging.info(f"📐 Janelas construídas: {len(X_list)}")

        # Split aleatório 85/15 (seed fixo para reprodutibilidade)
        indices = list(range(len(X_list)))
        random.seed(42)
        random.shuffle(indices)
        split = int(len(indices) * 0.85)
        train_idx = indices[:split]
        val_idx   = indices[split:]
        train_X = [X_list[i] for i in train_idx]
        train_Y = [Y_list[i] for i in train_idx]
        val_X   = [X_list[i] for i in val_idx]
        val_Y   = [Y_list[i] for i in val_idx]
        logging.info(f"🔀 Split: {len(train_X)} treino | {len(val_X)} validação")

        # Modelo e otimizador
        model     = DeepSTGAT_64(num_nodes=N, in_channels=C_ext, time_steps=self.window, dropout=self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
        criterion = ContrastiveTopKLoss(k=self.k_eval, margin=self.margin)

        steps_per_epoch = (len(train_X) // self.grad_accum) + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs, pct_start=0.2
        )

        output_path = os.path.join(ROOT_DIR, 'models', 'active', self.output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_grads = []
            sample_idx = list(range(len(train_X)))
            random.shuffle(sample_idx)
            optimizer.zero_grad()

            for step, idx in enumerate(sample_idx):
                pred = model(train_X[idx].to(DEVICE), [adj_geo, adj_conf]).squeeze()
                loss = criterion(pred, train_Y[idx].to(DEVICE)) / self.grad_accum
                loss.backward()

                if (step + 1) % self.grad_accum == 0:
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
            pk_list = []
            with torch.no_grad():
                for vx, vy in zip(val_X, val_Y):
                    k_eff = min(self.k_eval, (vy > 0).sum().item())
                    if k_eff > 0:
                        vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                        _, t_idx = torch.topk(vy, k_eff)
                        _, p_idx = torch.topk(vpred, self.k_eval)
                        score = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k_eff
                        pk_list.append(score)

            avg_pk   = np.mean(pk_list) if pk_list else 0.0
            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            avg_grad = np.mean(epoch_grads) if epoch_grads else 0.0
            logging.info(
                f"\n---> ÉPOCA {epoch+1:03d} [{self.region_key.upper()}] | "
                f"Val P@{self.k_eval}: {avg_pk*100:.2f}% | "
                f"Loss: {avg_loss:.4f} | Grad: {avg_grad:.4f} | "
                f"Recorde: {max(self.best_pk, avg_pk)*100:.2f}% <---\n"
            )

            if avg_pk > self.best_pk:
                self.best_pk = avg_pk
                torch.save({
                    'model_state_dict': model.state_dict(),
                    f'p{self.k_eval}': avg_pk,
                    'config': {
                        'window':      self.window,
                        'nodes':       N,
                        'arch':        'DeepSTGAT_64',
                        'in_channels': C_ext,
                        'region':      self.region_key,
                    }
                }, output_path)
                logging.info(f"💎 NOVO RECORDE [{self.region_key.upper()}]: P@{self.k_eval}={self.best_pk*100:.2f}% → {self.output_name}")

            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        logging.info(f"\n✅ {self.region_key.upper()} concluído. Melhor P@{self.k_eval}: {self.best_pk*100:.2f}%")


# ─────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────
def main():
    # Treina sequencialmente: fortaleza → rmf → interior
    # Para treinar apenas uma região: comente as outras ou passe argv
    regions = sys.argv[1:] if len(sys.argv) > 1 else list(REGION_CONFIGS.keys())
    for region in regions:
        if region not in REGION_CONFIGS:
            logging.error(f"❌ Região desconhecida: {region}. Opções: {list(REGION_CONFIGS.keys())}")
            continue
        try:
            SpecialistTrainer(region).train()
        except Exception as exc:
            logging.error(f"❌ ERRO CRÍTICO em {region.upper()}: {exc}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
