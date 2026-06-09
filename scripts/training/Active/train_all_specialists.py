"""
train_all_specialists.py - Script oficial de retreino (FOCO CVLI - HONESTY PARADIGM).
Versao 2026-05-21: Janela 14d, MemPalace Universal, Honesty Constraint.
"""
import gc
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64, DeepSTGAT_80, FortalezaHeteroSTGAT, PureSTGCN_64, ShallowGAT
    from training_vault import TrainingVault
except ImportError:
    from src.core.architectures import DeepSTGAT_64, DeepSTGAT_80, FortalezaHeteroSTGAT, PureSTGCN_64, ShallowGAT
    from src.core.training_vault import TrainingVault

# Configuracao de Log
log_file = os.path.join(ROOT_DIR, 'logs', 'training_ALL_SPECIALISTS.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREDICT_HORIZON = 14
TRAIN_BATCH_LOG_EVERY = int(os.environ.get("TRAIN_BATCH_LOG_EVERY", "30"))

REGION_CONFIGS = {
    'fortaleza': dict(
        window=60,
        lr=2e-4,
        epochs=18,
        patience=6,
        dropout=0.30,
        margin=1.0,
        k_eval=10,
        use_momentum=True,
        grad_accum=64,
        output_name='fortaleza_model_active.pth',
        model_class='PureSTGCN_64',
        focal_alpha=0.55,
        focal_gamma=2.0,
        ranking_weight=12.0,
        indecision_weight=0.8,
        weight_decay=0.005,
        scheduler='static',
    ),
    'rmf': dict(
        window=14,
        lr=0.001,
        epochs=30,
        patience=15,
        dropout=0.5,
        margin=1.5,
        k_eval=5,
        use_momentum=True,
        grad_accum=8,
        output_name='rmf_model.pth',
        focal_alpha=0.50,
        focal_gamma=2.0,
        ranking_weight=10.0,
        scheduler='onecycle',
    ),
    'interior': dict(
        window=14,
        lr=0.001,
        epochs=30,
        patience=15,
        dropout=0.3,
        margin=1.0,
        k_eval=10,
        use_momentum=True,
        grad_accum=4,
        output_name='interior_model.pth',
        focal_alpha=0.40,
        focal_gamma=2.0,
        ranking_weight=15.0,
        scheduler='onecycle',
    ),
}


def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(adj)


def build_momentum_features(features):
    n_nodes, n_steps, _ = features.shape
    momentum_feat = np.zeros((n_nodes, n_steps, 4))
    cold_streak = np.zeros(n_nodes)
    for t in range(60, n_steps):
        crimes = features[:, t, 0]
        cold_streak = np.where(crimes > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
    return momentum_feat


def inject_momentum_channels(features):
    """
    Alinha o treino com o orquestrador/benchmark:
    os canais taticos 33:36 recebem momentum dentro do tensor base de 37 canais.
    """
    enriched = features.copy()
    momentum_feat = build_momentum_features(features)
    if enriched.shape[2] >= 37:
        enriched[:, :, 33:37] = momentum_feat[:, :, :4]
    return enriched


class BinaryFocalRankingLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        ranking_weight=1.0,
        honesty_weight=2.0,
        indecision_weight=1.5,
        l2_weight=0.01,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ranking_weight = ranking_weight
        self.honesty_weight = honesty_weight
        self.indecision_weight = indecision_weight
        self.l2_weight = l2_weight

    def forward(self, pred, target, cold_streak_signal=None):
        target_bin = (target > 0).float()
        probs = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_bin, reduction='none')
        p_t = probs * target_bin + (1 - probs) * (1 - target_bin)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * bce_loss).mean()
        rank_loss = F.mse_loss(pred[target_bin > 0], target[target_bin > 0]) if target_bin.sum() > 0 else 0.0

        # Pune scores excessivamente flat nos hotspots principais.
        top_vals, _ = torch.topk(pred, min(10, len(pred)))
        gap = top_vals.mean() - pred.mean()
        indecision_penalty = torch.exp(-gap)

        honesty_penalty = 0.0
        if cold_streak_signal is not None:
            calmness = torch.clamp(-cold_streak_signal / 30.0, 0, 1)
            honesty_penalty = (calmness * torch.relu(pred)).mean()

        return (
            focal_loss
            + self.ranking_weight * rank_loss
            + self.honesty_weight * honesty_penalty
            + self.indecision_weight * indecision_penalty
            + self.l2_weight * torch.norm(pred, 2)
        )


class SpecialistTrainer:
    def __init__(self, region_key):
        self.cfg = REGION_CONFIGS[region_key]
        self.region_key = region_key
        self.vault = None

    def train(self):
        logging.info(f"\nESPECIALISTA: {self.region_key.upper()} (SENTINELA V4 - HONESTY)")
        path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        nf = data['node_features']
        adj_geo_np = data['adj_geo']
        adj_conf_np = data['adj_conflict']
        n_nodes, total_steps, _ = nf.shape
        adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(normalize_adj(adj_conf_np), dtype=torch.float32).to(DEVICE)

        features = inject_momentum_channels(nf)

        self.vault = TrainingVault(n_nodes, ROOT_DIR)

        x_list, y_list = [], []
        window = self.cfg['window']
        total_windows = max(0, total_steps - PREDICT_HORIZON - window)
        build_start = time.time()
        logging.info(
            "Preparando janelas | region=%s | total_steps=%s | candidate_windows=%s",
            self.region_key,
            total_steps,
            total_windows,
        )
        for t in range(window, total_steps - PREDICT_HORIZON):
            x_win = features[:, t - window:t, :].copy()
            x = torch.tensor(x_win, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            y = torch.tensor(nf[:, t:t + PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)

            if x.shape[1] < 41:
                padding = torch.zeros((1, 41 - x.shape[1], n_nodes, window))
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] > 41:
                x = x[:, :41, :, :]

            x_list.append(x)
            y_list.append(y)
            built_windows = len(x_list)
            if built_windows == 1 or built_windows % TRAIN_BATCH_LOG_EVERY == 0 or built_windows == total_windows:
                logging.info(
                    "Janela %s/%s (%.1f%%) preparada | elapsed=%.1fs",
                    built_windows,
                    total_windows,
                    (built_windows / max(total_windows, 1)) * 100.0,
                    time.time() - build_start,
                )

        split = int(len(x_list) * 0.85)
        train_x, train_y = x_list[:split], y_list[:split]
        val_x, val_y = x_list[split:], y_list[split:]
        if not train_x or not val_x:
            raise RuntimeError(
                f"Janela insuficiente para treino/validacao em {self.region_key}: "
                f"train={len(train_x)} | val={len(val_x)}"
            )

        model = self._build_model(n_nodes=n_nodes, in_channels=41, window=window).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg.get('weight_decay', 0.005),
        )
        criterion = BinaryFocalRankingLoss(
            alpha=self.cfg['focal_alpha'],
            gamma=self.cfg['focal_gamma'],
            ranking_weight=self.cfg['ranking_weight'],
            indecision_weight=self.cfg.get('indecision_weight', 1.5),
        )

        if self.cfg.get('scheduler') == 'cosine_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cfg['cosine_T0'],
                T_mult=self.cfg['cosine_Tmult'],
                eta_min=self.cfg['eta_min'],
            )
        elif self.cfg.get('scheduler') == 'static':
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg['lr'],
                steps_per_epoch=len(train_x) // self.cfg['grad_accum'] + 1,
                epochs=self.cfg['epochs'],
            )

        best_pk = 0.0
        no_improve = 0
        optimizer_steps_per_epoch = max(1, (len(train_x) + self.cfg['grad_accum'] - 1) // self.cfg['grad_accum'])
        logging.info(
            "Dataset preparado | train=%s | val=%s | nodes=%s | base_channels=%s | total_channels=%s | grad_accum=%s | opt_steps/epoch=%s",
            len(train_x),
            len(val_x),
            n_nodes,
            nf.shape[2],
            features.shape[2],
            self.cfg['grad_accum'],
            optimizer_steps_per_epoch,
        )
        for epoch in range(self.cfg['epochs']):
            model.train()
            self.vault.clear_epoch()
            epoch_loss = 0.0
            epoch_start = time.time()
            running_loss = 0.0
            optimizer_step_count = 0
            idx_list = list(range(len(train_x)))
            random.shuffle(idx_list)
            logging.info(
                "E%03d start | region=%s | minibatches=%s | grad_accum=%s",
                epoch + 1,
                self.region_key,
                len(idx_list),
                self.cfg['grad_accum'],
            )

            for step, idx in enumerate(idx_list):
                xi = train_x[idx].to(DEVICE)
                if random.random() > 0.2:
                    memory_vector = torch.tensor(self.vault.get_memory_vector(), dtype=torch.float32).to(DEVICE)
                    xi[:, 37, :, :] = memory_vector.view(1, 1, n_nodes, 1).expand(-1, -1, -1, window)

                cold_streak = xi[0, 36, :, -1]
                pred = model(xi, [adj_geo, adj_conf]).squeeze()
                loss = criterion(
                    pred,
                    train_y[idx].to(DEVICE),
                    cold_streak_signal=cold_streak,
                ) / self.cfg['grad_accum']
                loss.backward()
                loss_value = loss.item() * self.cfg['grad_accum']
                running_loss += loss_value

                if (step + 1) % self.cfg['grad_accum'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None and self.cfg.get('scheduler') != 'cosine_restarts':
                        scheduler.step()
                    epoch_loss += running_loss
                    optimizer_step_count += 1
                    running_loss = 0.0

                should_log_batch = (
                    (step + 1) == 1
                    or (step + 1) % TRAIN_BATCH_LOG_EVERY == 0
                    or (step + 1) == len(idx_list)
                )
                if should_log_batch:
                    elapsed = time.time() - epoch_start
                    progress_pct = ((step + 1) / len(idx_list)) * 100.0
                    current_lr = optimizer.param_groups[0]['lr']
                    logging.info(
                        "E%03d batch %s/%s (%.1f%%) | opt_steps=%s/%s | lr=%.6g | loss_cur=%.4f | loss_epoch=%.4f | elapsed=%.1fs",
                        epoch + 1,
                        step + 1,
                        len(idx_list),
                        progress_pct,
                        optimizer_step_count,
                        optimizer_steps_per_epoch,
                        current_lr,
                        loss_value,
                        (epoch_loss + running_loss) / (step + 1),
                        elapsed,
                    )

            if len(idx_list) % self.cfg['grad_accum'] != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and self.cfg.get('scheduler') != 'cosine_restarts':
                    scheduler.step()
                epoch_loss += running_loss
                optimizer_step_count += 1

            model.eval()
            pk10 = []
            pk20 = []
            val_start = time.time()
            with torch.no_grad():
                for val_step, (vx, vy) in enumerate(zip(val_x, val_y), start=1):
                    if (vy > 0).sum() > 0:
                        vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                        k10 = min(10, (vy > 0).sum().item(), n_nodes)
                        _, t10_idx = torch.topk(vy.to(DEVICE), k10)
                        _, p10_idx = torch.topk(vpred, 10)
                        pk10.append(len(set(t10_idx.cpu().numpy()) & set(p10_idx.cpu().numpy())) / k10)

                        k20 = min(20, (vy > 0).sum().item(), n_nodes)
                        _, t20_idx = torch.topk(vy.to(DEVICE), k20)
                        _, p20_idx = torch.topk(vpred, min(20, n_nodes))
                        pk20.append(len(set(t20_idx.cpu().numpy()) & set(p20_idx.cpu().numpy())) / k20)

                    should_log_val = (
                        val_step == 1
                        or val_step % TRAIN_BATCH_LOG_EVERY == 0
                        or val_step == len(val_x)
                    )
                    if should_log_val:
                        running_p10 = np.mean(pk10) if pk10 else 0.0
                        running_p20 = np.mean(pk20) if pk20 else 0.0
                        logging.info(
                            "E%03d val %s/%s (%.1f%%) | P@10=%.2f%% | P@20=%.2f%% | elapsed=%.1fs",
                            epoch + 1,
                            val_step,
                            len(val_x),
                            (val_step / len(val_x)) * 100.0,
                            running_p10 * 100,
                            running_p20 * 100,
                            time.time() - val_start,
                        )

            avg_p10 = np.mean(pk10) if pk10 else 0.0
            avg_p20 = np.mean(pk20) if pk20 else 0.0
            logging.info(
                "E%03d done | Val P@10: %.2f%% | Val P@20: %.2f%% | Loss: %.4f | opt_steps=%s | elapsed=%.1fs",
                epoch + 1,
                avg_p10 * 100,
                avg_p20 * 100,
                epoch_loss / len(train_x),
                optimizer_step_count,
                time.time() - epoch_start,
            )
            if avg_p10 > best_pk:
                best_pk = avg_p10
                no_improve = 0
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'model_class': model.__class__.__name__,
                        'window': window,
                        'in_channels': 41,
                        'region': self.region_key,
                    },
                    os.path.join(ROOT_DIR, 'models', 'active', self.cfg['output_name']),
                )
                logging.info(f"NOVO RECORDE: {best_pk * 100:.2f}%")
            else:
                no_improve += 1
                if no_improve >= self.cfg['patience']:
                    break

            if self.cfg.get('scheduler') == 'cosine_restarts' and scheduler is not None:
                scheduler.step()

    def _build_model(self, n_nodes, in_channels, window):
        model_name = self.cfg.get('model_class', 'ShallowGAT')
        if model_name == 'FortalezaHeteroSTGAT':
            return FortalezaHeteroSTGAT(
                num_nodes=n_nodes,
                in_channels=in_channels,
                time_steps=window,
                dropout=self.cfg['dropout'],
            )
        if model_name == 'PureSTGCN_64':
            return PureSTGCN_64(
                num_nodes=n_nodes,
                in_channels=in_channels,
                time_steps=window,
                dropout=self.cfg['dropout'],
            )
        if model_name == 'DeepSTGAT_64':
            return DeepSTGAT_64(
                num_nodes=n_nodes,
                in_channels=in_channels,
                time_steps=window,
                dropout=self.cfg['dropout'],
            )
        if model_name == 'DeepSTGAT_80':
            return DeepSTGAT_80(
                num_nodes=n_nodes,
                in_channels=in_channels,
                time_steps=window,
                dropout=self.cfg['dropout'],
            )
        return ShallowGAT(
            num_nodes=n_nodes,
            in_channels=in_channels,
            time_steps=window,
            dropout=self.cfg['dropout'],
        )


def main():
    for region in REGION_CONFIGS.keys():
        try:
            SpecialistTrainer(region).train()
        except Exception as e:
            logging.error(f"Erro em {region}: {e}")


if __name__ == "__main__":
    main()
