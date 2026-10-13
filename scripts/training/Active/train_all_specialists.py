"""
train_all_specialists.py - Script oficial de retreino (FOCO CVLI - HONESTY PARADIGM).
Versao 2026-07-24: Horizonte 30d, MemPalace Universal, Honesty Constraint.
"""
import gc
import json
import logging
import math
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
    from architectures import DeepSTGAT_64, DeepSTGAT_80, DeepSTGAT_v5, FortalezaHeteroSTGAT, PureSTGCN_64, ShallowGAT
    from training_vault import TrainingVault
except ImportError:
    from src.core.architectures import DeepSTGAT_64, DeepSTGAT_80, DeepSTGAT_v5, FortalezaHeteroSTGAT, PureSTGCN_64, ShallowGAT
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
PREDICT_HORIZON = 30
TRAIN_BATCH_LOG_EVERY = int(os.environ.get("TRAIN_BATCH_LOG_EVERY", "30"))
VALIDATION_DAYS = 60

REGION_CONFIGS = {
    'fortaleza': dict(
        window=120,             # +contexto histórico (era 90)
        lr=0.000015,            # regime conservador: preserva o ranking validado cedo
        epochs=40,
        patience=6,
        dropout=0.40,
        margin=1.0,
        k_eval=10,
        use_momentum=True,
        grad_accum=4,           # 249 passos/época (era 125) — mais atualizações
        output_name=os.path.join('legacy_torch', 'fortaleza_model_active.pth'),
        model_class='DeepSTGAT_v5',
        loss_type='negbinom',
        nb_r_init=2.0,
        nb_r_lr=0.00005,        # evita que a dispersão absorva a otimização do ranking
        nb_r_min=0.75,
        nb_r_max=50.0,
        ranking_weight=2.0,
        indecision_weight=0.2,
        weight_decay=0.02,
        scheduler='onecycle',
        onecycle_pct_start=0.10,
        validation_p10_weight=0.60,
        validation_min_delta=0.0025,
        incumbent_validation_score=0.42734,  # E002 anterior: 0.6*P@10 + 0.4*P@20
        bimestral_filter=True,
        # Hiperparâmetros para fine-tuning RL (SCST)
        rl_lr=0.000005,
        rl_epochs=6,
        rl_patience=4,
        rl_weight=0.15,
        rl_fp_penalty=0.10,
        rl_w10=0.60,
        rl_w20=0.40,
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
        output_name=os.path.join('legacy_torch', 'rmf_model.pth'),
        model_class='DeepSTGAT_v5',
        focal_alpha=0.50,
        focal_gamma=2.0,
        ranking_weight=10.0,
        scheduler='onecycle',
        bimestral_filter=True,
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
        output_name=os.path.join('legacy_torch', 'interior_model.pth'),
        model_class='DeepSTGAT_v5',
        focal_alpha=0.40,
        focal_gamma=2.0,
        ranking_weight=15.0,
        scheduler='onecycle',
        bimestral_filter=True,
    ),
}


def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(adj)


def get_active_nodes_bimestral(nf, t, window, bimester_days=60, min_bimesters=1):
    """
    Retorna máscara booleana (N,) de nós ativos:
    bairros com >= 1 CVLI em pelo menos `min_bimesters` bimestres da janela.

    Filtra bairros perpetuamente inativos para evitar gradientes zerados que
    poluem o treinamento e reduzem a discriminação nos hotspots reais.

    nf:            (N, T, C) — node features; canal 0 = CVLI total (contagem)
    t:             índice do fim da janela histórica (exclusive)
    window:        tamanho da janela em dias
    bimester_days: tamanho do bimestre (padrão 60 dias)
    min_bimesters: mínimo de bimestres com atividade para o nó ser considerado ativo
    """
    cvli = nf[:, t - window:t, 0]          # (N, window)
    n_bimesters = max(1, window // bimester_days)
    bimester_active = np.zeros((nf.shape[0], n_bimesters), dtype=bool)
    for b in range(n_bimesters):
        start_b = b * bimester_days
        end_b = min((b + 1) * bimester_days, window)
        bimester_active[:, b] = cvli[:, start_b:end_b].sum(axis=1) >= 1
    return bimester_active.sum(axis=1) >= min_bimesters


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


def build_temporal_split_config(dates):
    """Cria um corte temporal recente sem usar alvos ainda desconhecidos.

    Cada alvo cobre ``PREDICT_HORIZON`` dias. Portanto, os últimos 30 dias da
    série servem apenas como contexto de inferência; a validação termina no
    último dia que já possui seu horizonte completamente observado.
    """
    series_start = pd.Timestamp(dates[0]).normalize()
    series_end = pd.Timestamp(dates[-1]).normalize()
    last_observed_target_end = series_end - pd.Timedelta(days=PREDICT_HORIZON)
    val_start = last_observed_target_end - pd.Timedelta(days=VALIDATION_DAYS - 1)
    train_end = val_start - pd.Timedelta(days=1)
    if train_end <= series_start:
        raise RuntimeError(
            "Série insuficiente para split temporal: "
            f"{series_start.date()}..{series_end.date()}"
        )
    return {
        'train_start': series_start,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': last_observed_target_end,
    }


def negative_binomial_from_mean(mu, r):
    """Create a PyTorch NegativeBinomial parameterized by its mean ``mu``."""
    probs = (mu / (r + mu)).clamp(1e-6, 1.0 - 1e-6)
    return torch.distributions.NegativeBinomial(total_count=r, probs=probs)


def calculate_batch_metrics(pred, target, loss_type, active_mask=None):
    """Return deterministic ranking and calibration metrics for one window."""
    pred = pred.detach().squeeze()
    target = target.detach().squeeze()
    if active_mask is not None and active_mask.sum() > 0:
        mask = active_mask.detach().bool()
        pred = pred[mask]
        target = target[mask]

    positive_count = int((target > 0).sum().item())
    metrics = {
        'nodes_evaluated': int(target.numel()),
        'target_positive_nodes': positive_count,
        'target_total_events': float(target.sum().item()),
    }
    for k in (10, 20):
        predicted_k = min(k, int(target.numel()))
        true_k = min(k, positive_count)
        if predicted_k == 0 or true_k == 0:
            hits = 0
        else:
            pred_idx = set(torch.topk(pred, predicted_k).indices.cpu().tolist())
            true_idx = set(torch.topk(target, true_k).indices.cpu().tolist())
            hits = len(pred_idx & true_idx)
        metrics[f'hits_at_{k}'] = hits
        metrics[f'misses_at_{k}'] = true_k - hits
        metrics[f'false_positives_at_{k}'] = predicted_k - hits
        metrics[f'precision_at_{k}'] = hits / true_k if true_k else 0.0

    if loss_type == 'negbinom':
        estimate = F.softplus(pred)
        expected = target.float()
        metrics['error_scale'] = 'count_mean'
    else:
        estimate = torch.sigmoid(pred)
        expected = (target > 0).float()
        metrics['error_scale'] = 'event_probability'
    residual = estimate - expected
    metrics['mae'] = float(residual.abs().mean().item())
    metrics['rmse'] = float(torch.sqrt((residual ** 2).mean()).item())
    metrics['prediction_mean'] = float(estimate.mean().item())
    metrics['target_mean'] = float(expected.mean().item())
    return metrics


def _detached_float(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


class NegBinomRankingLoss(nn.Module):
    """
    Loss para contagem de CVLI com distribuição Binomial Negativa + ListMLE.

    Vantagens sobre MSE/Focal para dados de crime:
    - Modela overdispersão (variância > média) naturalmente presente no CVLI
    - Zero-inflation implícita: P(y=0 | mu_baixo) alta sem penalização extra
    - Gradientes menores para bairros quietos (honesty embutido)
    - ListMLE alinha diretamente com P@K em vez de minimizar erro de magnitude

    Parâmetros:
        log_r: dispersão aprendível (r = exp(log_r), limitado por nb_r_min/nb_r_max)
                Começa em r=nb_r_init. Converge para a overdispersão real dos dados.
        ranking_weight: peso do componente ListMLE
        indecision_weight: peso do gap penalty (top-K vs média)
    """

    def __init__(
        self,
        nb_r_init=2.0,
        ranking_weight=2.0,
        indecision_weight=0.2,
        nb_r_min=0.1,
        nb_r_max=50.0,
    ):
        super().__init__()
        if not 0 < nb_r_min <= nb_r_init <= nb_r_max:
            raise ValueError('Esperado 0 < nb_r_min <= nb_r_init <= nb_r_max')
        self.log_r = nn.Parameter(torch.tensor(math.log(nb_r_init)))
        self.ranking_weight = ranking_weight
        self.indecision_weight = indecision_weight
        self.nb_r_min = float(nb_r_min)
        self.nb_r_max = float(nb_r_max)

    def dispersion(self):
        return self.log_r.exp().clamp(min=self.nb_r_min, max=self.nb_r_max)

    def constrain_dispersion_(self):
        with torch.no_grad():
            self.log_r.clamp_(math.log(self.nb_r_min), math.log(self.nb_r_max))

    def forward(self, pred, target, active_mask=None, cold_streak_signal=None):
        # Filtro de nós ativos
        if active_mask is not None and active_mask.sum() > 0:
            pred = pred[active_mask]
            target = target[active_mask]
            if cold_streak_signal is not None:
                cold_streak_signal = cold_streak_signal[active_mask]

        mu = F.softplus(pred.squeeze())                              # pred → μ positivo
        r = self.dispersion()                                      # dispersão aprendível

        # Binomial Negativa NLL: -log P(y | μ, r)
        dist = negative_binomial_from_mean(mu, r)
        nb_loss = -dist.log_prob(target.float()).mean()

        # Entropia cruzada listwise: aproxima a distribuição espacial do alvo.
        # Não é o ListMLE clássico (Plackett-Luce).
        listwise_ce_loss = -(
            F.softmax(target.float(), dim=0) * F.log_softmax(mu, dim=0)
        ).sum()

        # Gap penalty: top-K deve ter μ bem separado do restante
        top_vals, _ = torch.topk(mu, min(10, len(mu)))
        gap = top_vals.mean() - mu.mean()
        indecision_penalty = torch.exp(-gap)

        # Honesty: bairros com cold streak não devem ter μ alto
        honesty_penalty = 0.0
        if cold_streak_signal is not None:
            calmness = torch.clamp(-cold_streak_signal / 30.0, 0, 1)
            honesty_penalty = (calmness * mu).mean()

        total_loss = (
            nb_loss
            + self.ranking_weight * listwise_ce_loss
            + self.indecision_weight * indecision_penalty
            + honesty_penalty
        )
        self.last_components = {
            'nb_nll': _detached_float(nb_loss),
            'listwise_ce': _detached_float(listwise_ce_loss),
            'listwise_weighted': _detached_float(self.ranking_weight * listwise_ce_loss),
            'indecision': _detached_float(indecision_penalty),
            'indecision_weighted': _detached_float(self.indecision_weight * indecision_penalty),
            'honesty': _detached_float(honesty_penalty),
            'total': _detached_float(total_loss),
        }
        return total_loss


class SelfCriticalRankingLoss(nn.Module):
    """
    Fine-tuning via Aprendizado por Reforco (Self-Critical Policy Gradient / SCST).
    Otimiza diretamente as metricas operacionais de corte P@10 e P@20 combinadas com
    a probabilidade de Poisson / Negative Binomial para refinar a fronteira de ranqueamento.
    """

    def __init__(
        self,
        base_criterion,
        k1=10,
        k2=20,
        w10=0.6,
        w20=0.4,
        fp_penalty=0.10,
        rl_weight=0.15,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.k1 = k1
        self.k2 = k2
        self.w10 = w10
        self.w20 = w20
        self.fp_penalty = fp_penalty
        self.rl_weight = rl_weight

    def dispersion(self):
        if hasattr(self.base_criterion, 'dispersion'):
            return self.base_criterion.dispersion()
        return None

    def constrain_dispersion_(self):
        if hasattr(self.base_criterion, 'constrain_dispersion_'):
            self.base_criterion.constrain_dispersion_()

    @property
    def log_r(self):
        if hasattr(self.base_criterion, 'log_r'):
            return self.base_criterion.log_r
        return None

    def _compute_reward(self, selected_indices, target):
        pos_mask = (target > 0)
        pos_count = int(pos_mask.sum().item())
        if pos_count == 0:
            return 0.0

        target_k1 = min(self.k1, pos_count)
        target_k2 = min(self.k2, pos_count)
        target_top10 = set(torch.topk(target, target_k1).indices.cpu().tolist())
        target_top20 = set(torch.topk(target, target_k2).indices.cpu().tolist())

        sel10 = set(selected_indices[:min(self.k1, len(selected_indices))].cpu().tolist())
        sel20 = set(selected_indices[:min(self.k2, len(selected_indices))].cpu().tolist())

        p10 = len(sel10 & target_top10) / target_k1 if target_k1 else 0.0
        p20 = len(sel20 & target_top20) / target_k2 if target_k2 else 0.0

        # Penalidade por falso positivo no Top-10
        fp10 = sum(1 for idx in sel10 if target[idx] == 0) / max(1, len(sel10))
        return (self.w10 * p10) + (self.w20 * p20) - (self.fp_penalty * fp10)

    def forward(self, pred, target, active_mask=None, cold_streak_signal=None):
        base_loss = self.base_criterion(pred, target, active_mask=active_mask, cold_streak_signal=cold_streak_signal)

        # Filtrar nós para a amostragem de RL
        p_work = pred.squeeze()
        t_work = target.squeeze()
        if active_mask is not None and active_mask.sum() > 0:
            p_work = p_work[active_mask]
            t_work = t_work[active_mask]

        scores = F.softplus(p_work)
        n_nodes = scores.numel()
        if n_nodes < self.k1:
            self.last_components = dict(getattr(self.base_criterion, 'last_components', {}))
            self.last_components['rl_policy'] = 0.0
            self.last_components['rl_advantage'] = 0.0
            return base_loss

        log_probs = F.log_softmax(scores, dim=0)

        # 1. Trajetoria Gulosa (Baseline de comparacao sem gradiente)
        greedy_indices = torch.topk(scores, min(self.k2, n_nodes)).indices
        with torch.no_grad():
            r_greedy = self._compute_reward(greedy_indices, t_work)

        # 2. Amostragem Exploratoria com Ruido Gumbel (Plackett-Luce)
        gumbel = -torch.empty_like(scores).exponential_().log()
        perturbed_scores = (scores + 1e-6).log() + gumbel
        sampled_indices = torch.topk(perturbed_scores, min(self.k2, n_nodes)).indices

        with torch.no_grad():
            r_sample = self._compute_reward(sampled_indices, t_work)
            advantage = r_sample - r_greedy

        # 3. Policy gradient: desencoraja escolhas piores e incentiva acertos na fronteira
        sampled_log_probs = log_probs[sampled_indices]
        policy_loss = -advantage * sampled_log_probs.sum()

        total_loss = base_loss + self.rl_weight * policy_loss

        self.last_components = dict(getattr(self.base_criterion, 'last_components', {}))
        self.last_components['rl_policy'] = _detached_float(policy_loss)
        self.last_components['rl_advantage'] = _detached_float(advantage)
        self.last_components['total'] = _detached_float(total_loss)
        return total_loss


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

    def forward(self, pred, target, active_mask=None, cold_streak_signal=None):
        """
        active_mask: tensor booleano (N,) — se fornecido, calcula o loss apenas nos nós
                     com atividade CVLI bimestral consistente, eliminando gradientes
                     de bairros perpetuamente inativos que poluem o treinamento.
        """
        # Aplicar filtro de nós ativos antes de qualquer cálculo preditivo
        if active_mask is not None and active_mask.sum() > 0:
            pred = pred[active_mask]
            target = target[active_mask]
            if cold_streak_signal is not None:
                cold_streak_signal = cold_streak_signal[active_mask]

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

        total_loss = (
            focal_loss
            + self.ranking_weight * rank_loss
            + self.honesty_weight * honesty_penalty
            + self.indecision_weight * indecision_penalty
            + self.l2_weight * torch.norm(pred, 2)
        )
        self.last_components = {
            'focal': _detached_float(focal_loss),
            'positive_mse': _detached_float(rank_loss),
            'positive_mse_weighted': _detached_float(self.ranking_weight * rank_loss),
            'honesty': _detached_float(honesty_penalty),
            'indecision': _detached_float(indecision_penalty),
            'l2': _detached_float(self.l2_weight * torch.norm(pred, 2)),
            'total': _detached_float(total_loss),
        }
        return total_loss


class SpecialistTrainer:
    def __init__(self, region_key, fine_tune_rl=False):
        self.cfg = dict(REGION_CONFIGS[region_key])
        self.region_key = region_key
        self.fine_tune_rl = fine_tune_rl
        self.vault = None
        if self.fine_tune_rl:
            self.cfg['lr'] = float(self.cfg.get('rl_lr', 5e-6))
            self.cfg['epochs'] = int(self.cfg.get('rl_epochs', 5))
            self.cfg['patience'] = int(self.cfg.get('rl_patience', 4))
            self.cfg['scheduler'] = 'static'

    def train(self):
        logging.info(f"\nESPECIALISTA: {self.region_key.upper()} (SENTINELA V4 - HONESTY)")
        path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        nf = data['node_features']
        dates = pd.to_datetime(data['dates'])
        adj_geo_np = data['adj_geo']
        adj_conf_np = data['adj_conflict']
        n_nodes, total_steps, _ = nf.shape
        adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(normalize_adj(adj_conf_np), dtype=torch.float32).to(DEVICE)

        features = inject_momentum_channels(nf)

        self.vault = TrainingVault(n_nodes, ROOT_DIR)

        x_list, y_list = [], []
        train_samples, val_samples = [], []
        window = self.cfg['window']
        total_windows = max(0, total_steps - PREDICT_HORIZON - window)
        split_cfg = build_temporal_split_config(dates)
        build_start = time.time()
        logging.info(
            (
                "Preparando janelas | region=%s | total_steps=%s | candidate_windows=%s "
                "| train=%s..%s | val=%s..%s"
            ),
            self.region_key,
            total_steps,
            total_windows,
            split_cfg['train_start'].date(),
            split_cfg['train_end'].date(),
            split_cfg['val_start'].date(),
            split_cfg['val_end'].date(),
        )
        for t in range(window, total_steps - PREDICT_HORIZON):
            history_start = dates[t - window]
            target_start = dates[t]
            target_end = dates[t + PREDICT_HORIZON - 1]

            x_win = features[:, t - window:t, :].copy()
            x = torch.tensor(x_win, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            y = torch.tensor(nf[:, t:t + PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)

            if x.shape[1] < 41:
                padding = torch.zeros((1, 41 - x.shape[1], n_nodes, window))
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] > 41:
                x = x[:, :41, :, :]

            # Filtro bimestral: máscara de nós com >= 1 CVLI por bimestre
            if self.cfg.get('bimestral_filter', False):
                active_mask = torch.from_numpy(
                    get_active_nodes_bimestral(nf, t, window)
                )
            else:
                active_mask = None

            sample = {
                'x': x,
                'y': y,
                'active_mask': active_mask,
                'history_start': history_start,
                'target_start': target_start,
                'target_end': target_end,
            }

            if (
                history_start >= split_cfg['train_start']
                and target_end <= split_cfg['train_end']
            ):
                train_samples.append(sample)
            elif (
                target_start >= split_cfg['val_start']
                and target_end <= split_cfg['val_end']
            ):
                val_samples.append(sample)

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

        train_x = [sample['x'] for sample in train_samples]
        train_y = [sample['y'] for sample in train_samples]
        train_masks = [sample['active_mask'] for sample in train_samples]
        val_x = [sample['x'] for sample in val_samples]
        val_y = [sample['y'] for sample in val_samples]
        val_masks = [sample['active_mask'] for sample in val_samples]

        if train_masks[0] is not None:
            active_ratio = np.mean([m.float().mean().item() for m in train_masks])
            logging.info(
                "Filtro bimestral | nós ativos médio: %.1f%% do total (%d)",
                active_ratio * 100,
                n_nodes,
            )
        if not train_x or not val_x:
            raise RuntimeError(
                f"Janela insuficiente para treino/validacao em {self.region_key}: "
                f"train={len(train_x)} | val={len(val_x)}"
            )
        logging.info(
            (
                "Split temporal aplicado | region=%s | train_windows=%s | val_windows=%s "
                "| first_train_target=%s | last_train_target=%s | "
                "first_val_target=%s | last_val_target=%s"
            ),
            self.region_key,
            len(train_samples),
            len(val_samples),
            train_samples[0]['target_start'].date(),
            train_samples[-1]['target_end'].date(),
            val_samples[0]['target_start'].date(),
            val_samples[-1]['target_end'].date(),
        )

        model = self._build_model(n_nodes=n_nodes, in_channels=41, window=window).to(DEVICE)

        # Se modo Fine-Tuning RL estiver ativo, carregar pesos consolidados do checkpoint incumbente
        output_path = os.path.join(ROOT_DIR, 'models', 'active', self.cfg['output_name'])
        if self.fine_tune_rl:
            if os.path.exists(output_path):
                logging.info("Carregando checkpoint para Fine-Tuning RL: %s", output_path)
                ckpt = torch.load(output_path, map_location=DEVICE, weights_only=False)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    logging.info("Pesos do modelo carregados com sucesso para etapa de Reforco.")
            else:
                logging.warning("Checkpoint %s nao encontrado. Fine-Tuning RL comecara dos pesos atuais.", output_path)

        # Construir criterion antes do optimizer para incluir log_r se NegBinom
        if self.cfg.get('loss_type') == 'negbinom':
            criterion = NegBinomRankingLoss(
                nb_r_init=self.cfg.get('nb_r_init', 2.0),
                ranking_weight=self.cfg.get('ranking_weight', 2.0),
                indecision_weight=self.cfg.get('indecision_weight', 0.2),
                nb_r_min=self.cfg.get('nb_r_min', 0.1),
                nb_r_max=self.cfg.get('nb_r_max', 50.0),
            ).to(DEVICE)
            logging.info(
                "Loss: NegBinomRankingLoss | r_init=%.2f | r_bounds=[%.2f, %.2f] | ranking_w=%.2f | indecision_w=%.2f",
                self.cfg.get('nb_r_init', 2.0),
                self.cfg.get('nb_r_min', 0.1),
                self.cfg.get('nb_r_max', 50.0),
                self.cfg.get('ranking_weight', 2.0),
                self.cfg.get('indecision_weight', 0.2),
            )
        else:
            criterion = BinaryFocalRankingLoss(
                alpha=self.cfg['focal_alpha'],
                gamma=self.cfg['focal_gamma'],
                ranking_weight=self.cfg['ranking_weight'],
                indecision_weight=self.cfg.get('indecision_weight', 1.5),
            )

        if self.fine_tune_rl:
            criterion = SelfCriticalRankingLoss(
                base_criterion=criterion,
                k1=10,
                k2=20,
                w10=self.cfg.get('rl_w10', 0.6),
                w20=self.cfg.get('rl_w20', 0.4),
                fp_penalty=self.cfg.get('rl_fp_penalty', 0.10),
                rl_weight=self.cfg.get('rl_weight', 0.15),
            ).to(DEVICE)
            logging.info(
                "Fine-Tuning RL Ativo: SelfCriticalRankingLoss | rl_weight=%.2f | fp_pen=%.2f | lr=%.2e",
                self.cfg.get('rl_weight', 0.15),
                self.cfg.get('rl_fp_penalty', 0.10),
                self.cfg['lr'],
            )

        # Optimizer com grupos de parâmetros separados:
        # log_r recebe LR muito maior para convergir para dispersão real rápido
        if hasattr(criterion, 'log_r') and self.cfg.get('nb_r_lr'):
            param_groups = [
                {
                    'params': model.parameters(),
                    'lr': self.cfg['lr'],
                    'weight_decay': self.cfg.get('weight_decay', 0.005),
                },
                {
                    'params': [criterion.log_r],
                    'lr': self.cfg['nb_r_lr'],
                    'weight_decay': 0.0,  # sem decay no escalar de dispersão
                },
            ]
            optimizer = torch.optim.AdamW(param_groups)
            logging.info(
                "Optimizer param_groups: modelo lr=%.2e | log_r lr=%.2e",
                self.cfg['lr'], self.cfg['nb_r_lr'],
            )
        else:
            all_params = list(model.parameters()) + list(criterion.parameters())
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.cfg['lr'],
                weight_decay=self.cfg.get('weight_decay', 0.005),
            )

        trainable_parameters = list(model.parameters()) + list(criterion.parameters())
        metrics_path = os.path.join(ROOT_DIR, 'logs', 'training_batch_metrics.jsonl')

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
            # max_lr por grupo de parâmetros se houver múltiplos grupos
            if len(optimizer.param_groups) > 1:
                max_lrs = [pg['lr'] for pg in optimizer.param_groups]
            else:
                max_lrs = self.cfg['lr']
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                steps_per_epoch=len(train_x) // self.cfg['grad_accum'] + 1,
                epochs=self.cfg['epochs'],
                pct_start=self.cfg.get('onecycle_pct_start', 0.3),
            )

        best_validation_score = self.cfg.get('incumbent_validation_score', 0.0)
        no_improve = 0
        if best_validation_score > 0:
            logging.info(
                "Piso do checkpoint incumbente: score composto=%.2f%%",
                best_validation_score * 100,
            )
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
                active_mask_t = train_masks[idx].to(DEVICE) if train_masks[idx] is not None else None
                pred = model(xi, [adj_geo, adj_conf]).squeeze()
                loss = criterion(
                    pred,
                    train_y[idx].to(DEVICE),
                    active_mask=active_mask_t,
                    cold_streak_signal=cold_streak,
                ) / self.cfg['grad_accum']
                loss.backward()
                loss_value = loss.item() * self.cfg['grad_accum']
                running_loss += loss_value

                batch_metrics = calculate_batch_metrics(
                    pred,
                    train_y[idx].to(DEVICE),
                    self.cfg.get('loss_type', 'binary_focal'),
                    active_mask=active_mask_t,
                )
                component_metrics = dict(getattr(criterion, 'last_components', {}))
                current_lrs = [group['lr'] for group in optimizer.param_groups]
                metric_record = {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'phase': 'train',
                    'region': self.region_key,
                    'epoch': epoch + 1,
                    'batch': step + 1,
                    'batches_in_epoch': len(idx_list),
                    'loss_total': loss_value,
                    'loss_components': component_metrics,
                    'lr_model': current_lrs[0],
                    'lr_loss': current_lrs[1] if len(current_lrs) > 1 else current_lrs[0],
                    'learned_nb_r': (
                        criterion.dispersion().item()
                        if hasattr(criterion, 'log_r')
                        else None
                    ),
                    **batch_metrics,
                }
                with open(metrics_path, 'a', encoding='utf-8') as metrics_file:
                    metrics_file.write(json.dumps(metric_record, ensure_ascii=False) + '\n')

                if (step + 1) % self.cfg['grad_accum'] == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                    optimizer.step()
                    if hasattr(criterion, 'constrain_dispersion_'):
                        criterion.constrain_dispersion_()
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
                    loss_lr = (
                        optimizer.param_groups[1]['lr']
                        if len(optimizer.param_groups) > 1
                        else current_lr
                    )
                    logging.info(
                        "E%03d batch %s/%s (%.1f%%) | opt_steps=%s/%s | lr_model=%.6g | lr_loss=%.6g | loss=%.4f | P@10=%.1f%% (%s hit/%s miss/%s FP) | MAE=%.4f RMSE=%.4f | loss_epoch=%.4f | elapsed=%.1fs",
                        epoch + 1,
                        step + 1,
                        len(idx_list),
                        progress_pct,
                        optimizer_step_count,
                        optimizer_steps_per_epoch,
                        current_lr,
                        loss_lr,
                        loss_value,
                        batch_metrics['precision_at_10'] * 100,
                        batch_metrics['hits_at_10'],
                        batch_metrics['misses_at_10'],
                        batch_metrics['false_positives_at_10'],
                        batch_metrics['mae'],
                        batch_metrics['rmse'],
                        (epoch_loss + running_loss) / (step + 1),
                        elapsed,
                    )

            if len(idx_list) % self.cfg['grad_accum'] != 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                if hasattr(criterion, 'constrain_dispersion_'):
                    criterion.constrain_dispersion_()
                optimizer.zero_grad()
                if scheduler is not None and self.cfg.get('scheduler') != 'cosine_restarts':
                    scheduler.step()
                epoch_loss += running_loss
                optimizer_step_count += 1

            model.eval()
            pk10 = []
            pk20 = []
            val_start = time.time()
            k_eval = self.cfg.get('k_eval', 10)
            with torch.no_grad():
                for val_step, (vx, vy) in enumerate(zip(val_x, val_y), start=1):
                    n_events = int((vy > 0).sum().item())
                    # CAUSA C fix: so calcula P@K quando ha eventos suficientes para
                    # a metrica ser estatisticamente significativa (evita ruido)
                    if n_events >= k_eval:
                        vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                        k10 = min(10, n_events, n_nodes)
                        _, t10_idx = torch.topk(vy.to(DEVICE), k10)
                        _, p10_idx = torch.topk(vpred, 10)
                        pk10.append(len(set(t10_idx.cpu().numpy()) & set(p10_idx.cpu().numpy())) / k10)

                        k20 = min(20, n_events, n_nodes)
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
            p10_weight = self.cfg.get('validation_p10_weight', 0.6)
            validation_score = p10_weight * avg_p10 + (1.0 - p10_weight) * avg_p20
            validation_min_delta = self.cfg.get('validation_min_delta', 0.0)
            # Log do parâmetro r aprendido (apenas para NegBinom)
            r_str = ""
            if hasattr(criterion, 'log_r'):
                r_val = criterion.dispersion().item()
                r_str = f" | r={r_val:.3f}"
            logging.info(
                "E%03d done | Val P@10: %.2f%% | Val P@20: %.2f%% | Score: %.2f%% | Loss: %.4f | opt_steps=%s | elapsed=%.1fs%s",
                epoch + 1,
                avg_p10 * 100,
                avg_p20 * 100,
                validation_score * 100,
                epoch_loss / len(train_x),
                optimizer_step_count,
                time.time() - epoch_start,
                r_str,
            )
            if validation_score > best_validation_score + validation_min_delta:
                best_validation_score = validation_score
                no_improve = 0
                output_path = os.path.join(ROOT_DIR, 'models', 'active', self.cfg['output_name'])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'criterion_state_dict': criterion.state_dict(),
                        'loss_type': self.cfg.get('loss_type', 'binary_focal'),
                        'loss_config': {
                            'nb_r_init': self.cfg.get('nb_r_init'),
                            'nb_r_lr': self.cfg.get('nb_r_lr'),
                            'nb_r_min': self.cfg.get('nb_r_min'),
                            'nb_r_max': self.cfg.get('nb_r_max'),
                            'ranking_weight': self.cfg.get('ranking_weight'),
                            'indecision_weight': self.cfg.get('indecision_weight', 1.5),
                            'focal_alpha': self.cfg.get('focal_alpha'),
                            'focal_gamma': self.cfg.get('focal_gamma'),
                        },
                        'learned_nb_r': (
                            criterion.dispersion().item()
                            if hasattr(criterion, 'log_r')
                            else None
                        ),
                        'validation_metrics': {
                            'p10': float(avg_p10),
                            'p20': float(avg_p20),
                            'composite_score': float(validation_score),
                            'p10_weight': float(p10_weight),
                            'min_delta': float(validation_min_delta),
                        },
                        'model_class': model.__class__.__name__,
                        'window': window,
                        'predict_horizon': PREDICT_HORIZON,
                        'in_channels': 41,
                        'region': self.region_key,
                        'train_start': str(split_cfg['train_start'].date()),
                        'train_end': str(split_cfg['train_end'].date()),
                        'val_start': str(split_cfg['val_start'].date()),
                        'val_end': str(split_cfg['val_end'].date()),
                    },
                    output_path,
                )
                logging.info(
                    "NOVO RECORDE COMPOSTO: %.2f%% | P@10=%.2f%% | P@20=%.2f%%",
                    best_validation_score * 100,
                    avg_p10 * 100,
                    avg_p20 * 100,
                )
            else:
                no_improve += 1
                logging.info(
                    "Sem melhora composta: %s/%s | score=%.2f%% | best=%.2f%% | min_delta=%.2fpp",
                    no_improve,
                    self.cfg['patience'],
                    validation_score * 100,
                    best_validation_score * 100,
                    validation_min_delta * 100,
                )
                if no_improve >= self.cfg['patience']:
                    logging.info(
                        "Early stopping | region=%s | best_score=%.2f%%",
                        self.region_key,
                        best_validation_score * 100,
                    )
                    break

            if self.cfg.get('scheduler') == 'cosine_restarts' and scheduler is not None:
                scheduler.step()

    def _build_model(self, n_nodes, in_channels, window):
        model_name = self.cfg.get('model_class', 'ShallowGAT')
        if model_name == 'DeepSTGAT_v5':
            return DeepSTGAT_v5(
                num_nodes=n_nodes,
                in_channels=in_channels,
                time_steps=window,
                num_graphs=2,
                dropout=self.cfg['dropout'],
            )
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
    import argparse
    parser = argparse.ArgumentParser(description='Retreino de especialistas regionais')
    parser.add_argument(
        '--region', nargs='+',
        default=list(REGION_CONFIGS.keys()),
        choices=list(REGION_CONFIGS.keys()),
        help='Regiões a treinar (default: todas)',
    )
    parser.add_argument(
        '--fine-tune-rl', action='store_true',
        help='Executa fine-tuning via Aprendizado por Reforço (SCST) sobre o checkpoint incumbente',
    )
    args, _ = parser.parse_known_args()
    metrics_path = os.path.join(ROOT_DIR, 'logs', 'training_batch_metrics.jsonl')
    with open(metrics_path, 'w', encoding='utf-8'):
        pass
    logging.info("Regiões selecionadas: %s | Modo RL Fine-Tuning: %s", args.region, args.fine_tune_rl)
    failures = []
    for region in args.region:
        try:
            SpecialistTrainer(region, fine_tune_rl=args.fine_tune_rl).train()
        except Exception as e:
            logging.error(f"Erro em {region}: {e}", exc_info=True)
            failures.append((region, str(e)))
    if failures:
        details = '; '.join(f'{region}: {message}' for region, message in failures)
        raise RuntimeError(f"Falha no retreino de {len(failures)} região(ões): {details}")


if __name__ == "__main__":
    main()
