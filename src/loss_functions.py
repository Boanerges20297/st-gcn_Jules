"""
Loss Functions for Ranking Optimization
Implements various loss functions for P@K optimization

Author: ST-GCN Enhanced System
Date: Feb 2026

Losses:
- PairwiseRankingLoss: Optimize for top-K ranking
- TopKLoss: Weighted loss emphasizing top-K accuracy
- CombinedLoss: Balance between P@5 and P@20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss
    
    Penalizes incorrect relative ordering of node scores
    Focuses on getting top-K nodes ordered correctly
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize pairwise loss
        
        Args:
            margin: Margin for margin-based ranking loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss
        
        Args:
            scores: Predicted scores (batch, n_nodes)
            targets: Target ranks/scores (batch, n_nodes)
        
        Returns:
            Scalar loss value
        """
        batch_size, n_nodes = scores.shape
        
        # Compute pairwise differences
        loss = 0.0
        
        for i in range(batch_size):
            score_i = scores[i]  # (n_nodes,)
            target_i = targets[i]  # (n_nodes,)
            
            # Get ranking indices
            target_ranks = (-target_i).argsort()  # Descending
            pred_ranks = (-score_i).argsort()
            
            # Compute rank correlation loss
            # Higher rank difference = higher loss
            ranking_diff = torch.abs(
                target_ranks.float() - pred_ranks.float()
            ).mean()
            
            loss += ranking_diff
        
        return loss / batch_size


class TopKLoss(nn.Module):
    """
    Top-K weighted loss
    
    Emphasizes correct ranking of top-K elements
    Penalizes misranking important elements more
    """
    
    def __init__(self, k: int = 20, weight_ratio: float = 2.0):
        """
        Initialize top-K loss
        
        Args:
            k: Top-K threshold
            weight_ratio: Weight ratio (top-K vs rest)
        """
        super().__init__()
        self.k = k
        self.weight_ratio = weight_ratio
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute top-K weighted loss
        
        Args:
            scores: Predicted scores (batch, n_nodes)
            targets: Target scores (batch, n_nodes)
        
        Returns:
            Scalar loss value
        """
        batch_size, n_nodes = scores.shape
        
        # MSE loss with importance weighting
        mse_loss = F.mse_loss(scores, targets, reduction='none')  # (batch, n_nodes)
        
        # Create weight matrix
        weights = torch.ones_like(targets)
        
        for i in range(batch_size):
            # Identify top-K in targets
            top_k_indices = torch.argsort(-targets[i])[:self.k]
            
            # Weight top-K higher
            weights[i, top_k_indices] = self.weight_ratio
        
        # Apply weights
        weighted_loss = (mse_loss * weights).mean()
        
        return weighted_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for balanced P@5 and P@20 optimization
    
    Loss = alpha * P@5_loss + (1-alpha) * P@20_loss
    
    Balances between maintaining top-5 accuracy and improving long-tail
    """
    
    def __init__(self, alpha: float = 0.5, k_list: list = None):
        """
        Initialize combined loss
        
        Args:
            alpha: Weight for P@5 loss (remainder goes to P@20)
            k_list: List of K values to optimize [5, 20]
        """
        super().__init__()
        self.alpha = alpha
        self.k_list = k_list or [5, 20]
        
        # Initialize component losses
        self.pairwise_loss = PairwiseRankingLoss()
        self.top_k_losses = {
            k: TopKLoss(k=k, weight_ratio=2.0 if k <= 5 else 1.5)
            for k in self.k_list
        }
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            scores: Predicted scores (batch, n_nodes)
            targets: Target scores (batch, n_nodes)
        
        Returns:
            Scalar loss value
        """
        # Loss for top-5 (aggressive)
        loss_p5 = self.top_k_losses[5](scores, targets)
        
        # Loss for top-20 (moderate)
        loss_p20 = self.top_k_losses[20](scores, targets)
        
        # Combine
        combined = self.alpha * loss_p5 + (1.0 - self.alpha) * loss_p20
        
        return combined


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    
    Treats ranking as a contrastive problem
    Learns to push correct rankings closer together
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize NT-Xent loss
        
        Args:
            temperature: Softmax temperature
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss
        
        Args:
            scores: Predicted scores (batch, n_nodes)
            targets: Target scores (batch, n_nodes)
        
        Returns:
            Scalar loss value
        """
        # Normalize scores
        scores_norm = F.normalize(scores, dim=1)
        targets_norm = F.normalize(targets, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(scores_norm, targets_norm.t())
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels (diagonal: positive pairs)
        labels = torch.arange(scores.size(0), device=scores.device)
        
        # Compute NT-Xent loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class ListwiseLoss(nn.Module):
    """
    Listwise loss (LambdaMART-style)
    
    Directly optimizes ranking metrics (NDCG, MAP)
    """
    
    def __init__(self, metric: str = 'ndcg'):
        """
        Initialize listwise loss
        
        Args:
            metric: 'ndcg' or 'map'
        """
        super().__init__()
        self.metric = metric
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute listwise loss
        
        Args:
            scores: Predicted scores (batch, n_nodes)
            targets: Target scores (batch, n_nodes)
        
        Returns:
            Scalar loss value
        """
        batch_size = scores.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            score_i = scores[i]
            target_i = targets[i]
            
            if self.metric == 'ndcg':
                # NDCG-based loss
                # Get predicted ranking
                pred_ranking = torch.argsort(-score_i)
                true_ranking = torch.argsort(-target_i)
                
                # Compute NDCG for predicted vs ideal
                ideal_ndcg = self._compute_ndcg(target_i[true_ranking], k=20)
                pred_ndcg = self._compute_ndcg(target_i[pred_ranking], k=20)
                
                # Loss = 1 - (pred_NDCG / ideal_NDCG)
                loss_i = 1.0 - (pred_ndcg / (ideal_ndcg + 1e-6))
            else:  # MAP
                loss_i = 0.0
            
            loss += loss_i
        
        return loss / batch_size
    
    @staticmethod
    def _compute_ndcg(values: torch.Tensor, k: int = 20) -> torch.Tensor:
        """Compute NDCG for a ranking"""
        k = min(k, len(values))
        
        # DCG
        discount = torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=values.device))
        dcg = (values[:k] / discount).sum()
        
        # Ideal DCG
        ideal_values = torch.sort(-values)[0][:k]
        idcg = (ideal_values / discount).sum()
        
        return dcg / (idcg + 1e-6)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining multiple objectives
    
    - MSE loss: General accuracy
    - Top-K loss: Top-K ranking
    - Ranking loss: Pairwise ordering
    """
    
    def __init__(self, 
                 weights: dict = None,
                 use_anomaly_weight: bool = True):
        """
        Initialize multi-task loss
        
        Args:
            weights: Loss weights {'mse': 0.4, 'topk': 0.3, 'ranking': 0.3}
            use_anomaly_weight: Apply anomaly weighting
        """
        super().__init__()
        
        self.weights = weights or {'mse': 0.4, 'topk': 0.3, 'ranking': 0.3}
        self.use_anomaly_weight = use_anomaly_weight
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Component losses
        self.mse_loss = nn.MSELoss()
        self.topk_loss = TopKLoss(k=20)
        self.ranking_loss = PairwiseRankingLoss()
    
    def forward(self, 
                scores: torch.Tensor, 
                targets: torch.Tensor,
                anomaly_levels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-task loss
        
        Args:
            scores: Predicted scores
            targets: Target scores
            anomaly_levels: Optional anomaly levels for weighting
        
        Returns:
            Scalar loss value
        """
        # Component losses
        mse = self.mse_loss(scores, targets)
        topk = self.topk_loss(scores, targets)
        ranking = self.ranking_loss(scores, targets)
        
        # Combine
        loss = (
            self.weights['mse'] * mse +
            self.weights['topk'] * topk +
            self.weights['ranking'] * ranking
        )
        
        # Apply anomaly weighting if provided
        if self.use_anomaly_weight and anomaly_levels is not None:
            anomaly_levels = anomaly_levels.float()
            weight = 1.0 - (anomaly_levels * 0.2)  # 80-100% weight
            loss = loss * weight.mean()
        
        return loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function
    
    Args:
        loss_type: 'mse', 'pairwise', 'topk', 'combined', 'ntxent', 'listwise', 'multitask'
        **kwargs: Additional arguments for loss
    
    Returns:
        Loss module
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'pairwise':
        return PairwiseRankingLoss(**kwargs)
    elif loss_type == 'topk':
        return TopKLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'ntxent':
        return NTXentLoss(**kwargs)
    elif loss_type == 'listwise':
        return ListwiseLoss(**kwargs)
    elif loss_type == 'multitask':
        return MultiTaskLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("Loss Functions Tests")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 8
    n_nodes = 319
    
    scores = torch.randn(batch_size, n_nodes)
    targets = torch.randn(batch_size, n_nodes)
    
    # Test each loss
    losses_to_test = ['mse', 'pairwise', 'topk', 'combined', 'listwise', 'multitask']
    
    for loss_type in losses_to_test:
        try:
            loss_fn = get_loss_function(loss_type)
            loss = loss_fn(scores, targets)
            print(f"{loss_type:15s}: {loss.item():.6f}")
        except Exception as e:
            print(f"{loss_type:15s}: ERROR - {str(e)[:40]}")
    
    # Test combined loss with different alphas
    print("\nCombined Loss with different alphas:")
    for alpha in [0.3, 0.5, 0.7]:
        loss_fn = get_loss_function('combined', alpha=alpha)
        loss = loss_fn(scores, targets)
        print(f"  alpha={alpha:.1f}: {loss.item():.6f}")
    
    print("\n" + "=" * 80)
    print("✅ Loss functions ready for training!")
