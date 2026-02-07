"""
Comprehensive metrics for ranking evaluation
Author: ST-GCN Enhanced System
Date: Feb 2026

Implements:
- Precision@K (P@5, P@10, P@20)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Recall@K
- Mean Reciprocal Rank (MRR)
"""

import numpy as np
from typing import Tuple, Dict, List


class MetricReporter:
    """
    Calculates comprehensive ranking metrics
    
    P@K: How many of top-K predicted are actually in top-K real?
    NDCG@K: Quality of ranking (discounted by position)
    Recall@K: How many actual top-K did we find?
    """
    
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        Precision@K: What fraction of top-K predicted are actually top-K?
        
        Args:
            y_true: Actual scores (shape: N,)
            y_pred: Predicted scores (shape: N,)
            k: Top-K threshold
        
        Returns:
            P@K value in [0, 1]
        """
        # Get top-K indices for both arrays
        real_top_k_indices = set(np.argsort(-y_true)[:k])
        pred_top_k_indices = set(np.argsort(-y_pred)[:k])
        
        # Count overlap
        overlap = len(real_top_k_indices & pred_top_k_indices)
        
        return overlap / k
    
    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        Recall@K: What fraction of actual top-K did we identify?
        
        Args:
            y_true: Actual scores (shape: N,)
            y_pred: Predicted scores (shape: N,)
            k: Top-K threshold
        
        Returns:
            Recall@K value in [0, 1]
        """
        real_top_k_indices = set(np.argsort(-y_true)[:k])
        pred_top_k_indices = set(np.argsort(-y_pred)[:k])
        
        overlap = len(real_top_k_indices & pred_top_k_indices)
        
        return overlap / len(real_top_k_indices) if len(real_top_k_indices) > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Measures ranking quality considering position importance.
        DCG = sum(relevance_i / log2(i+1)) for i in 1..K
        NDCG = DCG / Ideal_DCG
        
        Args:
            y_true: Actual scores (shape: N,)
            y_pred: Predicted scores (shape: N,)
            k: Top-K threshold
        
        Returns:
            NDCG@K value in [0, 1]
        """
        # Get predicted ranking
        pred_indices = np.argsort(-y_pred)[:k]
        pred_relevances = y_true[pred_indices]
        
        # Calculate DCG
        discount = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)
        dcg = np.sum(pred_relevances / discount)
        
        # Calculate Ideal DCG (perfect ranking)
        ideal_indices = np.argsort(-y_true)[:k]
        ideal_relevances = y_true[ideal_indices]
        idcg = np.sum(ideal_relevances / discount)
        
        # Normalize
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    @staticmethod
    def mean_reciprocal_rank(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
        """
        MRR@K: Average of reciprocal ranks of first relevant item
        
        Used to measure if top-K items are well-ranked on average.
        
        Args:
            y_true: Actual scores (shape: N,)
            y_pred: Predicted scores (shape: N,)
            k: Top-K threshold
        
        Returns:
            MRR@K value in (0, 1]
        """
        pred_indices = np.argsort(-y_pred)[:k]
        true_top_k = set(np.argsort(-y_true)[:k])
        
        for rank, idx in enumerate(pred_indices, 1):
            if idx in true_top_k:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Generate comprehensive report with all metrics
        
        Args:
            y_true: Actual scores (shape: N,)
            y_pred: Predicted scores (shape: N,)
        
        Returns:
            Dictionary with P@K, NDCG@K, Recall@K, MRR
        """
        return {
            # Precision metrics
            'p_at_5': MetricReporter.precision_at_k(y_true, y_pred, 5),
            'p_at_10': MetricReporter.precision_at_k(y_true, y_pred, 10),
            'p_at_20': MetricReporter.precision_at_k(y_true, y_pred, 20),
            
            # NDCG metrics
            'ndcg_at_5': MetricReporter.ndcg_at_k(y_true, y_pred, 5),
            'ndcg_at_10': MetricReporter.ndcg_at_k(y_true, y_pred, 10),
            'ndcg_at_20': MetricReporter.ndcg_at_k(y_true, y_pred, 20),
            
            # Recall metrics
            'recall_at_5': MetricReporter.recall_at_k(y_true, y_pred, 5),
            'recall_at_10': MetricReporter.recall_at_k(y_true, y_pred, 10),
            'recall_at_20': MetricReporter.recall_at_k(y_true, y_pred, 20),
            
            # MRR
            'mrr_at_20': MetricReporter.mean_reciprocal_rank(y_true, y_pred, 20),
        }
    
    @staticmethod
    def report_detailed(y_true: np.ndarray, y_pred: np.ndarray, 
                       node_ids: List[int] = None) -> Dict:
        """
        Extended report with per-node analysis
        
        Args:
            y_true: Actual scores
            y_pred: Predicted scores
            node_ids: Optional node IDs for tracking
        
        Returns:
            Dictionary with aggregated + per-window metrics
        """
        metrics = MetricReporter.report(y_true, y_pred)
        
        # Add per-node error analysis
        pred_ranking = np.argsort(-y_pred)
        true_ranking = np.argsort(-y_true)
        
        rank_positions = np.argsort(pred_ranking)
        
        errors = {
            'mae_rank': np.mean(np.abs(true_ranking - rank_positions)),
            'spearman_corr': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0,
            'mean_score_diff': np.mean(np.abs(y_true - y_pred)),
            'std_score_diff': np.std(np.abs(y_true - y_pred))
        }
        
        metrics.update(errors)
        
        return metrics


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Convenience function: P@K"""
    return MetricReporter.precision_at_k(y_true, y_pred, k)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Convenience function: NDCG@K"""
    return MetricReporter.ndcg_at_k(y_true, y_pred, k)


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Convenience function: Recall@K"""
    return MetricReporter.recall_at_k(y_true, y_pred, k)


if __name__ == "__main__":
    # Example usage
    y_true = np.array([8, 7, 6, 5, 4, 3, 2, 1])
    y_pred = np.array([7.9, 6.8, 5.7, 4.6, 4.5, 3.4, 2.3, 1.2])
    
    metrics = MetricReporter.report(y_true, y_pred)
    
    print("Comprehensive Metrics Report")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:15s}: {metric_value:.4f}")
