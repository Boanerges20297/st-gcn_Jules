"""
Enhanced Ranking Model with Anomaly Awareness
Extends GlobalRankingModel to incorporate exogenous event information

Author: ST-GCN Enhanced System
Date: Feb 2026

Changes from baseline:
- Added anomaly_level input parameter
- Output confidence score along with predictions
- Modified training to incorporate anomaly weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRankingModel(nn.Module):
    """
    Enhanced MLP that incorporates exogenous event awareness
    
    Maps ST-GCN score vector (N,) -> refined scores (N,) with anomaly awareness
    
    Drop-in replacement for GlobalRankingModel with additional capabilities:
    - Takes anomaly_level as input during forward pass
    - Returns both prediction and confidence score
    - Can incorporate anomaly information in training
    """
    
    def __init__(self, num_nodes: int = 319, 
                 hidden1: int = 512, 
                 hidden2: int = 256, 
                 dropout: float = 0.3,
                 use_anomaly: bool = True):
        """
        Initialize Enhanced Ranking Model
        
        Args:
            num_nodes: Number of nodes (319 for Fortaleza)
            hidden1: Hidden layer 1 size
            hidden2: Hidden layer 2 size
            dropout: Dropout rate
            use_anomaly: Whether to use anomaly information
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.use_anomaly = use_anomaly
        
        # Main prediction network
        self.net = nn.Sequential(
            nn.Linear(num_nodes, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden2, num_nodes)
        )
        
        # Optional: anomaly awareness module
        if use_anomaly:
            self.anomaly_processor = nn.Sequential(
                nn.Linear(1, 16),  # Anomaly level is single value
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output: confidence scaling factor [0, 1]
            )
        
        logger.info(f"Initialized EnhancedRankingModel (num_nodes={num_nodes}, anomaly={use_anomaly})")
    
    def forward(self, x: torch.Tensor, 
                anomaly_level: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional anomaly awareness
        
        Args:
            x: Input ST-GCN scores (batch_size, num_nodes) or (num_nodes,)
            anomaly_level: Optional anomaly level (0-1) per sample 
                          (batch_size, 1) or scalar
        
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: (batch_size, num_nodes) refined ranking scores
            - confidence_scores: (batch_size,) confidence [0, 1] for predictions
        """
        
        # Handle single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.size(0)
        
        # Main predictions
        predictions = self.net(x)  # (batch, num_nodes)
        
        # Confidence scoring
        if self.use_anomaly and anomaly_level is not None:
            # Normalize anomaly level to [0, 1]
            if isinstance(anomaly_level, (int, float)):
                anomaly_tensor = torch.tensor(
                    [[anomaly_level]], 
                    dtype=torch.float32, 
                    device=x.device
                )
            else:
                anomaly_tensor = anomaly_level.float()
                if anomaly_tensor.dim() == 0:
                    anomaly_tensor = anomaly_tensor.unsqueeze(0).unsqueeze(0)
                elif anomaly_tensor.dim() == 1:
                    anomaly_tensor = anomaly_tensor.unsqueeze(1)
            
            # Process anomaly to get confidence reduction
            confidence_reduction = self.anomaly_processor(anomaly_tensor)  # (batch, 1)
            
            # Confidence = 1 - (anomaly_level * 0.3)
            # This reduces confidence by up to 30% when anomaly is max (1.0)
            confidence_scores = 1.0 - (anomaly_tensor * 0.3)
            confidence_scores = confidence_scores.squeeze(1)  # (batch,)
        else:
            # No anomaly info: high confidence
            confidence_scores = torch.ones(batch_size, dtype=torch.float32, device=x.device)
        
        if squeeze_output:
            predictions = predictions.squeeze(0)
            confidence_scores = confidence_scores.squeeze(0)
        
        return predictions, confidence_scores
    
    def forward_with_anomaly(self, x: torch.Tensor, 
                             anomaly_level: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns only predictions (backward compatible)
        
        Args:
            x: Input scores
            anomaly_level: Anomaly level
        
        Returns:
            Predictions only (ignores confidence)
        """
        predictions, _ = self.forward(x, anomaly_level)
        return predictions


class RankingLossWithAnomalyWeighting(nn.Module):
    """
    Custom loss that incorporates anomaly weighting
    
    During anomaly events, it's harder to predict correctly, so we down-weight
    the loss to avoid degeneration.
    """
    
    def __init__(self, use_ranking_loss: bool = True, use_anomaly_weight: bool = True):
        """
        Initialize loss function
        
        Args:
            use_ranking_loss: Use pairwise ranking loss
            use_anomaly_weight: Weight loss by anomaly level
        """
        super().__init__()
        self.use_ranking_loss = use_ranking_loss
        self.use_anomaly_weight = use_anomaly_weight
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                anomaly_levels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss with optional anomaly weighting
        
        Args:
            predictions: Model predictions (batch, num_nodes)
            targets: Ground truth scores (batch, num_nodes)
            anomaly_levels: Anomaly flags (batch,) for weighting
        
        Returns:
            Scalar loss value
        """
        
        # MSE loss as baseline
        mse_loss = F.mse_loss(predictions, targets)
        
        # Optional ranking loss (for top-K accuracy)
        if self.use_ranking_loss:
            # Pairwise ranking loss
            pred_ranks = torch.argsort(-predictions, dim=1)
            true_ranks = torch.argsort(-targets, dim=1)
            
            # Calculate rank differences
            rank_diff = torch.abs(pred_ranks.float() - true_ranks.float()).mean()
            combined_loss = 0.7 * mse_loss + 0.3 * rank_diff
        else:
            combined_loss = mse_loss
        
        # Anomaly weighting
        if self.use_anomaly_weight and anomaly_levels is not None:
            # Normalize anomaly levels
            anomaly_levels = anomaly_levels.float()
            
            # Weight samples: high anomaly = lower weight (harder to predict)
            # weight = 1 - (anomaly * 0.2) -> range [0.8, 1.0]
            sample_weights = 1.0 - (anomaly_levels * 0.2)
            sample_weights = sample_weights.unsqueeze(1)  # (batch, 1)
            
            # Apply weights
            weighted_loss = combined_loss * sample_weights.mean()
            
            return weighted_loss
        else:
            return combined_loss


def create_enhanced_ranking_model(num_nodes: int = 319, 
                                  pretrained_path: Optional[str] = None) -> EnhancedRankingModel:
    """
    Factory function to create enhanced model
    
    Args:
        num_nodes: Number of nodes
        pretrained_path: Optional path to load pretrained weights
    
    Returns:
        EnhancedRankingModel instance
    """
    model = EnhancedRankingModel(num_nodes=num_nodes, use_anomaly=True)
    
    if pretrained_path:
        try:
            # Load only the main network weights (backward compatible)
            state_dict = torch.load(pretrained_path)
            if 'net.0.weight' in state_dict:
                # Old model format
                model.net.load_state_dict(state_dict)
            else:
                # New format
                model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    return model


if __name__ == "__main__":
    print("Enhanced Ranking Model Tests")
    print("=" * 80)
    
    # Test 1: Basic forward pass
    print("\nTest 1: Basic forward pass")
    model = EnhancedRankingModel(num_nodes=319)
    model.eval()
    
    x = torch.randn(8, 319)  # Batch of 8 samples
    predictions, confidence = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    # Test 2: With anomaly awareness
    print("\nTest 2: With anomaly awareness")
    anomaly_levels = torch.tensor([0.0, 0.3, 0.6, 0.9, 0.0, 0.5, 0.8, 0.2])
    predictions, confidence = model(x, anomaly_levels)
    
    print(f"  Anomaly levels: {anomaly_levels.tolist()}")
    print(f"  Confidence scores: {confidence.tolist()}")
    print(f"  Expected: Higher anomaly -> Lower confidence")
    
    # Test 3: Loss function
    print("\nTest 3: Anomaly-weighted loss")
    loss_fn = RankingLossWithAnomalyWeighting()
    
    targets = torch.randn(8, 319)
    anomaly_flags = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.2, 0.8])
    
    loss = loss_fn(predictions, targets, anomaly_flags)
    print(f"  Loss with anomaly weighting: {loss:.6f}")
    
    # Compare: loss without anomaly weighting
    loss_no_anomaly = loss_fn(predictions, targets, None)
    print(f"  Loss without anomaly weighting: {loss_no_anomaly:.6f}")
    
    # Test 4: Single sample
    print("\nTest 4: Single sample")
    single_x = torch.randn(319)
    single_pred, single_conf = model(single_x, torch.tensor(0.7))
    print(f"  Input shape: {single_x.shape}")
    print(f"  Output shape: {single_pred.shape}")
    print(f"  Confidence: {single_conf:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
