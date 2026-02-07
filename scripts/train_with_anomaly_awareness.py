"""
Training Script with Anomaly Awareness
Trains EnhancedRankingModel on data with exogenous event consideration

Author: ST-GCN Enhanced System
Date: Feb 2026

Procedure:
1. Load training data
2. Load events and calculate daily anomaly levels
3. Create EnhancedRankingModel
4. Train with anomaly-weighted loss
5. Evaluate on test set
6. Save trained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Tuple, Dict, List
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ranking_model_enhanced import (
    EnhancedRankingModel, 
    RankingLossWithAnomalyWeighting,
    create_enhanced_ranking_model
)
from event_manager import EventManager
from metrics import MetricReporter


class AnomalyAwareTrainer:
    """
    Trainer for EnhancedRankingModel with event integration
    
    Handles:
    - Loading data and events
    - Computing anomaly levels by window/date
    - Training with anomaly-weighted loss
    - Evaluation and metric reporting
    """
    
    def __init__(self, num_nodes: int = 319, 
                 device: str = 'cpu',
                 events_manager: EventManager = None):
        """
        Initialize trainer
        
        Args:
            num_nodes: Number of nodes
            device: 'cpu' or 'cuda'
            events_manager: EventManager instance
        """
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self.events_manager = events_manager or EventManager()
        
        # Model and optimizer
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        logger.info(f"Initialized trainer (device={device})")
    
    def create_model(self, pretrained_path: str = None) -> EnhancedRankingModel:
        """
        Create enhanced ranking model
        
        Args:
            pretrained_path: Optional path to pretrained weights
        
        Returns:
            Initialized model
        """
        self.model = create_enhanced_ranking_model(
            num_nodes=self.num_nodes,
            pretrained_path=pretrained_path
        )
        self.model.to(self.device)
        
        # Loss function with anomaly weighting
        self.loss_fn = RankingLossWithAnomalyWeighting(
            use_ranking_loss=True,
            use_anomaly_weight=True
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
        logger.info(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def prepare_data(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray,
                     X_val: np.ndarray, 
                     y_val: np.ndarray,
                     dates_train: List[date] = None,
                     dates_val: List[date] = None,
                     batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders with anomaly levels
        
        Args:
            X_train: Training features (n_samples, n_nodes)
            y_train: Training targets (n_samples, n_nodes)
            X_val: Validation features
            y_val: Validation targets
            dates_train: Training dates for anomaly calculation
            dates_val: Validation dates
            batch_size: Batch size
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        
        # Calculate anomaly levels if dates provided
        if dates_train is not None:
            train_anomaly_levels = np.array([
                self.events_manager.get_anomaly_level_for_date(d) 
                for d in dates_train
            ])
            logger.info(f"Train anomaly levels: min={train_anomaly_levels.min():.3f}, "
                       f"max={train_anomaly_levels.max():.3f}, "
                       f"mean={train_anomaly_levels.mean():.3f}")
        else:
            train_anomaly_levels = np.zeros(len(X_train))
        
        if dates_val is not None:
            val_anomaly_levels = np.array([
                self.events_manager.get_anomaly_level_for_date(d) 
                for d in dates_val
            ])
        else:
            val_anomaly_levels = np.zeros(len(X_val))
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        anomaly_train_t = torch.from_numpy(train_anomaly_levels).float().to(self.device)
        
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)
        anomaly_val_t = torch.from_numpy(val_anomaly_levels).float().to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t, anomaly_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t, anomaly_val_t)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average epoch loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (X, y, anomaly) in enumerate(train_loader):
            # Forward pass
            predictions, confidence = self.model(X, anomaly)
            
            # Compute loss with anomaly weighting
            loss = self.loss_fn(predictions, y, anomaly)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_confidence = []
        
        with torch.no_grad():
            for X, y, anomaly in val_loader:
                # Forward pass
                predictions, confidence = self.model(X, anomaly)
                
                # Loss
                loss = self.loss_fn(predictions, y, anomaly)
                total_loss += loss.item()
                
                # Collect for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_confidence.append(confidence.cpu().numpy())
        
        # Aggregate predictions
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_targets)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate per-sample metrics and average
        for k in [5, 10, 20]:
            p_at_k_list = []
            for i in range(len(y_pred)):
                pk = MetricReporter.precision_at_k(y_true[i], y_pred[i], k)
                p_at_k_list.append(pk)
            metrics[f'p_at_{k}'] = np.mean(p_at_k_list)
            
            ndcg_k_list = []
            for i in range(len(y_pred)):
                ndcg = MetricReporter.ndcg_at_k(y_true[i], y_pred[i], k)
                ndcg_k_list.append(ndcg)
            metrics[f'ndcg_at_{k}'] = np.mean(ndcg_k_list)
        
        metrics['avg_loss'] = total_loss / len(val_loader)
        metrics['avg_confidence'] = np.mean(all_confidence)
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10,
              early_stopping_patience: int = 3) -> Dict:
        """
        Complete training loop
        
        Args:
            train_loader: Training loader
            val_loader: Validation loader
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['avg_loss']
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Logging
            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info(f"  Train loss: {train_loss:.6f}")
            logger.info(f"  Val loss:   {val_loss:.6f}")
            logger.info(f"  P@5:  {val_metrics['p_at_5']:.4f}")
            logger.info(f"  P@10: {val_metrics['p_at_10']:.4f}")
            logger.info(f"  P@20: {val_metrics['p_at_20']:.4f}")
            logger.info(f"  NDCG@20: {val_metrics['ndcg_at_20']:.4f}")
            logger.info(f"  Avg confidence: {val_metrics['avg_confidence']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info("  ✅ New best model!")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"  Early stopping (patience={early_stopping_patience})")
                    break
        
        logger.info(f"\n✅ Training complete!")
        return self.history
    
    def save_model(self, save_path: str = 'models/ranking_model_with_anomaly.pth'):
        """Save trained model"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


def main():
    """Main training script"""
    
    logger.info("Starting Training with Anomaly Awareness (Week 2, Task 2.4)")
    
    # Initialize
    trainer = AnomalyAwareTrainer(device='cpu')  # Use 'cuda' if available
    
    # Create model
    trainer.create_model()
    
    # Dummy data (replace with real data loading)
    logger.info("Generating dummy training data...")
    n_train = 200
    n_val = 50
    
    X_train = np.random.randn(n_train, 319).astype(np.float32)
    y_train = np.random.exponential(2.0, (n_train, 319)).astype(np.float32)
    
    X_val = np.random.randn(n_val, 319).astype(np.float32)
    y_val = np.random.exponential(2.0, (n_val, 319)).astype(np.float32)
    
    # Dummy dates (last n_train days)
    end_date = date.today()
    dates_train = [end_date - timedelta(days=i) for i in range(n_train, 0, -1)]
    dates_val = [end_date - timedelta(days=i) for i in range(n_train + n_val, n_train, -1)]
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        X_train, y_train,
        X_val, y_val,
        dates_train=dates_train,
        dates_val=dates_val,
        batch_size=32
    )
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=5,  # Short training for testing
        early_stopping_patience=2
    )
    
    # Save model
    trainer.save_model('models/ranking_model_with_anomaly.pth')
    
    logger.info("✅ Training completed!")
    
    # Print final metrics
    final_metrics = history['val_metrics'][-1]
    print("\nFinal Validation Metrics")
    print("=" * 50)
    for metric_name, value in sorted(final_metrics.items()):
        print(f"{metric_name:20s}: {value:.4f}")


if __name__ == "__main__":
    main()
