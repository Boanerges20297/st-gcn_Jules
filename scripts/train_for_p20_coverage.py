"""
Training for P@20 Coverage
Trains model with combined loss to optimize both P@5 and P@20

Author: ST-GCN Enhanced System
Date: Feb 2026

Strategy:
- Use CombinedLoss (alpha=0.5) to balance P@5 and P@20
- Train for ~10 epochs
- Evaluate on extended metrics
- Track P@K and NDCG@K across all K
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ranking_model_enhanced import create_enhanced_ranking_model
from loss_functions import get_loss_function
from metrics import MetricReporter


class P20OptimizationTrainer:
    """
    Trainer for optimizing P@20 coverage while maintaining P@5
    
    Uses CombinedLoss to balance multiple objectives
    """
    
    def __init__(self, num_nodes: int = 319, 
                 device: str = 'cpu',
                 config: Dict = None):
        """
        Initialize trainer
        
        Args:
            num_nodes: Number of nodes
            device: 'cpu' or 'cuda'
            config: Training configuration
        """
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        
        # Default config
        self.config = config or {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'epochs': 15,
            'early_stopping_patience': 5,
            'loss_alpha': 0.5,  # Balance P@5 vs P@20
            'warmup_epochs': 2,
        }
        
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'best_epoch': 0,
            'best_loss': float('inf')
        }
        
        logger.info(f"Initialized P@20 Optimizer Trainer")
        logger.info(f"Config: {json.dumps(self.config, indent=2)}")
    
    def create_model(self, pretrained_path: str = None) -> nn.Module:
        """
        Create model for training
        
        Args:
            pretrained_path: Path to pretrained model
        
        Returns:
            Model instance
        """
        self.model = create_enhanced_ranking_model(
            num_nodes=self.num_nodes,
            pretrained_path=pretrained_path
        )
        self.model.to(self.device)
        
        # Loss function: CombinedLoss with configurable alpha
        self.loss_fn = get_loss_function(
            'combined',
            alpha=self.config['loss_alpha'],
            k_list=[5, 20]
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=1e-6
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def prepare_data(self, 
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     batch_size: int = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size (uses config if None)
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        batch_size = batch_size or self.config['batch_size']
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data loaders created: train={len(train_loader)} batches, "
                   f"val={len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training loader
            epoch: Current epoch
        
        Returns:
            Average epoch loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            # Forward pass (no anomaly info during P@20 training)
            predictions, _ = self.model(X)
            
            # Compute loss (combined P@5 + P@20)
            loss = self.loss_fn(predictions, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                current_loss = loss.item()
                logger.info(f"Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)}: "
                           f"loss={current_loss:.6f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """
        Evaluate on validation set
        
        Args:
            val_loader: Validation loader
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in val_loader:
                # Forward pass
                predictions, _ = self.model(X)
                
                # Loss
                loss = self.loss_fn(predictions, y)
                total_loss += loss.item()
                
                # Collect for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Aggregate predictions
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_targets)
        
        # Calculate all metrics
        metrics = {}
        
        # Per-sample metrics
        for k in [5, 10, 15, 20]:
            p_at_k_list = []
            ndcg_k_list = []
            recall_k_list = []
            
            for i in range(len(y_pred)):
                pk = MetricReporter.precision_at_k(y_true[i], y_pred[i], k)
                ndcg = MetricReporter.ndcg_at_k(y_true[i], y_pred[i], k)
                recall = MetricReporter.recall_at_k(y_true[i], y_pred[i], k)
                
                p_at_k_list.append(pk)
                ndcg_k_list.append(ndcg)
                recall_k_list.append(recall)
            
            metrics[f'p_at_{k}'] = float(np.mean(p_at_k_list))
            metrics[f'ndcg_at_{k}'] = float(np.mean(ndcg_k_list))
            metrics[f'recall_at_{k}'] = float(np.mean(recall_k_list))
        
        metrics['avg_loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Complete training loop
        
        Args:
            train_loader: Training loader
            val_loader: Validation loader
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config['epochs']} epochs")
        logger.info(f"Loss: Combined (alpha={self.config['loss_alpha']})")
        
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['avg_loss']
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Learning rate decay
            self.scheduler.step()
            
            # Logging
            logger.info(f"\nEpoch {epoch}/{self.config['epochs']}")
            logger.info(f"  Train loss: {train_loss:.6f}")
            logger.info(f"  Val loss:   {val_loss:.6f}")
            logger.info(f"  P@5:  {val_metrics['p_at_5']:.4f}")
            logger.info(f"  P@20: {val_metrics['p_at_20']:.4f}")
            logger.info(f"  NDCG@20: {val_metrics['ndcg_at_20']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.history['best_epoch'] = epoch
                self.history['best_loss'] = val_loss
                logger.info("  ✅ New best model!")
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"  Early stopping (patience={self.config['early_stopping_patience']})")
                    break
        
        logger.info(f"\n✅ Training complete at epoch {self.history['best_epoch']}")
        return self.history
    
    def save_model(self, save_path: str = 'models/model_with_p20_optimization.pth'):
        """Save trained model"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
        print(f"✅ Saved to {save_path}")
    
    def save_history(self, save_path: str = 'p20_training_history.json'):
        """Save training history"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Make history JSON serializable
        history_json = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'best_epoch': self.history['best_epoch'],
            'best_loss': float(self.history['best_loss']),
            'final_metrics': self.history['val_metrics'][-1] if self.history['val_metrics'] else {}
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        logger.info(f"History saved to {save_path}")


def main():
    """Main training script"""
    
    logger.info("Starting P@20 Optimization Training (Week 3, Task 3.3)")
    
    # Initialize trainer
    trainer = P20OptimizationTrainer(device='cpu')
    
    # Create model
    trainer.create_model()
    
    # Generate dummy data (replace with real data)
    logger.info("Generating training data...")
    n_train = 300
    n_val = 100
    
    X_train = np.random.randn(n_train, 319).astype(np.float32)
    y_train = np.random.exponential(2.0, (n_train, 319)).astype(np.float32)
    
    X_val = np.random.randn(n_val, 319).astype(np.float32)
    y_val = np.random.exponential(2.0, (n_val, 319)).astype(np.float32)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(X_train, y_train, X_val, y_val)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Save model and history
    trainer.save_model('models/model_with_p20_optimization.pth')
    trainer.save_history('p20_training_history.json')
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    best_idx = trainer.history['best_epoch'] - 1
    if best_idx < len(history['val_metrics']):
        best_metrics = history['val_metrics'][best_idx]
        print(f"\nBest Epoch: {trainer.history['best_epoch']}")
        print(f"  Loss: {trainer.history['best_loss']:.6f}")
        print(f"  P@5:  {best_metrics['p_at_5']:.4f}")
        print(f"  P@20: {best_metrics['p_at_20']:.4f}")
        print(f"  NDCG@20: {best_metrics['ndcg_at_20']:.4f}")
    
    print(f"\n✅ Training completed!")


if __name__ == "__main__":
    main()
