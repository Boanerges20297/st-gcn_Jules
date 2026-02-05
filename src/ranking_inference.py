#!/usr/bin/env python
"""
ranking_inference.py
Carregar e executar modelo de ranking para validar/corrigir ST-GCN em tempo de execução
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional

class RankingInference:
    """Carregar e executar modelo de ranking para validar predições de ST-GCN"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize ranking model for inference
        
        Args:
            model_path: Path to ranking_model_window30_final.pkl
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.config = None
        self.input_dim = None
        
        if not os.path.exists(model_path):
            print(f"[WARNING] Ranking model not found: {model_path}")
            return
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # Import model class
            from src.ranking_model_v2 import RankingModel
            
            # Load config
            self.config = data.get('config', {})
            self.input_dim = self.config.get('input_dim', 780)
            
            # Load scaler params
            self.scaler_mean = data.get('scaler_mean', None)
            self.scaler_scale = data.get('scaler_scale', None)
            
            # Recreate model
            self.model = RankingModel(
                input_dim=self.input_dim,
                hidden_dim=self.config.get('hidden_dim', 512),
                dropout_main=self.config.get('dropout', 0.2),
                dropout_small=0.1
            )
            
            # Load weights
            self.model.load_state_dict(data.get('model_state', {}))
            self.model.to(device)
            self.model.eval()
            
            metrics = data.get('metrics', {})
            print(f"[OK] Ranking model loaded")
            print(f"      Config: input={self.input_dim}, hidden={self.config.get('hidden_dim')}")
            print(f"      Performance: P@5={metrics.get('p5', 'N/A')}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load ranking model: {e}")
            self.model = None
    
    def validate_stgcn_predictions(self, 
                                   stgcn_scores: np.ndarray,
                                   node_features: np.ndarray,
                                   top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and re-rank ST-GCN predictions using ranking model
        
        Args:
            stgcn_scores: (N,) ST-GCN predictions (raw scores)
            node_features: (N, feature_dim) Node features for ranking model
            top_k: Number of top nodes to return
        
        Returns:
            validated_scores: (N,) Re-ranked scores
            reordered_indices: Top-k node indices after re-ranking
        """
        
        if self.model is None:
            print("[WARNING] Ranking model not loaded, using ST-GCN scores as-is")
            top_indices = np.argsort(-stgcn_scores)[:top_k]
            return stgcn_scores, top_indices
        
        try:
            # Validate input dimensions
            if node_features.shape[0] != len(stgcn_scores):
                print(f"[WARNING] Shape mismatch: features {node_features.shape[0]} vs scores {len(stgcn_scores)}")
                return stgcn_scores, np.argsort(-stgcn_scores)[:top_k]
            
            # Normalize features using scaler
            if self.scaler_mean is not None and self.scaler_scale is not None:
                X_scaled = (node_features - self.scaler_mean) / self.scaler_scale
            else:
                X_scaled = node_features
            
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Inference
            with torch.no_grad():
                ranking_scores = self.model(X_tensor).cpu().numpy()
            
            # Combine: ST-GCN + Ranking (weighted average)
            # Normalize both to [0, 1]
            stgcn_norm = (stgcn_scores - stgcn_scores.min()) / (stgcn_scores.max() - stgcn_scores.min() + 1e-6)
            ranking_norm = (ranking_scores.flatten() - ranking_scores.min()) / (ranking_scores.max() - ranking_scores.min() + 1e-6)
            
            # Combined score: 70% ST-GCN (primary model) + 30% Ranking (validator)
            combined_scores = 0.7 * stgcn_norm + 0.3 * ranking_norm
            
            # Get top-k
            top_indices = np.argsort(-combined_scores)[:top_k]
            
            return combined_scores, top_indices
        
        except Exception as e:
            print(f"[ERROR] Ranking validation failed: {e}")
            return stgcn_scores, np.argsort(-stgcn_scores)[:top_k]
    
    def get_validation_report(self,
                             stgcn_scores: np.ndarray,
                             combined_scores: np.ndarray,
                             top_k: int = 5) -> dict:
        """
        Generate validation report comparing ST-GCN vs Ranking-validated
        """
        
        stgcn_top = np.argsort(-stgcn_scores)[:top_k]
        combined_top = np.argsort(-combined_scores)[:top_k]
        
        overlap = len(set(stgcn_top) & set(combined_top))
        
        return {
            'overlap': overlap,
            'concordance': overlap / top_k,
            'stgcn_top5': stgcn_top.tolist(),
            'validated_top5': combined_top.tolist(),
        }


def load_ranking_model(model_path: str) -> Optional[RankingInference]:
    """Helper to load ranking model"""
    if not os.path.exists(model_path):
        return None
    
    try:
        return RankingInference(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load ranking: {e}")
        return None


if __name__ == '__main__':
    # Test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    model_path = 'models/ranking_model_window30_final.pkl'
    print(f"Testing ranking inference from: {model_path}")
    
    ranker = RankingInference(model_path)
    
    if ranker.model is not None:
        # Synthetic test
        N = 319
        stgcn_scores = np.random.rand(N) * 2
        node_features = np.random.randn(N, 780)
        
        combined, top_indices = ranker.validate_stgcn_predictions(stgcn_scores, node_features, top_k=5)
        report = ranker.get_validation_report(stgcn_scores, combined)
        
        print(f"\nValidation Report:")
        print(f"  Overlap (ST-GCN vs Validated): {report['concordance']:.1%}")
        print(f"  ST-GCN Top-5: {report['stgcn_top5']}")
        print(f"  Validated Top-5: {report['validated_top5']}")
