"""
ModelFusion: Combina outputs de múltiplos modelos (ST-GCN + ST-GAT).

Implementa estratégias de ensemble:
1. Weighted Average: confiança baseada em histórico de precisão
2. Attention-based: aprende dinamicamente quais modelos confiar
3. Anomaly-adjusted: reduz confiança quando evento ativo
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelConfidenceTracker:
    """
    Rastreia precisão histórica de cada modelo para ponderar ensemble.
    
    Mantém:
    - Acurácia por modelo (rolling window)
    - Ranking agreement scores
    - Error patterns por contexto
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: número de predições para manter na janela
        """
        self.window_size = window_size
        self.prediction_history: List[Dict] = []
        
        # Scores acumulativos
        self.gcn_accuracy = 0.8  # baseline do ST-GCN
        self.gat_accuracy = 0.75  # baseline do ST-GAT (novo)
    
    def record_prediction(self,
                         gcn_output: np.ndarray,
                         gat_output: np.ndarray,
                         ground_truth: Optional[np.ndarray] = None,
                         timestamp: Optional[str] = None):
        """
        Registra uma predição para cálculo de confiança.
        
        Args:
            gcn_output: (N, 1) scores do ST-GCN
            gat_output: (N, 1) scores do ST-GAT
            ground_truth: (N, 1) valores reais (opcional)
            timestamp: timestamp da predição
        """
        record = {
            'timestamp': timestamp,
            'gcn_score': gcn_output.copy(),
            'gat_score': gat_output.copy(),
            'truth': ground_truth.copy() if ground_truth is not None else None
        }
        
        self.prediction_history.append(record)
        
        # Manter janela
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
        
        # Atualizar scores se temos ground truth
        if ground_truth is not None:
            self._update_model_scores(gcn_output, gat_output, ground_truth)
    
    def _update_model_scores(self,
                            gcn_output: np.ndarray,
                            gat_output: np.ndarray,
                            ground_truth: np.ndarray):
        """Atualiza precisão dos modelos baseado em erros."""
        gcn_error = np.abs(gcn_output - ground_truth).mean()
        gat_error = np.abs(gat_output - ground_truth).mean()
        
        # Smoothed update
        alpha = 0.1  # taxa de aprendizado
        self.gcn_accuracy = 0.95 * self.gcn_accuracy - 0.05 * gcn_error
        self.gat_accuracy = 0.95 * self.gat_accuracy - 0.05 * gat_error
        
        # Clip to [0.5, 1.0]
        self.gcn_accuracy = np.clip(self.gcn_accuracy, 0.5, 1.0)
        self.gat_accuracy = np.clip(self.gat_accuracy, 0.5, 1.0)
    
    def get_confidence_weights(self) -> Tuple[float, float]:
        """
        Retorna (gcn_weight, gat_weight) para ensemble.
        
        Returns:
            tuple normalizado que soma 1.0
        """
        total = self.gcn_accuracy + self.gat_accuracy
        return (self.gcn_accuracy / total, self.gat_accuracy / total)
    
    def get_state(self) -> Dict:
        """Exporta estado para logging/debugging."""
        return {
            'gcn_accuracy': float(self.gcn_accuracy),
            'gat_accuracy': float(self.gat_accuracy),
            'history_size': len(self.prediction_history)
        }


class AttentionFusionLayer(nn.Module):
    """
    Aprende a combinar dinamicamente outputs de múltiplos modelos.
    
    Para cada timestep/contexto, aprende pesos: w_gcn(context), w_gat(context)
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 16):
        """
        Args:
            input_dim: dimensão de cada score de modelo (1)
            hidden_dim: hidden dimension da rede
        """
        super(AttentionFusionLayer, self).__init__()
        
        self.input_dim = input_dim
        
        # Rede para aprender pesos de fusão
        # Input: [score_gcn, score_gat] (concatenado)
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # 2 pesos: um para GCN, um para GAT
        )
        
        # Softmax para normalizar pesos
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,
               gcn_scores: torch.Tensor,
               gat_scores: torch.Tensor,
               anomaly_flags: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combina scores via aprendizado de pesos.
        
        Args:
            gcn_scores: (B, N, 1)
            gat_scores: (B, N, 1)
            anomaly_flags: (B, N) boolean mask (True=anomalia ativa)
            
        Returns:
            (fused_scores, fusion_weights)
        """
        B, N, _ = gcn_scores.shape
        
        # Concatenar inputs
        combined = torch.cat([gcn_scores, gat_scores], dim=-1)  # (B, N, 2)
        
        # Aprender pesos
        weights = self.fusion_net(combined)  # (B, N, 2)
        weights = self.softmax(weights)  # (B, N, 2)
        
        # Aplicar pesos
        w_gcn = weights[..., 0:1]  # (B, N, 1)
        w_gat = weights[..., 1:2]  # (B, N, 1)
        
        fused = w_gcn * gcn_scores + w_gat * gat_scores  # (B, N, 1)
        
        # Reduzir confiança se anomalia ativa
        if anomaly_flags is not None:
            # anomaly_flags: (B, N) boolean
            # Reduzir weight do fusion para GAT (menos confiável em anomalias)
            anomaly_mask = anomaly_flags.float()[:, :, None]  # (B, N, 1)
            
            # Dar mais peso para GCN em anomalias (mais estável)
            adjusted_w_gcn = w_gcn + anomaly_mask * 0.1
            adjusted_w_gat = w_gat - anomaly_mask * 0.1
            
            # Renormalizar
            total = adjusted_w_gcn + adjusted_w_gat + 1e-10
            adjusted_w_gcn = adjusted_w_gcn / total
            adjusted_w_gat = adjusted_w_gat / total
            
            fused = adjusted_w_gcn * gcn_scores + adjusted_w_gat * gat_scores
        
        return fused, weights
    
    def get_attention_info(self) -> Dict:
        """Retorna informações sobre pesos de atenção para interpretabilidade."""
        return {
            'fusion_net': str(self.fusion_net),
            'parameterized': True
        }


class ModelFusion:
    """
    Orquestra ensemble de múltiplos modelos de crime prediction.
    
    Suporta:
    1. ST-GCN (modelo baseline, rápido)
    2. ST-GAT (novo, dinâmico com atenção)
    3. Fusion layer (aprende combinação ótima)
    4. Anomaly adjustment (reduz confiança em eventos)
    """
    
    def __init__(self,
                 model_gcn: nn.Module,
                 model_gat: nn.Module,
                 device: str = 'cpu',
                 fusion_strategy: str = 'weighted_average'):
        """
        Args:
            model_gcn: ST-GCN model
            model_gat: ST-GAT model
            device: 'cpu' ou 'cuda'
            fusion_strategy: 'weighted_average' ou 'attention'
        """
        self.model_gcn = model_gcn
        self.model_gat = model_gat
        self.device = device
        self.fusion_strategy = fusion_strategy
        
        # Tracker de confiança
        self.confidence_tracker = ModelConfidenceTracker()
        
        # Fusion layer (se usando attention strategy)
        if fusion_strategy == 'attention':
            self.fusion_layer = AttentionFusionLayer().to(device)
        else:
            self.fusion_layer = None
        
        # Move models to device
        self.model_gcn.to(device)
        self.model_gat.to(device)
        
        # Modo inference
        self.model_gcn.eval()
        self.model_gat.eval()
        
        logger.info(f"ModelFusion initialized: strategy={fusion_strategy}")
    
    def predict(self,
               x: torch.Tensor,
               adj_list: List[torch.Tensor],
               anomaly_flags: Optional[np.ndarray] = None,
               return_components: bool = False) -> Dict:
        """
        Faz predição usando ensemble de modelos.
        
        Args:
            x: (B, 26, 319, 30) features
            adj_list: list de matrizes de adjacência
            anomaly_flags: (B, 319) booleano, True=evento ativo naquele nó
            return_components: se True, retorna outputs de cada modelo separadamente
            
        Returns:
            {
                'fusion': (B, 319, 1) scores finais,
                'gcn': (B, 319, 1) scores ST-GCN,
                'gat': (B, 319, 1) scores ST-GAT,
                'confidence': (B, 319) scores de confiança [0-1],
                'weights': (B, 319, 2) pesos de fusion,
                'anomaly_flags': (B, 319) flags de anomalia
            }
        """
        with torch.no_grad():
            # Forward pass em cada modelo
            gcn_scores = self.model_gcn(x, adj_list)  # (B, 319, 1)
            gat_scores = self.model_gat(x, adj_list)  # (B, 319, 1)
        
        # Normalizar scores para [0, 1] range
        gcn_scores_norm = torch.sigmoid(gcn_scores)
        gat_scores_norm = torch.sigmoid(gat_scores)
        
        # Fusion
        if self.fusion_strategy == 'weighted_average':
            fused = self._weighted_average_fusion(
                gcn_scores_norm,
                gat_scores_norm,
                anomaly_flags
            )
            weights = None
        elif self.fusion_strategy == 'attention':
            fused, weights = self._attention_fusion(
                gcn_scores_norm,
                gat_scores_norm,
                anomaly_flags
            )
        else:
            # Default: just return GCN
            fused = gcn_scores_norm
            weights = None
        
        # Converter anomaly flags para tensor
        if anomaly_flags is not None:
            anomaly_tensor = torch.from_numpy(anomaly_flags).float().to(self.device)
        else:
            anomaly_tensor = torch.zeros(x.shape[0], 319, device=self.device)
        
        # Confidence: baseado em accuracy histórica + anomalia
        confidence = self._compute_confidence(
            fused,
            anomaly_tensor,
            gcn_scores_norm,
            gat_scores_norm
        )
        
        # Build output
        result = {
            'fusion': fused.detach().cpu().numpy(),
            'gcn': gcn_scores_norm.detach().cpu().numpy(),
            'gat': gat_scores_norm.detach().cpu().numpy(),
            'confidence': confidence.detach().cpu().numpy(),
            'anomaly_flags': anomaly_tensor.cpu().numpy(),
        }
        
        if weights is not None:
            result['weights'] = weights.detach().cpu().numpy()
        
        # Log de predição para tracking
        self.confidence_tracker.record_prediction(
            result['gcn'][:, :, 0],
            result['gat'][:, :, 0],
            timestamp=None
        )
        
        return result
    
    def _weighted_average_fusion(self,
                                gcn_scores: torch.Tensor,
                                gat_scores: torch.Tensor,
                                anomaly_flags: Optional[np.ndarray]) -> torch.Tensor:
        """
        Combina via weighted average baseado em histórico de precisão.
        """
        w_gcn, w_gat = self.confidence_tracker.get_confidence_weights()
        
        # Aplicar pesos
        fused = w_gcn * gcn_scores + w_gat * gat_scores
        
        # Reduzir confiança em anomalias
        if anomaly_flags is not None:
            anomaly_tensor = torch.from_numpy(anomaly_flags).float().to(self.device)
            # Reduzir fused score em 30% onde há anomalia
            reduction = (1 - anomaly_tensor.unsqueeze(-1)) * 0.7 + \
                       anomaly_tensor.unsqueeze(-1) * 0.5
            fused = fused * reduction
        
        return fused
    
    def _attention_fusion(self,
                         gcn_scores: torch.Tensor,
                         gat_scores: torch.Tensor,
                         anomaly_flags: Optional[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combina via attention layer (parametrizado).
        """
        if anomaly_flags is not None:
            anomaly_tensor = torch.from_numpy(anomaly_flags).bool().to(self.device)
        else:
            anomaly_tensor = None
        
        fused, weights = self.fusion_layer(gcn_scores, gat_scores, anomaly_tensor)
        
        return fused, weights
    
    def _compute_confidence(self,
                          fused: torch.Tensor,
                          anomaly_mask: torch.Tensor,
                          gcn_scores: torch.Tensor,
                          gat_scores: torch.Tensor) -> torch.Tensor:
        """
        Calcula confidence score para cada predição.
        
        Baseado em:
        1. Agreement entre GCN e GAT
        2. Magnitude de anomalia
        3. Histórico de precisão
        """
        # Agreement score: quão próximos estão os modelos?
        agreement = 1 - torch.abs(gcn_scores - gat_scores).squeeze(-1)
        agreement = torch.clamp(agreement, 0, 1)
        
        # Anomaly reduction
        anomaly_reduction = 1 - anomaly_mask * 0.2  # -20% em anomalias
        
        # Combine
        confidence = agreement * anomaly_reduction
        confidence = torch.clamp(confidence, 0, 1)
        
        return confidence
    
    def train_fusion_layer(self,
                          train_loader,
                          epochs: int = 5,
                          lr: float = 0.001):
        """
        Treina o fusion layer em dados históricos.
        
        Args:
            train_loader: DataLoader com (x, y_true)
            epochs: número de epochs
            lr: learning rate
        """
        if self.fusion_layer is None:
            logger.warning("Fusion layer not available for this strategy")
            return
        
        optimizer = torch.optim.Adam(self.fusion_layer.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model_gcn.eval()
        self.model_gat.eval()
        self.fusion_layer.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for x, y_true in train_loader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                
                # Forward passes
                with torch.no_grad():
                    gcn_out = self.model_gcn(x, None)
                    gat_out = self.model_gat(x, None)
                
                # Fusion
                fused, _ = self.fusion_layer(
                    torch.sigmoid(gcn_out),
                    torch.sigmoid(gat_out),
                    None
                )
                
                # Loss
                loss = criterion(fused, y_true)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Fusion Layer Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.6f}")
        
        logger.info("Fusion layer training completed")
    
    def get_ensemble_stats(self) -> Dict:
        """Retorna estatísticas do ensemble."""
        return {
            'fusion_strategy': self.fusion_strategy,
            'confidence_tracker': self.confidence_tracker.get_state(),
            'models': {
                'gcn': str(self.model_gcn.__class__.__name__),
                'gat': str(self.model_gat.__class__.__name__),
            }
        }
