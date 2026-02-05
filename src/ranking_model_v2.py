#!/usr/bin/env python
"""
ranking_model_v2.py - Modelo de Ranking com Pairwise Loss
Otimiza diretamente para ordenacao de pares (hotspot A > hotspot B)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class PairwiseLoss(nn.Module):
    """
    Pairwise Ranking Loss: Minimiza o numero de inversoes
    (i.e., quando um hotspot de menor risco eh rankado antes de outro de maior risco)
    """
    def __init__(self):
        super(PairwiseLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        pred: (batch_size, num_nodes) - scores preditos
        target: (batch_size, num_nodes) - valores reais (targets)
        
        Loss = sum(log(1 + exp((s_j - s_i) * (y_i - y_j))))
        onde y_i > y_j (ranking correto)
        """
        batch_size, num_nodes = pred.shape
        
        # Versao vetorizada simplificada
        # Para cada batch, comparar top-ranked com bottom-ranked
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        for b in range(batch_size):
            targets = target[b]
            scores = pred[b]
            
            # Encontrar indices ordenados
            target_sorted_idx = torch.argsort(targets, descending=True)
            pred_sorted_idx = torch.argsort(scores, descending=True)
            
            # Inversao de ranking: quando pred nao concorda com target
            for rank_pos in range(min(10, num_nodes - 1)):  # Top-10
                correct_node = target_sorted_idx[rank_pos]
                predicted_node = pred_sorted_idx[rank_pos]
                
                if correct_node != predicted_node:
                    # Penalizar: s_wrong deveria ser menor que s_correct
                    s_correct = scores[correct_node]
                    s_wrong = scores[predicted_node]
                    loss = torch.log(1 + torch.exp(s_wrong - s_correct))
                    total_loss = total_loss + loss
        
        return total_loss / (batch_size * min(10, num_nodes - 1))

class RankingModel(nn.Module):
    """Modelo neural para ranking de hotspots"""
    def __init__(self, input_dim=26, hidden_dim=128, dropout_main=0.3, dropout_small=0.2):
        super(RankingModel, self).__init__()

        h1 = hidden_dim
        h2 = max(8, hidden_dim // 2)
        h3 = max(4, hidden_dim // 4)

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(dropout_main),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(dropout_main),

            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout_small),

            nn.Linear(h3, 1),
            nn.Sigmoid()  # Output entre 0 e 1
        )
    
    def forward(self, x):
        """x: (batch_size, input_dim) -> (batch_size, 1)"""
        return self.net(x).squeeze(1)

class RankingTrainerV2:
    """Trainer para modelo de ranking com Pairwise Loss"""

    def __init__(self, model, device='cpu', lr=0.01, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = PairwiseLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        self.scaler = StandardScaler()
    
    def prepare_batches(self, X, Y, batch_size=8, num_epochs=10):
        """
        Cria multiplas amostras/epocas dos dados para treinamento
        """
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        Y_tensor = torch.FloatTensor(Y)
        
        # Expandir dados: criar multiplas permutacoes aleatorias
        batches = []
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = np.random.permutation(len(X))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                X_batch = X_tensor[batch_indices]
                Y_batch = Y_tensor[batch_indices]
                
                batches.append((X_batch, Y_batch))
        
        return batches
    
    def precision_at_k(self, pred, target, k=5):
        """Calcula P@k: acurácia do ranking top-k"""
        batch_size = len(pred)
        
        if batch_size < k:
            k = batch_size
        
        # Top-k predito vs top-k real
        _, pred_topk = torch.topk(pred, k)
        _, target_topk = torch.topk(target, k)
        
        hits = len(set(pred_topk.tolist()) & set(target_topk.tolist()))
        return hits / k
    
    def train_epoch(self, train_batches):
        """Treina uma epoca"""
        self.model.train()
        total_loss = 0.0
        total_p5 = 0.0
        num_batches = 0
        
        for X_batch, Y_batch in train_batches:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            
            # Forward
            pred = self.model(X_batch)
            loss = self.criterion(pred.unsqueeze(0), Y_batch.unsqueeze(0))
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_p5 += self.precision_at_k(pred.detach(), Y_batch.detach(), k=5)
            num_batches += 1
        
        return total_loss / num_batches, total_p5 / num_batches
    
    def validate(self, X, Y):
        """Valida modelo"""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_tensor)
            loss = self.criterion(pred.unsqueeze(0), Y_tensor.unsqueeze(0))
            p5 = self.precision_at_k(pred, Y_tensor, k=5)
        
        return loss.item(), p5
    
    def predict(self, X):
        """Prediz rankings para dados novos"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            scores = self.model(X_tensor).cpu().numpy()
        
        # Retornar ranking (indices ordenados por score)
        ranking = np.argsort(-scores)
        
        return ranking, scores

if __name__ == "__main__":
    print("Teste do modelo de ranking v2...")
    
    # Teste basico
    model = RankingModel(input_dim=26)
    X_test = torch.randn(10, 26)
    output = model(X_test)
    
    print(f"OK: output shape = {output.shape}")
    print(f"OK: PairwiseLoss disponivel")
    print(f"OK: Trainer v2 pronto para usar")
