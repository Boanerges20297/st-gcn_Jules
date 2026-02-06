#!/usr/bin/env python
"""
ranking_correction_system.py

Sistema que carrega os modelos de ranking treinados e fornece:
1. Predições de ranking para cada dia
2. Score de confiança (quanto confiar no ranking vs ST-GCN)
3. Correção automática quando ST-GCN desvia
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

class RankingCorrectionSystem:
    """Sistema de correção por ranking"""
    
    def __init__(self):
        self.models_by_day = {}
        self.scalers_by_day = {}
        self.device = 'cpu'
        self._load_models()
    
    def _load_models(self):
        """Carrega modelos treinados"""
        model_dir = Path(ROOT) / 'models' / 'ranking_by_day'
        
        if not model_dir.exists():
            print("[WARNING] Modelo de ranking não encontrado. Sistema desativado.")
            return
        
        # Carregar scalers
        scalers_path = model_dir / 'scalers.pkl'
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers_by_day = pickle.load(f)
        
        # Carregar modelos
        for day in range(7):
            model_path = model_dir / f'ranking_model_day{day}.pth'
            if model_path.exists():
                try:
                    # Carregar arquivo (novo formato com config/model_state)
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Se for novo formato (dict com 'model_state'), extrair
                    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                        state_dict = checkpoint['model_state']
                    else:
                        # Formato antigo (state_dict direto)
                        state_dict = checkpoint
                    
                    model = self._build_model()
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.models_by_day[day] = model
                except Exception as e:
                    print(f"[WARNING] Falha ao carregar modelo dia {day}: {e}")
        
        print(f"[RANKING] Sistema de correção carregado com {len(self.models_by_day)} modelos")
    
    def _build_model(self):
        """Constrói arquitetura do modelo"""
        class RankingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(12, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )
            
            def forward(self, x):
                return self.fc(x).squeeze()
        
        return RankingModel()
    
    def extract_features(self, cvli_timeseries):
        """
        Extrai features de uma série temporal CVLI
        cvli_timeseries: array (num_nodes,) com últimos N dias de CVLI
        """
        num_nodes = len(cvli_timeseries) if isinstance(cvli_timeseries, np.ndarray) else 1
        
        if num_nodes == 1:
            # Caso de um único nó
            ts = np.array(cvli_timeseries).reshape(-1)
        else:
            ts = cvli_timeseries
        
        features = np.zeros((num_nodes if num_nodes > 1 else 1, 12))
        
        if num_nodes == 1:
            ts = np.array(cvli_timeseries).reshape(-1)
            features[0, 0] = ts.mean()
            features[0, 1] = np.sqrt(np.var(ts))
            features[0, 2] = ts.max()
            features[0, 3] = ts.min()
            features[0, 4] = (ts > 0).sum() / len(ts)
            features[0, 5] = ts.sum() / len(ts)
            
            if len(ts) > 5:
                recent = ts[-5:].mean()
                old = ts[:5].mean()
                features[0, 6] = recent - old
            
            if len(ts) > 1:
                features[0, 7] = np.mean(np.abs(np.diff(ts)))
            
            features[0, 8] = np.percentile(ts, 75) - np.percentile(ts, 25)
            features[0, 9] = ts.sum()
            
            if len(ts) > 3 and ts.sum() > 0:
                top3 = np.sum(np.sort(ts)[-3:])
                features[0, 10] = top3 / ts.sum()
            
            if ts.mean() > 0:
                features[0, 11] = ts.max() / ts.mean()
        else:
            for i in range(num_nodes):
                ts = cvli_timeseries[i, :]
                features[i, 0] = ts.mean()
                features[i, 1] = np.sqrt(np.var(ts))
                features[i, 2] = ts.max()
                features[i, 3] = ts.min()
                features[i, 4] = (ts > 0).sum() / len(ts)
                features[i, 5] = ts.sum() / len(ts)
                
                if len(ts) > 5:
                    recent = ts[-5:].mean()
                    old = ts[:5].mean()
                    features[i, 6] = recent - old
                
                if len(ts) > 1:
                    features[i, 7] = np.mean(np.abs(np.diff(ts)))
                
                features[i, 8] = np.percentile(ts, 75) - np.percentile(ts, 25)
                features[i, 9] = ts.sum()
                
                if len(ts) > 3 and ts.sum() > 0:
                    top3 = np.sum(np.sort(ts)[-3:])
                    features[i, 10] = top3 / ts.sum()
                
                if ts.mean() > 0:
                    features[i, 11] = ts.max() / ts.mean()
        
        features = np.nan_to_num(features, 0.0)
        return features
    
    def get_ranking_scores(self, cvli_timeseries, day_of_week=None):
        """
        Retorna scores de ranking para cada nó
        
        Args:
            cvli_timeseries: (num_nodes, num_days) ou (num_nodes,) com histórico CVLI
            day_of_week: dia da semana (0-6). Se None, usa data atual
        
        Returns:
            scores: array de scores (num_nodes,)
            confidence: score de confiança (0-1)
        """
        
        if len(self.models_by_day) == 0:
            # Fallback: retorna média simples
            if len(cvli_timeseries.shape) == 2:
                scores = cvli_timeseries.mean(axis=1)
            else:
                scores = cvli_timeseries
            return scores, 0.0
        
        # Determinar dia da semana
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        
        day_of_week = day_of_week % 7
        
        if day_of_week not in self.models_by_day:
            # Fallback
            if len(cvli_timeseries.shape) == 2:
                scores = cvli_timeseries.mean(axis=1)
            else:
                scores = cvli_timeseries
            return scores, 0.0
        
        # Extrair features
        if len(cvli_timeseries.shape) == 1:
            cvli_timeseries = cvli_timeseries.reshape(-1, 1)
        
        features = self.extract_features(cvli_timeseries)
        
        # Normalizar
        scaler = self.scalers_by_day.get(day_of_week)
        if scaler is not None:
            features_norm = scaler.transform(features)
        else:
            features_norm = features
        
        # Predição
        model = self.models_by_day[day_of_week]
        X_t = torch.FloatTensor(features_norm).to(self.device)
        
        with torch.no_grad():
            scores = model(X_t).cpu().numpy()
        
        # Confidence: média dos scores (quanto mais disperso, mais confiável)
        if len(scores) > 5:
            top_scores = np.sort(scores)[-5:]
            gap = top_scores[-1] - top_scores[0]
            confidence = min(1.0, gap / (np.mean(scores) + 1e-6))
        else:
            confidence = 0.7
        
        return scores, float(confidence)
    
    def correct_stgcn_prediction(self, stgcn_top5, cvli_timeseries, 
                                 day_of_week=None, confidence_threshold=0.6):
        """
        Corrige predição do ST-GCN usando ranking
        
        Args:
            stgcn_top5: top-5 nós preditos pelo ST-GCN
            cvli_timeseries: série temporal de CVLI
            day_of_week: dia da semana
            confidence_threshold: só corrige se confidence > threshold
        
        Returns:
            corrected_top5: top-5 corrigido
            confidence: confiança do ranking
            was_corrected: boolean indicando se foi corrigido
        """
        
        scores, confidence = self.get_ranking_scores(cvli_timeseries, day_of_week)
        
        if confidence < confidence_threshold:
            # Não é confiável, mantém ST-GCN
            return stgcn_top5, confidence, False
        
        # Ranking do modelo
        ranking_top5 = np.argsort(-scores)[:5]
        
        # Comparar
        overlap = len(set(stgcn_top5) & set(ranking_top5))
        
        if overlap >= 4:
            # Muito similar, não precisa corrigir
            return stgcn_top5, confidence, False
        
        # Corrigir: usar 80% ST-GCN + 20% Ranking
        # Isso mantém a maioria das predições mas corrige outliers
        corrected = []
        
        # Pega 4 do ST-GCN
        for node in stgcn_top5[:4]:
            if node not in corrected:
                corrected.append(node)
        
        # Pega 1 do Ranking (que ST-GCN não pegou)
        for node in ranking_top5:
            if node not in corrected and len(corrected) < 5:
                corrected.append(node)
        
        # Se ainda faltam, completa com ST-GCN
        for node in stgcn_top5:
            if node not in corrected and len(corrected) < 5:
                corrected.append(node)
        
        was_corrected = len(set(corrected) - set(stgcn_top5)) > 0
        
        return np.array(corrected[:5]), confidence, was_corrected

# Instância global
_ranking_system = None

def get_ranking_system():
    """Retorna instância do sistema"""
    global _ranking_system
    if _ranking_system is None:
        _ranking_system = RankingCorrectionSystem()
    return _ranking_system

if __name__ == "__main__":
    # Teste
    system = get_ranking_system()
    
    # Simular dados
    test_cvli = np.random.rand(10, 30) * 5
    
    scores, confidence = system.get_ranking_scores(test_cvli, day_of_week=0)
    print(f"Scores: {scores[:5]}")
    print(f"Confidence: {confidence:.4f}")
    
    stgcn_top5 = np.array([0, 1, 2, 3, 4])
    corrected, conf, was_corrected = system.correct_stgcn_prediction(
        stgcn_top5, test_cvli, day_of_week=0
    )
    print(f"ST-GCN top-5: {stgcn_top5}")
    print(f"Corrected top-5: {corrected}")
    print(f"Was corrected: {was_corrected}")
