import torch
import torch.nn as nn

class GlobalRankingModel(nn.Module):
    """MLP that maps full ST-GCN score vector (N) -> refined score vector (N)."""
    def __init__(self, num_nodes=319, hidden1=512, hidden2=256, dropout=0.3):
        super(GlobalRankingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout/2),
            nn.Linear(hidden2, num_nodes)
        )

    def forward(self, x):
        # x: (batch, num_nodes) -> (batch, num_nodes)
        return self.net(x)
