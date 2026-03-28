import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, adj):
        super(STGCNBlock, self).__init__()
        self.adj = adj
        # Temporal Convolution 1
        self.tmp_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        # Spatial Graph Convolution (Strictly Local)
        self.gcn_conv = nn.Linear(out_channels, spatial_channels)
        # Temporal Convolution 2
        self.tmp_conv2 = nn.Conv2d(spatial_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [Batch, Channels, Nodes, Time]
        
        # 1. Temporal
        x = self.relu(self.tmp_conv1(x))
        
        # 2. Spatial (GCN logic: A * X * W)
        # Reshape para [Batch, Time, Nodes, Channels]
        x = x.permute(0, 3, 2, 1)
        # Multiplicação pela adjacência (filtro local)
        x = torch.matmul(self.adj, x)
        x = self.relu(self.gcn_conv(x))
        
        # 3. Temporal
        x = x.permute(0, 3, 2, 1)
        x = self.relu(self.tmp_conv2(x))
        
        return x

class DeepSTGCN_CVP(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, out_channels=64, dropout=0.5):
        super(DeepSTGCN_CVP, self).__init__()
        # Registra a adjacência como buffer para não ser treinada mas estar no device correto
        self.register_buffer('adj_buffer', torch.zeros(num_nodes, num_nodes))
        
        self.block1 = STGCNBlock(in_channels, 32, 64, num_nodes, self.adj_buffer)
        self.block2 = STGCNBlock(64, 32, 64, num_nodes, self.adj_buffer)
        
        self.fcn = nn.Sequential(
            nn.Linear(64 * time_steps, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) # Predição por nó
        )

    def set_adj(self, adj):
        self.adj_buffer.copy_(adj)

    def forward(self, x, adjs=None):
        # x original: [1, Channels, Nodes, Time]
        # STGCN espera: [Batch, Channels, Nodes, Time]
        
        x = self.block1(x)
        x = self.block2(x)
        
        # Final layers - Agregação temporal para predição final
        # x: [Batch, 64, Nodes, Time] -> [Batch, Nodes, 64*Time]
        batch_size, channels, nodes, time = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, nodes, -1)
        
        out = self.fcn(x).squeeze(-1) # [Batch, Nodes]
        return out
