import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (B, N, in_features)
        # adj: (B, N, N) or (N, N)
        
        B, N, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W) # (B, N, out_features)
        
        # Attention Mechanism
        # Prepare for broadcasting
        # Wh1: (B, N, 1, out_features), Wh2: (B, 1, N, out_features)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) # (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) # (B, N, 1)
        
        # Broadcast add -> (B, N, N)
        e = self.leakyrelu(Wh1 + Wh2.transpose(1, 2)) 

        # Mask attention with adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        
        # Ensure adj matches batch dim if necessary, though usually adj is (N,N) static or (B,N,N) dynamic
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
            
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MultiGraphAttention(nn.Module):
    """
    Applies GAT on multiple graphs (e.g. Geo + Faction) and sums/fuses results.
    """
    def __init__(self, in_features, out_features, num_graphs=2, dropout=0.5, alpha=0.2):
        super(MultiGraphAttention, self).__init__()
        self.num_graphs = num_graphs
        self.gats = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha) 
            for _ in range(num_graphs)
        ])
    
    def forward(self, x, adj_list):
        # x: (B, N, C)
        # adj_list: list of (N,N) or (B,N,N) tensors
        outputs = []
        for i, gat in enumerate(self.gats):
            if i < len(adj_list):
                outputs.append(gat(x, adj_list[i]))
            else:
                # Fallback if fewer adjs than expected
                outputs.append(gat(x, adj_list[0]))
        
        # Sum aggregation (could be concat)
        return sum(outputs)

class STGATLayer(nn.Module):
    """
    Spatio-Temporal GAT Layer:
    Temporal Conv -> Graph Attention -> Residual
    """
    def __init__(self, in_channels, out_channels, num_graphs=2, time_steps=12, dropout=0.5):
        super(STGATLayer, self).__init__()
        
        # Temporal convolution (same as STGCN)
        self.time_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        
        # Graph Attention
        self.gat = MultiGraphAttention(out_channels, out_channels, num_graphs, dropout)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, adj_list):
        # x: (B, C, N, T)
        B, C, N, T = x.size()
        
        res = self.residual(x)
        
        # Temporal Conv
        x_time = self.time_conv(x) # (B, out, N, T)
        x_time = F.relu(x_time)
        
        # Permute for GAT: (B, T, N, C_out) -> flatten T into Batch? 
        # Or apply GAT frame by frame.
        # Standard STGCN often does GCN on each frame or treats T as channels.
        # Here we'll treat each timestep as a sample for GAT.
        
        x_perm = x_time.permute(0, 3, 2, 1).contiguous() # (B, T, N, out)
        x_reshaped = x_perm.view(B*T, N, -1) # (B*T, N, out)
        
        # GAT
        # We need to expand adj to match B*T
        adj_expanded = []
        for adj in adj_list:
            if len(adj.shape) == 2:
                # (N, N)
                adj_expanded.append(adj) # GAT layer handles broadcasting
            else:
                # (B, N, N) -> repeat for T?
                # For simplicity, assume static adj or adj matches B
                adj_expanded.append(adj)

        x_gat = self.gat(x_reshaped, adj_expanded) # (B*T, N, out)
        
        # Reshape back
        x_gat = x_gat.view(B, T, N, -1).permute(0, 3, 2, 1) # (B, out, N, T)
        
        return F.relu(x_gat + res)

class STGAT(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, num_classes=1, num_graphs=2, dropout=0.5):
        super(STGAT, self).__init__()
        
        self.layer1 = STGATLayer(in_channels, 16, num_graphs, time_steps, dropout)
        self.layer2 = STGATLayer(16, 32, num_graphs, time_steps, dropout)
        
        self.conv_final = nn.Conv2d(32, 64, kernel_size=(1, time_steps))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, adj_list):
        # x: (B, C, N, T)
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        
        x = self.conv_final(x) # (B, 64, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1) # (B, N, 64)
        
        return self.fc(x) # (B, N, 1)
