import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# NÚCLEO NEURAL REPORT PREVIEW: DEEP-STGAT (Spatial-Temporal Graph Attention)
# ============================================================================

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, channels, heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, N, T = x.shape
        x_temp = x.mean(dim=2).permute(0, 2, 1) 
        attn_out, _ = self.mha(x_temp, x_temp, x_temp)
        x_temp = self.norm(x_temp + attn_out)
        attn_weights = x_temp.permute(0, 2, 1).unsqueeze(2) 
        return x * torch.sigmoid(attn_weights)

class FastRelationalGCN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.4):
        super().__init__()
        self.W_self = nn.Linear(in_features, out_features)
        self.W_geo = nn.Linear(in_features, out_features)
        self.W_conf = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj_list):
        adj_geo, adj_conf = adj_list[0], adj_list[1]
        h_self = self.W_self(x)
        h_geo = torch.matmul(adj_geo, self.W_geo(x))
        h_conf = torch.matmul(adj_conf, self.W_conf(x))
        out = h_self + h_geo + h_conf
        BT, N, C = out.shape
        out = self.bn(out.view(-1, C)).view(BT, N, C)
        return self.dropout(F.elu(out))

class GlobalSpatialAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps, dropout=0.4):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.spatial_transformer = GlobalSpatialAttention(out_channels)
        self.gcn = FastRelationalGCN(out_channels, out_channels, dropout)
        self.temp_attn = MultiHeadTemporalAttention(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj_list):
        res = self.residual(x)
        x = F.elu(self.time_conv(x))
        B, C, N, T = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x_spatial = self.spatial_transformer(x_flat)
        x_spatial = self.gcn(x_spatial, adj_list)
        x = x_spatial.reshape(B, T, N, C).permute(0, 3, 2, 1)
        x = self.temp_attn(x)
        return x + res

# --- ARQUITETURAS DISPONÍVEIS ---

class DeepSTGAT_64(nn.Module):
    """ESPECIALISTA FORTALEZA (Versao Original Estavel - 63.2%)."""
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.4):
        super().__init__()
        self.layer1 = STGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2 = STGCNBlock(32, 64, time_steps, dropout)
        self.layer3 = STGCNBlock(64, 64, time_steps, dropout)
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.final_conv(x).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)

class TemperatureExpertGAT(nn.Module):
    """MODELO ESPECIALISTA LEVE (64 NEURONIOS FC)."""
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.2):
        super().__init__()
        self.layer1 = STGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2 = STGCNBlock(32, 64, time_steps, dropout)
        self.layer3 = STGCNBlock(64, 64, time_steps, dropout)
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        
        # FC CLASSICA: 64 -> 32 -> 1
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.final_conv(x).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)

class DeepSTGAT_32(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.4):
        super().__init__()
        self.layer1 = STGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2 = STGCNBlock(32, 32, time_steps, dropout)
        self.layer3 = STGCNBlock(32, 32, time_steps, dropout)
        self.final_conv = nn.Conv2d(32, 32, kernel_size=(1, time_steps))
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.final_conv(x).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)
