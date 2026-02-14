import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTemporalAttention(nn.Module):
    """Atenção temporal leve otimizada para CPU."""
    def __init__(self, channels, heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, N, T = x.shape
        # Pooling espacial para focar na cronologia global do bairro
        x_temp = x.mean(dim=2).permute(0, 2, 1) # (B, T, C)
        attn_out, _ = self.mha(x_temp, x_temp, x_temp)
        x_temp = self.norm(x_temp + attn_out)
        attn_weights = x_temp.permute(0, 2, 1).unsqueeze(2) # (B, C, 1, T)
        return x * torch.sigmoid(attn_weights)

class FastRelationalGCN(nn.Module):
    """
    Substituto cirúrgico para o GAT pesado.
    Processa conexões distantes (Conflito) e locais (Geo) via matrizes pré-calculadas.
    """
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.W_self = nn.Linear(in_channels, out_channels)
        self.W_geo = nn.Linear(in_channels, out_channels)
        self.W_conf = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, adj_list):
        # x shape: (B*T, N, C)
        adj_geo, adj_conf = adj_list[0], adj_list[1]
        
        # 1. Influência Própria
        h_self = self.W_self(x)
        
        # 2. Influência Geográfica (Vizinhos)
        # Usamos matmul direto pois as matrizes são fixas por semana
        h_geo = torch.matmul(adj_geo, self.W_geo(x))
        
        # 3. Influência Distante (Conflito/Facção) - O que o ST-GAT supria
        h_conf = torch.matmul(adj_conf, self.W_conf(x))
        
        # Fusão das influências
        out = h_self + h_geo + h_conf
        
        # Normalização e Ativação
        BT, N, C = out.shape
        out = self.bn(out.view(-1, C)).view(BT, N, C)
        return self.dropout(F.elu(out))

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps, dropout=0.3):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.gcn = FastRelationalGCN(out_channels, out_channels, dropout)
        self.temp_attn = MultiHeadTemporalAttention(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj_list):
        res = self.residual(x)
        x = F.elu(self.time_conv(x))
        
        B, C, N, T = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x_spatial = self.gcn(x_flat, adj_list)
        x = x_spatial.reshape(B, T, N, C).permute(0, 3, 2, 1)
        
        x = self.temp_attn(x)
        return x + res

class DeepSTGAT(nn.Module): # Mantemos o nome para compatibilidade
    """Arquitetura Relacional de Alta Velocidade (Phase 5 Surgical)."""
    def __init__(self, num_nodes, in_channels, time_steps, num_graphs=2, dropout=0.4):
        super().__init__()
        self.layer1 = STGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2 = STGCNBlock(32, 64, time_steps, dropout)
        self.layer3 = STGCNBlock(64, 64, time_steps, dropout)
        
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.final_conv(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        return self.fc(x)
