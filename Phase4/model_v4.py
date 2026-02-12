import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, channels, time_steps, heads=2): # Reduzi heads
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, N, T = x.shape
        x_temp = x.mean(dim=2).permute(0, 2, 1) # (B, T, C)
        attn_out, _ = self.mha(x_temp, x_temp, x_temp)
        x_temp = self.norm(x_temp + attn_out)
        attn_weights = x_temp.permute(0, 2, 1).unsqueeze(2) 
        return x * torch.sigmoid(attn_weights)

class VectGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_graphs, dropout=0.6): # Aumentei dropout
        super().__init__()
        self.num_graphs = num_graphs
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(num_graphs, in_features, out_features))
        self.a = nn.Parameter(torch.empty(num_graphs, 2 * out_features, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        if isinstance(adj_list, list): adjs = torch.stack(adj_list)
        else: adjs = adj_list
        Wh = torch.einsum('bnc,gcf->gbnf', x, self.W)
        a_src, a_dst = self.a[:, :self.out_features, :], self.a[:, self.out_features:, :]
        f_src = torch.einsum('gbnf,gfz->gbnz', Wh, a_src)
        f_dst = torch.einsum('gbnf,gfz->gbnz', Wh, a_dst)
        logits = self.leakyrelu(f_src + f_dst.transpose(-2, -1))
        mask = (adjs.unsqueeze(1) == 0)
        logits = logits.masked_fill(mask, -9e15)
        attention = F.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        return torch.einsum('gbnn,gbnf->gbnf', attention, Wh).mean(dim=0)

class STGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_graphs, time_steps, dropout=0.6):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.gat = VectGATLayer(out_channels, out_channels, num_graphs, dropout)
        self.temp_attn = MultiHeadTemporalAttention(out_channels, time_steps)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj_list):
        res = self.residual(x)
        x = self.time_conv(x)
        B, C, N, T = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x_spatial = self.gat(x_flat, adj_list)
        x = x_spatial.reshape(B, T, N, C).permute(0, 3, 2, 1)
        x = self.temp_attn(x)
        return self.bn(self.elu(x + res))

class DeepSTGAT(nn.Module):
    """Arquitetura simplificada para evitar overfitting."""
    def __init__(self, num_nodes, in_channels, time_steps, num_graphs=2, dropout=0.6):
        super().__init__()
        self.layer1 = STGATBlock(in_channels, 32, num_graphs, time_steps, dropout)
        self.layer2 = STGATBlock(32, 64, num_graphs, time_steps, dropout)
        # Removida Layer 3 para maior generalização
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.final_conv(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        return self.fc(self.dropout(x))
