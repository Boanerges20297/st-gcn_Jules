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
        self.prelu = nn.PReLU()

    def forward(self, x, adj_list):
        adj_geo, adj_conf = adj_list[0], adj_list[1]
        h_self = self.W_self(x)
        h_geo = torch.matmul(adj_geo, self.W_geo(x))
        h_conf = torch.matmul(adj_conf, self.W_conf(x))
        out = h_self + h_geo + h_conf
        BT, N, C = out.shape
        out = self.bn(out.view(-1, C)).view(BT, N, C)
        return self.dropout(self.prelu(out))

class GlobalSpatialAttention(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps, dropout=0.4, heads=8):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.prelu = nn.PReLU()
        self.spatial_transformer = GlobalSpatialAttention(out_channels, heads=heads)
        self.gcn = FastRelationalGCN(out_channels, out_channels, dropout)
        self.temp_attn = MultiHeadTemporalAttention(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj_list):
        res = self.residual(x)
        x = self.prelu(self.time_conv(x))
        B, C, N, T = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x_spatial = self.spatial_transformer(x_flat)
        x_spatial = self.gcn(x_spatial, adj_list)
        x = x_spatial.reshape(B, T, N, C).permute(0, 3, 2, 1)
        x = self.temp_attn(x)
        return x + res


class PureSTGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps, dropout=0.3):
        super().__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.prelu = nn.PReLU()
        self.gcn = FastRelationalGCN(out_channels, out_channels, dropout)
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj_list):
        res = self.residual(x)
        x = self.prelu(self.time_conv(x))
        B, C, N, T = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x_spatial = self.gcn(x_flat, adj_list)
        x = x_spatial.reshape(B, T, N, C).permute(0, 3, 2, 1)
        x = self.out_norm(x)
        return x + res


class ChannelSelector(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = tuple(int(i) for i in indices)

    def forward(self, x):
        valid = [idx for idx in self.indices if idx < x.shape[1]]
        if not valid:
            raise ValueError("Nenhum canal valido disponivel para o seletor heterogeneo.")
        return x[:, valid, :, :]

# --- ARQUITETURAS DISPONÍVEIS ---

class DeepSTGAT_64(nn.Module):
    """ESPECIALISTA FORTALEZA (Versao Original Estavel - 63.2%)."""
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.4):
        super().__init__()
        self.layer1 = STGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2 = STGCNBlock(32, 64, time_steps, dropout)
        self.layer3 = STGCNBlock(64, 64, time_steps, dropout)
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.prelu_final = nn.PReLU()
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.PReLU(), nn.Linear(32, 1))

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.prelu_final(self.final_conv(x)).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)

class DeepSTGAT_80(nn.Module):
    """ESPECIALISTA ELITE (EXPANSÃO 20%+ - 80 NEURÔNIOS + CANAL 38 MEMPALACE)."""
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.5):
        super().__init__()
        # in_channels agora será 38 (37 originais + 1 Vault Memory)
        self.layer1 = STGCNBlock(in_channels, 40, time_steps, dropout)
        self.layer2 = STGCNBlock(40, 80, time_steps, dropout)
        self.layer3 = STGCNBlock(80, 80, time_steps, dropout)
        self.final_conv = nn.Conv2d(80, 80, kernel_size=(1, time_steps))
        self.prelu_final = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(80, 40), 
            nn.PReLU(), 
            nn.Dropout(dropout),
            nn.Linear(40, 1)
        )

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.prelu_final(self.final_conv(x)).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)

class ShallowGAT(nn.Module):
    """MODELO TÁTICO RESIDUAL (ResGAT) - 2 camadas com skip connection para máxima extração tática."""
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.3, heads=16):
        super().__init__()
        self.layer1 = STGCNBlock(in_channels, 64, time_steps, dropout, heads=heads)
        self.layer2 = STGCNBlock(64, 64, time_steps, dropout, heads=heads)
        
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.prelu_final = nn.PReLU()
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32), 
            nn.PReLU(), 
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj_list):
        # Primeira camada extrai inteligência imediata
        out1 = self.layer1(x, adj_list)
        # Segunda camada extrai correlações de vizinhança de 2º grau
        out2 = self.layer2(out1, adj_list)
        
        # Residual: Preserva o sinal da primeira camada
        x = out1 + out2
        
        # Colapso temporal e FC
        x = self.prelu_final(self.final_conv(x)).squeeze(-1).permute(0, 2, 1)
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
            nn.LeakyReLU(0.2),
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
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.final_conv(x).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)


class PureSTGCN_64(nn.Module):
    """ST-GCN puro para Fortaleza: convolucao temporal + grafo relacional, sem atencao."""
    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.3):
        super().__init__()
        self.layer1 = PureSTGCNBlock(in_channels, 32, time_steps, dropout)
        self.layer2 = PureSTGCNBlock(32, 64, time_steps, dropout)
        self.layer3 = PureSTGCNBlock(64, 64, time_steps, dropout)
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.prelu_final = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.layer3(x, adj_list)
        x = self.prelu_final(self.final_conv(x)).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)


class FortalezaHeteroSTGAT(nn.Module):
    """
    Especialista heterogeneo para Fortaleza.
    Separa sinais dinamicos/taticos dos canais contextuais para evitar que o
    calendario e a malha estatica abafem os pulsos operacionais.
    """
    DYNAMIC_CHANNELS = (0, 1, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36)
    CONTEXT_CHANNELS = (
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 26, 29, 30, 37, 38, 39, 40,
    )

    def __init__(self, num_nodes, in_channels, time_steps, dropout=0.35):
        super().__init__()
        dyn_channels = len([c for c in self.DYNAMIC_CHANNELS if c < in_channels])
        ctx_channels = len([c for c in self.CONTEXT_CHANNELS if c < in_channels])

        self.dynamic_selector = ChannelSelector(self.DYNAMIC_CHANNELS)
        self.context_selector = ChannelSelector(self.CONTEXT_CHANNELS)

        self.dynamic_stem = nn.Sequential(
            nn.Conv2d(dyn_channels, 32, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.context_stem = nn.Sequential(
            nn.Conv2d(ctx_channels, 16, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(dropout * 0.25),
        )

        self.dynamic_block1 = STGCNBlock(32, 48, time_steps, dropout=dropout, heads=8)
        self.dynamic_block2 = STGCNBlock(48, 64, time_steps, dropout=dropout, heads=8)

        self.context_block1 = STGCNBlock(16, 24, time_steps, dropout=dropout * 0.75, heads=4)
        self.context_block2 = STGCNBlock(24, 32, time_steps, dropout=dropout * 0.75, heads=4)
        self.context_to_shared = nn.Conv2d(32, 64, kernel_size=1)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid(),
        )
        self.post_fusion = STGCNBlock(64, 64, time_steps, dropout=dropout, heads=8)
        self.final_conv = nn.Conv2d(64, 64, kernel_size=(1, time_steps))
        self.prelu_final = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, adj_list):
        x_dyn = self.dynamic_stem(self.dynamic_selector(x))
        x_ctx = self.context_stem(self.context_selector(x))

        x_dyn = self.dynamic_block1(x_dyn, adj_list)
        x_dyn = self.dynamic_block2(x_dyn, adj_list)

        x_ctx = self.context_block1(x_ctx, adj_list)
        x_ctx = self.context_block2(x_ctx, adj_list)
        x_ctx = self.context_to_shared(x_ctx)

        gate = self.fusion_gate(torch.cat([x_dyn, x_ctx], dim=1))
        x = gate * x_dyn + (1.0 - gate) * x_ctx
        x = self.post_fusion(x, adj_list)
        x = self.prelu_final(self.final_conv(x)).squeeze(-1).permute(0, 2, 1)
        return self.fc(x)


MODEL_REGISTRY = {
    "DeepSTGAT_64": DeepSTGAT_64,
    "DeepSTGAT_80": DeepSTGAT_80,
    "ShallowGAT": ShallowGAT,
    "TemperatureExpertGAT": TemperatureExpertGAT,
    "DeepSTGAT_32": DeepSTGAT_32,
    "PureSTGCN_64": PureSTGCN_64,
    "FortalezaHeteroSTGAT": FortalezaHeteroSTGAT,
}


def get_model_class(name):
    return MODEL_REGISTRY.get(name, ShallowGAT)
