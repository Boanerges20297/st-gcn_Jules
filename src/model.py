import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiGraphConvolution(nn.Module):
    """Convolução gráfica que aceita múltiplas adjacências (multi-grafo).

    A saída é a soma das convoluções aplicadas por cada grafo, cada um com
    seu próprio peso aprendível.
    """
    def __init__(self, in_features, out_features, num_graphs=2, bias=True):
        super(MultiGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_graphs = num_graphs
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(num_graphs)
        ])
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            nn.init.kaiming_uniform_(w)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_list):
        # x shape: (batch, N, in_features)
        output = 0
        for i, adj in enumerate(adj_list):
            w = self.weights[i]
            support = torch.matmul(x, w)
            output = output + torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_graphs=2):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2))
        self.gcn = MultiGraphConvolution(out_channels, out_channels, num_graphs=num_graphs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, adj_list):
        x = self.temporal_conv(x)
        B, C, N, T = x.size()
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B*T, N, C)
        x = self.gcn(x, adj_list)
        x = x.reshape(B, T, N, C).permute(0, 3, 2, 1)
        return self.dropout(self.relu(x))

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, num_classes=2, num_graphs=2):
        super(STGCN, self).__init__()
        self.layer1 = STGCNLayer(in_channels, 16, num_graphs=num_graphs)
        self.layer2 = STGCNLayer(16, 32, num_graphs=num_graphs)
        self.conv_final = nn.Conv2d(32, 64, kernel_size=(1, time_steps))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, adj_list):
        x = self.layer1(x, adj_list)
        x = self.layer2(x, adj_list)
        x = self.conv_final(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.fc(x)
        return x
