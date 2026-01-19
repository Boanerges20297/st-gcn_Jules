import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2))
        self.gcn = GraphConvolution(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, adj):
        x = self.temporal_conv(x)
        B, C, N, T = x.size()
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B*T, N, C)
        x = self.gcn(x, adj)
        x = x.reshape(B, T, N, C).permute(0, 3, 2, 1)
        return self.dropout(self.relu(x))

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, num_classes=2):
        super(STGCN, self).__init__()
        self.layer1 = STGCNLayer(in_channels, 16)
        self.layer2 = STGCNLayer(16, 32)
        self.conv_final = nn.Conv2d(32, 64, kernel_size=(1, time_steps))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.conv_final(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.fc(x)
        return x
