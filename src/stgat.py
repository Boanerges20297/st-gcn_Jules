import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer - Aprende pesos dinâmicos de adjacência via atenção.
    
    Input:
        h: (N, in_features) - features dos nós
        adj: (N, N) - matriz de adjacência (máscara booleana)
    
    Output:
        (N, out_features) - features transformadas com atenção espacial
    """
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Matriz de transformação linear
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Coeficientes de atenção (concatenação bilinear)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Args:
            h: (N, in_features)
            adj: (N, N) matriz binária de adjacência
        Returns:
            out: (N, out_features)
        """
        # Transformação linear: h' = h @ W
        Wh = torch.mm(h, self.W)  # (N, out_features)
        
        # Compute attention coefficients
        e = self._prepare_attentional_mechanism_input(Wh)  # (N, N)

        # Mask: apenas edges existentes na adjacência
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # (N, N)
        
        # Softmax por linha (normaliza atenção por nó)
        attention = F.softmax(attention, dim=1)  # (N, N)
        
        # Dropout na atenção (regularização)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aplica atenção aos features transformados
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Calcula coeficientes de atenção via concatenação bilinear.
        Para cada par (i, j): a_ij = [Wh_i || Wh_j]^T * a
        
        Args:
            Wh: (N, out_features)
        Returns:
            e: (N, N) matriz de scores de atenção (pre-softmax)
        """
        N = Wh.size()[0]
        
        # Repetir cada linha N vezes (acima e abaixo na diagonal)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # Cada linha repetida N vezes
        Wh_repeated_alternating = Wh.repeat(N, 1)  # Padrão alternado
        
        # Concatenation: [Wh_i || Wh_j]
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # Aplicar matriz de atenção: a^T @ [Wh_i || Wh_j]
        e = torch.matmul(all_combinations_matrix, self.a)  # (N*N, 1)
        e = self.leakyrelu(e)  # Apply activation
        
        return e.view(N, N)  # Reshape para (N, N)

class MultiGraphAttention(nn.Module):
    """
    Multi-head Graph Attention - Aplica atenção a múltiplas matrizes de adjacência.
    
    Útil para grafos com múltiplas relações:
    - adj_geo: proximidade geográfica
    - adj_faction: proximidade de controle territorial
    
    Cada head aprende pesos diferentes; outputs são mediados.
    """
    def __init__(self, in_features, out_features, num_graphs, dropout=0.5, alpha=0.2):
        super(MultiGraphAttention, self).__init__()
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(num_graphs)
        ])
        self.out_features = out_features
        self.num_graphs = num_graphs

    def forward(self, x, adj_list):
        """
        Args:
            x: (N, in_features) - features dos nós
            adj_list: list de (N, N) matrizes de adjacência
        Returns:
            out: (N, out_features) - features após atenção espacial
        """
        outputs = []
        for i, adj in enumerate(adj_list):
            out_i = self.attentions[i](x, adj)
            outputs.append(out_i)
        
        # Combinar outputs de múltiplos heads por média
        # Alternativa: concatenação + projeção (mais paramétrica)
        combined = torch.stack(outputs, dim=0)  # (num_graphs, N, out_features)
        return combined.mean(dim=0)  # (N, out_features)

class STGATLayer(nn.Module):
    """
    Spatio-Temporal Graph Attention Layer.
    
    Processa dados com formato: (batch, channels, nodes, time)
    
    Fluxo:
    1. Convolução temporal 1D: channels → out_channels
    2. Atenção espacial (GAT) por timestep
    3. Layer norm
    4. Conexão residual
    
    Args:
        in_channels: canais de entrada (26 para CVLI+features)
        out_channels: canais de saída
        num_graphs: número de matrizes de adjacência (2: geo + faction)
        time_steps: tamanho da janela temporal (30 dias)
        dropout: taxa de dropout
    """
    def __init__(self, in_channels, out_channels, num_graphs=2, time_steps=30, dropout=0.5):
        super(STGATLayer, self).__init__()
        
        # Componentes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_steps = time_steps
        
        # Temporal: Convolução 1D ao longo do tempo
        # Input: (B, C_in, N, T) ou (B, C_in, 1, T) após reshape
        # Output: (B, C_out, 1, T) ou (B, C_out, N, T)
        self.time_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),  # 1x3: não combina nós, apenas convolui no tempo
            padding=(0, 1)       # preserva time_steps
        )
        
        # Spatial: Graph Attention
        self.gat = MultiGraphAttention(
            in_features=out_channels,
            out_features=out_channels,
            num_graphs=num_graphs,
            dropout=dropout
        )
        
        # Residual: adapta dimensions se in_channels ≠ out_channels
        self.residual = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
        # Normalização
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, adj_list):
        """
        Args:
            x: (B, C, N, T) - batch, channels, nodes, time
            adj_list: list de (N, N) matrizes de adjacência
        
        Returns:
            out: (B, C_out, N, T) - outputs pós-processados
        """
        B, C_in, N, T = x.shape
        
        # ========== TEMPORAL CONVOLUTION ==========
        # Aplicar convolução 1D ao longo do tempo
        x_temporal = self.time_conv(x)  # (B, C_out, N, T)
        
        # ========== SPATIAL ATTENTION (por timestep) ==========
        # Aplicar GAT a cada timestep independentemente
        spatial_outputs = []
        
        for t in range(T):
            # Extrair features para timestep t
            x_t = x_temporal[:, :, :, t]  # (B, C_out, N)
            
            # Processar cada batch
            batch_outputs = []
            for b in range(B):
                # x_t[b] é (C_out, N)
                # Transpor para (N, C_out) como esperado por GAT
                x_b = x_t[b].t()  # (N, C_out)
                
                # Aplicar GAT multi-graph
                out_b = self.gat(x_b, adj_list)  # (N, C_out)
                
                # Transpor de volta para (C_out, N)
                batch_outputs.append(out_b.t())
            
            # Stack batch: list de (C_out, N) → (B, C_out, N)
            x_t_spatial = torch.stack(batch_outputs, dim=0)
            spatial_outputs.append(x_t_spatial)
        
        # Stack timesteps: list de (B, C_out, N) → (B, C_out, N, T)
        x_spatial = torch.stack(spatial_outputs, dim=-1)
        
        # ========== RESIDUAL CONNECTION ==========
        # Transformar entrada original para match dimensions
        x_residual = self.residual(x)  # (B, C_out, N, T)
        
        # ========== COMBINATION + NORMALIZATION ==========
        # Combinar spatial + residual
        x_combined = x_spatial + x_residual  # (B, C_out, N, T)
        
        # Layer norm: (B, C_out, N, T) → (B, N, T, C_out) para norm → (B, C_out, N, T)
        B, C_out, N, T = x_combined.shape
        x_norm = x_combined.permute(0, 2, 3, 1)  # (B, N, T, C_out)
        x_norm = self.ln(x_norm)                  # aplica norm na última dim (C_out)
        x_out = x_norm.permute(0, 3, 1, 2)      # volta para (B, C_out, N, T)
        
        return x_out

class STGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network (ST-GAT).
    
    Aprende representações de crime combinando:
    1. Atenção espacial (GAT): aprende quais vizinhos são relevantes dinamicamente
    2. Processamento temporal: convolução + atenção temporal
    3. Múltiplas relações: geograficamente + territorialmente
    
    Entrada: (batch, 26, 319, 30)
        - batch: número de amostras
        - 26: canais (CVLI, CVP, Tension + features onehot)
        - 319: nós (bairros de Fortaleza + interior Ceará)
        - 30: timesteps (últimos 30 dias)
    
    Saída: (batch, 319, 1)
        - scores de risco para cada nó
    
    Args:
        num_nodes: número de nós no grafo (319)
        in_channels: canais de entrada (26)
        time_steps: tamanho da janela temporal (30)
        num_classes: outputs por nó (1 para regressão de risco)
        num_graphs: número de matrizes de adjacência (2)
        dropout: taxa de dropout
    """
    def __init__(self, num_nodes, in_channels, time_steps, num_classes=1, num_graphs=2, dropout=0.5):
        super(STGAT, self).__init__()
        
        # Layer 1: 26 → 16 canais
        self.layer1 = STGATLayer(
            in_channels=in_channels,
            out_channels=16,
            num_graphs=num_graphs,
            time_steps=time_steps,
            dropout=dropout
        )
        
        # Layer 2: 16 → 32 canais
        self.layer2 = STGATLayer(
            in_channels=16,
            out_channels=32,
            num_graphs=num_graphs,
            time_steps=time_steps,
            dropout=dropout
        )
        
        # Convolução final: agregates sobre tempo
        # Input: (B, 32, N, T)
        # Output: (B, 64, N, 1) via kernel_size=(1, time_steps)
        self.conv_final = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, time_steps)
        )
        
        # Fully connected: 64 → 1 score por nó
        self.fc = nn.Linear(64, num_classes)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        """
        Args:
            x: (B, 26, 319, 30) tensor de features
            adj_list: list de 2 matrizes (319, 319) - adjacência
        
        Returns:
            out: (B, 319, 1) scores de risco
        """
        # Layer 1
        x = self.layer1(x, adj_list)  # (B, 16, 319, 30)
        x = self.dropout(x)
        
        # Layer 2
        x = self.layer2(x, adj_list)  # (B, 32, 319, 30)
        x = self.dropout(x)
        
        # Convolução final: reduce time dimension
        x = self.conv_final(x)  # (B, 64, 319, 1)
        
        # Squeeze time dimension
        x = x.squeeze(-1)  # (B, 64, 319)
        
        # Transpor para (B, 319, 64)
        x = x.permute(0, 2, 1)
        
        # Fully connected: project to score
        x = self.fc(x)  # (B, 319, 1)
        
        return x
    
    def get_attention_weights(self, x, adj_list):
        """
        Retorna pesos de atenção para interpretabilidade.
        Allows visualizing which spatial connections matter.
        
        Returns:
            dict com matrices de atenção por layer
        """
        # TODO: Implementar para interpretabilidade visual
        pass

