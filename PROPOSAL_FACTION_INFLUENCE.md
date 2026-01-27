# Proposta Técnica: Inclusão de Influência Faccional à Distância no Treino (Multi-Graph ST-GCN)

## 1. Objetivo
O objetivo é alterar o pipeline de treinamento para que localidades influenciadas pela mesma facção exerçam influência mútua na predição de risco, independentemente da distância geográfica entre elas.

## 2. Diagnóstico Atual
Atualmente, a construção do grafo em `src/data_processing.py` (`create_adjacency_matrix`) utiliza estritamente um critério espacial:
```python
if dist < threshold_degrees: # ~500 metros
    adj_matrix[i, j] = 1
```
Isso significa que o modelo ST-GCN atual é "cego" para conexões semânticas. Se dois bairros distantes são dominados pela mesma facção e sofrem ataques coordenados, o modelo não tem um caminho direto no grafo para propagar essa informação.

## 3. Solução Proposta: ST-GCN Multi-Grafo (Multi-View)

A melhor abordagem técnica não é apenas somar pesos na matriz existente (o que confundiria proximidade física com conexão semântica), mas sim adotar uma arquitetura **Multi-Grafo**. Isso permite que o modelo aprenda *pesos diferentes* para a influência geográfica versus a influência faccional.

### Passo 1: Processamento de Dados (`src/data_processing.py`)
Precisamos gerar duas matrizes de adjacência distintas:
1.  **`adj_geo`**: A matriz atual (proximidade física).
2.  **`adj_faction`**: Uma nova matriz onde `A[i,j] = 1` se `nodes_gdf.iloc[i]['faction'] == nodes_gdf.iloc[j]['faction']`, caso contrário 0.

**Alteração sugerida:**
```python
def create_dual_adjacency_matrices(gdf, threshold_degrees=0.005):
    n = len(gdf)
    adj_geo = np.zeros((n, n))
    adj_faction = np.zeros((n, n))
    
    centroids = gdf.geometry.centroid
    factions = gdf['faction'].values
    
    for i in range(n):
        # Conexão Geográfica
        for j in range(i + 1, n):
            if centroids.iloc[i].distance(centroids.iloc[j]) < threshold_degrees:
                adj_geo[i, j] = adj_geo[j, i] = 1
        
        # Conexão Faccional (Otimizável com broadcasting)
        # Cria arestas entre todos os nós da mesma facção
        same_faction_indices = np.where(factions == factions[i])[0]
        adj_faction[i, same_faction_indices] = 1
        
    # Normalização e Self-loops
    np.fill_diagonal(adj_geo, 1)
    np.fill_diagonal(adj_faction, 1)
    
    return adj_geo, adj_faction
```

### Passo 2: Arquitetura do Modelo (`src/model.py`)
O modelo ST-GCN deve ser atualizado para aceitar uma lista de matrizes de adjacência. A camada de convolução gráfica (`GraphConvolution`) processará cada grafo independentemente e somará os resultados (ou concatenará).

**Equação da nova convolução:**
$$ H^{(l+1)} = \sigma \left( \sum_{k \in \{geo, fac\}} \hat{D}_k^{-1/2} \hat{A}_k \hat{D}_k^{-1/2} H^{(l)} W_k^{(l)} \right) $$

Onde $W_{geo}$ e $W_{fac}$ são matrizes de pesos aprendíveis distintas. Isso permite que o modelo decida, por exemplo, que a influência geográfica é forte para crimes de oportunidade (CVP), enquanto a influência faccional é forte para crimes violentos (CVLI).

**Alteração sugerida (`GraphConvolution`):**
```python
class MultiGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_graphs=2):
        super().__init__()
        # Um peso para cada tipo de grafo (Geo e Faccional)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features)) 
            for _ in range(num_graphs)
        ])
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def forward(self, x, adj_list):
        output = 0
        for i, adj in enumerate(adj_list):
            support = torch.matmul(x, self.weights[i])
            output += torch.matmul(adj, support)
        return output + self.bias
```

### Passo 3: Loop de Treinamento (`src/train_separate.py`)
1.  Carregar ambas as matrizes do arquivo pickle.
2.  Aplicar a normalização Laplaciana ($\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$) em ambas.
3.  Passar a lista `[norm_adj_geo, norm_adj_faction]` para o modelo no `forward`.

## 4. Benefícios Esperados
1.  **Captura de Padrões Coordenados**: Se uma facção inicia uma ofensiva em múltiplos territórios, o grafo faccional propagará esse sinal de risco instantaneamente entre os nós aliados, mesmo que estejam em lados opostos da cidade.
2.  **Robustez**: O modelo não perde a capacidade de modelar o contágio local (geográfico), pois mantém o canal geográfico separado.
3.  **Flexibilidade**: Se a facção X é mais organizada que a facção Y, os pesos $W_{fac}$ aprenderão essa dinâmica globalmente.

## 5. Próximos Passos
Para implementar esta sugestão:
1.  Refatorar `src/data_processing.py` para gerar e salvar `adj_faction`.
2.  Atualizar `src/model.py` para usar `MultiGraphConvolution`.
3.  Ajustar `src/train_separate.py` para injetar os dois grafos.
