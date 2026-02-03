# Comparação de Arquiteturas para Previsão Espaço-Temporal

## 1. ST-GCN (Spatio-Temporal Graph Convolutional Network) - ATUAL

### Descrição
Rede neural que combina **Graph Convolutions** (espacial) com **Temporal Convolutions** (temporal) para capturar padrões em grafos dinâmicos.

### Vantagens
- ✅ **Modelagem Espacial Explícita**: Usa matrizes de adjacência (geográfica + facções) para capturar relações entre bairros
- ✅ **Eficiência Computacional**: Convolução 1D temporal é rápida (~25s/época em CPU)
- ✅ **Interpretabilidade**: Podemos visualizar quais conexões espaciais o modelo está usando
- ✅ **Adequado para Grafos Irregulares**: Funciona bem com 319 nós de tamanhos diferentes
- ✅ **Multi-Graph**: Pode combinar múltiplas relações (vizinhança geográfica + rivalidade de facções)

### Desvantagens
- ❌ **Dependência de Adjacências**: Precisa de matrizes de adjacência bem definidas
- ❌ **Janela Temporal Limitada**: Dificuldade com dependências muito longas (>30 dias)
- ❌ **Não Aprende Relações Espaciais**: As adjacências são fixas (não aprende novas conexões)

### Quando Usar
- Dados com estrutura de grafo conhecida (bairros, redes sociais, sensores)
- Relações espaciais importantes e estáveis
- Necessidade de interpretabilidade

---

## 2. LSTM (Long Short-Term Memory)

### Descrição
Rede recorrente com células de memória que capturam dependências temporais de longo prazo.

### Vantagens
- ✅ **Memória de Longo Prazo**: Pode lembrar padrões de 60-90 dias atrás
- ✅ **Aprendizado Sequencial**: Ideal para séries temporais com tendências e sazonalidade
- ✅ **Flexível**: Funciona sem necessidade de definir adjacências
- ✅ **Comprovado**: Arquitetura madura com muitas implementações

### Desvantagens
- ❌ **Sem Modelagem Espacial**: Trata cada nó independentemente (ignora vizinhança)
- ❌ **Treinamento Sequencial**: Mais lento que convoluções (não paralelizável no tempo)
- ❌ **Vanishing Gradient**: Pode esquecer padrões muito antigos
- ❌ **Mais Parâmetros**: Consome mais memória que ST-GCN

### Quando Usar
- Séries temporais sem estrutura espacial
- Dependências temporais muito longas (>30 dias)
- Dados irregulares no tempo

---

## 3. Transformer (Attention-Based)

### Descrição
Arquitetura baseada em **Self-Attention** que aprende relações entre todos os nós e todos os timesteps simultaneamente.

### Vantagens
- ✅ **Atenção Global**: Aprende automaticamente quais bairros influenciam outros (sem adjacências fixas)
- ✅ **Paralelizável**: Treina muito mais rápido em GPU que LSTM
- ✅ **Captura Dependências Longas**: Sem limitação de janela temporal
- ✅ **State-of-the-Art**: Arquitetura mais moderna e poderosa

### Desvantagens
- ❌ **Custo Computacional**: O(N²) em memória e tempo (N = nodes × timesteps)
- ❌ **Precisa de MUITO Dado**: Tendência a overfitting com <100k amostras
- ❌ **Difícil Interpretar**: A matriz de atenção é menos intuitiva que adjacências geográficas
- ❌ **Requer GPU**: Praticamente inviável treinar em CPU com 319 nós

### Quando Usar
- Dataset grande (>100k amostras)
- GPU disponível
- Relações espaciais desconhecidas ou dinâmicas
- Orçamento computacional alto

---

## Comparação Direta

| Critério | ST-GCN | LSTM | Transformer |
|----------|--------|------|-------------|
| **Modelagem Espacial** | ✅ Explícita (adjacências) | ❌ Nenhuma | ✅ Aprendida (atenção) |
| **Modelagem Temporal** | ⚠️ Convolução (curto prazo) | ✅ Memória (longo prazo) | ✅ Atenção (qualquer prazo) |
| **Velocidade (CPU)** | ✅ Rápido (~25s/época) | ⚠️ Moderado (~45s/época) | ❌ Lento (>300s/época) |
| **Memória** | ✅ Baixa | ⚠️ Moderada | ❌ Alta |
| **Dados Necessários** | ⚠️ Moderado (~10k) | ⚠️ Moderado (~10k) | ❌ Alto (>100k) |
| **Interpretabilidade** | ✅ Alta | ⚠️ Média | ❌ Baixa |
| **Overfitting** | ⚠️ Risco médio | ⚠️ Risco médio | ❌ Alto risco |

---

## Recomendação para Este Projeto

### Manter ST-GCN ✅

**Motivos:**
1. **Estrutura Espacial Rica**: Temos 2 grafos (geográfico + facções) que são cruciais
2. **Dataset Moderado**: 74k ocorrências é suficiente para ST-GCN mas insuficiente para Transformer
3. **CPU-Only**: ST-GCN treina em ~20 minutos; Transformer levaria horas
4. **Interpretabilidade Operacional**: Polícia precisa entender POR QUE um bairro é perigoso (vizinhança com facção rival)

### Melhorias Possíveis SEM Trocar Arquitetura

1. **Aumentar Janela Temporal** (✅ JÁ FEITO: 14 → 30 dias)
   - Captura padrões mensais e sazonais
   
2. **Features Temporais Ricas** (✅ JÁ FEITO: 8 canais)
   - Dia da semana (sin/cos)
   - Mês (sin/cos)
   - Final de semana
   
3. **Attention em Temporal** (🔧 POSSÍVEL)
   - Substituir convolução 1D por self-attention apenas no eixo temporal
   - Mantém eficiência espacial do GCN
   - Adiciona capacidade de longo prazo do Transformer
   
4. **Multi-Head Graph Attention** (🔧 POSSÍVEL)
   - Substituir GCN por GAT (Graph Attention Network)
   - Aprende pesos para cada adjacência (algumas facções podem ser mais influentes)
   
5. **Arquitetura Híbrida: ST-GAT** (🔧 RECOMENDADO)
   - Graph Attention (espacial) + Temporal Attention
   - Melhor dos dois mundos: aprende relações espaciais + captura dependências longas
   - Custo computacional moderado

---

## Próximos Passos

### Curto Prazo (Esta Sessão)
- ✅ Retreinar com janela 30 dias
- ✅ Verificar se P@5 > 15%
- ⏳ Avaliar necessidade de mais melhorias

### Médio Prazo (Se P@5 < 20%)
- Implementar ST-GAT (Graph Attention + Temporal Attention)
- Adicionar features exógenas (eventos prisionais, clima)
- Experimento com loss function (Ranking Loss vs MSE)

### Longo Prazo (Se Recursos Permitirem)
- Avaliar Transformer em GPU com dataset expandido (2020-2027)
- Ensemble de múltiplos modelos ST-GCN com diferentes janelas
- Transfer learning de outras cidades (Rio, São Paulo)

---

## Referências

- **ST-GCN Original**: Yan et al. (2018) - Spatial Temporal Graph Convolutional Networks
- **GAT**: Veličković et al. (2018) - Graph Attention Networks
- **Temporal Fusion Transformer**: Lim et al. (2021) - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **Crime Prediction**: Wang et al. (2020) - Crime Rate Inference with Big Data
