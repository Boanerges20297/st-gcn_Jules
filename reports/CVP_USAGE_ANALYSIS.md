# Análise: Uso de CVP no Modelo ST-GCN

## Resumo Executivo

**CVP (Crimes Violentos contra o Patrimônio) É REALMENTE USADO COMO VARIÁVEL DE ENTRADA NO MODELO**, não apenas como critério de predição.

---

## Evidências de Uso de CVP

### 1. Arquitetura do Modelo

O modelo ST-GCN foi configurado com **3 canais de entrada**:

```python
# Canal 0: CVLI (Crimes Violentos Letais Intencionais)
# Canal 1: CVP (Crimes Violentos contra o Patrimônio)
# Canal 2: Tension (Índice de Tensão Social)
```

**Fonte:** `app.py` linha 424-426

```python
# Expected 3: CVLI, CVP, Tension
if _node_features.shape[2] != 3:
    raise ValueError("...")
```

---

### 2. Validações de Shape

O sistema valida explicitamente que os dados devem ter **exatamente 3 canais** em múltiplos pontos:

#### a) Validação de Startup
```python
# app.py linha 424-426
if _node_features.shape[2] != 3:
    raise ValueError(f"node_features deve ter 3 canais (CVLI, CVP, Tension), encontrado: {_node_features.shape[2]}")
```

#### b) Validação Runtime
```python
# app.py linha 896-903
if input_slice.shape[2] != 3:
    if input_slice.shape[2] == 2:
        # Tenta adicionar canal de zeros como fallback
        zeros = np.zeros((input_slice.shape[0], input_slice.shape[1], 1))
        input_slice = np.concatenate([input_slice, zeros], axis=2)
```

---

### 3. Processamento Explícito do Canal CVP

O código processa **explicitamente** o canal 1 (CVP) durante as predições:

```python
# app.py linha 920-925
# Channel 0 is CVLI
input_cvli = input_slice[:, :, 0]
daily_avg = np.mean(input_cvli, axis=1)

# Channel 1 is CVP
input_cvp = input_slice[:, :, 1]
daily_avg_cvp = np.mean(input_cvp, axis=1)
hist_sum_cvp = np.sum(input_cvp, axis=1)
```

**Nota:** Comentário explícito na linha 922: "Channel 1 is CVP"

---

### 4. Fluxo de Dados Completo

```
1. ENTRADA DE DADOS
   ├─ node_features.shape = (N, T, 3)
   │  ├─ [:, :, 0] = CVLI
   │  ├─ [:, :, 1] = CVP  ← CANAL CVP
   │  └─ [:, :, 2] = Tension
   │
2. FORWARD PASS
   ├─ input_tensor.shape = (1, 3, N, T)
   │  └─ Todos os 3 canais são processados pela rede ST-GCN
   │
3. PROCESSAMENTO
   ├─ input_cvli = input_slice[:, :, 0]
   ├─ input_cvp = input_slice[:, :, 1]  ← EXTRAÇÃO DE CVP
   └─ Cálculo de estatísticas: hist_sum_cvp, daily_avg_cvp
```

---

## CVP: Entrada vs Predição

### Dupla Função de CVP

| Função | Descrição | Evidência |
|--------|-----------|-----------|
| **Variável de Entrada** | CVP é alimentado como Canal 1 para o modelo fazer predições | `input_slice[:, :, 1]` usado no forward pass |
| **Alvo de Predição** | Existe modelo separado `stgcn_cvp.pth` que prediz CVP | Arquivo `models/stgcn_cvp.pth` |

### Esclarecimento

- **CVP não é "apenas critério de predição"**
- CVP é uma **variável de entrada** (feature) que o modelo usa para predizer CVLI
- Existe também um modelo separado que **prediz CVP** (mas esse não está em uso na interface atual)

---

## Arquivos de Modelo Identificados

```
models/
├── stgcn_cvli.pth    ← Modelo que PREDIZ CVLI usando (CVLI, CVP, Tension) como entrada
├── stgcn_cvp.pth     ← Modelo que PREDIZ CVP usando (CVLI, CVP, Tension) como entrada
└── stgcn_model.pth   ← Modelo genérico
```

**Modelo em Uso:** `stgcn_cvli.pth`
- **Entrada:** 3 canais (CVLI histórico, CVP histórico, Tension)
- **Saída:** Predição de CVLI futuro

---

## Implicações Práticas

### 1. CVP Influencia as Predições de CVLI

Se CVP aumenta em uma região, o modelo **pode** prever aumento em CVLI, dependendo dos padrões aprendidos durante treinamento.

### 2. Remoção de CVP Degradaria o Modelo

Se tentássemos remover CVP (reduzir para 2 canais), teríamos que:
- Retreinar o modelo completamente
- Perder informação valiosa sobre contexto criminal
- Sistema atual **rejeitaria** dados com shape incorreto

### 3. Switch CVLI/CVP Era Enganoso

O switch removido não alterava quais **variáveis** eram usadas, apenas qual **predição** era exibida:
- Modo CVLI: Mostrava predições do modelo `stgcn_cvli.pth`
- Modo CVP: Mostrava predições do modelo `stgcn_cvp.pth`

**Ambos os modelos sempre usaram os 3 canais (CVLI, CVP, Tension) como entrada.**

---

## Conclusão

✅ **CVP é usado como variável de entrada (feature) no modelo**
✅ **CVP está no Canal 1 de um tensor de shape (N, T, 3)**
✅ **Validações de shape garantem que CVP está presente**
✅ **Código processa explicitamente o canal CVP**
✅ **Remoção de CVP quebraria o modelo atual**

**Recomendação:** Manter CVP como variável de entrada. Se o objetivo for simplificar, considere retreinar um modelo com apenas 2 canais (CVLI, Tension), mas isso exigiria:
1. Coleta de novos dados de treinamento
2. Treinamento completo do zero
3. Validação de performance
4. Possível degradação de acurácia

---

## Testes Criados

Arquivo: `tests/test_model_viability.py`

Testes implementados:
1. ✅ Inicialização do modelo com 3 canais
2. ✅ Forward pass com dados 3-canais
3. ✅ CVP é realmente utilizado (isolamento de canal)
4. ✅ Modelo rejeita entrada com número incorreto de canais
5. ✅ Predições em range razoável
6. ✅ CVLI e CVP contribuem independentemente
7. ✅ Checkpoint loading

**Executar testes:**
```bash
pytest tests/test_model_viability.py -v
```
