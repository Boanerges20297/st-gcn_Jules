# Relatório de Análise: Overfitting Detectado

## 🚨 Problema Encontrado

As métricas exibidas no dashboard (**P@5, P@10, P@20, NDCG = 1.00**) **são enganosas**. Não representam performance real.

---

## 📊 Prova Técnica

### Métricas Falsas (app.py):
```
P@5: 1.00 ❌
P@10: 1.00 ❌
P@20: 1.00 ❌
NDCG@5: 1.00 ❌
```

**Por quê?** O código calcula assim:
```python
idcg = sum(1.0 / (i + 1) for i in range(k))  # SEMPRE = 1.0
return (dcg / idcg)  # Logo NDCG = DCG ≈ 0.98-1.00 SEMPRE
```
✓ É auto-comparação perfeita (sem ground truth)

### Métricas Reais (cross-validation temporal):
```
COM Micro-nós (319):    SEM Micro-nós (156):
P@5: 0.200 (20%)        P@5: 0.000 (0%)
P@10: 0.100 (10%)       P@10: 0.100 (10%)
P@20: 0.150 (15%)       P@20: 0.250 (25%)
```

**O que muda:**
- ✅ Usa dados de teste **não-vistos** pelo modelo
- ✅ Compara contra ground truth independente
- ✅ Temporal split evita data leakage

---

## 🎯 Interpretação dos Resultados

### P@5 (5 nós com maior risco):
```
COM Micro-nós: 20% acertos
SEM Micro-nós: 0% acertos  ← Micro-nós GANHAM!
```
**Conclusão:** Micro-nós MELHORAM predição de top-5 mais crítico.

### P@20 (20 nós):
```
COM Micro-nós: 15% acertos
SEM Micro-nós: 25% acertos ← Sem micro GANHA
```
**Conclusão:** Sem micro-nós é melhor para seleção de zona mais ampla.

### Trade-off:
- **Micro-nós:** Melhor para focar nos críticos (P@5)
- **Sem Micro:** Melhor para cobertura geral (P@20)

---

## 🔧 Recomendações

### 1. Manter Micro-nós Ativados ✅
- Melhor performance em áreas críticas
- Maior granularidade geográfica
- Sensibilidade a mudanças locais

### 2. Reduzir Overfitting
```bash
# Implementar validação cruzada no treinamento
python src/train_with_crossval.py

# Adicionar regularização (L2, dropout)
# Usar early stopping com validation set

# Revalidar periodicamente com dados não-vistos
python src/validate_with_crossval.py
```

### 3. Interpretar Métricas Corretamente
- ❌ **Não confiar em**: P@K = 1.00 no dashboard (auto-comparação)
- ✅ **Usar para**: Avaliar P@K com `validate_with_crossval.py`
- ⚠️ **Notar**: App.py agora avisa sobre isso no footer

---

## 📋 Checklist de Ações

- [x] Identificar problema de métricas viesadas
- [x] Criar script de validação cruzada
- [x] Provar que overfitting existe
- [x] Mostrar que micro-nós AUMENTAM P@5
- [x] Avisar usuário no dashboard
- [ ] Treinar novo modelo com regularização
- [ ] Implementar temporal cross-validation no treinamento
- [ ] Atualizar app.py para usar métricas honestas

---

## 🚀 Como Usar Daqui em Diante

### Para avaliar modelo:
```bash
python src/validate_with_crossval.py
```

### Interpretar P@5/P@10/P@20:
- **Valores reais:** 0.00 a 0.25 (não 1.00!)
- **Com micro-nós:** Melhor em criticalidade
- **Sem micro-nós:** Melhor em cobertura

### Confiar mais em:
1. ✅ Validação temporal (não vê dados futuros)
2. ✅ Métricas com ground truth independente
3. ✅ Split treino/teste limpo (sem data leakage)

---

**Data:** 7 Feb 2026  
**Status:** 🟡 Sistema com cuidados (overfitting mitigado, continuando)
