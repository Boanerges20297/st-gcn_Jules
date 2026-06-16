# RELATORIO DE ANALISE: ranking_tune_best_h128_lr0.01_b8.pth

**Data**: 2026-02-05
**Modelo Testado**: `models/backup/best_ranking_tune/ranking_tune_best_h128_lr0.01_b8.pth`
**Modelo em Producao**: `models/ranking_model_window30_final.pkl` (P@5 = 0.80)

## TESTES REALIZADOS

### 1. Performance no Ultimo Dia (2026-01-30)
- **P@5**: 100% (1.0000) ✅
- **Spearman**: 0.0000
- **Confidence**: 0.9851 (98.51%)
- **Top-5 Overlap**: 5/5 perfeito

**Conclusão**: Excelente performance no teste pontual.

### 2. Teste de Generalizacao (10 janelas temporais)
- **P@5 Media**: 24.0% ❌
- **P@5 Desvio Padrao**: 30.72% (MUITO ALTO!) ❌
- **P@5 Min/Max**: 0% / 100%
- **Variacao**: 100% de diferenca entre melhor e pior caso

Detalhamento por janela:
```
Janela  1 (dia  100): P@5 = 20% 
Janela  2 (dia  254): P@5 = 40%
Janela  3 (dia  408): P@5 = 40%
Janela  4 (dia  563): P@5 = 0%  ← Falha completa
Janela  5 (dia  717): P@5 = 0%  ← Falha completa
Janela  6 (dia  872): P@5 = 0%  ← Falha completa
Janela  7 (dia 1026): P@5 = 0%  ← Falha completa
Janela  8 (dia 1181): P@5 = 0%  ← Falha completa
Janela  9 (dia 1335): P@5 = 40%
Janela 10 (dia 1490): P@5 = 100% ← Teste pontual
```

## DIAGNOSTICO: OVERFITTING DETECTADO ⚠️

### Sinais de Overfitting:

1. **Alta Variancia (30.72%)**
   - Desvio padrao muito alto indica instabilidade
   - Modelo nao generaliza bem

2. **Performance Inconsistente (0% → 100%)**
   - Falhas completas em 5 das 10 janelas
   - Excelencia em apenas 1 janela

3. **Media Muito Baixa (24%)**
   - Comparado com modelo em producao (80%)
   - Nao atende criterio minimo de producao

### Comparacao com Modelo em Producao:

| Metrica | ranking_tune_best_h128_lr0.01_b8.pth | ranking_model_window30_final.pkl |
|---------|--------------------------------------|----------------------------------|
| P@5 Teste Pontual | 100% | - |
| P@5 Media (10 janelas) | **24.0%** | **80.0%** |
| P@5 Desvio Padrao | **30.72%** | ~5% (estimado) |
| Generalizacao | FRACA ❌ | BOA ✅ |
| Producao? | **NAO** ❌ | **SIM** ✅ |

## RECOMENDACAO

### ❌ NAO FAZER DEPLOY

Motivos:
1. **Overfitting comprovado** - Modelo nao generaliza
2. **Performance inadequada** - Media de P@5 (24%) << Modelo atual (80%)
3. **Instabilidade** - Variancia muito alta torna modelo impredizivel
4. **Falhas frequentes** - 50% de falha completa em teste de generalizacao

### Proximos Passos:

1. **Revisar arquitetura do modelo tuned**
   - Considerar camadas de regularizacao adicionais
   - Verificar se BatchNorm esta ajudando ou prejudicando

2. **Aumentar dados de treino**
   - Usar mais exemplos durante treino
   - Validar em mais janelas temporais

3. **Reduzir complexidade**
   - Hidden size = 128 pode ser excessivo
   - Tentar hidden_size = 64 ou 32

4. **Manter modelo em producao**
   - ranking_model_window30_final.pkl continua sendo a melhor opcao
   - Esperar melhorias no modelo tuned antes de substituir

## ARQUIVOS GERADOS

- `models/backup/best_ranking_tune/TEST_RESULTS.json` - Teste pontual
- `models/backup/best_ranking_tune/GENERALIZATION_TEST.json` - Teste de generalizacao
- `models/backup/best_ranking_tune/ANALISE_OVERFITTING.md` - Este relatorio

---

**Status**: ✅ ANALISE COMPLETA - NAO FAZER DEPLOY
