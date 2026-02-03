# FASE 2: Integração de Embeddings Semânticos com LLM

**Status**: 🚀 Iniciando
**Data**: 03/02/2026
**Duração Estimada**: 3-4 dias
**Objetivo**: Adicionar embeddings semânticos dos bairros via Google Generative AI + manter P@5≈1.0

---

## 📋 Resumo Executivo

### Problema Atual (Fase 1)
- ✅ Modelo ranking alcançou NDCG@5=0.9995 (praticamente perfeito)
- ⚠️ Features limitadas a 26 canais (métricas crime + calendário)
- ⚠️ Sem informações semânticas sobre os bairros (localização, demografia, infraestrutura)
- ⚠️ Pode não generalizar bem para novos padrões/cenários

### Solução Fase 2
- 🎯 Adicionar embeddings semânticos (384D) do Google Generative AI
- 🎯 Combinar com features existentes (26D) → 410D total
- 🎯 Repensar o modelo se necessário
- 🎯 Validar que mantém P@5≈1.0 com melhor generalização

### Impacto Esperado
```
P@5:            1.0000 → 1.0000 (manter)
NDCG@5:         0.9995 → 0.9995+ (manter/melhorar)
Generalização:  ??? → ✅ (melhorar com semântica)
Robustez:       ??? → ✅ (menos dependente de padrões numéricos)
```

---

## 🎯 Objetivos Específicos

### Objetivo 1: Extrair Embeddings Semânticos
**Tarefa**: Usar Google Generative AI para criar embeddings dos bairros
- Input: Nome do bairro + descrição geográfica/demográfica
- Output: Embedding 384D por bairro
- Armazenar em cache para reutilização

**Arquivos Envolvidos**:
- `src/llm_service.py` - Já existe, estender para embeddings
- Nova função: `get_semantic_embedding(bairro_nome) → array 384D`

**Exemplo de Prompt**:
```
Gere um embedding semântico que capture as características do bairro:
Bairro: Aldeota, Fortaleza, Ceará
Características: Bairro nobre, zona sul, comércio, residencial de alto padrão, 
  próximo ao centro, infraestrutura completa, segurança moderada
```

### Objetivo 2: Integrar Embeddings ao Dataset
**Tarefa**: Estender `processed_graph_data.pkl` com embeddings semânticos
- Expandir de 26 → 410 canais
- Canais 0-25: Features atuais (crime + calendário)
- Canais 26-409: Embedding semântico (384D)
- Manter compatibilidade com dados históricos

**Arquivos Envolvidos**:
- `src/ranking_features.py` - Estender `extract_ranking_features()`
- `src/data_processing.py` - Adicionar embedding cache
- Nova função: `expand_features_with_semantics(features_26d) → features_410d`

### Objetivo 3: Treinar Novo Modelo
**Tarefa**: Treinar ranking_model_v2 com 410D features
- Input: 410D (26 crime/calendário + 384 semântico)
- Arquitetura: Testar multiple configs
  - Opção 1: Manter MLP simples (26→64→1) mas com 410 input
  - Opção 2: Usar camadas adicionais (410→256→128→64→1)
  - Opção 3: Usar attention entre features crime vs semânticas

**Arquivos Envolvidos**:
- `train_ranking_v2.py` - Adicionar suporte a 410D
- Novo modelo: `ranking_model_v3.py` (se arquitetura mudar)
- Output: `models/ranking_model_semantic_best.pkl`

### Objetivo 4: Validar e Comparar
**Tarefa**: Avaliar se melhora generalização sem perder ranking
- Treinar com validation split (80/20)
- Comparar:
  - Modelo v2 (26D): NDCG@5 no test set
  - Modelo v3 (410D): NDCG@5 no test set
- Testar em dados fora do período de treinamento

**Arquivos Envolvidos**:
- `eval_ranking_models.py` - Estender para validação cross-temporal
- Nova métrica: Generalização temporal (teste em dados 2026)

---

## 📅 Timeline Detalhada

### Dia 1: Setup + Extração de Embeddings (4-5h)

**1.1 Estender llm_service.py** (1h)
- [ ] Função `get_semantic_embedding(bairro_nome: str) → np.array`
- [ ] Cache em `data/processed/bairro_embeddings.json`
- [ ] Tratamento de erros (rate limits, timeouts)
- [ ] Logging de progresso

**1.2 Gerar Embeddings para 319 Bairros** (2-3h)
- [ ] Loop através de todos os nós (319)
- [ ] Chamar LLM com descrição de cada bairro
- [ ] Salvar com rate limiting (ex: 1 req/seg para não exceder quotas)
- [ ] Validar formato (384D por bairro)

**1.3 Criar Dataset 410D** (1-2h)
- [ ] Função em `ranking_features.py`: `expand_with_semantics()`
- [ ] Testrar: expandir alguns exemplos
- [ ] Validar shapes: (319, 1491, 410)

---

### Dia 2: Integração + Treinamento (4-5h)

**2.1 Adaptar train_ranking_v2.py** (1h)
- [ ] Aceitar input de 410D
- [ ] Manter backward compatibility com 26D
- [ ] Adicionar flag: `use_semantic_features=True/False`

**2.2 Testar Treinamento** (1h)
- [ ] Treinar 1 epoch com 410D
- [ ] Verificar convergência
- [ ] Benchmark tempo (vs 26D)

**2.3 Grid Search 410D** (2-3h)
- [ ] Testar 6-8 configs principais:
  - batch=[4,8], lr=[0.001,0.0005], hidden=[64,128]
- [ ] Salvar todas as checkpoints
- [ ] Registrar NDCG@5, Spearman por config

---

### Dia 3: Validação + Análise (3-4h)

**3.1 Eval Comparativa** (1-2h)
- [ ] Carregar best models (v2 26D vs v3 410D)
- [ ] Testar no mesmo test set
- [ ] Comparar: NDCG@5, P@5, Spearman, correlação

**3.2 Teste Temporal** (1-2h)
- [ ] Treinar com dados 2022-2025
- [ ] Testar com dados Janeiro 2026
- [ ] Medir quanto generaliza para "futuro"

**3.3 Análise de Features** (1h)
- [ ] Qual importância: crime vs semântica?
- [ ] Features semânticas estão ajudando ou apenas ruído?
- [ ] Visualizar projeções PCA

---

### Dia 4: Refinement + Documentation (2-3h)

**4.1 Decisão Final** (1h)
- [ ] V2 (26D) vs V3 (410D): qual melhor?
- [ ] Se V3 ganhar, adotar como padrão
- [ ] Se V2 ganhar, documentar por quê

**4.2 Otimizações Finais** (1h)
- [ ] Se necessário: ajustar arquitetura
- [ ] Treinar modelo final
- [ ] Salvar best checkpoint

**4.3 Documentação** (1h)
- [ ] Atualizar ARCHITECTURE_REFERENCE.md
- [ ] Criar FASE2_RESULTS.md com findings
- [ ] Documentar impactos no production

---

## 🛠️ Arquivos a Criar/Modificar

### Criar Novos
- `FASE2_SEMANTIC_EMBEDDINGS.py` - Gerador de embeddings (pode ter limite de requisições)
- `FASE2_VALIDATION.py` - Eval comparativa 26D vs 410D

### Modificar Existentes
- `src/llm_service.py` - Adicionar embedding generation
- `src/ranking_features.py` - Estender com semantics
- `train_ranking_v2.py` - Aceitar 410D input
- `eval_ranking_models.py` - Adicionar comparações

### Possível Criar
- `src/ranking_model_v3.py` - Se mudar arquitetura
- `hyperparam_search_phase2.py` - Grid search 410D

---

## 💾 Data Flow Fase 2

```
Bairros (319 nomes)
    ↓
LLM (Google Generative AI)
    ↓
Embeddings 384D × 319 bairros
    ↓
Cache: data/processed/bairro_embeddings.json
    ↓
expand_with_semantics()
    ↓
Features 410D (26 crime/cal + 384 semântico)
    ↓
Tensor (319 nós × 1491 timesteps × 410 features)
    ↓
train_ranking_v2.py (com input 410D)
    ↓
ranking_model_semantic_best.pkl
    ↓
eval_ranking_models.py
    ↓
Comparação vs V2 (26D)
```

---

## ⚠️ Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|--------|-----------|
| Rate limit do LLM | Alta | Médio | Cache, retry com backoff exponencial |
| Embeddings genéricos demais | Média | Alto | Usar prompts específicos sobre crime/segurança |
| Performance cai com 410D | Baixa | Alto | Testar com subset primeiro, dropout regularização |
| Features semânticas redundantes | Média | Baixo | Análise PCA, correlação com crime |
| Tempo de treino aumenta 10x | Média | Médio | Usar GPU/otimizações, paralelizar |

---

## ✅ Checklist de Sucesso

- [ ] **Embeddings extraídos**: 319 bairros com 384D embeddings
- [ ] **Dataset 410D criado**: processed_graph_data_semantic.pkl
- [ ] **Treinamento convergente**: Epoch 1 sem erros, loss decreasing
- [ ] **NDCG@5 ≥ 0.99**: Mantém performance de fase 1
- [ ] **Generalização melhor**: Test temporal 2026 com score > v2
- [ ] **Documentação completa**: FASE2_RESULTS.md com findings
- [ ] **Modelo salvo**: models/ranking_model_semantic_best.pkl
- [ ] **Deploy ready**: Código pronto para produção

---

## 📊 Métricas a Acompanhar

### Training
- Loss por epoch (deve convergir rápido)
- Tempo por epoch (baseline 26D vs 410D)

### Validation
- NDCG@5 (deve manter ≥ 0.99)
- P@5 (deve ser 1.0)
- Spearman ρ (deve manter ≥ 0.97)

### Generalization
- NDCG@5 em Janeiro 2026 (novo período)
- Degradação temporal (quanto piora em dados fora do treinamento)
- Confiança em top-5 predictions

---

## 🚀 Próximos Passos Imediatos

1. **Agora**: Revisar `src/llm_service.py` - verificar se já tem embedding support
2. **Próxima tarefa**: Implementar `get_semantic_embedding()` 
3. **Depois**: Gerar embeddings para 319 bairros (com rate limiting)
4. **Integração**: Estender dataset a 410D

---

## 📚 Referências

- **Fase 1 Report**: PHASE1_FINAL_REPORT.md
- **Best Model v2**: models/ranking_model_best_Config_01_Small.pkl
- **LLM Service**: src/llm_service.py
- **Feature Extraction**: src/ranking_features.py

---

## 💬 Notas

- Embeddings são **estáticos por bairro** (não variam por tempo)
- Diferente de features de crime que variam por timestep
- Objetivo: capturar contexto semântico permanente dos bairros
- Esperamos que ajude em generalização para novos períodos

