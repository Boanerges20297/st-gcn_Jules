# FASE 2: Checklist de Implementação

**Data Início**: 03/02/2026
**Objetivo**: Adicionar embeddings semânticos mantendo integração limpa com LLM existente

---

## ✅ PRÉ-REQUISITOS

- [ ] Revisar `FASE2_INTEGRACAO_SEMANTICA.md` (arquitectura integrada)
- [ ] Verificar `llm_service.py` - estrutura atual
- [ ] Confirmar `.env` tem API keys válidas (testar com parse_ciops_report)
- [ ] Backupar `llm_service.py` antes de editar

---

## 📝 FASE 2.1: Estender `llm_service.py`

### Task 2.1.1: Adicionar `_get_bairro_description()`
```
Status: [ ] TODO
Arquivo: src/llm_service.py (fim do arquivo, antes de FASE2)
Reqs: 
  - Mapeamento de 319 bairros com descrição
  - Formato: "Bairro nobre em zona sul, comércio..."
  - Fallback: "{bairro_name} in Fortaleza, Ceará"
Testes:
  - [ ] `_get_bairro_description('Aldeota')` retorna string
  - [ ] Fallback funciona para bairro desconhecido
```

### Task 2.1.2: Adicionar `_mock_embedding_for_bairro()`
```
Status: [ ] TODO
Arquivo: src/llm_service.py
Reqs:
  - Gera embedding 384D determinístico via MD5 hash
  - Same seed → same embedding (reproducible)
  - Escala pequena (stddev ~0.1)
Testes:
  - [ ] Mesmo bairro → mesmo embedding
  - [ ] Diferentes bairros → diferentes embeddings
  - [ ] Shape (384,)
  - [ ] Determinístico (rodar 2x, verificar igualdade)
```

### Task 2.1.3: Adicionar `_mock_semantic_embeddings()`
```
Status: [ ] TODO
Arquivo: src/llm_service.py
Reqs:
  - Wrapper que chama _mock_embedding_for_bairro() para lista
  - Usado quando sem API keys
Testes:
  - [ ] Retorna dict {bairro: [384D]}
  - [ ] Todos os bairros têm embedding
```

### Task 2.1.4: Adicionar `get_semantic_embeddings_batch()`
```
Status: [ ] TODO
Arquivo: src/llm_service.py (main function)
Reqs:
  - Load cache de disk (json format)
  - Para cada bairro missing:
    * Gera prompt descritivo
    * Chama _call_model_with_rotation() (COMPARTILHADO!)
    * Extrai JSON (384D array)
    * Sleep 1.5s rate limiting
  - Salva cache atualizado (incremental)
  - Fallback mock se LLM falhar por bairro
  
Integração Crítica:
  - [ ] Usar get_gemini_api_keys() (não duplicar)
  - [ ] Chamar _call_model_with_rotation() (não _call_model direto)
  - [ ] Cache em data/processed/bairro_embeddings.json (compartilhado)
  - [ ] Logging com logger existente
  
Testes:
  - [ ] Gera embedding para 5 bairros
  - [ ] Cache criado em disk
  - [ ] Segunda chamada carrega cache (mais rápido)
  - [ ] Rate limiting respeitado (~1.5s entre calls)
  - [ ] Fallback mock funciona se API falhar
  - [ ] Return dict {bairro: [384,]}
```

---

## 🔗 FASE 2.2: Integrar com `ranking_features.py`

### Task 2.2.1: Adicionar import
```
Status: [ ] TODO
Arquivo: src/ranking_features.py (topo)
Código:
  from src.llm_service import get_semantic_embeddings_batch

Testes:
  - [ ] Import sem erro
```

### Task 2.2.2: Criar `extract_ranking_features_with_semantics()`
```
Status: [ ] TODO
Arquivo: src/ranking_features.py
Reqs:
  - Input: node_features (319, 1491, 26)
  - Input: node_ids [0, 1, ..., 318]
  - Input: bairro_names {0: 'Aldeota', 1: 'Barroso', ...}
  - Output: (319, 1491, 410)
  
  - Paso 1: get_semantic_embeddings_batch() para unique bairros
  - Paso 2: Map node_id → embedding
  - Paso 3: Expandir features
    * Canais 0-25: copiar features_26d idênticas
    * Canais 26-409: repetir embedding para todos os timesteps
  
Testes:
  - [ ] Output shape (319, 1491, 410)
  - [ ] Canais 0-25 idênticos ao input
  - [ ] Canais 26-409 são embeddings válidos (384D)
  - [ ] Embeddings repetidos em todas timesteps (cada node é idêntico por timestep)
  - [ ] Sem NaN, sem inf
```

### Task 2.2.3: Testar integração
```
Status: [ ] TODO
Arquivo: test_semantic_features.py (novo)
Reqs:
  - Load processed_graph_data.pkl (26D)
  - Chamar extract_ranking_features_with_semantics()
  - Validar output shape, valores
  - Salvar como processed_graph_data_semantic.pkl (test)
  
Testes:
  - [ ] Features expandidas sem erros
  - [ ] Shapes corretos
  - [ ] Valores razoáveis (sem extremos)
  - [ ] Tempo de processamento aceitável
```

---

## 📦 FASE 2.3: Treinar Modelo 410D

### Task 2.3.1: Adaptar `train_ranking_v2.py`
```
Status: [ ] TODO
Arquivo: train_ranking_v2.py
Reqs:
  - Aceitar tanto 26D quanto 410D input
  - Flag: --use_semantics (default False para compatibilidade)
  - Auto-detect input shape
  
Testes:
  - [ ] python train_ranking_v2.py (sem flag → 26D, como antes)
  - [ ] python train_ranking_v2.py --use_semantics (com 410D)
  - [ ] Treina 1 epoch sem erro em ambos os casos
```

### Task 2.3.2: Testar 1 epoch 410D
```
Status: [ ] TODO
Reqs:
  - Carregar processed_graph_data_semantic.pkl
  - Treinar 1 epoch
  - Verificar convergência (loss decreasing)
  - Benchmark tempo/epoch vs 26D
  
Testes:
  - [ ] Loss decreasing
  - [ ] Tempo por epoch < 5s (ou documentar se > 5s)
  - [ ] P@5 computável
```

### Task 2.3.3: Grid search 410D (4-6 configs)
```
Status: [ ] TODO
Arquivo: hyperparam_search_phase2.py
Configs a testar:
  - batch=4, lr=0.001, hidden=64
  - batch=4, lr=0.0005, hidden=64
  - batch=8, lr=0.001, hidden=128
  - batch=4, lr=0.001, hidden=128
  - (+ 2-3 mais se tempo permitir)

Reqs:
  - Salvar melhor modelo como ranking_model_semantic_best.pkl
  - CSV com resultados (NDCG@5, P@5, Spearman, tempo)
  
Testes:
  - [ ] Todos configs convergem
  - [ ] Salvar resultados
  - [ ] Best model identificado
```

---

## 📊 FASE 2.4: Validação Comparativa

### Task 2.4.1: Eval 26D vs 410D (mesmo test set)
```
Status: [ ] TODO
Arquivo: eval_ranking_models_phase2.py
Reqs:
  - Carregar v2 best (26D): ranking_model_best_Config_01_Small.pkl
  - Carregar v3 best (410D): ranking_model_semantic_best.pkl
  - Testar em mesmo test set
  - Comparar: NDCG@5, P@5, Spearman, Loss
  
Testes:
  - [ ] Ambos modelos carregam sem erro
  - [ ] Métricas computáveis
  - [ ] Gerar tabela comparativa
  - [ ] P@5 ≥ 0.99 para ambos
```

### Task 2.4.2: Teste temporal (2026)
```
Status: [ ] TODO
Reqs:
  - Treinar v2 (26D) com dados 2022-2025
  - Treinar v3 (410D) com dados 2022-2025
  - Testar ambos em Janeiro 2026 (fora do período)
  - Comparar degradação temporal
  
Métrica: 
  - Qual modelo generaliza melhor para "futuro"?
  - Esperado: v3 (semântica) melhor que v2 (numérico)
  
Testes:
  - [ ] Treino 2022-2025 funciona
  - [ ] Test Janeiro 2026 funciona
  - [ ] Degradação temporal documentada
```

---

## 📄 FASE 2.5: Documentação + Decision

### Task 2.5.1: Criar FASE2_RESULTS.md
```
Status: [ ] TODO
Arquivo: FASE2_RESULTS.md
Conteúdo:
  - Resumo executivo
  - Findings (v2 vs v3, qual melhor)
  - Métricas comparativas
  - Recomendação (manter v2 ou usar v3?)
  - Impacto em produção
  - Timeline gasto
  
Testes:
  - [ ] Documento claro e convincente
```

### Task 2.5.2: Update ARCHITECTURE_REFERENCE.md
```
Status: [ ] TODO
Reqs:
  - Adicionar seção "FASE 2: Semantic Embeddings"
  - Documentar v3 model
  - Data flow atualizado
  - Decisão final (v2 vs v3)
  
Testes:
  - [ ] Documento atualizado
```

### Task 2.5.3: Decisão + Checkpoint
```
Status: [ ] TODO
Reqs:
  - Decision: V2 (26D) vs V3 (410D)?
    * Se V3 ganhou → adotar como novo padrão
    * Se V2 melhor → manter v2, documentar learning
  - Salvar melhor model
  - Git commit com phase 2 completo
  
Testes:
  - [ ] Decisão clara documentada
  - [ ] Best model identificado e salvo
```

---

## 🔍 Validação Final (Antes de Commitar)

- [ ] Todos os tasks acima marcados como DONE
- [ ] `llm_service.py` integrado corretamente (sem funções soltas)
- [ ] `ranking_features.py` estendido
- [ ] `train_ranking_v2.py` compatível com 410D
- [ ] Cache de embeddings criado em disk
- [ ] Resultados documentados
- [ ] ARCHITECTURE_REFERENCE.md atualizado
- [ ] Git status limpo (commits organizado)

---

## 📞 Pontos de Contato

**Se houver problema**:
1. Verificar API key rate limit (testar com `parse_ciops_report` primeiro)
2. Verificar cache file permissions
3. Verificar shapes de tensor (319, 1491, 26) → (319, 1491, 410)
4. Verificar logs em `llm_service.py` (logging.INFO)

**Decision Point**:
- Se v3 (410D) pior que v2 (26D): documentar por quê (features redundantes?)
- Se v3 melhor: considerar remover v2 em fase produção

