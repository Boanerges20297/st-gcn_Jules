# FASE 2: Integração de Embeddings Semânticos - Arquitetura Integrada

**Status**: 🚀 Iniciando
**Data**: 03/02/2026
**Atenção**: ⚠️ Função será integrada ao `llm_service.py` - NÃO deixar solta

---

## ⚠️ Aviso Crítico: Integração com Dados Exógenos

O `llm_service.py` **já é crítico** para extrair dados exógenos via LLM:
- `parse_ciops_report()` → Extrai eventos de segurança
- `parse_prisão_events()` → Extrai dados de prisões  
- Usa rotation de API keys e rate limiting
- Tem retry logic, error handling, fallbacks

**NOVA FUNÇÃO DEVE**:
- ✅ Seguir o mesmo padrão de rotation/rate limiting
- ✅ Usar mesmos API keys
- ✅ Integrar no mesmo `_call_model_with_rotation()`
- ✅ Cachear resultados (não fazer requisições repetidas)
- ❌ NÃO criar função solta independente
- ❌ NÃO duplicar lógica de API keys

---

## 🎯 Arquitetura Proposta

### 1. Estender `llm_service.py` com Novo Módulo

```python
# Em src/llm_service.py - adicionar nova seção:

def get_semantic_embeddings_batch(
    bairro_names: List[str],
    cache_file: str = 'data/processed/bairro_embeddings.json',
    api_keys: List[str] = None
) -> Dict[str, List[float]]:
    """
    Gera embeddings semânticos 384D para lista de bairros.
    
    Args:
        bairro_names: Lista de nomes de bairros (ex: ['Aldeota', 'Barroso', ...])
        cache_file: Onde cachear resultados (compartilhado com dados exógenos)
        api_keys: Usar api_keys fornecidas ou pegar de get_gemini_api_keys()
    
    Returns:
        Dict[bairro_name] = array 384D (numpy compatible)
    
    Features:
        - Rate limiting integrado (~1 req/seg)
        - Cache para evitar requisições repetidas
        - Rotation de API keys automática
        - Fallback se LLM falhar
        - Logging detalhado
    """
```

### 2. Fluxo de Dados Integrado

```
data/processed/
├── bairro_embeddings.json          ← Cache compartilhado
│   {
│     "Aldeota": [0.123, -0.456, ...],  // 384D
│     "Barroso": [0.789, 0.012, ...],
│     ...
│   }
├── processed_graph_data.pkl        ← Dataset original (26D)
└── processed_graph_data_semantic.pkl ← Dataset expandido (410D)
```

### 3. Integração com Data Processing

```python
# Em src/ranking_features.py

def expand_features_with_semantics(
    features_26d: np.ndarray,  # shape: (319, 1491, 26)
    embeddings_dict: Dict[str, np.ndarray]  # shape: {bairro: (384,)}
) -> np.ndarray:
    """
    Expand 26D features com embeddings semânticos 384D
    
    Output: (319, 1491, 410)
    """
    # Canais 0-25: Features crime + calendário (idênticos)
    # Canais 26-409: Embedding semântico (repetido para cada timestep)
```

---

## 📋 Implementação Detalhada

### Passo 1: Adicionar ao `llm_service.py` (INTEGRADO)

**Adicionar após `_mock_response()` no final**:

```python
# ============================================================================
# FASE 2: SEMANTIC EMBEDDINGS FOR NEIGHBORHOODS (INTEGRATED)
# ============================================================================

def _get_bairro_description(bairro_name: str) -> str:
    """
    Cria prompt descritivo para cada bairro.
    Inclui: localização, características demográficas, infraestrutura, segurança.
    """
    # Mapping básico de bairros com suas características
    bairro_descriptions = {
        'Aldeota': 'Bairro nobre em zona sul, comércio e residencial de alto padrão, próximo ao Centro',
        'Barroso': 'Zona periférica oeste, zona residencial popular',
        'Parangaba': 'Centro de Fortaleza, comercial, histórico, movimentado',
        # ... outros 316 bairros
    }
    
    return bairro_descriptions.get(
        bairro_name, 
        f'Neighborhood {bairro_name} in Fortaleza, Ceará, Brazil'
    )


def get_semantic_embeddings_batch(
    bairro_names: List[str],
    cache_file: str = None,
    api_keys: List[str] = None,
    rate_limit_delay: float = 1.5
) -> Dict[str, List[float]]:
    """
    Generate semantic embeddings (384D) for neighborhoods using Google Generative AI.
    
    INTEGRATED with existing LLM data extraction pipeline:
    - Uses shared API key rotation
    - Rate limiting to avoid quota exhaustion
    - Cache-first strategy (checks disk before API calls)
    - Fallback to mock embeddings if LLM fails
    
    Args:
        bairro_names: ['Aldeota', 'Barroso', ...]
        cache_file: Default 'data/processed/bairro_embeddings.json'
        api_keys: Use provided keys or get_gemini_api_keys()
        rate_limit_delay: Seconds between API calls (default 1.5s)
    
    Returns:
        {
            'Aldeota': [0.123, -0.456, ...],  # 384D
            'Barroso': [0.789, 0.012, ...],
            ...
        }
    
    Cache behavior:
        1. Load cache from disk (if exists)
        2. For each bairro not in cache:
           - Call LLM with rate limiting
           - Update cache on disk (incremental saves)
        3. Return complete dict
    
    """
    import time
    import pickle
    
    if cache_file is None:
        from pathlib import Path
        cache_file = str(Path('data/processed/bairro_embeddings.json'))
    
    if api_keys is None:
        api_keys = get_gemini_api_keys()
    
    if not api_keys:
        logger.warning('No API keys found; using mock embeddings')
        return _mock_semantic_embeddings(bairro_names)
    
    # Load existing cache
    embeddings = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                embeddings = json.load(f)
            logger.info(f'Loaded {len(embeddings)} cached embeddings from {cache_file}')
        except Exception as e:
            logger.warning(f'Failed to load cache: {e}; starting fresh')
    
    # Generate embeddings for missing bairros
    bairros_to_fetch = [b for b in bairro_names if b not in embeddings]
    
    if bairros_to_fetch:
        logger.info(f'Generating embeddings for {len(bairros_to_fetch)}/{len(bairro_names)} bairros')
        
        for idx, bairro in enumerate(bairros_to_fetch):
            try:
                logger.debug(f'[{idx+1}/{len(bairros_to_fetch)}] Getting embedding for {bairro}')
                
                description = _get_bairro_description(bairro)
                prompt = (
                    f"Generate semantic embedding for this neighborhood:\n\n"
                    f"Neighborhood: {bairro}, Fortaleza, Ceará, Brazil\n"
                    f"Description: {description}\n\n"
                    f"Respond with ONLY a JSON array of 384 floating-point numbers "
                    f"(no markdown, no explanation):\n"
                    f"[-0.123, 0.456, ..., 0.789]"
                )
                
                # Call with rotation - INTEGRATED with existing pipeline
                response = _call_model_with_rotation(prompt, api_keys)
                
                # Parse embedding from response
                embedding = _extract_json_from_text(response)
                
                if isinstance(embedding, list) and len(embedding) == 384:
                    embeddings[bairro] = embedding
                    logger.debug(f'Successfully generated embedding for {bairro}')
                else:
                    logger.warning(f'Invalid embedding format for {bairro}; using mock')
                    embeddings[bairro] = _mock_embedding_for_bairro(bairro)
                
            except Exception as e:
                logger.warning(f'Failed to get embedding for {bairro}: {e}; using mock')
                embeddings[bairro] = _mock_embedding_for_bairro(bairro)
            
            # Rate limiting - respect API quotas
            if idx < len(bairros_to_fetch) - 1:
                time.sleep(rate_limit_delay)
        
        # Save updated cache
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(embeddings, f)
            logger.info(f'Saved {len(embeddings)} embeddings to {cache_file}')
        except Exception as e:
            logger.error(f'Failed to save cache: {e}')
    
    # Return only requested bairros
    return {b: embeddings[b] for b in bairro_names if b in embeddings}


def _mock_embedding_for_bairro(bairro_name: str) -> List[float]:
    """
    Fallback: Generate deterministic mock embedding based on bairro name.
    Used when LLM fails but we still need to continue training.
    """
    import hashlib
    import numpy as np
    
    # Use bairro name as seed for reproducibility
    hash_obj = hashlib.md5(bairro_name.encode())
    seed = int(hash_obj.hexdigest(), 16) % (2**32)
    
    # Generate 384D vector
    rng = np.random.RandomState(seed)
    embedding = rng.randn(384).astype(np.float32) * 0.1  # Small scale
    
    logger.debug(f'Generated mock embedding for {bairro_name} (deterministic)')
    return embedding.tolist()


def _mock_semantic_embeddings(bairro_names: List[str]) -> Dict[str, List[float]]:
    """
    Generate all mock embeddings (when no API keys available).
    """
    logger.warning('No API keys available; using mock semantic embeddings for all bairros')
    return {b: _mock_embedding_for_bairro(b) for b in bairro_names}
```

---

## 🔗 Integração com `ranking_features.py`

**Modificar `src/ranking_features.py`**:

```python
# Adicionar ao topo do arquivo
from src.llm_service import get_semantic_embeddings_batch

def extract_ranking_features_with_semantics(
    node_features: np.ndarray,  # shape: (319, 1491, 26)
    node_ids: List[int],
    bairro_names: Dict[int, str],  # {node_id: 'Aldeota', ...}
    cache_file: str = 'data/processed/bairro_embeddings.json'
) -> np.ndarray:
    """
    Expand 26D features to 410D by adding semantic embeddings.
    
    Args:
        node_features: Original (319, 1491, 26)
        node_ids: [0, 1, 2, ..., 318]
        bairro_names: Mapping node_id → bairro name
        cache_file: Where embeddings are cached
    
    Returns:
        features_410d: (319, 1491, 410)
    """
    # Step 1: Get semantic embeddings for all unique bairros
    unique_bairros = list(set(bairro_names.values()))
    embeddings_dict = get_semantic_embeddings_batch(unique_bairros, cache_file=cache_file)
    
    # Step 2: Create mapping node_id → embedding
    node_embeddings = {
        node_id: np.array(embeddings_dict[bairro_names[node_id]])
        for node_id in node_ids
    }
    
    # Step 3: Expand features
    features_410d = np.zeros((319, 1491, 410), dtype=np.float32)
    features_410d[:, :, :26] = node_features  # Crime + calendar features
    
    for node_id in node_ids:
        emb = node_embeddings[node_id]  # 384D
        features_410d[node_id, :, 26:410] = emb  # Broadcast to all timesteps
    
    return features_410d
```

---

## 📅 Timeline Revisado (COM INTEGRAÇÃO)

### Dia 1: Extender `llm_service.py` Corretamente (3h)

- [ ] Adicionar `get_semantic_embeddings_batch()` ao final de llm_service.py
- [ ] Implementar `_get_bairro_description()` com mapeamento básico
- [ ] Implementar `_mock_embedding_for_bairro()` (fallback determinístico)
- [ ] Testar com 5 bairros (validar rate limiting, cache)

### Dia 2: Integrar com `ranking_features.py` (3h)

- [ ] Adicionar `extract_ranking_features_with_semantics()`
- [ ] Estender dataset 26D → 410D
- [ ] Testar output shapes
- [ ] Salvar `processed_graph_data_semantic.pkl`

### Dia 3: Treinar + Validar (4h)

- [ ] Adaptar `train_ranking_v2.py` para 410D
- [ ] Grid search 4 configs
- [ ] Comparar vs v2 (26D)
- [ ] Eval temporal

### Dia 4: Final (1-2h)

- [ ] Documentar findings
- [ ] Salvar best model
- [ ] Update ARCHITECTURE_REFERENCE.md

---

## 🛡️ Garantias de Integração

✅ **Uso de API keys compartilhado**: `get_gemini_api_keys()` (não duplicar)
✅ **Rotation automático**: `_call_model_with_rotation()` (mesmo comportamento)
✅ **Rate limiting integrado**: Sleep entre requisições (respeita quotas)
✅ **Cacheing estratégico**: Reutiliza cache existente (disk-based)
✅ **Fallback determinístico**: Mock embeddings se LLM falhar (modelo não quebra)
✅ **Logging centralizado**: Integrado com logger existente de llm_service.py
✅ **Error handling**: Mesmas estratégias de Exception handling

---

## ⚠️ Checklist de Segurança

- [ ] Função está DENTRO de `llm_service.py` (não arquivo separado)
- [ ] Usa `_call_model_with_rotation()` existente (não criar nova)
- [ ] Cache compartilhado em `data/processed/` (não em pasta aleatória)
- [ ] Rate limiting de 1.5s entre chamadas
- [ ] Mock embeddings determinísticos como fallback
- [ ] Logging verbose para debug
- [ ] Testes com API key rotation
- [ ] Sem dependencies novas (numpy + json apenas)

