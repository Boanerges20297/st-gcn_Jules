# FASE 2: Passo-a-Passo de Início Imediato

**Data**: 03/02/2026
**Objetivo**: Começar a implementação em 30 minutos

---

## 📖 Leitura Obrigatória (Ordem)

1. **README.md** (2 min) - Visão geral do projeto
2. **FASE2_INTEGRACAO_SEMANTICA.md** (10 min) - Arquitetura integrada com LLM
3. **FASE2_CHECKLIST.md** (5 min) - Checklist detalhado

**Total**: ~17 minutos de leitura

---

## 🎯 Tarefa Imediata: Task 2.1.1 (Primeira Função)

### O que fazer:

Adicionar esta função ao final do arquivo `src/llm_service.py`:

```python
# ============================================================================
# FASE 2: SEMANTIC EMBEDDINGS FOR NEIGHBORHOODS (INTEGRATED)
# ============================================================================

def _get_bairro_description(bairro_name: str) -> str:
    """
    Cria prompt descritivo para cada bairro.
    Usado para gerar embeddings semânticos via Google Generative AI.
    """
    bairro_descriptions = {
        'Aldeota': 'Bairro nobre na zona sul, comércio e residencial de alto padrão, próximo ao Centro',
        'Barroso': 'Zona periférica oeste, residencial popular',
        'Parangaba': 'Centro de Fortaleza, comercial, histórico, movimentado',
        'Meireles': 'Bairro próximo à praia, classe média, turismo',
        'Praia de Iracema': 'Zona turística, comércio, vida noturna',
        # ... continuar com outros bairros
    }
    
    return bairro_descriptions.get(
        bairro_name,
        f'Bairro {bairro_name} em Fortaleza, Ceará, Brasil'
    )
```

### Por quê:

- Precisamos descrever cada bairro para o LLM gerar embedding semântico
- Descrição captura características: localização, tipo residencial, infraestrutura
- LLM então cria embedding 384D baseado na descrição

### Testar:

```python
# No Python REPL ou script:
from src.llm_service import _get_bairro_description

desc = _get_bairro_description('Aldeota')
print(desc)  # Deve imprimir descrição

# Fallback:
desc = _get_bairro_description('BairroDesconhecido')
print(desc)  # Deve imprimir fallback genérico
```

### Acceptance Criteria:

- [ ] Função adicionada ao final de `src/llm_service.py`
- [ ] Retorna string para bairros conhecidos
- [ ] Retorna fallback genérico para desconhecidos
- [ ] Sem erros no import
- [ ] Sem quebras em outras funções de `llm_service.py`

---

## 🔍 Verificar Antes de Começar

### 1. API Keys Funcionando?

```bash
# Testar com função existente
python -c "from src.llm_service import parse_ciops_report; print('LLM OK')"
```

Se der erro: API keys não configuradas. Verificar `.env`:
```
GEMINI_API_KEY=...
```

### 2. Arquivo `src/llm_service.py` Está Acessível?

```bash
# Verificar
ls -la src/llm_service.py
```

Deve existir e ter ~930 linhas (já visto antes)

### 3. Git Status Limpo?

```bash
git status
```

Deve estar clean ou com apenas modificações pequenas. Se houver muitas mudanças:
```bash
git stash
```

---

## ⏱️ Timeline para Esta Semana

### Segunda 03/02 (Hoje):
- [ ] Ler documentação (17 min)
- [ ] Implementar Task 2.1.1 (30 min)
- [ ] Testar função (15 min)
- **Total**: 1h

### Terça 04/02:
- [ ] Task 2.1.2: `_mock_embedding_for_bairro()` (1h)
- [ ] Task 2.1.3: `_mock_semantic_embeddings()` (30 min)
- **Total**: 1.5h

### Quarta 05/02:
- [ ] Task 2.1.4: `get_semantic_embeddings_batch()` (2h) - CRITICAL
- [ ] Testar com 5 bairros (30 min)
- **Total**: 2.5h

### Quinta 06/02:
- [ ] Task 2.2: Integração com `ranking_features.py` (1.5h)
- [ ] Task 2.3: Teste de treino 410D (1h)
- **Total**: 2.5h

### Sexta 07/02:
- [ ] Grid search 410D (2-3h)
- [ ] Eval comparativa (1h)
- [ ] Documentação (1h)
- **Total**: 4-5h

**Semana Total**: ~11-12 horas (realistic estimate)

---

## 🚨 Pontos de Atenção Críticos

### 1. NÃO criar arquivo separado

❌ ERRADO:
```
phase2_embeddings.py (novo arquivo solto)
```

✅ CERTO:
```
src/llm_service.py (adicionar função ao final)
```

### 2. Usar API keys compartilhadas

❌ ERRADO:
```python
my_key = os.environ.get('GEMINI_API_KEY')  # duplica lógica
```

✅ CERTO:
```python
api_keys = get_gemini_api_keys()  # usa função existente
```

### 3. Usar rate limiting

❌ ERRADO:
```python
for bairro in bairros:
    embedding = _call_model_with_rotation(prompt, keys)  # sem delay
```

✅ CERTO:
```python
for bairro in bairros:
    embedding = _call_model_with_rotation(prompt, keys)
    time.sleep(1.5)  # rate limiting
```

### 4. Cache em lugar certo

❌ ERRADO:
```python
cache_file = 'my_cache.json'
```

✅ CERTO:
```python
cache_file = 'data/processed/bairro_embeddings.json'
```

---

## 💡 Dicas Práticas

### Para Debugar:

```python
# Ver o que está em llm_service.py
python -c "import src.llm_service; print(dir(src.llm_service))"

# Ver estrutura de API keys
from src.llm_service import get_gemini_api_keys
keys = get_gemini_api_keys()
print(f'Found {len(keys)} API keys')
```

### Para Testar Integração:

```python
# Verificar que função não quebra resto
from src.llm_service import parse_ciops_report, get_semantic_embeddings_batch
print('Both functions import OK')
```

### Para Cache:

```python
# Verificar se arquivo criou
import os
import json

cache_file = 'data/processed/bairro_embeddings.json'
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        data = json.load(f)
    print(f'Cache tem {len(data)} bairros')
```

---

## 📞 Se der problema...

### Problema: Import Error em llm_service.py

**Solução**: Verificar indentação (Python é sensível)
```bash
python -m py_compile src/llm_service.py
```

### Problema: API Key não funciona

**Solução**: Testar com função existente primeiro
```python
from src.llm_service import parse_ciops_report
# Se isso funcionar, API keys OK
```

### Problema: Rate limit excedido

**Solução**: Aumentar delay entre chamadas
```python
time.sleep(2.0)  # aumentar de 1.5 para 2.0
```

### Problema: Cache gigante

**Solução**: Cache é OK, cada embedding é ~384 floats (~1.5KB)
319 bairros × 1.5KB = ~480KB total (pequeno)

---

## ✅ Checklist Antes de Começar

- [ ] Leu README.md
- [ ] Leu FASE2_INTEGRACAO_SEMANTICA.md
- [ ] Leu FASE2_CHECKLIST.md
- [ ] Verificou que `src/llm_service.py` existe
- [ ] API keys configuradas no `.env`
- [ ] Git status limpo
- [ ] Verificou que `parse_ciops_report` funciona

**Se tudo ✅, está pronto para começar Task 2.1.1**

---

## 🎯 Próximo Passo (AGORA)

1. Abrir `src/llm_service.py`
2. Ir ao final do arquivo
3. Adicionar `_get_bairro_description()` conforme acima
4. Salvar
5. Testar com código acima
6. Commitar: `git commit -m "feat: phase2 - add _get_bairro_description()"`

**Tempo esperado**: 30 minutos

---

## 🚀 Boa Sorte!

Você tem isso. A arquitetura está pronta, documentação clara, integração segura. Agora é "just code it". 

Se tiver dúvidas durante implementação, referência rápida:
- FASE2_INTEGRACAO_SEMANTICA.md → Arquitetura
- FASE2_CHECKLIST.md → Tarefas específicas
- llm_service.py → Exemplos de padrões (api rotation, error handling, logging)

Bora codar! 💻

