# 🔴 Diagnóstico: Erros do Dashboard - Feb 8, 2026

## ERROS IDENTIFICADOS

### 1️⃣ Erro: `POST /api/exogenous/parse` → 400 BAD REQUEST

**Localização**: Frontend em `templates/index.html` linha 1104  
**Severidade**: CRÍTICA (bloqueia upload de eventos exógenos)

#### Root Cause:
O endpoint retorna 400 quando:
```python
# app.py linha 2662
if data is None:
    return jsonify({'error': 'JSON inválido ou cabecalho Content-Type ausente.'}), 400
if missing_city:
    return jsonify({
        'error': 'Falta a cidade na sua ocorrência!',
        'missing_city': missing_city
    }), 400
```

**Debug necessário**: Verificar qual erro está ocorrendo exatamente:
- ❓ JSON inválido?
- ❓ Content-Type ausente?
- ❓ Cidade não identificada pelo parser?

#### Solução:
```javascript
// templates/index.html - Adicionar logging para diagnosticar
$.ajax({
    url: '/api/exogenous/parse',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ text: text }),
    success: function(resp) { /* ... */ },
    error: function(xhr, status, err) {
        // NOVO: Logs detalhados
        console.error('400 Error Details:', {
            status: xhr.status,
            responseText: xhr.responseText,
            response: xhr.responseJSON,
            requestText: text.substring(0, 100)
        });
        alert('❌ Parse Error (400):\n' + 
              (xhr.responseJSON?.error || 'Erro desconhecido'));
    }
});
```

---

### 2️⃣ Erro: `Cannot read properties of null (reading 'getLayers')`

**Localização**: Frontend em `templates/index.html` linha 1450  
**Severidade**: CRÍTICA (crash no dashboard)

#### Root Cause:
```javascript
// templates/index.html:1450
function updateTopCriticalAreas() {
    var layers = geojsonLayer.getLayers();  // ❌ geojsonLayer é NULL aqui!
    // ...
}
```

**O problema**: `updateTopCriticalAreas()` é chamada ANTES de `geojsonLayer` ser carregado:

```
SEQUÊNCIA INCORRETA:
1. updateDashboard() chamado → tenta chamar updateTopCriticalAreas()
2. geojsonLayer NÃO foi carregado yet (ainda não retornou de /api/polygons)
3. geojsonLayer.getLayers() → NULL REFERENCE ERROR!
```

**DIAGRAMA DE TIMING**:
```
Tempo (ms) | Evento
-----------|--------------------------------
0          | Page load inicia
50         | updateDashboard() chamado
100        | updateTopCriticalAreas() → CRASH (geojsonLayer=null)
200        | /api/polygons retorna ✓
250        | geojsonLayer = L.geoJson(...) ✓
300        | updateTopCriticalAreas() finalmente OK
```

#### Solução 1: Null Check (Imediato)
```javascript
function updateTopCriticalAreas() {
    // Proteção contra chamada prematura
    if (!geojsonLayer || !geojsonLayer.getLayers) {
        console.warn('geojsonLayer não carregado ainda, retry em 500ms');
        setTimeout(updateTopCriticalAreas, 500);
        return;
    }
    
    var layers = geojsonLayer.getLayers();
    // ... resto do código
}
```

#### Solução 2: Lazy Initialization (Recomendado)
```javascript
function updateTopCriticalAreas() {
    if (!geojsonLayer) {
        console.debug('updateTopCriticalAreas postponed: aguardando geojsonLayer');
        return; // Silenciosamente ignorar llamada prematura
    }
    
    if (!geojsonLayer.getLayers) {
        console.warn('updateTopCriticalAreas: geojsonLayer não tem método getLayers');
        return;
    }
    
    var layers = geojsonLayer.getLayers();
    // ... resto do código
}
```

---

### 3️⃣ Erro: Parsing JSON/Object Error

**Localização**: Console → "Parsing error Object"  
**Severidade**: MENOR (informativo)

#### Root Cause:
Provavelmente relacionado ao erro 400 acima. Quando o serverretorna 400 com JSON de erro, o frontend tenta fazer parse e falha.

#### Solução:
Já está coberta na Solução 1 acima (logs detalhados).

---

## 🛠️ IMPLEMENTAÇÃO DOS FIXES

### Fix #1: Adicionar null check em `updateTopCriticalAreas`

**Arquivo**: `templates/index.html` linha 1450

```javascript
// ANTES:
function updateTopCriticalAreas() {
    var layers = geojsonLayer.getLayers();
    ...
}

// DEPOIS:
function updateTopCriticalAreas() {
    // Proteção: geojsonLayer pode não estar carregado (race condition)
    if (!geojsonLayer) {
        console.debug('[updateTopCriticalAreas] geojsonLayer ainda não carregado, aguardando...');
        return;
    }
    
    try {
        var layers = geojsonLayer.getLayers();
        if (!layers || layers.length === 0) {
            console.debug('[updateTopCriticalAreas] Nenhuma camada disponível ainda');
            return;
        }
    } catch (e) {
        console.warn('[updateTopCriticalAreas] Erro ao obter camadas:', e);
        return;
    }
    
    // ... resto do código continua normal
}
```

### Fix #2: Melhorar logs em `/api/exogenous/parse`

**Arquivo**: `templates/index.html` linha 1104

```javascript
// ANTES:
$.ajax({
    url: '/api/exogenous/parse',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ text: text }),
    success: function(resp) { ... },
    error: function(err) {
        console.error("Simulation error", err);
        alert("Erro ao simular: " + err.statusText);
    }
});

// DEPOIS:
$.ajax({
    url: '/api/exogenous/parse',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ text: text }),
    success: function(resp) {
        console.log('[Parse Success]', resp);
        // ... resto do success
    },
    error: function(xhr, status, err) {
        var errorMsg = 'Erro desconhecido';
        var errorDetails = {};
        
        try {
            errorDetails = xhr.responseJSON;
            errorMsg = xhr.responseJSON.error || xhr.responseJSON.message || 'Erro no parse';
        } catch(e) {
            errorMsg = xhr.statusText || status || 'Erro ao processar';
        }
        
        console.error('[Parse Error]', {
            status: xhr.status,
            statusText: xhr.statusText,
            error: errorMsg,
            details: errorDetails,
            responseText: xhr.responseText.substring(0, 200)
        });
        
        showLoading();
        alert('❌ Erro ao geoposicionar (HTTP ' + xhr.status + '):\n' + errorMsg);
        hideLoading();
    }
});
```

### Fix #3: Garantir carregamento sequencial

**Arquivo**: `templates/index.html` linha ~700 (fim do `.done()` callback)

```javascript
// NO FINAL DO $.when().done() CALLBACK, ADICIONAR:
$.when(
    $.getJSON('/api/polygons'),
    $.getJSON('/api/risk')
).done(function(polygonsArgs, riskArgs) {
    // ... código existente de inicialização do geojsonLayer ...
    
    // Adicionar FINAL DO CALLBACK:
    geojsonLayer.addTo(map);
    
    // ✅ NOVO: Garantir que updateTopCriticalAreas seja chamado APÓS geojsonLayer estar ready
    setTimeout(function() {
        console.log('[Init] geojsonLayer carregado, chamando updateTopCriticalAreas');
        updateTopCriticalAreas();
    }, 100);
    
}).fail(function(err1, err2) {
    console.error("[INIT] Erro ao carregar dados iniciais:", err1, err2);
    alert("❌ Erro ao carregar polígonos ou dados de risco.");
});
```

---

## 📊 IMPACTO DOS FIXES

| Fix | Erro | Severidade | Impacto |
|:--|:--|:--:|:--|
| #1 | getLayers null | CRÍTICA | ❌ Crash → ✅ Recuperação automática |
| #2 | 400 Bad Request | CRÍTICA | ❌ Sem feedback → ✅ Logs detalhados |
| #3 | Race condition | ALTA | ❌ Inconsistência → ✅ Sequencial garantido |

---

## ✅ CHECKLIST

- [ ] Aplicar Fix #1 em `updateTopCriticalAreas`
- [ ] Aplicar Fix #2 em `/api/exogenous/parse` error handler
- [ ] Aplicar Fix #3 em carregamento inicial
- [ ] Testar novo upload de eventos exógenos
- [ ] Monitorar console para logs detalhados
- [ ] Confirmar que `updateTopCriticalAreas` é chamado após geojsonLayer

---

## 🔍 COMO TESTAR

1. **Abrir DevTools** (F12)
2. **Ir para Console**
3. **Tentar upload de evento**:
   - Se Fix #1 funciona: verá `[updateTopCriticalAreas] geojsonLayer ainda não carregado` (temporário)
4. **Se erro 400**: verá detalhes em `[Parse Error]`
5. **Após ~500ms**: dashboard atualiza corretamente

---

*Relatório: verify_overfitting_and_recommendations.py*  
*Data: 8 Feb 2026*
