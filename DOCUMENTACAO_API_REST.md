# 📡 API REST - REPORT PREVIEW

## Visão Geral

A **API REST** do REPORT PREVIEW fornece acesso programático a todas as funcionalidades do sistema de predição de risco. Construída com **Flask**, segue padrões RESTful e retorna respostas em **JSON**.

**Base URL:** `http://localhost:5050` (ou configurável em produção)

---

## 1. Endpoints Principais

### 🗺️ **1.1. Obter Mapa de Risco**
Retorna scores de risco para todos os bairros do estado.

```http
GET /api/risk
```

**Resposta (200 OK):**
```json
{
  "timestamp": "2026-03-01T19:08:36.955Z",
  "temperature_state": 62.5,
  "confidence": 0.87,
  "regions": {
    "fortaleza": {
      "name": "FORTALEZA (CAPITAL)",
      "avg_risk": 58.3,
      "neighborhoods": [
        {
          "id": 1,
          "name": "BOM JARDIM",
          "risk_score": 85.2,
          "risk_level": "ALTO",
          "coordinates": [-38.5432, -3.7456],
          "status": "monitored",
          "confidence_interval": 0.89
        },
        {
          "id": 2,
          "name": "PIRAMBU",
          "risk_score": 45.1,
          "risk_level": "MODERADO",
          "coordinates": [-38.5123, -3.7123]
        }
      ]
    },
    "rmf": {
      "name": "REGIÃO METROPOLITANA",
      "avg_risk": 42.1,
      "neighborhoods": [...]
    },
    "interior": {
      "name": "INTERIOR DO ESTADO",
      "avg_risk": 35.7,
      "neighborhoods": [...]
    }
  }
}
```

**Status Codes:**
- `200` - Sucesso
- `500` - Erro no cálculo de risco

---

### 📊 **1.2. Simular Cenário (What-If Analysis)**
Projeta o impacto de ações policiais ou eventos críticos no mapa de risco.

```http
POST /api/simulate
Content-Type: application/json

{
  "action_type": "suppression",
  "location_id": 1,
  "teams_deployed": 5,
  "hours_duration": 12
}
```

**Parâmetros:**

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-----------|-----------|
| `action_type` | string | ✅ | `"suppression"` ou `"conflict"` |
| `location_id` | integer | ✅ | ID do bairro alvo |
| `teams_deployed` | integer | ❌ | Número de equipes (para suppression) |
| `hours_duration` | integer | ❌ | Duração esperada em horas (padrão: 12) |
| `event_nature` | string | ❌ | Tipo de conflito (para conflict) |

**Resposta (200 OK):**
```json
{
  "simulation_id": "sim_20260301_001",
  "scenario": "suppression",
  "duration_hours": 12,
  "projected_changes": {
    "location_id": 1,
    "original_risk": 85.2,
    "projected_risk": 62.3,
    "risk_reduction": 22.9,
    "confidence": 0.78
  },
  "regional_impact": {
    "fortaleza": {
      "avg_risk_change": -8.5
    },
    "propagation": {
      "neighboring_zones": [2, 5, 8],
      "secondary_effect": -3.2
    }
  },
  "recommended_timing": "immediate",
  "notes": "Ação de supressão reduz risco em 22.9%. Efeito cascata em zonas vizinhas: -3.2%"
}
```

---

### 🔍 **1.3. Explicar Risco de um Bairro**
Gera justificativa em linguagem natural para o score de risco.

```http
GET /api/explain/<node_id>
```

**Exemplo:**
```http
GET /api/explain/1
```

**Resposta (200 OK):**
```json
{
  "node_id": 1,
  "neighborhood_name": "BOM JARDIM",
  "current_risk": 85.2,
  "explanation": {
    "primary_factors": [
      {
        "factor": "histórico",
        "weight": 0.45,
        "description": "45% do risco vem de padrão histórico elevado nos últimos 120 dias"
      },
      {
        "factor": "contágio_espacial",
        "weight": 0.35,
        "description": "35% de 'sombra de risco' de bairros vizinhos (PIRAMBU, PARQUE MANIBURA)"
      },
      {
        "factor": "evento_recente",
        "weight": 0.20,
        "description": "20% por evento exógeno crítico (Canal 25) há 4 horas: homicídio no bairro"
      }
    ],
    "recent_events": [
      {
        "timestamp": "2026-03-01T15:08:36Z",
        "type": "homicídio",
        "channel": 25,
        "impact": "+18.5%"
      }
    ],
    "neighboring_threat_propagation": [
      {
        "neighbor_id": 2,
        "neighbor_name": "PIRAMBU",
        "influence": "+5.2%"
      }
    ]
  }
}
```

---

### 💾 **1.4. Processar Evento Exógeno (Ingestão de Dados)**
Interpreta texto de evento policial via LLM e atualiza risco em tempo real.

```http
POST /api/exogenous/parse
Content-Type: application/json

{
  "text": "AÇÃO POLICIAL em Bom Jardim: Prisão qualificada, apreensão de 2 fuzis AK-47, munição. 14:30h",
  "source": "CIOPS",
  "manual_override": false
}
```

**Resposta (200 OK):**
```json
{
  "status": "processed",
  "event_id": "evt_20260301_0042",
  "extracted_data": {
    "date": "2026-03-01",
    "time": "14:30",
    "neighborhood": "BOM JARDIM",
    "event_nature": "Ação Policial - Prisão Qualificada",
    "severity": "high",
    "channel": 23,
    "extracted_items": [
      {
        "type": "weapons",
        "count": 2,
        "description": "fuzis AK-47",
        "weight_modifier": 2.5
      },
      {
        "type": "arrest",
        "severity": "qualified",
        "weight_modifier": 1.2
      }
    ]
  },
  "impact": {
    "affected_locations": [1],
    "risk_changes": [
      {
        "location_id": 1,
        "previous_risk": 85.2,
        "new_risk": 71.3,
        "change": -13.9,
        "effect_type": "suppression"
      }
    ],
    "propagation": {
      "neighboring_impact": [-2.1, -1.8],
      "timestamp": "2026-03-01T14:30:00Z"
    }
  },
  "validation": {
    "date_consistency": true,
    "data_quality": 0.96,
    "warnings": []
  }
}
```

**Status Codes:**
- `200` - Evento processado com sucesso
- `400` - Erro de validação (ex: texto vazio, data inválida)
- `422` - Evento rejeitado (data no futuro, inconsistência temporal)

---

### 💾 **1.5. Salvar Evento Exógeno (Modo Manual)**
Salva evento estruturado manualmente sem passar por LLM.

```http
POST /api/exogenous/save
Content-Type: application/json

{
  "date": "2026-03-01",
  "time": "14:30",
  "neighborhood": "BOM JARDIM",
  "event_nature": "Ação Policial - Prisão",
  "channel": 23,
  "details": {
    "weapons_seized": ["fuzil AK-47", "revólver .38"],
    "arrests": 3
  }
}
```

**Resposta (200 OK):**
```json
{
  "status": "saved",
  "event_id": "evt_20260301_0043",
  "timestamp": "2026-03-01T14:30:00Z",
  "risk_impact": {
    "affected_neighborhoods": ["BOM JARDIM"],
    "avg_risk_change": -13.9
  }
}
```

---

### 📍 **1.6. Obter Polígonos Geoespaciais (GeoJSON)**
Retorna geometrias dos bairros para visualização em mapas interativos.

```http
GET /api/polygons
```

**Resposta (200 OK):**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 1,
        "name": "BOM JARDIM",
        "region": "fortaleza",
        "risk_score": 85.2,
        "risk_level": "ALTO"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-38.5432, -3.7456], ...]]
      }
    }
  ]
}
```

---

### 📊 **1.7. Status do Modelo (Health Check)**
Verifica disponibilidade e estado do sistema.

```http
GET /api/model-update-status
```

**Resposta (200 OK):**
```json
{
  "status": "idle",
  "model_version": "2.0",
  "last_update": "2026-03-01T10:30:00Z",
  "training_instances": "fortaleza, rmf, interior",
  "confidence_global": 0.87,
  "data_freshness": {
    "exogenous_events": "2 horas atrás",
    "historical_baseline": "120 dias"
  }
}
```

---

### 🔔 **1.8. Status de Anomalias**
Detecta padrões anormais no risco.

```http
GET /api/anomaly_status
```

**Resposta (200 OK):**
```json
{
  "anomalies_detected": 3,
  "global_anomaly_score": 0.32,
  "regions": {
    "fortaleza": {
      "anomaly_score": 0.45,
      "anomalies": [
        {
          "location_id": 1,
          "type": "sudden_spike",
          "magnitude": 0.8,
          "timestamp": "2026-03-01T18:00:00Z",
          "cause": "Evento crítico detectado"
        }
      ]
    }
  }
}
```

---

### 📋 **1.9. Listar Eventos Exógenos Recentes**
Retorna histórico de eventos processados.

```http
GET /api/exogenous-events?limit=20&offset=0
```

**Query Parameters:**

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `limit` | integer | 20 | Número máximo de eventos a retornar |
| `offset` | integer | 0 | Índice de início para paginação |
| `region` | string | null | Filtro por região (fortaleza, rmf, interior) |
| `date_from` | string | null | Filtro por data inicial (YYYY-MM-DD) |
| `date_to` | string | null | Filtro por data final (YYYY-MM-DD) |

**Resposta (200 OK):**
```json
{
  "total": 245,
  "returned": 20,
  "offset": 0,
  "events": [
    {
      "event_id": "evt_20260301_0043",
      "date": "2026-03-01",
      "time": "14:30",
      "neighborhood": "BOM JARDIM",
      "event_nature": "Ação Policial - Prisão",
      "channel": 23,
      "impact": -13.9,
      "ingested_at": "2026-03-01T14:35:00Z"
    }
  ]
}
```

---

### 📊 **1.10. Última Avaliação de Eficiência (Backtesting)**
Retorna métrica P10/P20 mais recente do Monitor de Eficiência.

```http
GET /api/efficiency-latest
```

**Resposta (200 OK):**
```json
{
  "evaluation_date": "2026-02-22",
  "global": {
    "p5": 0.92,
    "p10": 0.88,
    "p20": 0.91,
    "confidence": 0.89
  },
  "regions": {
    "fortaleza": {
      "p10": 0.85,
      "p20": 0.89,
      "hits10": ["BOM JARDIM", "PIRAMBU"],
      "hits20": ["BOM JARDIM", "PIRAMBU", "PARQUE MANIBURA", "CONJUNTO ESPERANÇA"]
    },
    "rmf": {
      "p10": 0.80,
      "p20": 0.84
    },
    "interior": {
      "p10": 0.75,
      "p20": 0.78
    }
  }
}
```

---

## 2. Cache de Explicações do Gestor

Otimiza performance mantendo explicações geradas em cache.

### 2.1. Obter Cache Completo

```http
GET /api/manager_explanations/cache
```

**Resposta:**
```json
{
  "cache_size": 127,
  "last_updated": "2026-03-01T19:00:00Z",
  "cached_nodes": {
    "1": {
      "neighborhood": "BOM JARDIM",
      "explanation": "...",
      "timestamp": "2026-03-01T18:30:00Z"
    }
  }
}
```

### 2.2. Deletar um Node do Cache

```http
DELETE /api/manager_explanations/cache/1
```

### 2.3. Limpar Cache Completo

```http
POST /api/manager_explanations/cache/clear
```

---

## 3. Tratamento de Erros

Todos os erros retornam um objeto JSON estruturado:

```json
{
  "error": true,
  "code": "VALIDATION_ERROR",
  "message": "Campo obrigatório 'neighborhood' não fornecido",
  "details": {
    "missing_fields": ["neighborhood"],
    "received": ["date", "time"]
  },
  "timestamp": "2026-03-01T19:08:36.955Z"
}
```

### Códigos de Erro Comuns

| Código | Status HTTP | Descrição |
|--------|-----------|-----------|
| `VALIDATION_ERROR` | 400 | Parâmetros inválidos ou ausentes |
| `DATE_ERROR` | 422 | Data no futuro ou inconsistente |
| `LLM_ERROR` | 503 | Falha ao processar com Gemini |
| `MODEL_ERROR` | 500 | Erro no cálculo do modelo ST-GAT |
| `NOT_FOUND` | 404 | Recurso não encontrado |

---

## 4. Exemplo de Fluxo Completo

### 1️⃣ Obter Estado Atual
```bash
curl http://localhost:5050/api/risk | jq '.regions.fortaleza.neighborhoods[0]'
```

### 2️⃣ Processar Evento
```bash
curl -X POST http://localhost:5050/api/exogenous/parse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Homicídio em Bom Jardim, 22:15h",
    "source": "CIOPS"
  }' | jq '.impact.risk_changes[0]'
```

### 3️⃣ Simular Resposta
```bash
curl -X POST http://localhost:5050/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "suppression",
    "location_id": 1,
    "teams_deployed": 8,
    "hours_duration": 24
  }' | jq '.projected_changes'
```

### 4️⃣ Explicar Resultado
```bash
curl http://localhost:5050/api/explain/1 | jq '.explanation.primary_factors'
```

---

## 5. Autenticação e Segurança

*(Futuro)*

Para produção, implementar:
- **JWT Bearer Tokens** para autenticação
- **Rate Limiting:** 100 req/min por IP
- **HTTPS/TLS** para comunicação segura
- **RBAC:** Roles distintos para gestores, analistas, admins

**Header de Autenticação (futuro):**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 6. Rate Limiting (Futuro)

| Endpoint | Limite |
|----------|--------|
| `/api/risk` | 10 req/min |
| `/api/simulate` | 5 req/min |
| `/api/exogenous/parse` | 30 req/min |
| Outros | 100 req/min |

Respostas aceleradas retornarão:
```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

---

## 7. Changelog da API

| Versão | Data | Mudanças |
|--------|------|----------|
| 2.0 | 2026-03-01 | Adicionado `/api/efficiency-latest`, aperfeiçoado `/api/explain` |
| 1.5 | 2026-02-15 | Implementado cache de explicações |
| 1.0 | 2026-01-01 | Versão inicial |

---

**Última atualização:** 01 de Março de 2026
