# ⚡ QUICK START - ST-GCN Jules Production Setup

**Tempo estimado**: 15-20 minutos  
**Nível de dificuldade**: Iniciante  
**Requisitos**: Python 3.10+, pip, venv  

---

## 🚀 1. Instalação Rápida

```bash
# Clone/navigate to project
cd C:\Users\Boanerges\Desktop\Projetos\st-gcn_Jules

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify models exist
dir models\stgcn_model_v2.pth
dir models\ranking_model_window30_final.pkl
```

**Output esperado**:
```
Volume in drive C is Windows
Directory of C:\Users\Boanerges\Desktop\Projetos\st-gcn_Jules\models

02/03/2026  14:30    200 KB  stgcn_model_v2.pth
02/03/2026  14:35  2.5 MB  ranking_model_window30_final.pkl
```

---

## 🎯 2. Executar Aplicação

### Opção A: Flask Server (Recomendado)

```bash
# Terminal 1: Start Flask
python app.py

# Output esperado:
# [SETUP] Recarregamento periódico ajustado para 60 minutos
# Loaded 319 nodes from JSON
# [DEBUG] node_features shape: (319, 1491, 26)
# [PeriodicReload] Scheduled reload starting...
# WARNING:werkzeug: * Running on http://127.0.0.1:5000
```

### Opção B: Teste Rápido (Demo)

```bash
# Terminal 1: Run demo script (5-10 seg)
python scripts/demo_ranking_validation.py

# Output esperado:
# ===============================================================
# RANKING VALIDATION REPORT
# ===============================================================
# ST-GCN Top-5: [146 244 253 124 152]
# Ranking-Validated Top-5: [146 244 253 124 152]
# Concordance: 100.0%
# Overlap: 5/5 nodes
# Mean score boost: +0.42
# Status: VALIDACAO EM TEMPO DE EXECUCAO FUNCIONANDO ✓
```

---

## 📱 3. Acessar Dashboard

Após iniciar Flask, abra em navegador:

| Recurso | URL | O que faz |
|---------|-----|-----------|
| **Mapa Interativo** | http://127.0.0.1:5000/map | Visualiza 319 áreas com cores (Crítico/Alerta/Monitorado) |
| **Risk Forecast (JSON)** | http://127.0.0.1:5000/api/risk-forecast | Retorna scores de risco para todos os nodes |
| **Top-5 Critical** | http://127.0.0.1:5000/api/rank-top-k | Top-5 áreas críticas (JSON) |
| **Eventos Exógenos** | http://127.0.0.1:5000/api/events | Lista 20+ eventos ativos |

### Exemplo: Top-5 Críticas

```bash
# No terminal 2:
curl http://127.0.0.1:5000/api/rank-top-k

# Resposta (JSON):
{
  "top_5_nodes": [
    {"node_id": 146, "risk_score": 95.2, "status": "CRÍTICO"},
    {"node_id": 244, "risk_score": 92.1, "status": "CRÍTICO"},
    {"node_id": 253, "risk_score": 88.5, "status": "CRÍTICO"},
    {"node_id": 124, "risk_score": 85.3, "status": "ALERTA"},
    {"node_id": 152, "risk_score": 82.1, "status": "ALERTA"}
  ],
  "timestamp": "2026-02-03T15:30:00Z",
  "validation": "100% concordance (ST-GCN + RankingModel)"
}
```

---

## 🎓 4. Entender os Modelos (30 segundos cada)

### ST-GCN v2 (Preditor Principal)

```
O quê?    Rede neural que aprende padrões espaço-temporais
Entrada?  Últimos 30 dias de crime (26D por dia)
Saída?    Score de risco para cada bairro (0-100)
Perda?    Predição de homicídios (CVLI)
P@5?      0.70 (70% do top-5 correto)
Peso?     70% na predição final
```

### RankingModel (Validador em Tempo Real)

```
O quê?    Rede que otimiza ordem de ranking
Entrada?  780D features (30 dias × 26 canais flattened)
Saída?    Scores re-ordenados para ranking
Perda?    PairwiseLoss (otimiza ordem, não absoluto)
P@5?      0.80 (80% do top-5 correto) ⭐
Peso?     30% na predição final
```

### Combinação (70/30)

```
final_score = 0.7 * st_gcn_normalized + 0.3 * ranking_normalized
Resultado:  P@5 = 0.80 com 100% concordância top-5 ✓
```

---

## 📊 5. Features (26 Canais)

| # | Nome | Tipo | Range | Usar para |
|---|------|------|-------|-----------|
| 0 | CVLI (Homicídios) | Count | [0,5+] | **TARGET** (prediz isto) |
| 1 | CVP (Roubos) | Count | [0,20+] | Indicador secundário |
| 2 | Tension Index | Contínuo | [0,1] | Risco combinado |
| 3-9 | Dia da semana | One-hot | {0,1} | Padrão semanal |
| 10-21 | Mês | One-hot | {0,1} | Padrão sazonal |
| 22 | Weekend | Binary | {0,1} | Simplificado |
| 23-25 | Reservados | Zero | [0,0] | Futuro (LLM?) |

---

## ⚙️ 6. Estrutura de Dados Essencial

```
data/
├── processed/
│   ├── processed_graph_data.pkl     ← (319, 1491, 26) histórico completo
│   └── adjacency_matrices/
│       ├── adj_geo.pkl              ← Mapa de vizinhança geográfica
│       └── adj_conflict.pkl         ← Mapa de territórios em disputa
├── exogenous_events.json            ← 20+ eventos críticos
└── raw/
    └── AIS - CAPITAL.geojson        ← Polígonos dos 319 bairros

models/
├── stgcn_model_v2.pth               ← ST-GCN weights (200 KB)
└── ranking_model_window30_final.pkl ← RankingModel + scaler (2.5 MB)
```

---

## 🔧 7. Adicionando Novo Evento Exógeno

### Passo 1: Editar JSON

```bash
# Arquivo: data/exogenous_events.json
# Adicionar antes do último } :

{
  "id": "novo_evento_feb04",
  "lat": -3.7234,              # Latitude do evento
  "lng": -38.4567,             # Longitude do evento
  "date": "2026-02-04T10:00:00Z",
  "natureza": "Confrontação em [bairro]",
  "conflict_severity": "HIGH",  # HIGH/MEDIUM/LOW
  "source": "CIOPS",
  "radius_m": 500              # Raio de afetação (500-1000m)
}
```

### Passo 2: Reinicializar

```bash
# Opção A: Automático (próximo reload em 60 min)
# Opção B: Manual - reiniciar app.py
#   - Ctrl+C para parar
#   - python app.py para reiniciar
```

### Passo 3: Verificar

```bash
curl http://127.0.0.1:5000/api/events

# Procurar por seu evento na resposta JSON
# Bairros próximos ganham amplificação:
#   - HIGH: min 90% criticidade
#   - MEDIUM: min 65% criticidade
```

---

## 🐛 8. Troubleshooting Rápido

| Erro | Causa | Fix |
|------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch não instalado | `pip install torch` |
| `FileNotFoundError: models/stgcn_model_v2.pth` | Modelo faltando | Verificar path, download se necessário |
| `Shape mismatch: (319, 1491, 26)` | Dados obsoletos | `python src/data_processing.py` |
| `Port 5000 already in use` | Flask já rodando | Porta alternativa: `python app.py --port 5001` |
| `API response > 1s` | Computação lenta | Aumentar reload interval (60 → 120 min) |
| `Sem cores no mapa` | Cache JS obsoleto | Ctrl+Shift+R (hard refresh) no navegador |

---

## 📈 9. Verificar Performance

### Comando: Testar Inference

```bash
# Em novo terminal:
python scripts/test_ranking_load.py

# Output esperado:
# Successfully loaded!
# Keys: ['model_state', 'scaler_mean', 'scaler_scale', 'config', 'metrics']
# Config: {'input_dim': 780, 'hidden_dim': 512, 'dropout': 0.2, ...}
# Metrics: {'p5': 0.8, 'epoch': 18}
```

### Verificar Logs

```bash
# App logs (Flask console)
# Procurar por:
# [OK] Ranking model loaded
# [DEBUG] Ranking validator carregado para validação em tempo de execução
# PeriodicReload completed successfully
```

---

## 📚 10. Próximos Passos

### Desenvolvimento
- [ ] Adicionar novo tipo de feature (canal 23+)
- [ ] Treinar novo modelo com dados 2026
- [ ] Implementar visualização de atenção (explainability)

### Monitoramento
- [ ] Configurar alertas (criticidade > 80 por 3 dias)
- [ ] Dashboard de performance (NDCG trend)
- [ ] Exportar relatórios diários (PDF)

### Produção
- [ ] Deploy em servidor (AWS/GCP/Azure)
- [ ] Integrar com sistema de despacho policial
- [ ] Setup CI/CD (GitHub Actions)
- [ ] Backup automático (3x por dia)

---

## 💬 FAQ Rápido

**P: Posso treinar um novo modelo?**  
R: Sim! Rode `python scripts/train_ranking_window30_final.py` (10 min). Salva em models/.

**P: Como atualizar dados com novos crimes?**  
R: `python src/data_processing.py` (reprocessa JSON). PeriodicReload automático (60 min).

**P: Posso desabilitar validação em tempo real?**  
R: Sim, em app.py linha ~775, set `ranking_validator = None`.

**P: Qual é a cobertura geográfica?**  
R: 319 bairros (Fortaleza + interior Ceará). Todos monitorados 24/7.

**P: Com qual frequência atualiza?**  
R: Dados recarregam a cada 60 minutos (configurável). API responde em <200ms.

---

## 📞 Suporte

**Documentation**: Veja README.md (1400+ linhas, muito detalhado!)  
**Config Reference**: Seção "Configurações Ideais em Produção" no README  
**Scripts Available**: `scripts/` (20+ ferramentas)  
**Logs Location**: Console (Flask) ou arquivo se configurado  

**Last Updated**: 03/02/2026 | **Version**: 2.0.0
